from abc import ABCMeta
from datetime import datetime
from typing import Sequence, Mapping

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from hhutil.io import fmt_path

from horch.common import CUDA, convert_tensor
from horch.train.base import StatefulList, Serializable
from horch.train.metric_history import MetricHistory
from horch.train.v3.callbacks import config_callbacks


def find_most_recent(work_dir, pattern):
    d = fmt_path(work_dir)
    pattern = pattern
    saves = list(d.glob(pattern))
    if len(saves) == 0:
        raise FileNotFoundError("No checkpoint to load in %s" % work_dir)
    fp = max(saves, key=lambda f: f.stat().st_mtime)
    return fp


class Learner(Serializable, metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers, lr_schedulers, train_metrics, eval_metrics, work_dir,
                 fp16=False, device='auto', **kwargs):
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        optimizers = list(optimizers)
        if not isinstance(lr_schedulers, Sequence):
            lr_schedulers = [lr_schedulers]
        if device == 'auto':
            device = 'cuda' if CUDA else 'cpu'
        device = torch.device(device)
        work_dir = fmt_path(work_dir)
        model.to(device)
        if isinstance(criterion, nn.Module):
            criterion.to(device)

        if fp16:
            from apex import amp
            model, optimizers = amp.initialize(model, optimizers, opt_level="O1", verbosity=0)

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.work_dir = work_dir
        self.fp16 = fp16
        self.device = device

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._log_dir = self.work_dir / "runs"
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self._writer = SummaryWriter(str(self._log_dir / current_time), flush_secs=10)

        self._verbose = True
        self._state = {
            "train": {},
            "eval": {},
            "test": {},
        }

        # epoch -> stage -> metric -> value
        self.metric_history = MetricHistory(["train", "eval", "test"])

    def train_batch(self, batch):
        pass

    def eval_batch(self, batch):
        pass

    def test_batch(self, batch):
        pass

    def state_dict(self):
        return self._state

    def to_save(self) -> Mapping[str, Serializable]:
        d = {'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers),
             "metric_history": self.metric_history,
             "learner": self}
        if self.fp16:
            from apex import amp
            d['amp'] = amp
        return d

    def save(self):
        path = str(self.work_dir / f"epoch_{self._state['train']['epoch'] + 1}.pth")
        state_dict = {}
        for k, v in self.to_save().items():
            state_dict[k] = v.state_dict()
        torch.save(state_dict, path)
        print('Save trainer to %s' % path)

    def load(self, fp=None):

        fp = fp or find_most_recent(self.work_dir, "epoch_*.pth")
        state_dict = torch.load(fp)

        if not self.fp16 and 'amp' in state_dict:
            del state_dict['amp']

        for k, v in self.to_save().items():
            v.load_state_dict(state_dict[k])

        print("Load trainer from %s" % fp)

    def load_state_dict(self, state_dict):
        self._state = state_dict

    def set_global_state(self, k, v):
        modes = ['train', 'eval', 'test']
        for m in modes:
            self._state[m][k] = v

    def fit(self, train_loader, epochs, val_loader=None, val_freq=1,
            save_freq=None, callbacks=None):

        do_eval = val_loader is not None

        cbks = config_callbacks(
            self,
            callbacks,
            save_freq=save_freq,
        )
        start_epoch = self._state['train'].get('epoch', 0)
        epochs = epochs + start_epoch
        self.set_global_state("epochs", epochs)

        state = self._state['train']
        state['metrics'] = {}
        cbks.begin_train(state)
        for epoch in range(start_epoch, epochs):
            self.set_global_state("epoch", epoch)
            cbks.begin_epoch(state)
            self._run_one_epoch(train_loader, cbks, 'train')
            cbks.after_epoch(state)

            if do_eval and (epoch + 1) % val_freq == 0:
                cbks.begin_eval(self._state['eval'])
                self._state['eval']['metrics'] = {}
                self._run_one_epoch(val_loader, cbks, 'eval')
                cbks.after_eval(self._state['eval'])

        cbks.after_train(state)

    def _run_one_epoch(self, data_loader, callbacks, mode):
        outputs = []
        state = self._state[mode]
        steps = len(data_loader)
        state.update({
            'steps': steps,
        })
        for metric in getattr(self, mode + '_metrics').values():
            metric.reset()
        for step, batch in enumerate(data_loader):
            state.update({
                "step": step,
                "batch": batch,
            })

            callbacks.begin_batch(state)
            if mode in ['train', 'eval']:
                getattr(self, mode + '_batch')(batch)
            else:
                pred = self.test_batch(batch)
                outputs.append(pred)

            for metric in getattr(self, mode + '_metrics').values():
                metric.update(metric._output_transform(state))

            callbacks.after_batch(state)

        for name, metric in getattr(self, mode + '_metrics').items():
            state['metrics'][name] = metric.compute()

        if mode == 'test':
            return outputs

    def evaluate(self, val_loader, callbacks=None):

        cbks = config_callbacks(self, callbacks, mode='eval')
        cbks.begin_eval(self._state['eval'])
        self._state['eval']['metrics'] = {}
        self._run_one_epoch(val_loader, cbks, 'eval')
        cbks.after_eval(self._state['eval'])

    def predict(self, test_loader):
        pass


def requires_grad(network: nn.Module, arch: bool, model: bool):
    for p in network.arch_parameters():
        p.requires_grad_(arch)
    for p in network.model_parameters():
        p.requires_grad_(model)


def backward(loss, optimizer, fp16):
    if fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return
