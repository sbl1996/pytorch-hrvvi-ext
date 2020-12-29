from abc import ABCMeta
from datetime import datetime
from typing import Sequence, Mapping

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from hhutil.io import fmt_path, time_now

from horch.common import CUDA
from horch.train.base import StatefulList, Serializable
from horch.train.metric_history import MetricHistory
from horch.train.callbacks import config_callbacks


def find_most_recent(work_dir, pattern):
    d = fmt_path(work_dir)
    saves = list(d.glob(pattern))
    if len(saves) == 0:
        raise FileNotFoundError("No checkpoint to load in %s" % work_dir)
    fp = max(saves, key=lambda f: f.stat().st_mtime)
    return fp


class Learner(Serializable, metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers, lr_schedulers, train_metrics, eval_metrics, work_dir,
                 fp16=False, device='auto', grad_clip_norm=0.0, optimized_execution=False):
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
            self.scaler = torch.cuda.amp.GradScaler()

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.work_dir = work_dir
        self.fp16 = fp16
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.optimized_execution = optimized_execution
        if self.optimized_execution:
            self.model = torch.jit.script(self.model)

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

        self._terminated = False

    def train_batch(self, batch):
        raise NotImplementedError

    def eval_batch(self, batch):
        raise NotImplementedError

    def test_batch(self, batch):
        raise NotImplementedError

    def state_dict(self):
        return self._state

    def to_save(self) -> Mapping[str, Serializable]:
        d = {'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers),
             "metric_history": self.metric_history,
             "learner": self}
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
        if do_eval:
            self._val_freq = val_freq

        cbks = config_callbacks(
            self,
            callbacks,
            save_freq=save_freq,
        )
        start_epoch = self._state['train'].get('epoch', 0)
        epochs = epochs + start_epoch
        self.set_global_state("epochs", epochs)

        print("%s Start training" % (time_now(),))

        state = self._state['train']
        state['metrics'] = {}
        cbks.begin_train(state)
        for epoch in range(start_epoch, epochs):
            self.set_global_state("epoch", epoch)
            cbks.begin_epoch(state)
            self._run_one_epoch(train_loader, cbks, 'train')
            cbks.after_epoch(state)

            if do_eval and (epoch + 1) % self._val_freq == 0:
                cbks.begin_eval(self._state['eval'])
                self._state['eval']['metrics'] = {}
                self._run_one_epoch(val_loader, cbks, 'eval')
                cbks.after_eval(self._state['eval'])

            if self._terminated:
                print("Terminated at epoch %d" % epochs)
                break
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
            if mode == 'train':
                self.train_batch(batch)
                if self.fp16:
                    self.scaler.update()
            elif mode == 'eval':
                self.eval_batch(batch)
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


def forward(learner: Learner, *args):
    if learner.optimized_execution:
        with torch.jit.optimized_execution(True):
            outputs = learner.model(*args)
    else:
        outputs = learner.model(*args)
    return outputs


def backward(learner: Learner, loss):
    if learner.fp16:
        scaler = learner.scaler
        loss = scaler.scale(loss)
    loss.backward()


def optimizer_step(learner: Learner, optimizer, grad_clip_params=None):
    if learner.fp16:
        scaler = learner.scaler
        if learner.grad_clip_norm and grad_clip_params:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(grad_clip_params, learner.grad_clip_norm)
        scaler.step(optimizer)
    else:
        if learner.grad_clip_norm and grad_clip_params:
            nn.utils.clip_grad_norm_(grad_clip_params, learner.grad_clip_norm)
        optimizer.step()
