from collections import defaultdict, OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict, Callable, Union, Optional, Any, List, Mapping

from toolz.curried import curry

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from horch.common import CUDA
from horch.io import fmt_path
from horch.train.v2.base import StatefulList, Serializable
from horch.train.v2.callback import Events, Checkpoint
from horch.train.v2.engine import Engine
from horch.train.v2.metrics import Metric
from horch.utils import time_now


class Epochs:

    def __init__(self, n: int):
        self.n = n


class Iters:

    def __init__(self, n: int):
        self.n = n


class MetricHistory(Serializable):

    def __init__(self, stages: Sequence[str]):
        self.stages = stages
        # stage -> epoch -> metric -> value
        self._history = {
            stage: {}
            for stage in stages
        }

    def state_dict(self) -> OrderedDict:
        return OrderedDict(deepcopy(self._history))

    def load_state_dict(self, state_dict: Mapping):
        for k in self._history:
            self._history[k] = deepcopy(state_dict[k])

    def record(self, stage, epoch, metric, value):
        h = self._history[stage]
        if epoch not in h:
            h[epoch] = {}
        h[epoch][metric] = value

    def get_metric(self, metric, stage=None, start=None, end=None):
        if stage is None:
            return {
                stage: self.get_metric(metric, stage, start, end)
                for stage in self.stages
            }
        else:
            h = self._history[stage]
            epochs = list(h.keys())
            min_epoch, max_epochs = min(epochs), max(epochs)
            if start is None:
                start = min_epoch
            if end is None:
                end = max_epochs
            values = [h[e].get(metric) for e in range(start, end + 1)]
            if all(v is None for v in values):
                return None
            elif len(values) == 1:
                return values[0]
            else:
                return values

    def get_epochs(self, start, end, stage=None):
        if stage is None:
            h = {
                stage: self.get_epochs(start, end, stage)
                for stage in self.stages
            }
            for k in h.keys():
                if h[k] is None:
                    del h[k]
            return h
        else:
            h = self._history[stage]
            metrics = set([m for e in range(start, end + 1) for m in h[e].keys()])
            return {
                m: self.get_metric(m, stage, start, end)
                for m in metrics
            }


class TrainerBase:

    def __init__(self,
                 model: nn.Module,
                 criterion: Union[nn.Module, Callable],
                 optimizers: Union[Optimizer, Sequence[Optimizer]],
                 lr_schedulers: Union[_LRScheduler, Sequence[_LRScheduler]],
                 metrics: Mapping[str, Metric],
                 test_metrics: Mapping[str, Metric],
                 work_dir: Union[Path, str] = ".",
                 fp16: bool = False,
                 lr_step_on_iter: bool = False,
                 device: Union[str, torch.device] = 'auto',
                 **kwargs):

        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
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
            model, optimizer = amp.initialize(model, optimizers, opt_level="O1", verbosity=0)

        # Set Arguments

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.metrics = metrics
        self.test_metrics = test_metrics
        self.work_dir = work_dir
        self.fp16 = fp16
        self.lr_step_on_iter = lr_step_on_iter
        self.device = device

        self.log_dir = self.work_dir / "runs"
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(str(self.log_dir / current_time), flush_secs=10)

        self.train_engine = None
        self._train_engine_state = None

        self._epochs = 0
        self._kwargs = kwargs
        self._verbose = True

        # epoch -> stage -> metric -> value
        self.metric_history = MetricHistory(["train", "valid", "test"])

    def to_save(self) -> Mapping[str, Serializable]:
        d = {'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers),
             "train_engine": lambda: self._train_engine_state,
             "metric_history": self.metric_history}
        if self.fp16:
            from apex import amp
            d['amp'] = amp
        return d

    def save(self):
        path = str(self.work_dir / f"epoch_{self._epochs + 1}.pth")
        state_dict = {}
        for k, v in self.to_save().items():
            if hasattr(v, "state_dict"):
                state_dict[k] = v.state_dict()
            else:
                state_dict[k] = v()
        torch.save(state_dict, path)
        print('Save trainer to %s' % path)

    def load(self, fp=None):

        if fp is None:
            d = Path(self.work_dir)
            pattern = "epoch_*.pt*"
            saves = list(d.glob(pattern))
            if len(saves) == 0:
                raise FileNotFoundError("No checkpoint to load in %s" % self.work_dir)
            fp = max(saves, key=lambda f: f.stat().st_mtime)

        state_dict = torch.load(fp)

        if not self.fp16 and 'amp' in state_dict:
            del state_dict['amp']

        for k, v in self.to_save().items():
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(state_dict[k])
        self._train_engine_state = state_dict["train_engine"]
        print("Load trainer from %s" % fp)

    def _create_train_engine(self) -> Engine:
        raise NotImplementedError

    def _create_eval_engine(self) -> Engine:
        raise NotImplementedError

    def log(self, s):
        if self._verbose:
            print(s)

    @curry
    def _log_epoch_start(self, engine: Engine):
        lrs = "".join(", lr %f" % lr_scheduler.get_last_lr()[0] for lr_scheduler in self.lr_schedulers)
        self.log("Epoch %d%s" % (engine.state.epoch + 1, lrs))

    @curry
    def _lr_scheduler_step(self, engine: Engine):
        iteration = engine.state.iteration
        iters_per_epoch = engine.state.epoch_length
        iterations = engine.state.epoch * iters_per_epoch + iteration
        for lr_scheduler in self.lr_schedulers:
            steps = iterations if self.lr_step_on_iter else iterations / iters_per_epoch
            lr_scheduler.step(steps)

    def _set_epochs(self, engine):
        self._epochs = engine.state.epoch

    @curry
    def log_metrics(self, engine: Engine, writer: Optional[SummaryWriter], metric_history: MetricHistory, stage: str):
        log_str = "%s %s - " % (time_now(), stage)
        metric_logs = []
        for k, v in engine.state.metrics.items():
            metric_logs.append("%s: %.4f" % (k, v))
            if writer:
                writer.add_scalar("%s/%s" % (k, stage), v, self._epochs + 1)
            metric_history.record(stage, self._epochs + 1, k, v)
        log_str += ", ".join(metric_logs)
        self.log(log_str)

    def fit(self,
            train_loader: DataLoader,
            epochs: Optional[int],
            val_loader: Optional[DataLoader] = None,
            eval_freq: Optional[int] = 1,
            save_by: Optional[Union[str, int, Epochs, Iters]] = None,
            callbacks: Sequence[Callable] = ()):

        self.train_engine = self._create_train_engine()
        eval_engine = self._create_eval_engine()
        if self._train_engine_state:
            self.train_engine.load_state_dict(self._train_engine_state)

        self.train_engine.call_on(
            Events.EPOCH_BEGIN, self._log_epoch_start),
        self.train_engine.call_on(
            Events.BATCH_END, self._lr_scheduler_step),
        self.train_engine.call_on(
            Events.EPOCH_END, self._set_epochs),
        self.train_engine.call_on(
            Events.EPOCH_END, self.log_metrics(
                writer=self.writer, metric_history=self.metric_history, stage='train'))

        if save_by:
            checkpoint = Checkpoint(self.to_save(), save_freq=1, save_dir=self.work_dir, n_saved=1)
            self.train_engine.call(checkpoint)

        validate = val_loader is not None and eval_freq
        if validate:
            self.train_engine.call_on(
                Events.EPOCH_END, lambda _: eval_engine.run(val_loader, 1), eval_freq)
            eval_engine.call_on(
                Events.EPOCH_END, self.log_metrics(writer=self.writer, stage='valid', metric_history=self.metric_history))

        for callback in callbacks:
            self.train_engine.call_on(
                Events.BATCH_BEGIN, callback)

        try:
            start_epoch, end_epoch = self.train_engine.run(train_loader, epochs)
            self._train_engine_state = self.train_engine.state_dict()
            return self.metric_history.get_epochs(start_epoch + 1, end_epoch, "valid" if validate else "train")
        except KeyboardInterrupt as e:
            self._train_engine_state = self.train_engine.state_dict()
            raise e

    def evaluate(self, val_loader):
        eval_engine = self._create_eval_engine()
        eval_engine.call_on(
            Events.EPOCH_END, self.log_metrics(writer=None, stage='test', metric_history=self.metric_history))
        eval_engine.run(val_loader, 1)
        return self.metric_history.get_epochs(self._epochs + 1, self._epochs + 1, "test")
