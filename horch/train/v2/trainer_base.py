from datetime import datetime, timezone, timedelta
from enum import Enum
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


class Epochs:

    def __init__(self, n: int):
        self.n = n


class Iters:

    def __init__(self, n: int):
        self.n = n


class TrainerBase:

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizers: Union[Optimizer, Sequence[Optimizer]],
                 lr_schedulers: Union[_LRScheduler, Sequence[_LRScheduler]],
                 metrics: List[Metric],
                 test_metrics: List[Metric],
                 save_path: Union[Path, str] = ".",
                 fp16: bool = False,
                 lr_step_on_iter: bool = False,
                 device: Optional[str, torch.device] = None,
                 **kwargs):

        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        if not isinstance(lr_schedulers, Sequence):
            lr_schedulers = [lr_schedulers]
        if device is None:
            device = 'cuda' if CUDA else 'cpu'
        device = torch.device(device)
        save_path = fmt_path(save_path)
        model.to(device)

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
        self.save_path = save_path
        self.fp16 = fp16
        self.lr_step_on_iter = lr_step_on_iter
        self.device = device

        self.log_path = self.save_path / "runs"
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(str(self.log_path / current_time), flush_secs=10)

        self._train_engine_state = None

        self._epochs = 0
        self._kwargs = kwargs

    def to_save(self) -> Mapping[str, Serializable]:
        d = {'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers),
             "engine": self.train_engine}
        if self.fp16:
            from apex import amp
            d['amp'] = amp
        return d

    def save(self):
        path = str(self.save_path / f"epoch_{self._epochs + 1}.pth")
        state_dict = {}
        for k, v in self.to_save():
            state_dict[k] = v.state_dict()
        torch.save(state_dict, path)
        print('Save trainer to %s' % path)

    def load(self, fp=None):

        if fp is None:
            d = Path(self.save_path)
            pattern = "epoch_*.pt*"
            saves = list(d.glob(pattern))
            if len(saves) == 0:
                raise FileNotFoundError("No checkpoint to load in %s" % self.save_path)
            fp = max(saves, key=lambda f: f.stat().st_mtime)

        state_dict = torch.load(fp)

        if not self.fp16 and 'amp' in state_dict:
            del state_dict['amp']

        for k, v in self.to_save():
            v.load_state_dict(state_dict[k])
        print("Load trainer from %s" % fp)

    def _create_train_engine(self) -> Engine:
        raise NotImplementedError

    def _create_eval_engine(self) -> Engine:
        raise NotImplementedError

    @curry
    def _log_epoch_start(self, engine: Engine):
        lrs = "".join(", lr %f" % lr_scheduler.get_last_lr()[0] for lr_scheduler in self.lr_schedulers)
        engine.log.info("Epoch %d%s" % (engine.state.epoch + 1, lrs))

    @curry
    def _lr_scheduler_step(self, engine: Engine):
        iteration = engine.state.iteration
        iters_per_epoch = engine.state.epoch_length
        for lr_scheduler in self.lr_schedulers:
            steps = iteration if self.lr_step_on_iter else iteration / iters_per_epoch
            lr_scheduler.step(steps)

    def _set_epochs(self, engine):
        self._epochs = engine.state.epoch

    @curry
    def log_metrics(self, engine: Engine, writer: Optional[SummaryWriter], stage: str):
        log_str = "%s %s - " % (
            datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S"), stage)
        metric_logs = []
        for k, v in engine.state.metrics.items():
            metric_logs.append("%s: %.4f" % (k, v))
            if writer:
                writer.add_scalar("%s/%s" % (k, stage), v, self._epochs + 1)
        log_str += ", ".join(metric_logs)
        print(log_str)

    def fit(self,
            train_loader: DataLoader,
            epochs: Optional[int],
            val_loader: Optional[DataLoader] = None,
            eval_freq: Optional[int] = 1,
            save_by: Optional[Union[str, int, Epochs, Iters]] = None,
            n_saved: int = 1,
            callbacks: Sequence[Callable] = ()):

        self.train_engine = self._create_train_engine()
        eval_engine = self._create_eval_engine()
        if self._train_engine_state:
            self.train_engine.load_state_dict(self._train_engine_state)

        self.train_engine.call_on(
            Events.ITERATION_COMPLETED, self._lr_scheduler_step),
        self.train_engine.call_on(
            Events.EPOCH_COMPLETED, self._set_epochs),
        self.train_engine.call_on(
            Events.EPOCH_COMPLETED, self.log_metrics(writer=self.writer, stage='train'))

        if save_by:
            checkpoint = Checkpoint(self.to_save(), save_freq=1, save_dir=self.save_path, n_saved=n_saved)
            self.train_engine.call(checkpoint)

        if val_loader is not None and eval_freq:
            self.train_engine.call_on(
                Events.EPOCH_COMPLETED, lambda _: eval_engine.run(val_loader, 1), eval_freq)
            eval_engine.call_on(
                Events.EPOCH_COMPLETED, self.log_metrics(writer=self.writer, stage='valid'))

        for callback in callbacks:
            self.train_engine.call_on(
                Events.ITERATION_COMPLETED, callback)

        try:
            self.train_engine.run(train_loader, epochs)
        except KeyboardInterrupt as e:
            self._train_engine_state = self.train_engine.state_dict()
            raise e

    def evaluate(self, val_loader):
        eval_engine = self._create_eval_engine()
        eval_engine.call_on(
            Events.EPOCH_COMPLETED, self.log_metrics(writer=None, stage='test'))
        eval_engine.run(val_loader, 1)
