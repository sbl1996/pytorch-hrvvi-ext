from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Sequence, Dict, Callable, Union, Optional

import torch
import torch.nn as nn
from toolz.curried import curry
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from horch.common import CUDA
from horch.io import fmt_path
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Metric


def backward(loss, optimizer, fp16):
    if fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return


class StatefulList:

    def __init__(self, xs):
        for x in xs:
            if not hasattr(x, "state_dict"):
                raise TypeError("Object {} should have `state_dict` method".format(type(x)))
        self.xs = xs

    def state_dict(self):
        d = {"states": []}
        for x in self.xs:
            d['states'].append(x.state_dict())
        return d

    def load_state_dict(self, d):
        for x, state_dict in zip(self.xs, d['states']):
            x.load_state_dict(state_dict)


class Epochs:

    def __init__(self, n: int):
        self.n = n


class Iters:

    def __init__(self, n: int):
        self.n = n


class TrainerState(Enum):
    INITIALIZED = 1
    FIRST_FITTING = 2
    CONTINUED_FITTING = 3


class TrainerBase:

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizers: Union[Optimizer, Sequence[Optimizer]],
                 lr_schedulers: Union[_LRScheduler, Sequence[_LRScheduler]],
                 metrics: Dict[str, Metric],
                 test_metrics: Dict[str, Metric],
                 save_path: Union[Path, str] = ".",
                 fp16: bool = False,
                 lr_step_on_iter: bool = False,
                 device: Optional[str] = None):

        # Check Arguments
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        if not isinstance(lr_schedulers, Sequence):
            lr_schedulers = [lr_schedulers]
        if device is None:
            device = 'cuda' if CUDA else 'cpu'
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
        self.writer = SummaryWriter(str(self.log_path / current_time))

        self.train_engine = self._create_train_engine()
        self.eval_engine = self._create_eval_engine()
        saver = DiskSaver(str(self.save_path), create_dir=True, require_empty=False)
        self.checkpoint_handler = Checkpoint(self.to_save(), saver)

        self._traier_state = TrainerState.INITIALIZED

    def to_save(self):
        d = {'train_engine': self.train_engine, 'eval_engine': self.eval_engine,
             'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers)}
        if self.fp16:
            from apex import amp
            d['amp'] = amp
        return d

    def resume(self):
        d = Path(self.save_path)
        pattern = "checkpoint_*.pt"
        saves = list(d.glob(pattern))
        if len(saves) == 0:
            raise FileNotFoundError("No checkpoint to load in %s" % self.save_path)
        fp = max(saves, key=lambda f: f.stat().st_mtime)
        checkpoint = torch.load(fp)
        if not self.fp16 and 'amp' in checkpoint:
            del checkpoint['amp']
        Checkpoint.load_objects(self.to_save(), checkpoint)
        print("Load trainer from %s" % fp)

    def _create_train_engine(self):
        raise NotImplementedError

    def _create_eval_engine(self):
        raise NotImplementedError

    @curry
    def _log_epoch_start(self, engine):
        lrs = "".join(", lr %f" % lr_scheduler.get_last_lr()[0] for lr_scheduler in self.lr_schedulers)
        print("Epoch %d%s" % (engine.state.epoch, lrs))

    @curry
    def _lr_scheduler_step(self, engine):
        iteration = engine.state.iteration
        iters_per_epoch = engine.state.epoch_length
        for lr_scheduler in self.lr_schedulers:
            steps = iteration if self.lr_step_on_iter else iteration / iters_per_epoch
            lr_scheduler.step(steps)

    def fit(self,
            train_loader: DataLoader,
            epochs: Optional[int],
            val_loader: Optional[DataLoader] = None,
            save_freq: Optional[Union[int, Epochs, Iters]] = None,
            eval_freq: Union[int, Epochs, Iters] = 1):

        fit_events = [
            self.train_engine.add_event_handler(
                Events.EPOCH_STARTED, self._log_epoch_start),
            self.train_engine.add_event_handler(
                Events.ITERATION_COMPLETED, self._lr_scheduler_step)
        ]

        if save_freq:
            fit_events.append(
                self.train_engine.add_event_handler(
                    get_event_by_freq(save_freq), self.checkpoint_handler))

        if val_loader is not None:
            fit_events.append(
                self.train_engine.add_event_handler(
                    get_event_by_freq(eval_freq), lambda _: self.eval_engine.run(val_loader)))

        try:
            self.train_engine.run(train_loader, epochs)
            for e in fit_events:
                e.remove()
        except KeyboardInterrupt as e:
            for e in fit_events:
                e.remove()
            raise e


def get_event_by_freq(freq: Union[int, Epochs, Iters]):
    if isinstance(freq, int):
        freq = Epochs(freq)
    if isinstance(freq, Epochs):
        return Events.EPOCH_COMPLETED(every=freq.n)
    elif isinstance(freq, Iters):
        return Events.ITERATION_COMPLETED(every=freq.n)
