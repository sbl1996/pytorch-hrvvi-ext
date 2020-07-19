from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Sequence, Dict, Callable, Union, Optional, Any

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from toolz.curried import curry
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from horch.common import CUDA
from horch.io import fmt_path
from ignite.engine import Events, Engine
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
    INIT = 1
    FITTING = 2


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
                 device: Optional[str] = None,
                 **kwargs):

        # Check Arguments
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
        self._eval_engine_state = None

        self._traier_state = TrainerState.INIT
        self._epochs = 0

        self._kwargs = kwargs

    def to_save(self):
        d = {'model': self.model, 'optimizers': StatefulList(self.optimizers),
             'lr_schedulers': StatefulList(self.lr_schedulers)}
        if self.fp16:
            from apex import amp
            d['amp'] = amp
        return d

    def resume(self, fp=None):
        assert self._traier_state == TrainerState.INIT

        if fp is None:
            d = Path(self.save_path)
            pattern = "checkpoint_*.pt*"
            saves = list(d.glob(pattern))
            if len(saves) == 0:
                raise FileNotFoundError("No checkpoint to load in %s" % self.save_path)
            fp = max(saves, key=lambda f: f.stat().st_mtime)

        checkpoint = torch.load(fp)

        if not self.fp16 and 'amp' in checkpoint:
            del checkpoint['amp']

        Checkpoint.load_objects(self.to_save(), checkpoint)

        self._train_engine_state = checkpoint['train_engine']
        self._eval_engine_state = checkpoint['eval_engine']
        self._traier_state = TrainerState.FITTING

        print("Load trainer from %s" % fp)

    def _create_train_engine(self) -> Engine:
        raise NotImplementedError

    def _create_eval_engine(self) -> Engine:
        raise NotImplementedError

    def _attach_prograss_bar(self, train_engine: Engine):
        pb = ProgressBar()
        pb.attach(train_engine)

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
                writer.add_scalar("%s/%s" % (k, stage), v, self._epochs)
        log_str += ", ".join(metric_logs)
        print(log_str)

    def fit(self,
            train_loader: DataLoader,
            epochs: Optional[int],
            val_loader: Optional[DataLoader] = None,
            eval_freq: Union[int, Epochs, Iters] = 1,
            save_freq: Optional[Union[int, Epochs, Iters]] = None,
            n_saved: int = 1,
            progress_bar: bool = False,
            callbacks: Sequence[Callable] = ()):

        train_engine = self._create_train_engine()
        eval_engine = self._create_eval_engine()

        if self._traier_state == TrainerState.FITTING:
            train_engine.load_state_dict(self._train_engine_state)
            eval_engine.load_state_dict(self._eval_engine_state)

        if not progress_bar:
            train_engine.add_event_handler(
                Events.EPOCH_STARTED, self._log_epoch_start),
        train_engine.add_event_handler(
            Events.ITERATION_COMPLETED, self._lr_scheduler_step),
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._set_epochs),
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_metrics(writer=self.writer, stage='train'))

        if save_freq:
            def global_step_transform(engine, event_name):
                return engine.state.iteration if isinstance(save_freq, Iters) else engine.state.epoch

            saver = DiskSaver(str(self.save_path), create_dir=True, require_empty=False)
            to_save = {**self.to_save(), "train_engine": train_engine, "eval_engine": eval_engine}

            checkpoint_handler = Checkpoint(to_save, saver, n_saved=n_saved,
                                            global_step_transform=global_step_transform)

            train_engine.add_event_handler(get_event_by_freq(save_freq), checkpoint_handler)

        if val_loader is not None:
            train_engine.add_event_handler(
                get_event_by_freq(eval_freq), lambda _: eval_engine.run(val_loader))
            eval_engine.add_event_handler(
                Events.EPOCH_COMPLETED, self.log_metrics(writer=self.writer, stage='valid'))

        if progress_bar:
            self._attach_prograss_bar(train_engine)

        for callback in callbacks:
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED, callback, self)

        try:
            max_epochs = epochs if self._traier_state == TrainerState.INIT else None
            self._traier_state = TrainerState.FITTING
            train_engine.run(train_loader, max_epochs)
            self._train_engine_state = train_engine.state_dict()
            self._eval_engine_state = eval_engine.state_dict()
        except KeyboardInterrupt as e:
            self._train_engine_state = train_engine.state_dict()
            self._eval_engine_state = eval_engine.state_dict()
            self._traier_state = TrainerState.FITTING
            raise e

    def evaluate(self, val_loader):
        eval_engine = self._create_eval_engine()
        eval_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_metrics(writer=None, stage='test'))
        eval_engine.run(val_loader)


def get_event_by_freq(freq: Union[int, Epochs, Iters]):
    if isinstance(freq, int):
        freq = Epochs(freq)
    if isinstance(freq, Epochs):
        return Events.EPOCH_COMPLETED(every=freq.n)
    elif isinstance(freq, Iters):
        return Events.ITERATION_COMPLETED(every=freq.n)

#
# def trainer_callback_wrap(f):
#     def func(engine, *args, **kwargs):
#         return f(*args, **kwargs)
#
#     return func
