import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from toolz.curried import get, curry

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from horch.common import CUDA

from horch.train.metrics import TrainLoss
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, TopKCategoricalAccuracy, Loss
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None):
    """Prepare batch for training: pass to a device with options.

    """
    if isinstance(batch[0], tuple):
        (x_train, y_train), (x_val, y_val) = batch
        x_train, y_train, x_val, y_val = [convert_tensor(x, device=device) for x in [x_train, y_train, x_val, y_val]]
        return (x_train, y_train), (x_val, y_val)
    else:
        x, y = batch
        x = convert_tensor(x, device=device)
        y = convert_tensor(y, device=device)
        return x, y


def create_darts_trainer(
        model, criterion, optimizer_model, optimizer_arch, lr_scheduler, metrics, device, clip_grad_norm=5):
    def step(engine, batch):
        model.train()

        (input, target), (input_search, target_search) = _prepare_batch(batch, device)

        optimizer_arch.zero_grad()
        for p in model.model_parameters():
            p.requires_grad_(False)
        for p in model.arch_parameters():
            p.requires_grad_(True)
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer_arch.step()

        optimizer_model.zero_grad()
        for p in model.arch_parameters():
            p.requires_grad_(False)
        for p in model.model_parameters():
            p.requires_grad_(True)
        logits_search = model(input_search)
        loss_search = criterion(logits_search, target_search)
        loss_search.backward()
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer_model.step()

        lr_scheduler.step(engine.state.iteration / engine.state.epoch_length)
        return {
            "loss": loss.item(),
            "batch_size": input.size(0),
            "y": target_search,
            "y_pred": logits_search.detach(),
        }

    engine = Engine(step)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_darts_evaluator(model, metrics, device):
    def step(engine, batch):
        model.eval()
        input, target = _prepare_batch(batch, device)
        with torch.no_grad():
            output = model(input)

        return {
            "batch_size": input.size(0),
            "y": target,
            "y_pred": output,
        }

    engine = Engine(step)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


@curry
def log_metrics(engine, stage):
    log_str = "%s: %s - " % (
        datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"), stage)
    log_str += ", ".join(["%s: %.4f" % (k, v) for k, v in engine.state.metrics.items()])
    print(log_str)


class DARTSTrainer:

    def __init__(self, model, criterion, optimizer_model, optimizer_arch, lr_scheduler,
                 metrics=None, test_metrics=None, save_path="checkpoints", device=None):
        self.device = device or ('cuda' if CUDA else 'cpu')
        model.to(self.device)

        self.model = model
        self.criterion = criterion
        self.optimizer_model = optimizer_model
        self.optimizer_arch = optimizer_arch
        self.lr_scheduler = lr_scheduler
        self._output_transform = get(["y_pred", "y"])
        self.metrics = metrics or {
            "loss": TrainLoss(),
            "acc": Accuracy(self._output_transform),
        }
        self.test_metrics = test_metrics or {
            "loss": Loss(self.criterion, self._output_transform),
            "acc": Accuracy(self._output_transform),
        }
        self.save_path = save_path
        self._log_path = os.path.join(self.save_path, "runs")

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(self._log_path, current_time)
        self.writer = SummaryWriter(log_dir)

        self.train_engine = self._create_train_engine()
        self.eval_engine = self._create_eval_engine()
        self.checkpoint_handler = Checkpoint(self.to_save(),
                                             DiskSaver(self.save_path, create_dir=True, require_empty=False))

    def to_save(self):
        return {'train_engine': self.train_engine, 'eval_engine': self.eval_engine,
                'model': self.model, 'optimizer_model': self.optimizer_model, 'optimizer_arch': self.optimizer_arch,
                'lr_scheduler': self.lr_scheduler}

    def resume(self):
        d = Path(self.save_path)
        pattern = "%checkpoint_*.pth"
        saves = list(d.glob(pattern))
        if len(saves) == 0:
            raise FileNotFoundError("No checkpoint to load in %s" % (self.save_path))
        fp = max(saves, key=lambda f: f.stat().st_mtime)
        checkpoint = torch.load(fp)
        Checkpoint.load_objects(self.to_save(), checkpoint)
        print("Load trainer from %s" % fp)

    def _create_train_engine(self):
        engine = create_darts_trainer(
            self.model, self.criterion, self.optimizer_model, self.optimizer_arch,
            self.lr_scheduler, self.metrics, self.device)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, log_metrics(stage='train'))
        return engine

    def _create_eval_engine(self):
        engine = create_darts_evaluator(self.model, self.test_metrics, self.device)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, log_metrics(stage='valid'))
        return engine

    def fit(self, train_loader, epochs=None, val_loader=None, save_every=5000):

        fit_events = [
            self.train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=save_every), self.checkpoint_handler),
            self.train_engine.add_event_handler(
                Events.EPOCH_COMPLETED, lambda _: self.eval_engine.run(val_loader)),
            self.train_engine.add_event_handler(
                Events.EPOCH_STARTED,
                lambda engine: print("Epoch %d, lr %f" % (engine.state.epoch, self.lr_scheduler.get_last_lr()[0])))]

        try:
            self.train_engine.run(train_loader, epochs)
            for e in fit_events:
                e.remove()
        except KeyboardInterrupt as e:
            for e in fit_events:
                e.remove()
            raise e
