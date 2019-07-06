import datetime
import os
from collections import defaultdict
from pathlib import Path

from horch.common import CUDA, one_hot
from horch.models.utils import unfreeze, freeze
from horch.train.trainer import wrap, _terminate_on_iterations
from toolz.curried import get, identity, curry, keyfilter

import torch
import torch.nn as nn

from ignite.engine import Engine, Events

from ignite.handlers import Timer, ModelCheckpoint

from horch.train._utils import _prepare_batch, set_lr, send_weixin, cancel_event, to_device


def create_gan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, metrics=None,
        device=None, prepare_batch=_prepare_batch):

    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)

    def _update(engine, batch):
        inputs, _ = prepare_batch(batch, device=device)
        real_x = inputs[0]

        batch_size = real_x.size(0)

        unfreeze(D)
        D.train()
        optimizerD.zero_grad()

        real_p = D(real_x)

        noise = torch.randn(batch_size, G.in_channels)
        noise = to_device(noise, device)
        with torch.no_grad():
            fake_x = G(noise)
        fake_p = D(fake_x)
        lossD = criterionD(real_p, fake_p)
        lossD.backward()
        optimizerD.step()

        freeze(D)
        G.train()
        optimizerG.zero_grad()

        noise = torch.randn(batch_size, G.in_channels)
        noise = to_device(noise, device)
        fake_p = D(G(noise))
        lossG = criterionG(fake_p)
        lossG.backward()
        optimizerG.step()

        output = {
            "lossD": lossD.item(),
            "lossG": lossG.item(),
            "batch_size": batch_size,
        }
        return output

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_acgan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, metrics=None,
        device=None, prepare_batch=_prepare_batch):

    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)

    def _update(engine, batch):
        inputs, targets = prepare_batch(batch, device=device)
        real_x = inputs[0]
        labels = targets[0]

        batch_size = real_x.size(0)
        num_classes = D.num_classes

        unfreeze(D)
        D.train()
        optimizerD.zero_grad()

        real_p, real_cp = D(real_x)

        noise = torch.randn(batch_size, G.in_channels - num_classes)
        noise = to_device(noise, device)
        z = torch.cat([noise, one_hot(labels, num_classes)], dim=1)
        with torch.no_grad():
            fake_x = G(z)
        fake_p, fake_cp = D(fake_x)
        lossD = criterionD(real_p, fake_p, real_cp, fake_cp, labels)
        lossD.backward()
        optimizerD.step()

        freeze(D)
        G.train()
        optimizerG.zero_grad()

        noise = torch.randn(batch_size, G.in_channels - num_classes)
        noise = to_device(noise, device)
        z = torch.cat([noise, one_hot(labels, num_classes)], dim=1)
        fake_p, fake_cp = D(G(z))
        lossG = criterionG(fake_p, fake_cp, labels)
        lossG.backward()
        optimizerG.step()

        output = {
            "lossD": lossD.item(),
            "lossG": lossG.item(),
            "batch_size": batch_size,
        }
        return output

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class GANTrainer:

    def __init__(self, G, D, criterionG, criterionD, optimizerG, optimizerD, lr_schedulerG=None, lr_schedulerD=None,
                 metrics=None, save_path=".", name="GAN", gan_type='gan'):

        self.G = G
        self.D = D
        self.criterionG = criterionG
        self.criterionD = criterionD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.lr_schedulerG = lr_schedulerG
        self.lr_schedulerD = lr_schedulerD
        self.metrics = metrics or {}
        self.name = name
        root = Path(save_path).expanduser().absolute()
        self.save_path = root / 'gan_trainer' / self.name

        self.metric_history = defaultdict(list)
        self.device = 'cuda' if CUDA else 'cpu'
        self._timer = Timer()
        self._iterations = 0

        self.G.to(self.device)
        self.D.to(self.device)

        assert gan_type in ['gan', 'acgan']
        if gan_type == 'gan':
            self.create_fn = create_gan_trainer
        elif gan_type == 'acgan':
            self.create_fn = create_acgan_trainer

    def _lr_scheduler_step(self, engine):
        if self.lr_schedulerG:
            self.lr_schedulerG.step()
        if self.lr_schedulerD:
            self.lr_schedulerD.step()

    def _attach_timer(self, engine):
        self._trained_time = 0
        self._timer.reset()

    def _increment_iteration(self, engine):
        self._iterations += 1

    def _log_results(self, engine, log_interval, max_iter):
        if self._iterations % log_interval != 0:
            return
        i = engine.state.iteration
        elapsed = self._timer.value()
        self._timer.reset()
        self._trained_time += elapsed
        trained_time = self._trained_time
        eta_seconds = int((trained_time / i) * (max_iter - i))
        it_fmt = "%" + str(len(str(max_iter))) + "d"
        print(("Iter: " + it_fmt + ", Cost: %.2fs, Eta: %s") % (
            self._iterations, elapsed, datetime.timedelta(seconds=eta_seconds)))

        for name, metric in self.metrics.items():
            metric.completed(engine, name)
            metric.reset()

        msg = ""
        for name, val in engine.state.metrics.items():
            msg += "%s: %.4f\t" % (name, val)
            self.metric_history[name].append(val)
        print(msg)

    def fit(self, it, max_iter, log_interval=100, save=None, callbacks=()):

        engine = self.create_fn(
            self.G, self.D, self.criterionG, self.criterionD, self.optimizerG, self.optimizerD,
            self.metrics, self.device)
        self._attach_timer(engine)

        engine.add_event_handler(
            Events.ITERATION_STARTED, self._lr_scheduler_step)

        engine.add_event_handler(Events.ITERATION_COMPLETED, self._increment_iteration)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_results, log_interval, max_iter)

        # Set checkpoint
        if save:
            checkpoint_handler = save.parse(self)
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_handler, {"trainer": self})

        for callback in callbacks:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, wrap(callback), self)

        engine.add_event_handler(
            Events.ITERATION_COMPLETED, _terminate_on_iterations, max_iter)

        # Run
        engine.run(it, 1)

        # Return history
        return self.metric_history

    def state_dict(self):
        s = {
            "iterations": self.iterations(),
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "optimizerG": self.optimizerG.state_dict(),
            "optimizerD": self.optimizerD.state_dict(),
            "criterionG": self.criterionG.state_dict(),
            "criterionD": self.criterionD.state_dict(),
            "lr_schedulerG": None,
            "lr_schedulerD": None,
            "metric_history": self.metric_history,
        }
        if self.lr_schedulerG:
            s["lr_schedulerG"] = self.lr_schedulerG.state_dict()
        if self.lr_schedulerD:
            s["lr_schedulerD"] = self.lr_schedulerD.state_dict()
        return s

    def load_state_dict(self, state_dict):
        iterations, G, D, optimizerG, optimizerD, criterionG, criterionD, lr_schedulerG, lr_schedulerD, metric_history = get(
            ["iterations", "G", "D", "optimizerG", "optimizerD", "criterionG", "criterionD",
             "lr_schedulerG", "lr_schedulerD", "metric_history"], state_dict)
        self._iterations = iterations
        self.G.load_state_dict()
        self.D.load_state_dict()
        self.optimizerG.load_state_dict(optimizerG)
        self.optimizerD.load_state_dict(optimizerD)
        self.criterionG.load_state_dict(criterionG)
        self.criterionD.load_state_dict(criterionD)
        if self.lr_schedulerG and lr_schedulerG:
            self.lr_schedulerG.load_state_dict(lr_schedulerG)
        if self.lr_schedulerD and lr_schedulerD:
            self.lr_schedulerD.load_state_dict(lr_schedulerD)
        self.metric_history = metric_history

    def iterations(self):
        return self._iterations