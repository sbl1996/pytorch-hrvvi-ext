import os
import datetime
import re
from collections import defaultdict
from pathlib import Path

from toolz.curried import get, identity, curry, keyfilter

import numpy as np

import torch
import torch.nn as nn

from torchvision.utils import make_grid

from ignite.engine import Events
from ignite.handlers import Timer, ModelCheckpoint

from horch.common import CUDA, one_hot
from horch.models.utils import unfreeze, freeze
from horch.train.engine import Engine
from horch.train.trainer import _trainer_callback_wrap, _terminate_on_iterations
from horch.train._utils import _prepare_batch, set_lr, send_weixin, cancel_event, to_device


def create_gan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, make_latent=None, metrics=None,
        device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)
    if make_latent is None:
        make_latent = lambda b: torch.randn(b, G.in_channels)

    def _update(engine, batch):
        inputs, _ = prepare_batch(batch, device=device)
        real_x = inputs[0]

        batch_size = real_x.size(0)

        unfreeze(D)
        D.train()
        optimizerD.zero_grad()

        real_p = D(real_x)

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        with torch.no_grad():
            fake_x = G(lat)
        fake_p = D(fake_x)
        lossD = criterionD(real_p, fake_p)
        lossD.backward()
        optimizerD.step()

        freeze(D)
        G.train()
        optimizerG.zero_grad()

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        fake_p = D(G(lat))
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


def create_infogan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, make_latent=None, metrics=None,
        device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)
    if make_latent is None:
        make_latent = lambda b: torch.randn(b, G.in_channels)

    def _update(engine, batch):
        inputs, _ = prepare_batch(batch, device=device)
        real_x = inputs[0]

        batch_size = real_x.size(0)

        D.q = False
        unfreeze(D.features)
        unfreeze(D.d_head)
        D.features.train()
        D.d_head.train()
        optimizerD.zero_grad()

        real_p = D(real_x)

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        with torch.no_grad():
            fake_x = G(lat)
        fake_p = D(fake_x)
        lossD = criterionD(real_p, fake_p)
        lossD.backward()
        optimizerD.step()

        freeze(D.features)
        freeze(D.d_head)
        G.train()
        optimizerG.zero_grad()

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        D.q = True
        fake_p, q_p = D(G(lat))
        lossG = criterionG(fake_p, q_p, lat)
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
        G, D, criterionG, criterionD, optimizerG, optimizerD, make_latent, metrics=None,
        device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)

    num_classes = D.out_channels - 1
    lat_dim = G.in_channels - num_classes

    if make_latent is None:
        make_latent = lambda b: torch.randn(b, lat_dim)

    def _update(engine, batch):
        inputs, targets = prepare_batch(batch, device=device)
        real_x = inputs[0]
        labels = targets[0]

        batch_size = real_x.size(0)

        unfreeze(D)
        D.train()
        optimizerD.zero_grad()

        real_p = D(real_x)
        real_cp = real_p[:, 1:]
        real_p = real_p[:, 0]

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        z = torch.cat([lat, one_hot(labels, num_classes)], dim=1)
        with torch.no_grad():
            fake_x = G(z)
        fake_p = D(fake_x)
        fake_cp = fake_p[:, 1:]
        fake_p = fake_p[:, 0]
        lossD = criterionD(real_p, fake_p, real_cp, fake_cp, labels)
        lossD.backward()
        optimizerD.step()

        freeze(D)
        G.train()
        optimizerG.zero_grad()

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        z = torch.cat([lat, one_hot(labels, num_classes)], dim=1)
        fake_p = D(G(z))
        fake_cp = fake_p[:, 1:]
        fake_p = fake_p[:, 0]
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


def create_cgan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, make_latent=None, metrics=None,
        device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        G.to(device)
        D.to(device)
    num_classes = D.out_channels - 1
    lat_dim = G.in_channels - num_classes

    if make_latent is None:
        make_latent = lambda b: torch.randn(b, lat_dim)

    def _update(engine, batch):
        inputs, targets = prepare_batch(batch, device=device)
        real_x = inputs[0]
        labels = targets[0]

        batch_size = real_x.size(0)

        unfreeze(D)
        D.train()
        optimizerD.zero_grad()

        real_p = D(real_x, labels)

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        with torch.no_grad():
            fake_x = G(lat, labels)
        fake_p = D(fake_x, labels)
        lossD = criterionD(real_p, fake_p)
        lossD.backward()
        optimizerD.step()

        freeze(D)
        # D.eval()
        G.train()
        optimizerG.zero_grad()

        lat = make_latent(batch_size)
        lat = to_device(lat, device)
        fake_p = D(G(lat, labels), labels)
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


class GANTrainer:

    def __init__(self, G, D, criterionG, criterionD, optimizerG, optimizerD, lr_schedulerG=None, lr_schedulerD=None,
                 make_latent=None, metrics=None, save_path=".", name="GAN", gan_type='gan'):

        self.G = G
        self.D = D
        self.criterionG = criterionG
        self.criterionD = criterionD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.lr_schedulerG = lr_schedulerG
        self.lr_schedulerD = lr_schedulerD
        self.make_latent = make_latent
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

        assert gan_type in ['gan', 'acgan', 'cgan', 'infogan']
        if gan_type == 'gan':
            self.create_fn = create_gan_trainer
        elif gan_type == 'acgan':
            self.create_fn = create_acgan_trainer
        elif gan_type == 'cgan':
            self.create_fn = create_cgan_trainer
        elif gan_type == 'infogan':
            self.create_fn = create_infogan_trainer

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

    def fit(self, it, max_iter, log_interval=100, callbacks=()):

        engine = self.create_fn(
            self.G, self.D, self.criterionG, self.criterionD, self.optimizerG, self.optimizerD,
            self.make_latent, self.metrics, self.device)
        self._attach_timer(engine)

        engine.add_event_handler(
            Events.ITERATION_STARTED, self._lr_scheduler_step)

        engine.add_event_handler(Events.ITERATION_COMPLETED, self._increment_iteration)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_results, log_interval, max_iter)

        for callback in callbacks:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, _trainer_callback_wrap(callback), self)

        # Run
        engine.run(it, max_iter)

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
        self.G.load_state_dict(G)
        self.D.load_state_dict(D)
        self.optimizerG.load_state_dict(optimizerG)
        self.optimizerD.load_state_dict(optimizerD)
        self.criterionG.load_state_dict(criterionG)
        self.criterionD.load_state_dict(criterionD)
        if self.lr_schedulerG and lr_schedulerG:
            self.lr_schedulerG.load_state_dict(lr_schedulerG)
        if self.lr_schedulerD and lr_schedulerD:
            self.lr_schedulerD.load_state_dict(lr_schedulerD)
        self.metric_history = metric_history

    def save(self, remove_prev=True):
        d = self.save_path
        d.mkdir(parents=True, exist_ok=True)

        if remove_prev:
            pattern = "%s_trainer*.pth" % self.name
            saves = list(d.glob(pattern))
            if len(saves) != 0:
                fp = max(saves, key=lambda f: f.stat().st_mtime)
                p = "%s_trainer_(?P<iters>[0-9]+).pth" % self.name
                iters = int(re.match(p, fp.name).group('iters'))
                if self.iterations() > iters:
                    fp.unlink()

        filename = "%s_trainer_%d.pth" % (self.name, self.iterations())
        fp = d / filename
        torch.save(self.state_dict(), fp)
        print("Save trainer as %s" % fp)

    def load(self):
        d = self.save_path
        pattern = "%s_trainer*.pth" % self.name
        saves = list(d.glob(pattern))
        if len(saves) == 0:
            raise FileNotFoundError("No checkpoint to load for %s in %s" % (self.name, self.save_path))
        fp = max(saves, key=lambda f: f.stat().st_mtime)
        self.load_state_dict(torch.load(fp, map_location=self.device))
        print("Load trainer from %s" % fp)

    def iterations(self):
        return self._iterations


def _gan_trainer_callback_wrap(f):
    def func(engine, *args, **kwargs):
        return f(*args, **kwargs)

    return func


@curry
def save_generated(trainer, save_interval, fixed_inputs, sharpen=True):
    if trainer.iterations() % save_interval != 0:
        return
    import matplotlib.pyplot as plt
    trainer.G.eval()
    if torch.is_tensor(fixed_inputs):
        fixed_inputs = (fixed_inputs,)
    fixed_inputs = to_device(fixed_inputs, trainer.device)
    with torch.no_grad():
        fake_x = trainer.G(*fixed_inputs).cpu()
    trainer.G.train()
    img = np.transpose(make_grid(fake_x, padding=2, normalize=True).numpy(), (1, 2, 0))
    if not sharpen:
        img = (img + 1) / 2
    fp = trainer.save_path / "images" / ("%d.jpg" % trainer.iterations())
    fp.parent.mkdir(exist_ok=True, parents=True)
    plt.imsave(fp, img)


@curry
def save_trainer(trainer, save_interval):
    if trainer.iterations() % save_interval != 0:
        return
    trainer.save()
