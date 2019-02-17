import os
from collections import defaultdict

import itchat
from toolz.curried import get, identity, curry, keyfilter

import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy as IgniteAccuracy

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.handlers import Timer, ModelCheckpoint

from hutil.train._utils import _prepare_batch, set_lr, send_weixin, cancel_event


def create_gan_trainer(
        G, D, criterionG, criterionD, optimizerG, optimizerD, lr_schedulerG=None, lr_schedulerD=None, metrics={},
        device=None, prepare_batch=_prepare_batch):

    def _update(engine, batch):
        x, y = prepare_batch(batch, device=device)
        real, noise = x
        y_real, y_fake = y

        D.train()
        G.train()

        D.zero_grad()
        y_pred_real = D(real)

        fake = G(noise)
        y_pred_fake = D(fake.detach())
        lossD = criterionD(y_pred_real, y_pred_fake, y_real, y_fake)

        lossD.backward()
        optimizerD.step()

        G.zero_grad()
        y_pred_fake = D(fake)
        lossG = criterionG(y_pred_fake, y_real)
        lossG.backward()
        optimizerG.step()

        output = {
            "lossD": lossD.item(),
            "lossG": lossG.item(),
            "batch_size": real.size(0),
        }
        return output

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class GANTrainer:

    def __init__(self, G, D, criterionG, criterionD, optimizerG, optimizerD, lr_schedulerG=None, lr_schedulerD=None,
                 metrics={}, device=None, save_path=".", name="Net"):

        self.G = G
        self.D = D
        self.criterionG = criterionG
        self.criterionD = criterionD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.lr_schedulerG = lr_schedulerG
        self.lr_schedulerD = lr_schedulerD
        self.metrics = metrics
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        self.name = name

        self.metric_history = defaultdict(list)
        self._print_callbacks = set([lambda msg: print(msg, end='')])
        self._weixin_logined = False
        self._timer = Timer()
        self._epochs = 0

        self.G.to(self.device)
        self.D.to(self.device)
        # self._create_evaluator()

    def _create_engine(self):
        engine = create_gan_trainer(
            self.G, self.D, self.criterionG, self.criterionD, self.optimizerG, self.optimizerD,
            self.lr_schedulerG, self.lr_schedulerD, self.metrics, self.device)
        engine.add_event_handler(
            Events.EPOCH_STARTED, self._lr_scheduler_step)
        self._timer.attach(engine,
                           start=Events.EPOCH_STARTED)
        return engine

    def _on(self, event_name, f, *args, **kwargs):
        cancel_event(self.engine, event_name, f)
        self.engine.add_event_handler(event_name, f, *args, **kwargs)

    def _fprint(self, msg):
        for f in self._print_callbacks:
            try:
                f(msg)
            except Exception as e:
                pass

    def _enable_send_weixin(self):
        if self._weixin_logined:
            self._print_callbacks.add(send_weixin)
        else:
            print("Weixin is not logged in.")

    def _disable_send_weixin(self):
        self._print_callbacks.discard(send_weixin)

    def _lr_scheduler_step(self, engine):
        if self.lr_schedulerG:
            self.lr_schedulerG.step()
        if self.lr_schedulerD:
            self.lr_schedulerD.step()

    def _log_epochs(self, engine, epochs):
        self._epochs += 1
        self._fprint("Epoch %d/%d\n" %
                     (self._epochs, self._epochs + epochs - engine.state.epoch))

    def _evaluate(self, engine, val_loader):
        self.evaluator.run(val_loader)

    def _log_results(self, engine, validate):
        elapsed = int(self._timer.value())
        msg = ""
        msg += "elapsed: %ds\t" % elapsed
        for name, val in engine.state.metrics.items():
            msg += "%s: %.4f\t" % (name, val)
            self.metric_history[name].append(val)
        msg += '\n'
        if validate:
            msg += "validate ------\t"
            for name, val in self.evaluator.state.metrics.items():
                msg += "%s: %.4f\t" % (name, val)
                self.metric_history["val_" + name].append(val)
            msg += "\n"
        self._fprint(msg)

    def fit(self, train_loader, epochs, val_loader=None, send_weixin=False, save_per_epochs=None, callbacks=[]):
        validate = val_loader is not None
        # Weixin
        if send_weixin:
            self._enable_send_weixin()

        # Create engine
        engine = self._create_engine()

        # Register events
        engine.add_event_handler(Events.EPOCH_STARTED,
                                 self._log_epochs, epochs)

        if validate:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, self._evaluate, val_loader)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._log_results, validate)

        # Set checkpoint
        if save_per_epochs:
            checkpoint_handler = ModelCheckpoint(
                self.save_path, self.name, save_per_epochs, save_as_state_dict=True, require_empty=False)
            checkpoint_handler._iteration = self.epochs()
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_handler, {"trainer": self})

        for callback in callbacks:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, _callback_wrapper(callback), self)

        # Run
        engine.run(train_loader, epochs)

        # Destroy
        self._disable_send_weixin()

        # Return history
        hist = {metric: hist[-epochs:]
                for metric, hist in self.metric_history.items()}
        if not validate:
            hist = keyfilter(lambda k: not k.startswith("val_"), hist)
        return hist

    def state_dict(self):
        s = {
            "epochs": self.epochs(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "lr_scheduler": None,
            "metric_history": self.metric_history,
        }
        if self.lr_scheduler:
            s["lr_scheduler"] = self.lr_scheduler.state_dict()
        return s

    def load_state_dict(self, state_dict):
        epochs, model, optimizer, criterion, lr_scheduler, metric_history = get(
            ["epochs", "model", "optimizer", "criterion", "lr_scheduler", "metric_history"], state_dict)
        self._epochs = epochs
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.criterion.load_state_dict(criterion)
        if self.lr_scheduler and lr_scheduler:
            self.lr_scheduler.load_state_dict(lr_scheduler)
        self.metric_history = metric_history

    def epochs(self):
        return self._epochs

    def login_weixin(self, save_path='.'):
        itchat.logout()
        itchat.auto_login(hotReload=True, enableCmdQR=2,
                          statusStorageDir=os.path.join(save_path, 'weixin.pkl'))
        self._weixin_logined = True

    def logout_weixin(self):
        itchat.logout()
        self._weixin_logined = False

    def set_lr(self, lr):
        set_lr(lr, self.optimizer, self.lr_scheduler)


def _callback_wrapper(f):
    def func(engine, *args, **kwargs):
        return f(*args, **kwargs)
    return func
