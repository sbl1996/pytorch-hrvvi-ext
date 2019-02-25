import re
import traceback
import time
import os
from collections import defaultdict

import itchat
from tqdm import tqdm
from toolz.curried import get, identity, curry, keyfilter

import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy as IgniteAccuracy
from ignite._utils import convert_tensor
from ignite.handlers import Timer

from hutil.common import CUDA
from hutil.functools import find, lmap
from hutil.ext.checkpoint import ModelCheckpoint
from hutil.train.metrics import TrainLoss, Loss
from hutil.train._utils import _prepare_batch, send_weixin, set_lr, cancel_event, detach


def create_supervised_evaluator(model, metrics={},
                                device=None, prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device)
            s1 = time.time()
            y_pred = model(*x)
            if torch.is_tensor(y_pred):
                y_pred = (y_pred,)
            output = {
                "y_pred": detach(y_pred),
                "y": detach(y),
                'batch_size': x[0].size(0),
            }
            return output

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_trainer(
        model, criterion, optimizer, lr_scheduler=None, metrics={},
        device=None, prepare_batch=_prepare_batch):

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device)
        y_pred = model(*x)
        if torch.is_tensor(y_pred):
            y_pred = (y_pred,)
        loss = criterion(*y_pred, *y)
        loss.backward()
        optimizer.step()
        output = {
            "y_pred": detach(y_pred),
            "y": detach(y),
            "loss": loss.item(),
            "batch_size": x[0].size(0),
        }
        return output

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Trainer:

    def __init__(self, model, criterion, optimizer, lr_scheduler=None,
                 metrics={}, evaluate_metrics=None, device=None, save_path=".", name="Net"):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.evaluate_metrics = evaluate_metrics
        if evaluate_metrics is None:
            self.evaluate_metrics = metrics.copy()
            if 'loss' in metrics and isinstance(metrics['loss'], TrainLoss):
                self.evaluate_metrics['loss'] = Loss(criterion=criterion)
        self.device = device or ('cuda' if CUDA else 'cpu')
        self.save_path = save_path
        self.name = name

        self.metric_history = defaultdict(list)
        self._print_callbacks = set([lambda msg: print(msg, end='')])
        self._weixin_logined = False
        self._timer = Timer()
        self._epochs = 0

        self.evaluator = create_supervised_evaluator(
            self.model, self.evaluate_metrics, self.device)

    def _fprint(self, msg):
        for f in self._print_callbacks:
            try:
                f(msg)
            except Exception:
                pass

    def _enable_send_weixin(self):
        if self._weixin_logined:
            self._print_callbacks.add(send_weixin)
        else:
            print("Weixin is not logged in.")

    def _disable_send_weixin(self):
        self._print_callbacks.discard(send_weixin)

    def _log_epochs(self, engine, epochs):
        self._fprint("Epoch %d/%d\n" %
                     (self._epochs + 1, self._epochs + 1 + epochs - engine.state.epoch))

    def _lr_scheduler_step(self, engine):
        self.lr_scheduler.step()

    def _evaluate(self, engine, val_loader):
        self.evaluator.run(val_loader)

    def _increment_epoch(self, engine):
        self._epochs += 1

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

    def fit(self, train_loader, epochs, val_loader=None, send_weixin=False, save_per_epochs=None, save_by_metric=None, patience=0, callbacks=[]):
        validate = val_loader is not None
        # Weixin
        if send_weixin:
            self._enable_send_weixin()
        else:
            self._disable_send_weixin()

        engine = create_supervised_trainer(
            self.model, self.criterion, self.optimizer, self.lr_scheduler,
            self.metrics, self.device)
        if self.lr_scheduler:
            engine.add_event_handler(
                Events.EPOCH_STARTED, self._lr_scheduler_step)
        self._timer.attach(engine,
                           start=Events.EPOCH_STARTED)
        engine.add_event_handler(Events.EPOCH_STARTED,
                                 self._log_epochs, epochs)

        if validate:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, self._evaluate, val_loader)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._increment_epoch)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._log_results, validate)

        # Set checkpoint
        if save_by_metric:
            mat = re.match(
                "(?P<sign>-?)(?P<metric>val_[a-zA-Z]+)", save_by_metric)
            assert mat, "save by metric must be of form `-?val_<evaluate_metric>`"
            sign = -1 if mat.group('sign') else 1
            save_metric = mat.group('metric')
            assert save_metric[4:] in self.evaluate_metrics, "the metric used must be one of \
                evaluate_metrics"

            def score_function(e): return sign * \
                self.metric_history[save_metric][-1]
            checkpoint_handler = ModelCheckpoint(
                self.save_path, self.name,
                score_name=save_metric, patience=patience,
                score_function=score_function,
                save_as_state_dict=True, require_empty=False)
            checkpoint_handler._iteration = self.epochs()
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_handler, {"trainer": self})
        elif save_per_epochs:
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
            "lr_scheduler": None,
            "metric_history": self.metric_history,
        }
        if self.lr_scheduler:
            s["lr_scheduler"] = self.lr_scheduler.state_dict()
        return s

    def load_state_dict(self, state_dict):
        epochs, model, optimizer, lr_scheduler, metric_history = get(
            ["epochs", "model", "optimizer", "lr_scheduler", "metric_history"], state_dict)
        self._epochs = epochs
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        if self.lr_scheduler and lr_scheduler:
            self.lr_scheduler.load_state_dict(lr_scheduler)
        self.metric_history = metric_history

    def epochs(self):
        return self._epochs

    def login_weixin(self, save_path='.'):
        itchat.auto_login(hotReload=True, enableCmdQR=2,
                          statusStorageDir=os.path.join(save_path, 'weixin.pkl'))
        self._weixin_logined = True

    def logout_weixin(self):
        itchat.logout()
        self._weixin_logined = False

    def evaluate(self, test_loader):
        return self.evaluator.run(test_loader).metrics

    def set_lr(self, lr):
        set_lr(lr, self.optimizer, self.lr_scheduler)


def _callback_wrapper(f):
    def func(engine, *args, **kwargs):
        return f(*args, **kwargs)
    return func
