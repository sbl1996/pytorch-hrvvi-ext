import os
from datetime import datetime
from collections import defaultdict

from toolz.curried import get, keyfilter

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer
from tensorboardX import SummaryWriter

from hutil.common import CUDA, detach
from hutil.train.metrics import TrainLoss, Loss
from hutil.train._utils import _prepare_batch, set_lr


def create_supervised_evaluator(model, metrics=None,
                                device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            input, target = prepare_batch(batch, device=device)
            if hasattr(model, 'inference'):
                preds = model.inference(*input)
            else:
                preds = model(*input)
            if torch.is_tensor(preds):
                preds = (preds,)
            output = {
                "preds": preds,
                "target": target,
                'batch_size': input[0].size(0),
            }
            return output

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_trainer(
        model, criterion, optimizer, metrics=None,
        device=None, prepare_batch=_prepare_batch):
    if metrics is None:
        metrics = {}
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        input, target = prepare_batch(batch, device=device)
        preds = model(*input)
        if torch.is_tensor(preds):
            preds = (preds,)
        loss = criterion(*preds, *target)
        loss.backward()
        optimizer.step()
        output = {
            "preds": detach(preds),
            "target": detach(target),
            "loss": loss.item(),
            "batch_size": input[0].size(0),
        }
        return output

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Trainer:

    def __init__(self, model, criterion, optimizer, lr_scheduler=None,
                 metrics=None, evaluate_metrics=None, save_path=".", name="Net"):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics or {}
        self.evaluate_metrics = evaluate_metrics
        if evaluate_metrics is None:
            self.evaluate_metrics = metrics.copy()
            if 'loss' in metrics and isinstance(metrics['loss'], TrainLoss):
                self.evaluate_metrics['loss'] = Loss(criterion=criterion)
        self.save_path = save_path
        self.name = name

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(save_path, 'runs', self.name, current_time)
        self._writer = SummaryWriter(log_dir)

        self.metric_history = defaultdict(list)
        self._device = 'cuda' if CUDA else 'cpu'
        self._timer = Timer()
        self._epochs = 0

        self.model.to(self._device)

    def _log_epochs(self, engine, epochs):
        print("Epoch %d/%d" %
              (self._epochs + 1, self._epochs + 1 + epochs - engine.state.epoch))

    def _lr_scheduler_step(self, engine):
        self.lr_scheduler.step(self.epochs())

    def _increment_epoch(self, engine):
        self._epochs += 1

    def _log_results(self, engine, evaluator=None):
        elapsed = int(self._timer.value())
        msg = "elapsed: %ds\t" % elapsed
        for name, val in engine.state.metrics.items():
            if isinstance(val, float):
                msg += "%s: %.4f\t" % (name, val)
                self._writer.add_scalar(name, val, self.epochs())
            else:
                msg += "%s: %s\t" % (name, val)
                for i, v in enumerate(val):
                    self._writer.add_scalar("%s-%d" % (name, i + 1), v, self.epochs())
            self.metric_history[name].append(val)
        print(msg)

    def _log_val_results(self, engine, evaluator, per_epochs=1):
        if engine.state.epoch % per_epochs != 0:
            return
        msg = "validate ------\t"
        for name, val in evaluator.state.metrics.items():
            if isinstance(val, float):
                msg += "%s: %.4f\t" % (name, val)
                self._writer.add_scalar(name, val, self.epochs())
            else:
                msg += "%s: %s\t" % (name, val)
                for i, v in enumerate(val):
                    self._writer.add_scalar("%s-%d" % (name, i + 1), v, self.epochs())
            self.metric_history["val_" + name].append(val)
        print(msg)

    def _evaluate(self, engine, evaluator, val_loader, per_epochs=1):
        if engine.state.epoch % per_epochs == 0:
            evaluator.run(val_loader)

    def _terminate_on_iterations(self, engine, iterations):
        if engine.state.iteration == iterations:
            engine.terminate()

    def fit(self, train_loader, epochs=1, val_loader=None, save=None, iterations=None, callbacks=None):
        if val_loader is not None:
            validate = True
            if isinstance(val_loader, tuple):
                val_loader, eval_per_epochs = val_loader
            else:
                eval_per_epochs = 1
        else:
            validate = False
        callbacks = callbacks or []

        engine = create_supervised_trainer(
            self.model, self.criterion, self.optimizer,
            self.metrics, self._device)
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, StepOnIteration):
                engine.add_event_handler(
                    Events.ITERATION_STARTED, self._lr_scheduler_step)
            else:
                engine.add_event_handler(
                    Events.EPOCH_STARTED, self._lr_scheduler_step)

        self._timer.attach(engine,
                           start=Events.EPOCH_STARTED)
        engine.add_event_handler(Events.EPOCH_STARTED,
                                 self._log_epochs, epochs)

        if validate:
            evaluator = create_supervised_evaluator(
                self.model, self.evaluate_metrics, self._device)
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, self._evaluate, evaluator, val_loader, eval_per_epochs)
        else:
            evaluator = None

        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._increment_epoch)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._log_results, evaluator)
        if validate:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, self._log_val_results, evaluator, eval_per_epochs)

        # Set checkpoint
        if save:
            checkpoint_handler = save.parse(self)
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_handler, {"trainer": self})

        for callback in callbacks:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, _callback_wrapper(callback), self)

        if iterations:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, self._terminate_on_iterations, iterations)
            epochs = 1000

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
            "models": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": None,
            "metric_history": self.metric_history,
        }
        if self.lr_scheduler:
            s["lr_scheduler"] = self.lr_scheduler.state_dict()
        return s

    def load_state_dict(self, state_dict):
        epochs, model, optimizer, lr_scheduler, metric_history = get(
            ["epochs", "models", "optimizer", "lr_scheduler", "metric_history"], state_dict)
        self._epochs = epochs
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        if self.lr_scheduler and lr_scheduler:
            self.lr_scheduler.load_state_dict(lr_scheduler)
        self.metric_history = metric_history

    def epochs(self):
        return self._epochs

    def evaluate(self, test_loader):
        evaluator = create_supervised_evaluator(
            self.model, self.evaluate_metrics, self._device)
        return evaluator.run(test_loader).metrics

    def set_lr(self, lr):
        set_lr(lr, self.optimizer, self.lr_scheduler)


def _callback_wrapper(f):
    def func(engine, *args, **kwargs):
        return f(*args, **kwargs)

    return func


class StepOnIteration:

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
