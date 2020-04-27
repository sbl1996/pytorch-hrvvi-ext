from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn
from toolz.curried import curry

from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from horch.train.trainer_base import TrainerBase, backward


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
        model, criterion, optimizer_arch, optimizer_model, lr_scheduler, metrics, device, clip_grad_norm=5, fp16=False):
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
        backward(loss, optimizer_arch, fp16)
        optimizer_arch.step()

        optimizer_model.zero_grad()
        for p in model.arch_parameters():
            p.requires_grad_(False)
        for p in model.model_parameters():
            p.requires_grad_(True)
        logits_search = model(input_search)
        loss_search = criterion(logits_search, target_search)
        backward(loss_search, optimizer_model, fp16)
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer_model.step()

        lr_scheduler.step(engine.state.iteration / engine.state.epoch_length)
        return {
            "loss": loss.item(),
            "batch_size": input.size(0),
            "y_true": target_search,
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
            "y_true": target,
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


class DARTSTrainer(TrainerBase):

    def _create_train_engine(self):
        engine = create_darts_trainer(
            self.model, self.criterion, self.optimizers[0], self.optimizers[1],
            self.lr_schedulers[0], self.metrics, self.device, fp16=self.fp16)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, log_metrics(stage='train'))
        return engine

    def _create_eval_engine(self):
        engine = create_darts_evaluator(self.model, self.test_metrics, self.device)
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, log_metrics(stage='valid'))
        return engine