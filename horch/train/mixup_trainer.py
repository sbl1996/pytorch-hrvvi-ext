import numpy as np

import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor
from torch.nn.utils import clip_grad_norm_

from horch.train.trainer_base import backward, TrainerBase


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_supervised_trainer(
        model, criterion, optimizer, metrics,
        device, alpha, clip_grad_norm=None, accumulation_steps=1, fp16=False):

    def step(engine, batch):
        model.train()
        x, y_true = convert_tensor(batch, device)
        x, y_a, y_b, lam = mixup_data(x, y_true, alpha)
        logits = model(x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        backward(loss, optimizer, fp16)
        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        outs = {
            "loss": loss.item(),
            "batch_size": x.size(0),
            "y_true_a": y_a,
            "y_true_b": y_true,
            "y_pred": y_b.detach(),
        }
        return outs

    engine = Engine(step)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_evaluator(model, metrics, device):

    def step(engine, batch):
        model.eval()
        x, y_true = convert_tensor(batch, device)
        with torch.no_grad():
            logits = model(x)
        output = {
            "y_pred": logits,
            "y_true": y_true,
            'batch_size': x[0].size(0),
        }
        return output

    engine = Engine(step)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Trainer(TrainerBase):

    def _create_train_engine(self):
        engine = create_supervised_trainer(
            self.model, self.criterion, self.optimizers[0], self.metrics, self.device, fp16=self.fp16)
        return engine

    def _create_eval_engine(self):
        engine = create_supervised_evaluator(self.model, self.test_metrics, self.device)
        return engine
