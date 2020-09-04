from typing import Callable, Sequence, Dict, Optional

from hhutil.functools import pick

from ignite.metrics import Metric

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.utils import convert_tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from horch.train.classification.mix import MixBase
from horch.train.trainer_base import backward, TrainerBase


def create_supervised_trainer(
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        metrics: Dict[str, Metric],
        device: torch.device,
        mix: Optional[MixBase] = None, clip_grad_norm=None, accumulation_steps=1, fp16=False):

    def step(engine, batch):
        model.train()
        x, y_true = convert_tensor(batch, device)

        if mix:
            x, y_true = mix(x, y_true)
            logits = model(x)
            loss = mix.loss(criterion, logits, y_true)
        else:
            logits = model(x)
            loss = criterion(logits, y_true)

        backward(loss, optimizer, fp16)
        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        outs = {
            "loss": loss.item(),
            "batch_size": x.size(0),
            "y_true": y_true,
            "y_pred": logits.detach(),
            "lr": optimizer.param_groups[0]['lr'],
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

    def _attach_prograss_bar(self, train_engine: Engine):
        pb = ProgressBar()
        # pb.attach(train_engine, output_transform=lambda x: print(x))
        pb.attach(train_engine, output_transform=pick(['lr', 'loss']))

    def _create_train_engine(self):
        engine = create_supervised_trainer(
            self.model, self.criterion, self.optimizers[0], self.metrics, self.device,
            self._kwargs.get('mix'), fp16=self.fp16)
        return engine

    def _create_eval_engine(self):
        engine = create_supervised_evaluator(self.model, self.test_metrics, self.device)
        return engine
