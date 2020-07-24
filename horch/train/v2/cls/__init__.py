from typing import Union, Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from ignite.utils import convert_tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from horch.common import CUDA
from horch.train.v2.engine import Engine, backward
from horch.train.v2.callback import DefaultTrainLogger, DefaultEvalLogger, ModelCheckpoint
from horch.train.v2.metrics import Metric


def create_supervised_trainer(
    model: nn.Module,
    criterion: Union[Callable, nn.Module],
    optimizer: Optimizer,
    metrics: List[Metric],
    clip_grad_norm: Optional[float] = None,
    fp16: bool = False,
    device: Union[str, torch.device] = 'auto',
):
    if device == 'auto':
        device = 'cuda' if CUDA else "cpu"

    def train_batch(engine, batch):
        engine.model.train()

        x, y_true = convert_tensor(batch, device)

        logits = engine.model(x)
        loss = engine.criterion(logits, y_true)
        backward(loss, engine.optimizer, fp16)
        engine.optimizer.step()
        engine.optimizer.zero_grad()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)

        outs = {
            "loss": loss.item(),
            "batch_size": x.size(0),
            "y_true": y_true,
            "y_pred": logits.detach(),
            "lr": engine.optimizer.param_groups[0]['lr'],
        }
        return outs

    callbacks = [*metrics]
    engine = Engine(
        train_batch, callbacks,
        model=model, criterion=criterion, optimizer=optimizer)
    return engine


def create_supervised_evaluator(
    model: nn.Module,
    criterion: Union[Callable, nn.Module],
    metrics: List[Metric],
    device: Union[str, torch.device] = 'auto',
):
    if device == 'auto':
        device = 'cuda' if CUDA else "cpu"

    def test_batch(engine, batch):
        engine.model.eval()

        x, y_true = convert_tensor(batch, device)

        with torch.no_grad():
            logits = engine.model(x)

        output = {
            "y_true": y_true,
            "y_pred": logits,
            "batch_size": x.shape[0],
        }

        return output

    callbacks = [*metrics]
    engine = Engine(test_batch, callbacks,
                    model=model, criterion=criterion)

    return engine
