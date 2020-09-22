import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor
from torch import nn as nn

from horch.legacy.train.trainer_base import backward, TrainerBase


def requires_grad(network: nn.Module, arch: bool, model: bool):
    for p in network.arch_parameters():
        p.requires_grad_(arch)
    for p in network.model_parameters():
        p.requires_grad_(model)


def create_darts_trainer(
    network, criterion, optimizer_arch, optimizer_model, metrics, device,
    clip_grad_norm=5, fp16=False):

    def step(engine, batch):
        network.train()

        (input, target), (input_search, target_search) = convert_tensor(batch, device)

        optimizer_arch.zero_grad()
        requires_grad(network, arch=True, model=False)
        logits = network(input)
        loss = criterion(logits, target)
        backward(loss, optimizer_arch, fp16)
        optimizer_arch.step()

        optimizer_model.zero_grad()
        requires_grad(network, arch=False, model=True)
        logits_search = network(input_search)
        loss_search = criterion(logits_search, target_search)
        backward(loss_search, optimizer_model, fp16)
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(network.parameters(), clip_grad_norm)
        optimizer_model.step()

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


def create_darts_evaluator(network, metrics, device):
    def step(engine, batch):
        network.eval()
        input, target = convert_tensor(batch, device)
        with torch.no_grad():
            output = network(input)

        return {
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": output,
        }

    engine = Engine(step)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class DARTSTrainer(TrainerBase):

    def _create_train_engine(self):
        engine = create_darts_trainer(
            self.model, self.criterion, self.optimizers[0], self.optimizers[1],
            self.metrics, self.device, fp16=self.fp16)
        return engine

    def _create_eval_engine(self):
        engine = create_darts_evaluator(self.model, self.test_metrics, self.device)
        return engine
