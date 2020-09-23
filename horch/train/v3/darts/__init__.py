import torch
import torch.nn as nn

from horch.common import convert_tensor
from horch.train.v3.learner import Learner, requires_grad, backward


class DARTSLearner(Learner):

    def __init__(self, network, criterion, optimizer_arch, optimizer_model, lr_scheduler,
                 clip_grad_norm=5, **kwargs):
        super().__init__(network, criterion, (optimizer_arch, optimizer_model),
                         lr_scheduler, clip_grad_norm=clip_grad_norm, train_arch=True, **kwargs)

    def train_batch(self, batch):
        state = self._state['train']
        network = self.model
        optimizer_arch, optimizer_model = self.optimizers
        lr_scheduler = self.lr_schedulers[0]

        network.train()
        input, target, input_search, target_search = convert_tensor(batch, self.device)

        if self.train_arch:
            requires_grad(network, arch=True, model=False)
            optimizer_arch.zero_grad()
            logits = network(input)
            loss = self.criterion(logits, target)
            backward(loss, optimizer_arch, self.fp16)
            optimizer_arch.step()

        requires_grad(network, arch=False, model=True)
        lr_scheduler.step(state['epoch'] + (state['step'] / state['steps']))
        optimizer_model.zero_grad()
        logits_search = network(input_search)
        loss_search = self.criterion(logits_search, target_search)
        backward(loss_search, optimizer_model, self.fp16)
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(network.parameters(), self.clip_grad_norm)
        optimizer_model.step()

        state.update({
            "loss": loss.item() if self.train_arch else loss_search.item(),
            "batch_size": input.size(0),
            "y_true": target_search,
            "y_pred": logits_search.detach(),
        })

    def eval_batch(self, batch):
        state = self._state['eval']
        network = self.model

        network.eval()
        input, target = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = network(input)

        state.update({
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": output,
        })

    def test_batch(self, batch):
        state = self._state['test']
        network = self.model

        network.eval()
        input = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = network(input)

        state.update({
            "batch_size": input.size(0),
            "y_pred": output,
        })
