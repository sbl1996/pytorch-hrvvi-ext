import torch
import torch.nn as nn

from horch.common import convert_tensor
from horch.train.v3.learner import Learner, backward


class CNNLearner(Learner):

    def __init__(self, model, criterion, optimizer, lr_scheduler,
                 clip_grad_norm=5, **kwargs):
        super().__init__(model, criterion, optimizer,
                         lr_scheduler, clip_grad_norm=clip_grad_norm, **kwargs)

    def train_batch(self, batch):
        state = self._state['train']
        model = self.model
        optimizer = self.optimizers[0]
        lr_scheduler = self.lr_schedulers[0]

        model.train()
        input, target = convert_tensor(batch, self.device)

        lr_scheduler.step(state['epoch'] + (state['step'] / state['steps']))
        optimizer.zero_grad()
        outputs = self.model(input)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            logits, logits_aux = outputs
        else:
            logits = outputs
        loss = self.criterion(outputs, target)
        backward(loss, optimizer, self.fp16)
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        optimizer.step()

        state.update({
            "loss": loss.item(),
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": logits.detach(),
        })

    def eval_batch(self, batch):
        state = self._state['eval']
        model = self.model

        model.eval()
        input, target = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = model(input)

        state.update({
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": output,
        })

    def test_batch(self, batch):
        state = self._state['test']
        model = self.model

        model.eval()
        input = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = model(input)

        state.update({
            "batch_size": input.size(0),
            "y_pred": output,
        })
