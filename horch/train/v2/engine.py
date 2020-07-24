import logging
from typing import Sequence, Mapping
from collections import OrderedDict

from torch.utils.data import DataLoader

from horch.train.v2.base import Serializable
from horch.train.v2.callback import Callback, CallbackList, CallOn


class State:

    def __init__(self, **kwargs):
        self.iteration = 0
        self.epoch = 0
        self.epoch_length = None

        self.output = None
        self.batch = None
        self.metrics = {}
        self.dataloader = None
        self.seed = None

        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(Serializable):
    _state_dict_all_req_keys = ("epoch",)

    def __init__(self, step_fn, callbacks: Sequence[Callback], **kwargs):
        self.step_fn = step_fn
        self.callbacks = CallbackList(callbacks)
        self.state = State()
        self.log = logging.getLogger(__name__)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def call(self, callback: Callback):
        self.callbacks.append(callback)

    def call_on(self, event, f, freq=1):
        self.callbacks.append(CallOn(event, f, freq))

    def run(self, dataloader: DataLoader, num_epochs: int):
        state = self.state
        cbks = self.callbacks

        state.dataloader = dataloader
        state.epoch_length = len(dataloader)
        state.total_epochs = state.epoch + num_epochs
        start_epoch = state.epoch
        total_epochs = state.total_epochs
        cbks.on_begin(self)
        for epoch in range(start_epoch, total_epochs):
            state.epoch = epoch
            self._run_epoch()
        cbks.on_end(self)

    def _run_epoch(self):
        state = self.state
        cbks = self.callbacks

        cbks.on_epoch_begin(self)
        for iteration, batch in enumerate(state.dataloader):
            state.iteration = iteration
            state.batch = batch

            cbks.on_batch_begin(self)
            output = self.step_fn(self, state.batch)
            state.output = output
            cbks.on_batch_end(self)
        cbks.on_epoch_end(self)

    def load_state_dict(self, state_dict: Mapping):
        super().load_state_dict(state_dict)

        self.state = State(epoch=state_dict["epoch"])

    def state_dict(self) -> OrderedDict:
        state = self.state
        return OrderedDict({
            "epoch": state.epoch,
        })


def backward(loss, optimizer, fp16):
    if fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return

