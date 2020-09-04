from enum import Enum
from queue import Queue
from typing import Sequence, Mapping, Callable, Union

from hhutil.io import fmt_path, rm, PathLike

import torch
from horch.train.v2.base import Serializable
from horch.utils import time_now


class Callback(object):

    def __init__(self):
        pass

    def on_begin(self, engine):
        pass

    def on_end(self, engine):
        pass

    def on_epoch_begin(self, engine):
        pass

    def on_epoch_end(self, engine):
        pass

    def on_batch_begin(self, engine):
        pass

    def on_batch_end(self, engine):
        pass


class CallbackList(object):
    def __init__(self, callbacks: Sequence[Callback]):
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def on_begin(self, engine):
        for c in self.callbacks:
            c.on_begin(engine)

    def on_end(self, engine):
        for c in self.callbacks:
            c.on_end(engine)

    def on_epoch_begin(self, engine):
        for c in self.callbacks:
            c.on_epoch_begin(engine)

    def on_epoch_end(self, engine):
        for c in self.callbacks:
            c.on_epoch_end(engine)

    def on_batch_begin(self, engine):
        for c in self.callbacks:
            c.on_batch_begin(engine)

    def on_batch_end(self, engine):
        for c in self.callbacks:
            c.on_batch_end(engine)


def join_metric_logs(metrics, delim=" - "):
    logs = []
    for k, v in metrics.items():
        logs.append("%s: %.4f" % (k, v))
    return delim.join(logs)


class DefaultTrainLogger(Callback):

    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq

    def on_epoch_begin(self, engine):
        state = engine.state
        engine.log.info('Epoch %d/%d, lr: %f' % (state.epoch + 1, state.max_epochs, state.output['lr']))

    def on_batch_end(self, engine):
        state = engine.state
        i = state.iteration

        def log():
            engine.log.info("%s train %d/%d - %s" % (
                time_now(), i + 1, state.epoch_length, join_metric_logs(state.metrics, delim=" - ")))

        if self.log_freq == -1:
            if i == state.epoch_length - 1:
                log()
        else:
            if (i + 1) % self.log_freq == 0 or i == state.epoch_length - 1:
                log()


class DefaultEvalLogger(Callback):

    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq

    def on_batch_end(self, engine):
        state = engine.state
        i = state.iteration

        def log():
            engine.log.info("%s %s %d/%d - %s" % (
                time_now(), engine.stage, i + 1, state.epoch_length, join_metric_logs(state.metrics, delim=" - ")))

        if self.log_freq == -1:
            if i == state.epoch_length - 1:
                log()
        else:
            if (i + 1) % self.log_freq == 0 or i == state.epoch_length - 1:
                log()


class CallOn(Callback):

    def __init__(self, event, handler, freq):
        super().__init__()
        self.event = event
        self.freq = freq
        self.handler = handler

    def on_begin(self, engine):
        if self.event == Events.BEGIN:
            self.handler(engine)

    def on_end(self, engine):
        if self.event == Events.END:
            self.handler(engine)

    def on_epoch_begin(self, engine):
        epoch = engine.state.epoch + 1
        if self.event == Events.EPOCH_BEGIN and epoch % self.freq == 0:
            self.handler(engine)

    def on_epoch_end(self, engine):
        epoch = engine.state.epoch + 1
        if self.event == Events.EPOCH_END and epoch % self.freq == 0:
            self.handler(engine)

    def on_batch_begin(self, engine):
        iteration = engine.state.iteration + 1
        if self.event == Events.BATCH_BEGIN and iteration % self.freq == 0:
            self.handler(engine)

    def on_batch_end(self, engine):
        iteration = engine.state.iteration + 1
        if self.event == Events.BATCH_END and iteration % self.freq == 0:
            self.handler(engine)


class Events(Enum):

    BEGIN = "begin"
    END = "end"

    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"

    BATCH_BEGIN = "batch_begin"
    BATCH_END = "batch_end"


def call_on(event, freq, f):
    return CallOn(event, freq, f)


class Checkpoint(Callback):

    def __init__(self,
                 to_save: Mapping[str, Union[Serializable, Callable[[], Mapping]]],
                 save_freq: int,
                 save_dir: PathLike,
                 n_saved: int = 1):
        super().__init__()
        self.to_save = to_save
        self.save_freq = save_freq
        self.save_dir = fmt_path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_saved = n_saved
        self._saved = Queue(n_saved)

    def save(self, engine):
        if self._saved.full():
            self.delete()
        path = str(self.save_dir / f"epoch_{engine.state.epoch + 1}.pth")

        state_dict = {}
        for k, v in self.to_save.items():
            if hasattr(v, "state_dict"):
                state_dict[k] = v.state_dict()
            else:
                state_dict[k] = v()
        torch.save(state_dict, path)
        self._saved.put(path)
        print('Save checkpoint at %s' % path)

    def delete(self):
        if not self._saved.empty():
            path = self._saved.get()
            rm(path)

    def on_epoch_end(self, engine):
        if (engine.state.epoch + 1) % self.save_freq == 0:
            self.save(engine)

    def on_train_end(self, engine):
        if (engine.state.epoch + 1) % self.save_freq != 0:
            self.save(engine)