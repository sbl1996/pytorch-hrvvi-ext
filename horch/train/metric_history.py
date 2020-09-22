from copy import deepcopy
from typing import Sequence, Mapping
from collections import OrderedDict

class MetricHistory:

    def __init__(self, stages: Sequence[str]):
        self.stages = stages
        # stage -> epoch -> metric -> value
        self._history = {
            stage: {}
            for stage in stages
        }

    def state_dict(self) -> OrderedDict:
        return OrderedDict(deepcopy(self._history))

    def load_state_dict(self, state_dict: Mapping):
        for k in self._history:
            self._history[k] = deepcopy(state_dict[k])

    def record(self, stage, epoch, metric, value):
        h = self._history[stage]
        if epoch not in h:
            h[epoch] = {}
        h[epoch][metric] = value

    def get_metric(self, metric, stage=None, start=None, end=None):
        if stage is None:
            return {
                stage: self.get_metric(metric, stage, start, end)
                for stage in self.stages
            }
        else:
            h = self._history[stage]
            epochs = list(h.keys())
            min_epoch, max_epochs = min(epochs), max(epochs)
            if start is None:
                start = min_epoch
            if end is None:
                end = max_epochs
            values = [h[e].get(metric) for e in range(start, end + 1)]
            if all(v is None for v in values):
                return None
            elif len(values) == 1:
                return values[0]
            else:
                return values

    def get_epochs(self, start, end, stage=None):
        if stage is None:
            h = {
                stage: self.get_epochs(start, end, stage)
                for stage in self.stages
            }
            for k in h.keys():
                if h[k] is None:
                    del h[k]
            return h
        else:
            h = self._history[stage]
            metrics = set([m for e in range(start, end + 1) for m in h[e].keys()])
            return {
                m: self.get_metric(m, stage, start, end)
                for m in metrics
            }