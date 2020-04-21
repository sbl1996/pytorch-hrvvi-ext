from toolz.curried import get

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Average(Metric):

    def __init__(self, output_transform):
        super().__init__(output_transform)
        self._num_examples = 0
        self._sum = 0

    def reset(self):
        self._num_examples = 0
        self._sum = 0

    def update(self, output):
        val, n = output
        self._sum += val * n
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Metric must have at least one example before it can be computed')
        return self._sum / self._num_examples


class TrainLoss(Average):
    r"""
    Reuse training loss to avoid extra computations.
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    @staticmethod
    def output_transform(output):
        loss, batch_size = get(["loss", "batch_size"], output)
        return loss, batch_size


class Loss(Average):
    r"""
    Reuse training loss to avoid extra computations.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y_true, batch_size = get(["y_pred", "y_true", "batch_size"], output)
        loss = self.criterion(y_pred, y_true).item()
        return loss, batch_size


class EpochSummary(Metric):

    def __init__(self, metric_func):
        super().__init__()
        self.metric_func = metric_func

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, output):
        preds, target = get(["preds", "target"], output)
        self.preds.append(preds[0])
        self.targets.append(target[0])

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.metric_func(preds, targets)