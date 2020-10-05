from abc import ABCMeta, abstractmethod
from typing import Any
from toolz.curried import get

class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


class Metric(metaclass=ABCMeta):

    def __init__(self, output_transform=lambda x: x, name=None):
        super().__init__()
        assert name is not None
        self.name = name
        self._output_transform = output_transform
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, output) -> None:
        pass

    @abstractmethod
    def compute(self) -> Any:
        pass


class Average(Metric):

    def __init__(self, output_transform, name):
        super().__init__(output_transform, name)
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

    def __init__(self, name="loss"):
        super().__init__(output_transform=self.output_transform, name=name)

    @staticmethod
    def output_transform(state):
        return get(["loss", "batch_size"], state)


class Loss(Average):

    def __init__(self, criterion, name="loss"):
        self.criterion = criterion
        super().__init__(output_transform=self.output_transform, name=name)

    def output_transform(self, state):
        y_true, y_pred = get(["y_true", "y_pred"], state)
        loss = self.criterion(y_pred, y_true).item()
        return loss, y_pred.shape[0]
