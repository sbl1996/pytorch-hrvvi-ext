from abc import ABCMeta, abstractmethod
from typing import Any, Mapping
from toolz import identity
from horch.train.v2.callback import Callback


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


class Metric(Callback, metaclass=ABCMeta):

    _required_output_keys = ("y", "y_pred")

    def __init__(self, output_transform=lambda x: x, name=None):
        super().__init__()
        assert name is not None
        self.name = name
        self._output_transform = output_transform
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to it's initial state.

        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output) -> None:
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function.
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest. However, if a :class:`~collections.abc.Mapping` is returned,
                 it will be (shallow) flattened into `engine.state.metrics` when
                 :func:`~ignite.metrics.Metric.completed` is called.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass

    def on_epoch_begin(self, engine):
        self.reset()

    def on_batch_end(self, engine):
        output = self._output_transform(engine.state.output)
        if isinstance(output, Mapping):
            if self._required_output_keys is None:
                raise TypeError(
                    "Transformed engine output for {} metric should be a tuple/list, but given {}".format(
                        self.__class__.__name__, type(output)
                    )
                )
            if not all([k in output for k in self._required_output_keys]):
                raise ValueError(
                    "When transformed engine's output is a mapping, "
                    "it should contain {} keys, but given {}".format(self._required_output_keys, list(output.keys()))
                )
            output = tuple(output[k] for k in self._required_output_keys)
        self.update(output)
        engine.state.metrics[self.name] = self.compute()


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
    _required_output_keys = ("loss", "batch_size")

    def __init__(self, name="loss"):
        super().__init__(identity, name)


class Loss(Average):
    _required_output_keys = ("y_pred", "y_true")

    def __init__(self, criterion, name="loss"):
        self.criterion = criterion
        super().__init__(output_transform=self.output_transform, name=name)

    def output_transform(self, output):
        y_pred, y_true = output
        loss = self.criterion(y_pred, y_true).item()
        return loss, y_pred.shape[0]