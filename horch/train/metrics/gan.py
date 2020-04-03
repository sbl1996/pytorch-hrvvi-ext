import torch
from ignite.engine import Events
from ignite.metrics import Metric
from toolz.curried import get


class IAverage(Metric):
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
        return self._sum / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine, name):
        result = self.compute()
        if torch.is_tensor(result) and len(result.shape) == 0:
            result = result.item()
        engine.state.metrics[name] = result

    def attach(self, engine, name):
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)


class LossD(IAverage):

    def __init__(self):
        super().__init__(self.output_transform)

    @staticmethod
    def output_transform(output):
        lossD, batch_size = get(["lossD", "batch_size"], output)
        return lossD, batch_size


class LossG(IAverage):

    def __init__(self):
        super().__init__(self.output_transform)

    @staticmethod
    def output_transform(output):
        lossG, batch_size = get(["lossG", "batch_size"], output)
        return lossG, batch_size