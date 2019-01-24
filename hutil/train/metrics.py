from toolz import curry
from toolz.curried import get

from ignite.metrics import Accuracy as IgniteAccuracy
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from hutil.functools import lmap


class Average(Metric):

    def __init__(self, output_transform):
        super().__init__(output_transform)

    def reset(self):
        self._num_examples = 0
        self._sum = 0

    def update(self, output):
        val, N = output
        self._sum += val * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Metric must have at least one example before it can be computed')
        return self._sum / self._num_examples


def get_loss(output):
    loss, y_pred = get(["loss", "y_pred"], output)
    return loss, y_pred[0].size(0)


def Loss():
    return Average(get_loss)


@curry
def take_until_eos(eos_index, tokens):
    for i, token in enumerate(tokens):
        if token == eos_index:
            return tokens[:i]
    return tokens


def bleu(y_pred, y, eos_index):
    y_pred = y_pred.argmax(dim=1)
    output = lmap(take_until_eos(eos_index), y_pred.tolist())
    target = lmap(take_until_eos(eos_index), y.tolist())
    target = lmap(lambda x: [x], target)
    score = corpus_bleu(
        target, output, smoothing_function=SmoothingFunction().method1)
    return score


@curry
def get_bleu(eos_index, output):
    y_pred, y = get(["y_pred", "y"], output)
    y_pred = y_pred[0]
    y = y[0]
    return bleu(y_pred, y, eos_index), y.size(0)


def Accuracy():
    def output_transform(output):
        y_pred, y = get(["y_pred", "y"], output)
        return y_pred[0], y[0]
    return IgniteAccuracy(output_transform)


def Bleu(eos_index):
    return Average(get_bleu(eos_index))


def get_lossD(output):
    lossD, batch_size = get(["lossD", "batch_size"], output)
    return lossD, batch_size


def LossD():
    return Average(get_lossD)


def get_lossG(output):
    lossG, batch_size = get(["lossG", "batch_size"], output)
    return lossG, batch_size


def LossG():
    return Average(get_lossG)
