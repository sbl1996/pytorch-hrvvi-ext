from toolz.curried import get

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import functional as F

from ignite.metrics import Accuracy as IgniteAccuracy, Metric

from horch.train.metrics import Average


def topk_accuracy(y_true, y_pred, k=5):
    r"""
    Parameters
    ----------
    input : torch.Tensor
        Tensor of shape (batch_size, num_classes, ...)
    target : torch.Tensor
        Tensor of shape (batch_size, ...)
    k : int
        Default: 5
    """
    num_examples = np.prod(y_true.size())
    topk_pred = torch.topk(y_pred, k=k, dim=1)[1]
    num_corrects = torch.sum(topk_pred == y_true.unsqueeze(1)).item()
    accuracy = num_corrects / num_examples
    return accuracy, num_examples


class TopKAccuracy(Average):
    r"""
    Args:
        k: default to 5
    Inputs:
        preds: (batch_size, C, ...) or (batch_size, ...)
        y:      (batch_size, ...)
    """

    def __init__(self, k=5):
        self.k = k
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_true, y_pred = get(["y_true", "y_pred"], output)
        return topk_accuracy(y_true, y_pred, k=self.k)


class Accuracy(IgniteAccuracy):
    r"""
    Inputs:
        preds: (batch_size, C, ...) or (batch_size, ...)
        y:      (batch_size, ...)
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    @staticmethod
    def output_transform(output):
        y_pred, y_true = get(["y_pred", "y_true"], output)
        return y_pred, y_true


class ROCAUC(Metric):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.y_preds = []
        self.y_trues = []

    def update(self, output):
        y_pred, y_true = get(["y_pred", "y_true"], output)
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_preds, dim=0)
        y_true = torch.cat(self.y_trues, dim=0)
        y_score = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
        y_true = y_true.cpu().numpy()
        return roc_auc_score(y_true, y_score)


class EpochSummary(Metric):

    def __init__(self, metric_func):
        super().__init__()
        self.metric_func = metric_func

    def reset(self):
        self.y_preds = []
        self.y_trues = []

    def update(self, output):
        y_pred, y_true = get(["y_pred", "y_true"], output)
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_preds, dim=0)
        y_true = torch.cat(self.y_trues, dim=0)
        return self.metric_func(y_pred, y_true)