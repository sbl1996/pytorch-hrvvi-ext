from toolz.curried import get

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import functional as F

from ignite.metrics import Accuracy as IgniteAccuracy, Metric

from horch.train.metrics import Average


def topk_accuracy(y_true, y_pred, k=5):

    num_examples = np.prod(y_true.size())
    topk_pred = torch.topk(y_pred, k=k, dim=1)[1]
    num_corrects = torch.sum(topk_pred == y_true.unsqueeze(1)).item()
    accuracy = num_corrects / num_examples
    return accuracy, num_examples


class TopKAccuracy(Average):
    _required_output_keys = ["y_pred", "y_true"]

    def __init__(self, k=5):
        self.k = k
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y_true = output
        return topk_accuracy(y_true, y_pred, k=self.k)


def accuracy(y_true, y_pred):
    num_examples = np.prod(y_true.size())
    pred = torch.argmax(y_pred, dim=1)
    num_corrects = torch.sum(pred == y_true).item()
    acc = num_corrects / num_examples
    return acc, num_examples


class Accuracy(Average):

    def __init__(self, mixup=False):
        self.mixup = mixup
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        if self.mixup:
            y_pred, y_true, batch_size, lam = get(["y_pred", "y_true", "batch_size", "mixup_lambda"], output)
            y_a, y_b = y_true
            pred = torch.argmax(y_pred, dim=1)
            num_corrects = (lam * pred.eq(y_a).cpu().sum().float()
                        + (1 - lam) * pred.eq(y_b).cpu().sum().float())
            acc = num_corrects / batch_size
            return acc, batch_size
        else:
            y_pred, y_true = get(["y_pred", "y_true"], output)
            return accuracy(y_true, y_pred)


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

    _required_output_keys = ["y_pred", "y_true"]

    def __init__(self, metric_func):
        super().__init__()
        self.metric_func = metric_func

    def reset(self):
        self.y_preds = []
        self.y_trues = []

    def update(self, output):
        y_pred, y_true = output
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_preds, dim=0)
        y_true = torch.cat(self.y_trues, dim=0)
        return self.metric_func(y_pred, y_true)