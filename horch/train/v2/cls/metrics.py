import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F

from horch.train.v2.metrics import Average, Metric


def topk_accuracy(y_true, y_pred, k=5):
    num_examples = np.prod(y_true.size())
    topk_pred = torch.topk(y_pred, k=k, dim=1)[1]
    num_corrects = torch.sum(topk_pred == y_true.unsqueeze(1)).item()
    accuracy = num_corrects / num_examples
    return accuracy, num_examples


def accuracy(y_true, y_pred):
    num_examples = np.prod(y_true.size())
    pred = torch.argmax(y_pred, dim=1)
    num_corrects = torch.sum(pred == y_true).item()
    acc = num_corrects / num_examples
    return acc, num_examples


class TopKAccuracy(Average):

    _required_output_keys = ("y_true", "y_pred")

    def __init__(self, k=5, name=None):
        self.k = k
        name = self._init_name(name)
        super().__init__(self.output_transform, name)

    def output_transform(self, output):
        y_true, y_pred = output
        return topk_accuracy(y_true, y_pred, self.k)

    def _init_name(self, name):
        if name is None:
            name = "acc" + str(self.k)
        return name


class Accuracy(Average):

    _required_output_keys = ("y_true", "y_pred")

    def __init__(self, name="acc"):
        super().__init__(self.output_transform, name)

    @staticmethod
    def output_transform(output):
        y_true, y_pred = output
        return accuracy(y_true, y_pred)


class ROCAUC(Metric):

    _required_output_keys = ("y_true", "y_pred")

    def __init__(self, name='auc'):
        super().__init__(name=name)

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
        y_score = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
        y_true = y_true.cpu().numpy()
        return roc_auc_score(y_true, y_score)
