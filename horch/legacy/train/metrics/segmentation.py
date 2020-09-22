from PIL import Image
from toolz.curried import get

import numpy as np

import torch
from ignite.metrics import Metric

from horch.legacy.train.metrics import Average


class MeanIoU(Metric):

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        super().__init__(self.output_transform)

    def reset(self):
        self.total_cm = np.zeros(self.num_classes, self.num_classes)

    def update(self, output):
        cm = output
        self.total_cm += cm

    def output_transform(self, output):
        y_true, y_pred = get(["y_true", "y_pred"], output)
        c = self.num_classes

        if isinstance(y_true, Image.Image):
            y_true = [np.array(img) for img in y_true]
        elif torch.is_tensor(y_true):
            y_true = y_true.cpu().byte().numpy()

        y_pred = y_pred.argmax(dim=1)
        y_pred = y_pred.cpu().byte().numpy()

        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        if self.ignore_index is not None:
            mask = np.not_equal(y_true, self.ignore_index)
            y_pred = np.where(mask, y_pred, c)
            y_true = np.where(mask, y_true, c)

        current_cm = confusion_matrix(y_true, y_pred, self.num_classes + 1)[:c, :c]
        return current_cm


def confusion_matrix(y_true, y_pred, num_classes):
    c = num_classes
    return np.reshape(np.bincount(y_true * c + y_pred, minlength=c * c), (c, c))


class PixelAccuracy(Average):

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        super().__init__(self.output_transform)

    def output_transform(self, output):
        y_true, y_pred, batch_size = get(
            ["y_true", "y_pred", "batch_size"], output)

        y_pred = y_pred.argmax(dim=1)

        accs = []
        for i in range(batch_size):
            y = y_true[i]
            p = y_pred[i]
            tp = (y == p).sum()
            if self.ignore_index is not None:
                tp += (y == self.ignore_index).sum()
            accs.append(tp.cpu().item() / np.prod(y.shape))
        acc = np.mean(accs)
        return acc, batch_size


class F1Score(Metric):
    r"""
    """

    def __init__(self, threshold=0.5, ignore_index=None, eps=1e-8, from_logits=True):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.eps = eps
        self.from_logits = from_logits
        super().__init__(self.output_transform)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, output):
        tp, fp, fn = output
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        p = self.tp / (self.tp + self.fp + self.eps)
        r = self.tp / (self.tp + self.fn + self.eps)

        f1 = 2 * p * r / (p + r + self.eps)
        return f1

    def output_transform(self, output):
        y, p = get(["y_true", "y_pred"], output)

        if p.ndim == 4:
            if p.size(1) == 1:
                p = p.squeeze(1)
                if self.from_logits:
                    p = torch.sigmoid(p)
            elif p.size(1) == 2:
                if self.from_logits:
                    p = torch.softmax(p, dim=1)[:, 1, :, :]
        elif p.ndim == 3:
            if self.from_logits:
                p = torch.sigmoid(p)
        p = p > self.threshold
        p = p.long()
        y = y.long()

        if self.ignore_index is None:
            w = torch.ones_like(y)
        else:
            w = (y != self.ignore_index).long()
        tp = torch.sum(p * y * w).item()
        fp = torch.sum((1 - p) * y * w).item()
        fn = torch.sum(p * (1 - y) * w).item()
        return tp, fp, fn