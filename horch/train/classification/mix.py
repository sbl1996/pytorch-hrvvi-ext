import math
import numpy as np

import torch


class MixBase:

    def __init__(self):
        pass

    def __call__(self, x, y):
        pass

    def loss(self, criterion, y_pred, y_true):
        pass


def calculate_gain(p):
    if p == 0 or p == 1:
        return 0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


class Mixup(MixBase):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.lam = None

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        self.lam = lam
        return mixed_x, (y_a, y_b)

    def loss(self, criterion, y_pred, y_true):
        y_a, y_b = y_true
        loss = self.lam * criterion(y_pred, y_a) + (1 - self.lam) * criterion(y_pred, y_b)
        return loss - calculate_gain(self.lam)


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


class CutMix(MixBase):

    def __init__(self, beta, prob):
        super().__init__()
        self.beta = beta
        self.prob = prob
        self.lam = None

    def __call__(self, x, y):
        r = np.random.rand()
        if r > self.prob:
            self.lam = None
            return x, y

        lam = np.random.beta(self.beta, self.beta) if self.beta > 0 else 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size, device=x.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        self.lam = lam
        return x, (y_a, y_b)

    def loss(self, criterion, y_pred, y_true):
        if self.lam:
            y_a, y_b = y_true
            loss = self.lam * criterion(y_pred, y_a) + (1 - self.lam) * criterion(y_pred, y_b)
            return loss - calculate_gain(self.lam)
        else:
            return criterion(y_pred, y_true)


def get_mix(cfg):
    if cfg is None:
        return None
    mix_type = cfg.type
    cfg.pop("type")
    mix = eval(mix_type)(**cfg)
    return mix

