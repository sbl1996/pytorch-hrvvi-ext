import math
import random
from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.ops import dims, unsqueeze


def inverse_sigmoid(x):
    return math.log(x / (1 - x))


def focal_loss(input, target, weight=None, ignore_index=None, gamma=2, reduction='mean'):
    logpt = -F.cross_entropy(input, target, weight=weight, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


def focal_loss2(input, target, gamma=2, beta=1, alpha=0.25, eps=1e-6, reduction='mean'):
    xt = gamma * input + beta * (2 * target - 1)
    eps = inverse_sigmoid(1 - eps)
    xt = torch.clamp(xt, -eps, eps)
    return F.binary_cross_entropy_with_logits(
        xt, target,
        reduction=reduction,
        pos_weight=input.new_tensor(alpha)) / gamma


def conf_penalty(input, beta=0.1):
    p = F.softmax(input, dim=1)
    loss = (torch.log(p) * (beta * p - 1)).sum(dim=1).mean()
    return loss


@curry
def cross_entropy(input, target, weight=None, confidence_penalty=None):
    loss = F.cross_entropy(input, target, weight)
    if confidence_penalty:
        loss = loss + conf_penalty(input, beta=confidence_penalty)
    return loss


def f1_loss(pred, target, eps=1e-8, average='micro'):
    assert pred.shape == target.shape
    if average == 'samples':
        dim = dims(pred)[1:]
        tp = torch.sum(pred * target, dim=dim)
        fp = torch.sum((1 - pred) * target, dim=dim)
        fn = torch.sum(pred * (1 - target), dim=dim)
    elif average == 'micro':
        tp = torch.sum(pred * target)
        fp = torch.sum((1 - pred) * target)
        fn = torch.sum(pred * (1 - target))
    else:
        raise ValueError("`average` must be one of [`samples`, 'micro'], got `%s`" % average)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    return 1 - torch.mean(f1)


def f_beta_loss(pred, target, eps=1e-8, beta=1, average='micro'):
    assert pred.shape == target.shape
    assert pred.dtype == target.dtype
    if average == 'samples':
        dim = dims(pred)[1:]
        tp = torch.sum(pred * target, dim=dim)
        fp = torch.sum((1 - pred) * target, dim=dim)
        fn = torch.sum(pred * (1 - target), dim=dim)
    elif average == 'micro':
        tp = torch.sum(pred * target)
        fp = torch.sum((1 - pred) * target)
        fn = torch.sum(pred * (1 - target))
    else:
        raise ValueError("`average` must be one of [`samples`, 'micro'], got `%s`" % average)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = (1 + beta * beta) * p * r / ((beta * beta * p) + r + eps)
    return 1 - torch.mean(f1)


def dice_loss(pred, target):
    dim = dims(pred)
    numerator = 2 * torch.sum(pred * target, dim=dim)
    denominator = torch.sum(pred + target, dim=dim)
    losses = 1 - (numerator + 1) / (denominator + 1)
    return torch.mean(losses)


def weighted_bce_loss(input, target, ignore_index=None, reduction='mean', from_logits=True):
    n = target.size()[1:].numel()
    n = torch.full((len(target),), n, dtype=target.dtype, device=target.device)
    dim = dims(target)[1:]
    if ignore_index is not None:
        weight = (target != ignore_index).float()
        n = torch.sum(weight, dim=dim)
        target = target.masked_fill(target == ignore_index, 0)
    else:
        weight = 1
    n_pos = torch.sum(target, dim=dim)
    neg_weight = n_pos / n
    neg_weight = unsqueeze(neg_weight, dim)
    pos_weight = 1 - neg_weight
    weight *= target * pos_weight + (1 - target) * neg_weight
    if from_logits:
        return F.binary_cross_entropy_with_logits(input, target, weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(input, target, weight, reduction=reduction)


def binary_focal_loss(input, target, pos_weight=0.25, gamma=2, reduction='mean'):
    logpt = -F.binary_cross_entropy_with_logits(input, target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    if pos_weight is not None:
        alpha = torch.full_like(target, pos_weight)
        alpha[target == 0] = 1 - pos_weight
        loss = loss * alpha
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


class SegmentationLoss:

    def __init__(self, p=0.01, loss='f1', scale=1.0, **kwargs):
        self.p = p
        self.loss = loss
        self.scale = scale
        self.kwargs = kwargs

    def __call__(self, input, target):
        input = input.squeeze(1)
        target = target.type_as(input)
        if self.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(
                input, target, **self.kwargs)
        elif self.loss == 'focal':
            loss = binary_focal_loss(input, target, **self.kwargs)
        elif self.loss == 'f1':
            pred = torch.sigmoid(input)
            loss = f1_loss(pred, target, **self.kwargs)
        elif self.loss == 'f_beta':
            pred = torch.sigmoid(input)
            loss = f_beta_loss(pred, target, **self.kwargs)
        elif self.loss == 'dice':
            pred = torch.sigmoid(input)
            loss = dice_loss(pred, target)
        elif self.loss == 'bce+dice':
            pred = torch.sigmoid(input)
            loss1 = F.binary_cross_entropy_with_logits(
                input, target, **self.kwargs)
            loss2 = dice_loss(pred, target)
            loss = loss1 + loss2
        loss = loss * self.scale
        if random.random() < self.p:
            print("loss: %.4f" % loss.item())
        return loss


def calculate_gain(p, c):
    pc = p / c
    p1 = 1 - p + pc
    return -(p - pc) * math.log(pc) - p1 * math.log(p1)


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', label_smoothing=None):
        super().__init__()
        self.label_smoothing, self.reduction = label_smoothing, reduction

    def forward(self, output, target):
        if self.label_smoothing:
            c = output.size(1)
            log_probs = F.log_softmax(output, dim=1)
            if self.reduction == 'sum':
                loss = -log_probs.sum()
            else:
                loss = -log_probs.sum(dim=1)
                if self.reduction == 'mean':
                    loss = loss.mean()
            loss = loss * self.label_smoothing / c + (1 - self.label_smoothing) * F.nll_loss(log_probs, target, reduction=self.reduction)
            return loss - calculate_gain(self.label_smoothing, c)
        else:
            return F.cross_entropy(output, target, reduction=self.reduction)