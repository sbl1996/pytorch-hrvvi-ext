import math
import random

import torch
import torch.nn.functional as F
from toolz import curry

from horch import cuda
from horch.ops import dims, unsqueeze


def inverse_sigmoid(x):
    return math.log(x / (1-x))


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
    eps = inverse_sigmoid(1-eps)
    xt = torch.clamp(xt, -eps, eps)
    return F.binary_cross_entropy_with_logits(
        xt, target,
        reduction=reduction,
        pos_weight=input.new_tensor(alpha)) / gamma


def iou_loss(prediction, ground_truth, reduction='mean'):
    l_p = prediction[..., 0]
    t_p = prediction[..., 1]
    r_p = prediction[..., 2]
    b_p = prediction[..., 3]
    l_t = ground_truth[..., 0]
    t_t = ground_truth[..., 1]
    r_t = ground_truth[..., 2]
    b_t = ground_truth[..., 3]

    area_p = (t_p + b_p) * (l_p + r_p)
    area_t = (t_t + b_t) * (l_t + r_t)
    inter_h = torch.min(t_p, t_t) + torch.min(b_p, b_t)
    inter_w = torch.min(l_p, l_t) + torch.min(r_p, r_t)
    inter = inter_h * inter_w
    iou = inter / (area_p + area_t - inter)
    loss = -torch.log(iou)
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError("reduction must be mean or sum")


def loc_kl_loss(loc_p, log_var_p, loc_t, reduction='sum'):
    r"""
    Parameters
    ----------
    loc_p : torch.Tensor
        (N, 4)
    log_var_p : torch.Tensor
        (N, 4)
    loc_t : torch.Tensor
        (N, 4)
    reduction : str
        `sum` or `mean`
    """
    loc_e = (loc_t - loc_p).abs()
    loss = (torch.pow(loc_e, 2) / 2).masked_fill(loc_e > 1, 0) + (loc_e - 1/2).masked_fill(loc_e <= 1, 0)
    loss = loss * torch.exp(-log_var_p) + log_var_p / 2
    if reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def normal_nll_loss(x, mu, var, eps=1e-6):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    logli = -0.5 * (var.mul(2 * math.pi) + eps).log() - (x - mu).pow(2).div(var.mul(2.0) + eps)
    nll = -(logli.sum(dim=1).mean())

    return nll


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


def f1_loss(pred, target, eps=1e-8, average='samples'):
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

    f1 = 2 * p * r / (p + r + eps)
    return 1 - torch.mean(f1)


def dice_loss(pred, target):
    assert pred.shape == target.shape
    assert pred.dtype == target.dtype
    dim = dims(pred)
    numerator = 2 * torch.sum(pred * target, dim=dim)
    denominator = torch.sum(pred + target, dim=dim)
    losses = 1 - (numerator + 1) / (denominator + 1)
    return torch.mean(losses)


def weighted_binary_cross_entropy_with_logits(input, target, ignore_index=None, reduction='mean'):
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
    pos_weight = n_pos / n
    pos_weight = unsqueeze(pos_weight, dim)
    neg_weight = 1 - pos_weight
    weight *= target * pos_weight + (1 - target) * neg_weight
    return F.binary_cross_entropy_with_logits(input, target, weight, reduction=reduction)


class SegmentationLoss:

    def __init__(self, weight=None, ignore_index=255, p=0.01, loss='ce'):
        if weight is not None:
            self.weight = cuda(torch.from_numpy(weight).float())
        else:
            self.weight = None
        self.ignore_index = ignore_index
        self.p = p
        self.loss = loss

    def __call__(self, input, target):
        input = input.squeeze(1)
        target = target.type_as(input)
        if self.loss == 'ce':
            loss = F.binary_cross_entropy_with_logits(
                input, target, pos_weight=self.weight)
        elif self.loss == 'focal':
            loss = focal_loss(input, target, weight=self.weight, ignore_index=self.ignore_index)
        elif self.loss == 'f1':
            pred = torch.sigmoid(input)
            loss = f1_loss(pred, target, average='micro')
        elif self.loss == 'dice':
            pred = torch.sigmoid(input)
            loss = dice_loss(pred, target)
        elif self.loss == 'ce+dice':
            pred = torch.sigmoid(input)
            loss1 = F.binary_cross_entropy_with_logits(
                input, target, pos_weight=self.weight)
            loss2 = dice_loss(pred, target)
            loss = loss1 + loss2
        if random.random() < self.p:
            print("seg: %.4f" % loss.item())
        return loss
