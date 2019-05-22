import math

import torch
import torch.nn.functional as F


def inverse_sigmoid(x):
    return math.log(x / (1-x))


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
    var_p : torch.Tensor
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
