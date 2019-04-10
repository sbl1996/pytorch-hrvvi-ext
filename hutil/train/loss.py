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
