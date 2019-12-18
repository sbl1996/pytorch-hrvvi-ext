import math

import torch


def one_hot(tensor, C=None, dtype=torch.float):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=dtype)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


def sample(t, n):
    if len(t) >= n:
        indices = torch.randperm(len(t), device=t.device)[:n]
    else:
        indices = torch.randint(len(t), size=(n,), device=t.device)
    return t[indices]


def _concat(xs, dim=1):
    if torch.is_tensor(xs):
        return xs
    elif len(xs) == 1:
        return xs[0]
    else:
        return torch.cat(xs, dim=dim)


def inverse_sigmoid(x, eps=1e-6, inplace=False):
    if not torch.is_tensor(x):
        if eps != 0:
            x = min(max(x, eps), 1 - eps)
        return math.log(x / (1 - x))
    if inplace:
        return inverse_sigmoid_(x, eps)
    if eps != 0:
        x = torch.clamp(x, eps, 1 - eps)
    return (x / (1 - x)).log()


def inverse_sigmoid_(x, eps=1e-6):
    if eps != 0:
        x = torch.clamp_(x, eps, 1 - eps)
    return x.div_(1 - x).log_()


def expand_last_dim(t, *size):
    return t.view(*t.size()[:-1], *size)


def select0(t, indices):
    arange = torch.arange(t.size(1), device=t.device)
    return t[indices, arange]


def select1(t, indices):
    arange = torch.arange(t.size(0), device=t.device)
    return t[arange, indices]


def select(t, dim, indices):
    if dim == 0:
        return select0(t, indices)
    elif dim == 1:
        return select1(t, indices)
    else:
        raise ValueError("dim could be only 0 or 1, not %d" % dim)