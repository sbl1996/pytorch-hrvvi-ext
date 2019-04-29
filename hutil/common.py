from collections.abc import Sequence, Mapping

import torch


class Args(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, tuple(args))

    def __repr__(self):
        return "Args" + super().__repr__()


def one_hot(tensor, C=None, dtype=torch.float):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=dtype)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


CUDA = torch.cuda.is_available()


def detach(t, clone=True):
    if torch.is_tensor(t):
        if clone:
            return t.clone().detach()
        else:
            return t.detach()
    elif isinstance(t, Args):
        return t
    elif isinstance(t, Sequence):
        return t.__class__(detach(x, clone) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, detach(v, clone)) for k, v in t.items())
    else:
        return t


def cuda(t):
    if torch.is_tensor(t):
        return t.cuda() if CUDA else t
    elif isinstance(t, Sequence):
        return t.__class__(cuda(x) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, cuda(v)) for k, v in t.items())
    else:
        return t


def cpu(t):
    if torch.is_tensor(t):
        return t.cpu()
    elif isinstance(t, Sequence):
        return t.__class__(cpu(x) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, cpu(v)) for k, v in t.items())
    else:
        return t
