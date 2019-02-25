from collections.abc import Sequence

import torch


def one_hot(tensor, C=None, dtype=torch.float):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=dtype)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


CUDA = torch.cuda.is_available()


def cuda(t):
    if torch.is_tensor(t):
        return t.cuda() if CUDA else t
    if isinstance(t, Sequence):
        return t.__class__(cuda(x) for x in t)
    return t


class Args(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, tuple(args))

    def __repr__(self):
        return "Args" + super().__repr__()
