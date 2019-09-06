from collections.abc import Sequence, Mapping

import torch


class ProtectedSeq:

    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, item):
        return self.seq

    def __len__(self):
        return 1

    def __repr__(self):
        return "Args(" + str(self.seq) + ")"

    def __iter__(self):
        return iter((self.seq,))


CUDA = torch.cuda.is_available()


def detach(t, clone=True):
    if torch.is_tensor(t):
        if clone:
            return t.clone().detach()
        else:
            return t.detach()
    elif isinstance(t, ProtectedSeq):
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


def tuplify(x, n=-1):
    if x is None:
        return ()
    elif torch.is_tensor(x):
        return (x,)
    elif not isinstance(x, Sequence):
        assert n > 0, "Length must be positive, but got %d" % n
        return (x,) * n
    else:
        if n == -1:
            n = len(x)
        else:
            assert len(x) == n, "The length of x is %d, not equal to the expected length %d" % (len(x), n)
        return tuple(x)


