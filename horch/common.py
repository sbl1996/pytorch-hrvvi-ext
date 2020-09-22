from typing import Sequence, Mapping


import torch

CUDA = torch.cuda.is_available()


def detach(t, clone=True):
    if torch.is_tensor(t):
        if clone:
            return t.clone().detach()
        else:
            return t.detach()
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


def convert_tensor(input_, device, non_blocking=False):

    def _func(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=non_blocking) if device is not None else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors.
    """
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(input_, input_type, func):
    """Apply a function on a object of `input_type` or mapping, or sequence of objects of `input_type`.
    """
    if isinstance(input_, input_type):
        return func(input_)
    if isinstance(input_, (str, bytes)):
        return input_
    if isinstance(input_, Mapping):
        return type(input_)({k: apply_to_type(sample, input_type, func) for k, sample in input_.items()})
    if isinstance(input_, tuple) and hasattr(input_, "_fields"):  # namedtuple
        return type(input_)(*(apply_to_type(sample, input_type, func) for sample in input_))
    if isinstance(input_, Sequence):
        return type(input_)([apply_to_type(sample, input_type, func) for sample in input_])
    raise TypeError(("input must contain {}, dicts or lists; found {}".format(input_type, type(input_))))

