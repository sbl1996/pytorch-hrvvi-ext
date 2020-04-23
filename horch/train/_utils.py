from collections.abc import Sequence

import torch


def to_device(args, device):
    if torch.is_tensor(args):
        return args.to(device=device)
    elif isinstance(args, Sequence):
        return args.__class__(to_device(arg, device)
                              for arg in args)
    else:
        return args


def wrap(x):
    if torch.is_tensor(x):
        return (x,)
    return x


def _prepare_batch(batch, device=None):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch
    x = wrap(x)
    y = wrap(y)
    return to_device(x, device), to_device(y, device)
