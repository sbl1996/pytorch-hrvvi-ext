from collections.abc import Sequence

import torch
from horch.common import Args
from horch.functools import find


def to_device(args, device):
    if torch.is_tensor(args):
        return args.to(device=device)
    elif isinstance(args, Args):
        for arg in args[0]:
            if torch.is_tensor(arg):
                return args[0]
        return args
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


def cancel_event(engine, event_name, f):
    if engine.has_event_handler(f, event_name):
        handlers = engine._event_handlers[event_name]
        index = find(lambda x: x[0] == f, handlers)
        del handlers[index]


def set_lr(lr, optimizer, lr_scheduler=None):
    if lr_scheduler:
        lr_scheduler.base_lrs[0] = lr
    else:
        for group in optimizer.param_groups:
            group['lr'] = lr


def send_weixin(msg):
    import itchat
    itchat.send(msg, toUserName='filehelper')
