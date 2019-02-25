from collections.abc import Sequence

import torch
from hutil.common import Args
from hutil.functools import find

import itchat


def detach(xs):
    return [
        torch.detach(x) if torch.is_tensor(x) else x
        for x in xs
    ]


def to_device(args, device):
    if torch.is_tensor(args):
        return args.to(device=device)
    if isinstance(args, Args):
        return args
    return [
        to_device(arg, device)
        for arg in args
    ]


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
    itchat.send(msg, toUserName='filehelper')
