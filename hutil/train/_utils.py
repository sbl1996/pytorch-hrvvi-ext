import torch
from ignite._utils import convert_tensor
from hutil.functools import find

import itchat


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    if len(batch) == 2:
        x, y = batch
        if torch.is_tensor(x):
            x = (x,)
        if torch.is_tensor(y):
            y = (y,)
    else:
        *x, y = batch
        y = (y,)
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


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
