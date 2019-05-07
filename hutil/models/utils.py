import torch.nn as nn

import torch
from toolz import curry
from hutil.ext.summary import summary


def _concat(preds, dim=1):
    if torch.is_tensor(preds):
        return preds
    elif len(preds) == 1:
        return preds[0]
    else:
        return torch.cat(preds, dim=dim)


def get_last_conv(m):
    r"""
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, nn.Conv2d), m.modules())
    return list(convs)[-1]


def get_out_channels(m):
    r"""
    Get the output channels of the last conv layer of a block.
    """
    return get_last_conv(m).out_channels


@curry
def conv_to_atrous(m, rate):
    r"""
    Convert a 3x3 Conv2d to Atrous Convolution.
    """
    if 'Conv2d' in type(m).__name__ and m.kernel_size != (1, 1):
        kh, kw = m.kernel_size
        ph = int(((kh - 1) * (rate - 1) + kh - 1) / 2)
        pw = int(((kw - 1) * (rate - 1) + kw - 1) / 2)
        m.padding = (ph, pw)
        m.stride = (1, 1)
        m.dilation = (rate, rate)
    return m


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model


def clip(model, tol=1e-6):
    for p in model.parameters():
        p[p.abs() < tol] = 0
    return model


def get_loc_cls_preds(ps, num_classes, concat=True):
    loc_preds = []
    cls_preds = []
    b = ps[0].size(0)
    for p in ps:
        p = p.permute(0, 3, 2, 1).contiguous().view(
            b, -1, 4 + num_classes)
        loc_preds.append(p[..., :4])
        cls_preds.append(p[..., 4:])
    if concat:
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p
    return loc_preds, cls_preds
