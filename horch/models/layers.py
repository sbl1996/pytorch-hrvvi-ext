import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

from horch.defaults import DEFAULTS
from horch.nn import HardSwish, Swish


def get_groups(channels, ref=32):
    if channels == 1:
        return 1
    xs = filter(lambda x: channels % x == 0, range(2, channels + 1))
    c = min(xs, key=lambda x: abs(x - ref))
    if c < 8:
        c = max(c, channels // c)
    return channels // c


def Norm(channels, type='default', **kwargs):
    assert isinstance(channels, int)
    if isinstance(type, nn.Module):
        return type
    if type in ['default', 'def']:
        return Norm(channels, DEFAULTS['norm'], **kwargs)
    elif type == 'bn':
        if 'affine' in kwargs:
            cfg_bn = {**DEFAULTS['bn'], 'affine': kwargs['affine']}
        else:
            cfg_bn = DEFAULTS['bn']
        return nn.BatchNorm2d(channels, **cfg_bn)
    elif type == 'gn':
        num_groups = get_groups(channels, 32)
        return nn.GroupNorm(num_groups, channels)
    else:
        raise NotImplementedError("No normalization named %s" % type)


def Act(type='default'):
    if isinstance(type, nn.Module):
        return type
    if type in ['default', 'def']:
        return Act(DEFAULTS['act'])
    elif type == 'relu':
        return nn.ReLU(**DEFAULTS['relu'])
    elif type == 'relu6':
        return nn.ReLU6(**DEFAULTS['relu6'])
    elif type == 'leaky_relu':
        return nn.LeakyReLU(**DEFAULTS['leaky_relu'])
    elif type == 'sigmoid':
        return nn.Sigmoid()
    elif type == 'hswish':
        return HardSwish(**DEFAULTS['hswisg'])
    elif type == 'swish':
        return Swish()
    else:
        raise NotImplementedError("Activation not implemented: %s" % type)


def calc_same_padding(kernel_size, dilation=(1, 1)):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    return ph, pw


def Conv2d(in_channels, out_channels, kernel_size, stride=1,
           padding='same', dilation=1, groups=1, bias=None, norm=None, act=None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    layers = []
    if bias is None:
        bias = norm is None

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)

    if norm is not None:
        layers.append(Norm(out_channels, norm))
    if act is not None:
        layers.append(Act(act))
    layers = [conv] + layers
    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def Linear(in_channels, out_channels, bias=None, norm=None, act=None):
    layers = []
    if bias is None:
        bias = norm is None
    fc = nn.Linear(
        in_channels, out_channels, bias=bias)

    if norm in ['default', 'def', 'bn']:
        layers.append(nn.BatchNorm1d(out_channels))
    if act is not None:
        layers.append(Act(act))
    layers = [fc] + layers
    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def Pool(type, kernel_size, stride=1, padding='same', ceil_mode=False):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding == 'same':
        padding = calc_same_padding(kernel_size)
    if type == 'avg':
        return nn.AvgPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=False)
    elif type == 'max':
        return nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
    else:
        raise NotImplementedError("No activation named %s" % type)


def DWConv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
             padding='same', bias=None, norm=None, act=None, mid_norm=None):
    return nn.Sequential(
        Conv2d(in_channels, in_channels, kernel_size, stride,
               groups=in_channels, dilation=dilation, padding=padding, bias=False, norm=mid_norm),
        Conv2d(in_channels, out_channels, 1, bias=bias, norm=norm, act=act),
    )


def Seq(*layers):
    layers = [
        l for l in layers if l is not None
    ]
    return nn.Sequential(*layers) if len(layers) != 1 else layers[0]