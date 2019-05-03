import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_add(x, y):
    r"""
    Upsample x and add it to y.

    Parameters
    ----------
    x : torch.Tensor
        tensor to upsample
    y : torch.Tensor
        tensor to be added
    """
    h, w = y.size()[2:4]
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y


def get_groups(channels, ref=32):
    xs = filter(lambda x: channels % x == 0, range(2, channels + 1))
    c = min(xs, key=lambda x: abs(x - ref))
    if c < 8:
        c = max(c, channels // c)
    return channels // c


def get_norm_layer(name, channels):
    if isinstance(name, nn.Module):
        return name
    if name == 'bn':
        return nn.BatchNorm2d(channels)
    elif name == 'gn':
        num_groups = get_groups(channels, 32)
        return nn.GroupNorm(num_groups, channels)
    else:
        raise NotImplementedError


def get_activation(name):
    if isinstance(name, nn.Module):
        return name
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    else:
        return NotImplementedError


def Conv2d(in_channels, out_channels,
           kernel_size=3, stride=1,
           padding='same', dilation=1, groups=1,
           norm_layer=None, activation=None):

    if padding == 'same':
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
            padding = (ph, pw)
        else:
            padding = (kernel_size - 1) // 2
    layers = []
    bias = norm_layer is None
    conv = nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, dilation, groups, bias)
    nn.init.kaiming_normal_(conv.weight, nonlinearity=activation)
    if bias:
        nn.init.zeros_(conv.bias)
    layers.append(conv)
    if norm_layer is not None:
        layers.append(get_norm_layer(norm_layer, out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.avgpool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s
