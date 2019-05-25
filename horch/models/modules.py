from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import _tuple
from horch.models.defaults import get_default_activation, get_default_norm_layer


def hardsigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace=inplace) / 6


def hardswish(x, inplace=False):
    return x * (F.relu6(x + 3, inplace=inplace) / 6)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hardsigmoid(x, self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hardswish(x, self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


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


def upsample_concat(x, y):
    h, w = y.size()[2:4]
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    return torch.cat((x, y), dim=1)


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
        raise NotImplementedError("No normalization named %s" % name)


def get_attention(name, **kwargs):
    if not name:
        return Identity()
    name = name.lower()
    if name == 'se':
        return SEModule(**kwargs)
    elif name == 'sem':
        return SELayerM(**kwargs)
    elif name == 'cbam':
        return CBAM(**kwargs)
    else:
        raise NotImplementedError("No attention module named %s" % name)


def get_activation(name):
    if isinstance(name, nn.Module):
        return name
    if name == 'default':
        return get_activation(get_default_activation())
    elif name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'relu6':
        return nn.ReLU6(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'hswish':
        return HardSwish(inplace=True)
    else:
        raise NotImplementedError("No activation named %s" % name)


def Conv2d(in_channels, out_channels,
           kernel_size, stride=1,
           padding='same', dilation=1, groups=1,
           norm_layer=None, activation=None, depthwise_separable=False, mid_norm_layer=None, transposed=False):
    if depthwise_separable:
        assert kernel_size != 1, "No need to use depthwise separable convolution in 1x1"
        if norm_layer is None:
            assert mid_norm_layer is not None, "`mid_norm_layer` must be provided when `norm_layer` is None"
        else:
            if mid_norm_layer is None:
                mid_norm_layer = norm_layer
        return DWConv2d(in_channels, out_channels, kernel_size, stride, padding, mid_norm_layer, norm_layer, activation, transposed)
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
    if transposed:
        conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
    else:
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
    if activation is not None:
        if activation == 'sigmoid':
            nn.init.xavier_normal_(conv.weight)
        elif activation == 'leaky_relu':
            nn.init.kaiming_normal_(conv.weight, a=0.1, nonlinearity='leaky_relu')
        else:
            try:
                nn.init.kaiming_normal_(conv.weight, nonlinearity=activation)
            except ValueError:
                nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    else:
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    if bias:
        nn.init.zeros_(conv.bias)

    if norm_layer is not None:
        if norm_layer == 'default':
            norm_layer = get_default_norm_layer()
        layers.append(get_norm_layer(norm_layer, out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    layers = [conv] + layers
    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def Linear(in_channels, out_channels,
           norm_layer=None, activation=None):
    layers = []
    bias = norm_layer is None
    fc = nn.Linear(in_channels, out_channels)
    nn.init.kaiming_normal_(fc.weight, nonlinearity=activation or 'default')
    if bias:
        nn.init.zeros_(fc.bias)
    layers.append(fc)
    if norm_layer == 'bn':
        layers.append(nn.BatchNorm1d(out_channels))
    elif norm_layer == 'gn':
        layers.append(nn.GroupNorm(get_groups(out_channels, 32), out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.pool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


class CBAMChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        aa = F.adaptive_avg_pool2d(x, 1).view(b, c)
        aa = self.mlp(aa)
        am = F.adaptive_max_pool2d(x, 1).view(b, c)
        am = self.mlp(am)
        a = torch.sigmoid(aa + am).view(b, c, 1, 1)
        return x * a


class CBAMSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(2, 1, kernel_size=7, norm_layer='bn')

    def forward(self, x):
        aa = x.mean(dim=1, keepdim=True)
        am = x.max(dim=1, keepdim=True)[0]
        a = torch.cat([aa, am], dim=1)
        a = torch.sigmoid(self.conv(a))
        return x * a


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.channel = CBAMChannelAttention(in_channels, reduction)
        self.spatial = CBAMSpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class SELayerM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        channels = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU6(True),
            nn.Linear(channels, in_channels),
            HardSigmoid(True),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.avgpool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


@curry
def DWConv2d(in_channels, out_channels,
             kernel_size=3, stride=1,
             padding='same', mid_norm_layer='bn',
             norm_layer=None, activation=None, transposed=False):
    return nn.Sequential(
        Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels,
               norm_layer=mid_norm_layer, transposed=transposed),
        Conv2d(in_channels, out_channels, kernel_size=1,
               norm_layer=norm_layer, activation=activation),
    )


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if 'inference' in kwargs:
            self._inference = kwargs['inference']

    def forward(self, *xs):
        for module in self._modules.values():
            xs = module(*_tuple(xs))
        return xs

    def inference(self, *xs):
        self.eval()
        with torch.no_grad():
            xs = self.forward(*xs)
        preds = self._inference(*_tuple(xs))
        self.train()
        return preds


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
