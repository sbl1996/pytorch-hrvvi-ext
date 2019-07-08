from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import tuplify
from horch.models.defaults import get_default_activation, get_default_norm_layer


def hardsigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace=inplace) / 6


def hardswish(x, inplace=False):
    return x * (F.relu6(x + 3, inplace=inplace) / 6)


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return swish(x)


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
    elif hasattr(name, '__call__'):
        return name(channels)
    elif name == 'default':
        return get_norm_layer(get_default_norm_layer(), channels)
    elif name == 'bn':
        return nn.BatchNorm2d(channels)
    elif name == 'gn':
        num_groups = get_groups(channels, 32)
        return nn.GroupNorm(num_groups, channels)
    else:
        raise NotImplementedError("No normalization named %s" % name)


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
    elif name == 'swish':
        return Swish()
    else:
        raise NotImplementedError("No activation named %s" % name)


def PreConv2d(in_channels, out_channels,
              kernel_size, stride=1,
              padding='same', dilation=1, groups=1, bias=False,
              norm_layer='default', activation='default'):
    if padding == 'same':
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
            padding = (ph, pw)
        else:
            padding = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ("bn", get_norm_layer(norm_layer, in_channels)),
        ("relu", get_activation(activation)),
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups, bias=bias))
    ]))


def Conv2d(in_channels, out_channels,
           kernel_size, stride=1,
           padding='same', dilation=1, groups=1, bias=None,
           norm_layer=None, activation=None, depthwise_separable=False, mid_norm_layer=None, transposed=False):
    if depthwise_separable:
        assert kernel_size != 1, "No need to use depthwise separable convolution in 1x1"
        # if norm_layer is None:
        #     assert mid_norm_layer is not None, "`mid_norm_layer` must be provided when `norm_layer` is None"
        # else:
        if mid_norm_layer is None:
            mid_norm_layer = norm_layer
        return DWConv2d(in_channels, out_channels, kernel_size, stride, padding, bias, mid_norm_layer, norm_layer,
                        activation, transposed)
    if padding == 'same':
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
            padding = (ph, pw)
        else:
            padding = (kernel_size - 1) // 2
    layers = []
    if bias is None:
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


def Linear(in_channels, out_channels, bias=None, norm_layer=None, activation=None):
    layers = []
    if bias is None:
        bias = norm_layer is None
    fc = nn.Linear(
        in_channels, out_channels, bias=bias)
    if activation is not None:
        if activation == 'sigmoid':
            nn.init.xavier_normal_(fc.weight)
        elif activation == 'leaky_relu':
            nn.init.kaiming_normal_(fc.weight, a=0.1, nonlinearity='leaky_relu')
        else:
            try:
                nn.init.kaiming_normal_(fc.weight, nonlinearity=activation)
            except ValueError:
                nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
    else:
        nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
    if bias:
        nn.init.zeros_(fc.bias)

    if norm_layer == 'default' or norm_layer == 'bn':
        layers.append(nn.BatchNorm1d(out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    layers = [fc] + layers
    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def Pool(name, kernel_size, stride=1, padding='same', ceil_mode=False):
    if padding == 'same':
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
            padding = (ph, pw)
        else:
            padding = (kernel_size - 1) // 2
    if name == 'avg':
        return nn.AvgPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=False)
    elif name == 'max':
        return nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
    else:
        raise NotImplementedError("No activation named %s" % name)


def DWConv2d(in_channels, out_channels,
             kernel_size=3, stride=1,
             padding='same', bias=True, mid_norm_layer='default',
             norm_layer=None, activation=None, transposed=False):
    return nn.Sequential(
        Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels,
               norm_layer=mid_norm_layer, transposed=transposed, activation=activation),
        Conv2d(in_channels, out_channels, kernel_size=1,
               norm_layer=norm_layer, bias=bias),
    )


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def forward(self, *xs):
        for module in self._modules.values():
            xs = module(*tuplify(xs))
        return xs


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def seq(*modules):
    mods = []
    for k, v in modules:
        if v is not None:
            mods.append((k, v))
    return nn.Sequential(OrderedDict(mods))


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.zeros(self.n_channels, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_theta = Conv2d(in_channels, in_channels // 8, kernel_size=1)

        self.conv_phi = Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.pool_phi = nn.MaxPool2d(kernel_size=2, stride=(2, 2))

        self.conv_g = Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.pool_g = nn.MaxPool2d(kernel_size=2, stride=(2, 2))

        self.conv_attn = Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        theta = self.conv_theta(x)
        theta = theta.view(b, -1, h * w)

        phi = self.conv_phi(x)
        phi = self.pool_phi(phi)
        phi = phi.view(b, -1, h * w // 4)

        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = F.softmax(attn, dim=-1)

        g = self.conv_g(x)
        g = self.pool_g(g)
        g = g.view(b, -1, h * w // 4)

        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(b, -1, h, w)
        attn_g = self.conv_attn(attn_g)

        x = x + self.sigma * attn_g
        return x


class SelfAttention2(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.proj_f = nn.Linear(in_channels, channels)
        self.proj_g = nn.Linear(in_channels, channels)
        self.proj_h = nn.Linear(in_channels, channels)
        self.proj_v = nn.Linear(channels, in_channels)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        identity = x
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        ff = self.proj_f(x)
        fg = self.proj_g(x)
        s = torch.bmm(ff, fg.permute(0, 2, 1))
        s = torch.softmax(s, dim=-1)
        fh = self.proj_h(x)
        fv = self.proj_v(torch.bmm(s, fh))
        x = fv.permute(0, 2, 1).reshape(b, c, h, w)
        x = identity + self.sigma * x
        return x
