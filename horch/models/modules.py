from collections import OrderedDict

from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import tuplify
from horch.models.defaults import get_default_activation, get_default_norm_layer
from torch import nn as nn


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


class DropConnect(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        assert 0 <= p <= 1, "drop probability has to be between 0 and 1, but got %f" % p
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1.0 - self.p
        batch_size = x.size(0)
        t = torch.rand(batch_size, 1, 1, 1, dtype=x.dtype, device=x.device) < keep_prob
        x = (x / keep_prob).masked_fill(t, 0)
        return x

    def extra_repr(self):
        return 'p={}'.format(self.p)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).to(x)

            # compute block mask
            block_mask, keeped = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * (block_mask.numel() / keeped).to(out)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        keeped = block_mask.numel() - block_mask.sum().to(torch.float32)  # prevent overflow in float16
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask, keeped

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).to(x)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)


class DropBlockScheduled(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps, start_step=0):
        super(DropBlockScheduled, self).__init__()
        self.dropblock = dropblock
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.start_step = start_step
        self.nr_steps = nr_steps
        self.step_size = (stop_value - start_value) / nr_steps

    def forward(self, x):
        if self.training:
            self.step()
        return self.dropblock(x)

    def step(self):
        idx = self.i.item()  # TODO (drop on restart)
        if self.start_step < idx < self.start_step + self.nr_steps:
            self.dropblock.drop_prob += self.step_size

        self.i += 1


class MBConv(nn.Sequential):
    def __init__(self, in_channels, channels, out_channels, kernel_size, stride=1, se_ratio=1 / 16):
        super().__init__()

        self.bn = get_norm_layer('default', in_channels)
        if in_channels != channels:
            self.expand = Conv2d(in_channels, channels, kernel_size=1,
                                 norm_layer='default', activation='default')

        self.dwconv = Conv2d(channels, channels, kernel_size, stride=stride, groups=channels,
                             norm_layer='default', activation='default')

        if se_ratio:
            assert 0 < se_ratio < 1
            self.se = SEModule(channels, reduction=int(1 / se_ratio))

        if out_channels is not None:
            self.project = Conv2d(channels, out_channels, kernel_size=1,
                                  norm_layer='default')
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = super().forward(x)
        if self.use_res_connect:
            x += identity
        return x


def seq(*modules):
    return nn.Sequential(OrderedDict(modules))
