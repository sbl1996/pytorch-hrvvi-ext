import torch.nn as nn
import torch.nn.functional as F
from horch.models.detection.nasfpn import ReLUConvBN

from horch.models.modules import upsample_add, Conv2d, Sequential, Pool, upsample_concat, MBConv
from horch.models.detection.nasfpn import NASFPN


class TopDown(nn.Module):
    def __init__(self, in_channels, f_channels, lite=False):
        super().__init__()
        self.lat = Conv2d(
            in_channels, f_channels, kernel_size=1,
            norm_layer='default')
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=5 if lite else 3,
            norm_layer='default', activation='default', depthwise_separable=lite)

    def forward(self, c, p):
        p = upsample_add(p, self.lat(c))
        p = self.conv(p)
        return p


class DeconvTopDown(nn.Module):
    def __init__(self, in_channels1, in_channels2, f_channels, lite=False):
        super().__init__()
        self.lat = Conv2d(
            in_channels1, f_channels, kernel_size=1,
            norm_layer='default')
        self.deconv = Conv2d(in_channels2, f_channels, kernel_size=4, stride=2,
                             norm_layer='default', depthwise_separable=lite, transposed=True)
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=5 if lite else 3,
            norm_layer='default', activation='default', depthwise_separable=lite)

    def forward(self, c, p):
        p = self.lat(c) + self.deconv(p)
        p = self.conv(p)
        return p


class FPNExtraLayers(nn.Module):
    def __init__(self, in_channels, extra_layers=(6, 7), f_channels=None, downsample='conv', lite=False):
        super().__init__()
        self.extra_layers = nn.ModuleList([])
        for _ in extra_layers:
            if downsample == 'conv':
                l = ReLUConvBN(in_channels, f_channels, stride=2, lite=lite)
            elif downsample == 'maxpool':
                l = Pool('max', kernel_size=1, stride=2)
            elif downsample == 'avgpool':
                l = Pool('avg', kernel_size=1, stride=2)
            else:
                raise ValueError("%s as downsampling is invalid." % downsample)
            in_channels = f_channels
            self.extra_layers.append(l)

    def forward(self, p):
        ps = []
        for l in self.extra_layers:
            p = l(p)
            ps.append(p)
        return tuple(ps)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, lite=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels // 2, kernel_size=1,
                            norm_layer='default', activation='default')
        padding = 1 if stride == 2 else 0
        self.conv2 = Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=padding,
                            norm_layer='default', activation='default', depthwise_separable=lite)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SSDExtraLayers(nn.Module):
    def __init__(self, in_channels_list, extra_layers=(6, 7), f_channels=None, no_padding=-1, lite=False):
        super().__init__()
        in_channels = in_channels_list[-1]
        self.extra_layers = nn.ModuleList([])
        for _ in extra_layers:
            l = BasicBlock(in_channels, f_channels, lite=lite)
            self.extra_layers.append(l)
            in_channels = f_channels

        for i in range(no_padding, 0):
            l = self.extra_layers[i].conv2[0]
            if lite:
                l = l[0]
            l.stride = (1, 1)
            l.padding = (0, 0)

        self.out_channels = in_channels_list + [f_channels] * len(extra_layers)

    def forward(self, *cs):
        ps = list(cs)
        for l in self.extra_layers:
            ps.append(l(ps[-1]))
        return tuple(ps)


class FPN(nn.Module):
    r"""
    Feature Pyramid Network which enhance features of different levels.

    Parameters
    ----------
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(256,512,1024)``
    f_channels : int
        Number of output channels.
    extra_layers : tuple of ints
        Extra layers to add, e.g., ``(6, 7)``
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    upsample : str
        Use bilinear upsampling when `interpolate` and ConvTransposed when `deconv`
        Default: `interpolate`
    """

    def __init__(self, in_channels_list, f_channels=256, extra_layers=(), downsample='conv', lite=False,
                 upsample='interpolate'):
        super().__init__()
        self.lat = Conv2d(in_channels_list[-1], f_channels, kernel_size=1, norm_layer='default')
        self.extra_layers = extra_layers
        if extra_layers:
            self.extras = FPNExtraLayers(f_channels, extra_layers, f_channels, downsample=downsample, lite=lite)
        if upsample == 'deconv':
            self.topdowns = nn.ModuleList([
                DeconvTopDown(c, f_channels, f_channels, lite=lite)
                for c in in_channels_list[:-1]
            ])
        else:
            self.topdowns = nn.ModuleList([
                TopDown(c, f_channels, lite=lite)
                for c in in_channels_list[:-1]
            ])
        self.out_channels = [f_channels] * (len(in_channels_list) + len(extra_layers))

    def forward(self, *cs):
        p = self.lat(cs[-1])
        ps = (p,)
        if self.extra_layers:
            ps = ps + self.extras(p)
        for c, topdown in zip(reversed(cs[:-1]), reversed(self.topdowns)):
            p = topdown(c, ps[0])
            ps = (p,) + ps
        return ps


class BottomUp(nn.Module):
    def __init__(self, f_channels, lite=False):
        super().__init__()
        self.down = Conv2d(
            f_channels, f_channels, kernel_size=3, stride=2,
            norm_layer='default', activation='default', depthwise_separable=lite)
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=3,
            norm_layer='default', activation='default', depthwise_separable=lite)

    def forward(self, p, n):
        n = p + self.down(n)
        n = self.conv(n)
        return n


class FPN2(nn.Module):
    r"""
    Bottom-up path augmentation.

    Parameters
    ----------
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(256,256,256)``
        Notice: they must be the same.
    f_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels_list, f_channels, lite=False):
        super().__init__()
        assert len(set(in_channels_list)) == 1, "Input channels of every level must be the same"
        assert in_channels_list[0] == f_channels, "Input channels must be the same as `f_channels`"
        self.bottomups = nn.ModuleList([
            BottomUp(f_channels, lite=lite)
            for _ in in_channels_list[1:]
        ])
        self.out_channels = [f_channels] * len(in_channels_list)

    def forward(self, *ps):
        ns = [ps[0]]
        for p, bottomup in zip(ps[1:], self.bottomups):
            n = bottomup(p, ns[-1])
            ns.append(n)
        return tuple(ns)


class ContextEnhance(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lats = nn.ModuleList([
            Conv2d(c, out_channels, kernel_size=1, norm_layer='default')
            for c in in_channels
        ])
        self.lat_glb = Conv2d(in_channels[-1], out_channels, kernel_size=1,
                              norm_layer='default')

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        p = self.lats[0](cs[0])
        for c, lat in zip(cs[1:], self.lats[1:]):
            p += F.interpolate(lat(c), size=size, mode='bilinear', align_corners=False)
        c_glb = F.adaptive_avg_pool2d(cs[-1], 1)
        p_glb = self.lat_glb(c_glb)
        p += p_glb
        return p


def stacked_fpn(num_stacked, in_channels_list, extra_layers=(), f_channels=256, lite=False, upsample='interpolate'):
    r"""
    Stacked FPN with alternant top down block and bottom up block.

    Parameters
    ----------
    num_stacked : int
        Number of stacked fpns.
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(128,256,512)``
    extra_layers : tuple of ints
        Extra layers to add, e.g., ``(6, 7)``
    f_channels : int
        Number of feature (output) channels.
        Default: 256
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    upsample : str
        Use bilinear upsampling if `interpolate` and ConvTransposed if `deconv`
        Default: `interpolate`
    """
    assert num_stacked >= 2, "Use FPN directly if `num_stacked` is smaller than 2."
    layers = [FPN(in_channels_list, f_channels, extra_layers, lite=lite, upsample=upsample)]
    for i in range(1, num_stacked):
        if i % 2 == 0:
            layers.append(FPN(layers[-1].out_channels, f_channels, lite=lite, upsample=upsample))
        else:
            layers.append(FPN2(layers[-1].out_channels, f_channels, lite=lite))
    m = Sequential(*layers)
    m.out_channels = m[-1].out_channels
    return m


def stacked_nas_fpn(num_stacked, in_channels_list, extra_layers=(), f_channels=256, lite=False, upsample='interpolate'):
    r"""
    Stacked FPN with alternant top down block and bottom up block.

    Parameters
    ----------
    num_stacked : int
        Number of stacked fpns.
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(128,256,512)``
    extra_layers : tuple of ints
        Extra layers to add, e.g., ``(6, 7)``
    f_channels : int
        Number of feature (output) channels.
        Default: 256
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    upsample : str
        Use bilinear upsampling if `interpolate` and ConvTransposed if `deconv`
        Default: `interpolate`
    """
    assert num_stacked >= 2, "Use FPN directly if `num_stacked` is smaller than 2."
    layers = [FPN(in_channels_list, f_channels, extra_layers, downsample='maxpool', lite=lite, upsample=upsample)]
    for i in range(1, num_stacked):
        layers.append(NASFPN(f_channels))
    m = Sequential(*layers)
    m.out_channels = m[-1].out_channels
    return m


class IDA(nn.Module):
    def __init__(self, in_channels_list, f_channels, lite=False):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.topdowns = nn.ModuleList([
            DeconvTopDown(in_channels_list[i], in_channels_list[i + 1], f_channels, lite=lite)
            for i in range(self.num_levels - 1)
        ])
        if self.num_levels > 2:
            self.deep = IDA([f_channels] * (self.num_levels - 1), f_channels)

    def forward(self, *xs):
        xs = [
            l(xs[i], xs[i + 1]) for i, l in enumerate(self.topdowns)
        ]
        if self.num_levels > 2:
            return self.deep(*xs)
        else:
            return xs[0]


class IDA2(nn.Module):
    def __init__(self, in_channels, lite=False):
        super().__init__()
        self.num_levels = len(in_channels)
        self.topdowns = nn.ModuleList([
            DeconvTopDown(in_channels[i], in_channels[i + 1], in_channels[i + 1], lite=lite)
            for i in range(self.num_levels - 1)
        ])
        if self.num_levels > 2:
            self.deep = IDA2(in_channels[1:], lite=lite)

    def forward(self, *xs):
        xs = [
            l(xs[i], xs[i + 1]) for i, l in enumerate(self.topdowns)
        ]
        if self.num_levels > 2:
            return self.deep(*xs)
        else:
            return xs[0]


class YOLOFPN(nn.Module):
    def __init__(self, in_channels_list, f_channels=256):
        super().__init__()
        assert len(in_channels_list) == 3
        channels = in_channels_list
        self.conv51 = nn.Sequential(
            MBConv(channels[-1], channels[-1], f_channels, kernel_size=5),
            MBConv(f_channels, f_channels * 4, f_channels, kernel_size=5),
        )
        self.conv52 = MBConv(f_channels, f_channels * 4, None, kernel_size=5)

        self.lat5 = Conv2d(f_channels, f_channels // 2, kernel_size=1,
                           norm_layer='default')

        self.conv41 = nn.Sequential(
            MBConv(channels[-2] + f_channels // 2, channels[-2] + f_channels // 2, f_channels // 2, kernel_size=5),
            MBConv(f_channels // 2, f_channels * 2, f_channels // 2, kernel_size=5),
        )
        self.conv42 = MBConv(f_channels // 2, f_channels * 2, None, kernel_size=5)

        self.lat4 = Conv2d(f_channels // 2, f_channels // 4, kernel_size=1,
                           norm_layer='default')

        self.conv31 = nn.Sequential(
            MBConv(channels[-3] + f_channels // 4, channels[-3] + f_channels // 4, f_channels // 4, kernel_size=5),
            MBConv(f_channels // 4, f_channels, f_channels // 4, kernel_size=5),
        )
        self.conv32 = MBConv(f_channels // 4, f_channels, None, kernel_size=5)

        self.out_channels = [f_channels, f_channels * 2, f_channels * 4]

    def forward(self, c3, c4, c5):
        p51 = self.conv51(c5)
        p52 = self.conv52(p51)

        p41 = upsample_concat(self.lat5(p51), c4)
        p42 = self.conv41(p41)
        p43 = self.conv42(p42)

        p31 = upsample_concat(self.lat4(p42), c3)
        p32 = self.conv31(p31)
        p33 = self.conv32(p32)

        return p33, p43, p52
