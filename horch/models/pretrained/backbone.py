from difflib import get_close_matches

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nn import L2Norm
from horch.models.pretrained.vovnet import vovnet27_slim, vovnet39, vovnet57

from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.efficientnet import calc_tf_padding

from horch.models.pretrained.mobilenetv3 import mobilenetv3_large
from horch.models.utils import get_out_channels, calc_out_channels, conv_to_atrous, decimate


def _check_levels(levels):
    assert tuple(range(levels[0], levels[-1] + 1)) == tuple(levels), "Feature levels must in ascending order."


def backbone_forward(self, x):
    outs = []
    x = self.layer1(x)
    if 1 in self.feature_levels:
        outs.append(x)
    x = self.layer2(x)
    if 2 in self.feature_levels:
        outs.append(x)

    x = self.layer3(x)
    if 3 in self.feature_levels:
        outs.append(x)

    x = self.layer4(x)
    if 4 in self.feature_levels:
        outs.append(x)

    if 5 in self.forward_levels:
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
    return outs


class ShuffleNetV2(nn.Module):
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

    width_mult  Top1    Top5    Params      FLOPs/2
    x0.5        40.99	18.65	1,366,792   43.31M
    x1.0        31.44	11.63	2,278,604   149.72M
    x1.5        27.47	9.42	4,406,098   320.77M
    x2.0        25.94	8.45	7,601,686   595.84M

    Parameters
    ----------
    mult: float
        Width multiplier which could one of [ 0.5, 1.0, 1.5, 2.0 ]
        Default: 0.5
    feature_levels : sequence of int
        Feature levels to output.
        Default: (3, 4, 5)
    """
    mult2name = {
        0.5: "shufflenetv2_wd2",
        1.0: "shufflenetv2_w1",
        1.5: "shufflenetv2_w3d2",
        2.0: "shufflenetv2_w2",
    }

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=True, include_final=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model(self.mult2name[mult], pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.stem
        self.layer2 = net.init_block.pool
        self.layer3 = net.stage1
        self.layer4 = net.stage2
        if include_final:
            self.layer5 = nn.Sequential(
                net.stage3,
                net.final_block,
            )
        else:
            self.layer5 = net.stage3
        out_channels = [
            get_out_channels(self.layer1),
            get_out_channels(self.layer1),
            calc_out_channels(self.layer3),
            calc_out_channels(self.layer4),
            calc_out_channels(self.layer5),
        ]

        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ShuffleNetV2b(nn.Module):
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    
    width_mult  Top1    Top5    Params      FLOPs/2
    x0.5        40.99	18.65	1,366,792   43.31M
    x1.0        31.44	11.63	2,278,604   149.72M
    x1.5        27.47	9.42	4,406,098   320.77M
    x2.0        25.94	8.45	7,601,686   595.84M

    Parameters
    ----------
    mult: float
        Width multiplier which could one of [ 0.5, 1.0, 1.5, 2.0 ]
        Default: 0.5
    feature_levels : sequence of int
        Feature levels to output.
        Default: (3, 4, 5)
    """
    mult2name = {
        0.5: "shufflenetv2b_wd2",
        1.0: "shufflenetv2b_w1",
        1.5: "shufflenetv2b_w3d2",
        2.0: "shufflenetv2b_w2",
    }

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=True, include_final=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model(self.mult2name[mult], pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.stem
        self.layer2 = net.init_block.pool
        self.layer3 = net.stage1
        self.layer4 = net.stage2
        if include_final:
            self.layer5 = nn.Sequential(
                net.stage3,
                net.final_block,
            )
        else:
            self.layer5 = net.stage3

        out_channels = [
            get_out_channels(self.layer1),
            get_out_channels(self.layer1),
            calc_out_channels(self.layer3),
            calc_out_channels(self.layer4),
            calc_out_channels(self.layer5),
        ]

        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class MobileNetV2(nn.Module):
    r"""MobileNetV2: Inverted Residuals and Linear Bottlenecks

    width_mult  Top1   Top5    Params      FLOPs/2
    x0.25       48.34	24.51	1,516,392   34.24M
    x0.5        35.98	14.93	1,964,736   100.13M
    x0.75       30.17	10.82	2,627,592   198.50M
    x1.0        26.97	8.87	3,504,960   329.36M

    Parameters
    ----------
    mult : float
        Width multiplier, [ 0.25, 0.5, 0.75, 1.0 ] is avaliable.
        Default: 1.0
    feature_levels : sequence of int
        features of which layers to output.
        Default: (3, 4, 5)
    """

    mult2name = {
        0.25: "mobilenetv2_wd4",
        0.5: "mobilenetv2_wd2",
        0.75: "mobilenetv2_w3d4",
        1.0: "mobilenetv2_w1",
    }

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels

        net = ptcv_get_model(self.mult2name[float(mult)], pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = nn.Sequential(
            net.init_block,
            net.stage1,
            net.stage2.unit1.conv1,
        )
        self.layer2 = nn.Sequential(
            net.stage2.unit1.conv2,
            net.stage2.unit1.conv3,
            net.stage2.unit2,
            net.stage3.unit1.conv1,
        )
        self.layer3 = nn.Sequential(
            net.stage3.unit1.conv2,
            net.stage3.unit1.conv3,
            *net.stage3[1:],
            net.stage4.unit1.conv1,
        )
        self.layer4 = nn.Sequential(
            net.stage4.unit1.conv2,
            net.stage4.unit1.conv3,
            *net.stage4[1:],
            net.stage5.unit1.conv1,
        )
        self.layer5 = nn.Sequential(
            net.stage5.unit1.conv2,
            net.stage5.unit1.conv3,
            *net.stage5[1:],
            net.final_block,
        )

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ResNeSt(nn.Module):
    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = torch.hub.load('zhanghang1989/ResNeSt', name, pretrained=pretrained)
        del net.fc
        self.layer1 = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
        )
        self.layer2 = nn.Sequential(
            net.maxpool,
            net.layer1,
        )
        self.layer3 = net.layer2
        self.layer4 = net.layer3
        self.layer5 = net.layer4
        self.out_channels = np.array([
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ])

    def forward(self, x):
        return backbone_forward(self, x)


class EfficientNet(nn.Module):
    r"""EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

    version Top1    Top5    Params      FLOPs/2
    b0      24.12   7.24    5,288,548	414.31M
    b1      22.07   6.14    7,794,184	608.59M
    b2      21.06   5.52    9,109,994	699.46M
    b3      19.63   4.89    12,233,232	1,016.10M

    Parameters
    ----------
    version : str
        b0, b1, b2, b3, b4, b5, b6, b7 are avaliable.
        Default: b0
    feature_levels (sequence of int): features of which layers to output
        Default: (3, 4, 5)
    """

    def __init__(self, version='b0', feature_levels=(3, 4, 5), pretrained=True, include_final=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        name = 'efficientnet_%sc' % version
        backbone = ptcv_get_model(name, pretrained=pretrained)
        del backbone.output
        features = backbone.features
        self._kernel_sizes = [3]
        self._strides = [2]
        self.layer1 = nn.Sequential(
            features.init_block.stem,
            features.stage1,
            features.stage2.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage2.unit1.kernel_size)
        self._strides.append(features.stage2.unit1.stride)
        self.layer2 = nn.Sequential(
            features.stage2.unit1.conv2,
            features.stage2.unit1.se,
            features.stage2.unit1.conv3,
            features.stage2[1:],
            features.stage3.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage3.unit1.kernel_size)
        self._strides.append(features.stage3.unit1.stride)
        self.layer3 = nn.Sequential(
            features.stage3.unit1.conv2,
            features.stage3.unit1.se,
            features.stage3.unit1.conv3,
            features.stage3[1:],
            features.stage4.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage4.unit1.kernel_size)
        self._strides.append(features.stage4.unit1.stride)
        self.layer4 = nn.Sequential(
            features.stage4.unit1.conv2,
            features.stage4.unit1.se,
            features.stage4.unit1.conv3,
            features.stage4[1:],
            features.stage5.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage5.unit1.kernel_size)
        self._strides.append(features.stage5.unit1.stride)
        self.layer5 = nn.Sequential(
            features.stage5.unit1.conv2,
            features.stage5.unit1.se,
            features.stage5.unit1.conv3,
            features.stage5[1:],
            *([features.final_block] if include_final else []),
        )

        self.out_channels = np.array([
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ])

    def forward(self, x):
        outs = []
        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[0], self._strides[0]))
        x = self.layer1(x)
        if 1 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[1], self._strides[1]))
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[2], self._strides[2]))
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[3], self._strides[3]))
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[4], self._strides[4]))
        if 5 in self.forward_levels:
            x = self.layer5(x)
            if 5 in self.feature_levels:
                outs.append(x)
        return tuple(outs)


class ProxylessNAS(nn.Module):
    r"""EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

    version     Top1    Top5    Params      FLOPs/2
    cpu	        24.71   7.61    4,361,648   459.96M
    gpu	        24.79   7.45    7,119,848   476.08M
    mobile      25.41   7.80    4,080,512   332.46M
    mobile14    23.29   6.62    6,857,568   597.10M

    Parameters
    ----------
    version : str
        gpu, cpu, mobile or mobile14
        Default: mobile
    feature_levels (sequence of int): features of which layers to output
        Default: (3, 4, 5)
    """

    def __init__(self, version='mobile', feature_levels=(3, 4, 5), pretrained=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        name = 'proxylessnas_%s' % version
        backbone = ptcv_get_model(name, pretrained=pretrained)
        del backbone.output
        features = backbone.features

        self.layer1 = nn.Sequential(
            features.init_block,
            features.stage1,
            features.stage2.unit1.body.bc_conv,
        )
        self.layer2 = nn.Sequential(
            features.stage2.unit1.body.dw_conv,
            features.stage2.unit1.body.pw_conv,
            features.stage2[1:],
            features.stage3.unit1.body.bc_conv,
        )
        self.layer3 = nn.Sequential(
            features.stage3.unit1.body.dw_conv,
            features.stage3.unit1.body.pw_conv,
            features.stage3[1:],
            features.stage4.unit1.body.bc_conv,
        )
        self.layer4 = nn.Sequential(
            features.stage4.unit1.body.dw_conv,
            features.stage4.unit1.body.pw_conv,
            features.stage4[1:],
            features.stage5.unit1.body.bc_conv,
        )
        self.layer5 = nn.Sequential(
            features.stage5.unit1.body.dw_conv,
            features.stage5.unit1.body.pw_conv,
            features.stage5[1:],
            features.final_block
        )

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class SqueezeNet(nn.Module):
    r"""SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB models size

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        from torchvision.models.squeezenet import squeezenet1_1, Fire
        backbone = squeezenet1_1(pretrained=pretrained)
        del backbone.classifier
        backbone = backbone.features
        backbone[0].padding = (1, 1)

        self.layer1 = backbone[:2]
        self.layer2 = backbone[2:5]
        self.layer3 = backbone[5:8]
        self.layer4 = backbone[8:]

        if 5 in feature_levels:
            self.layer5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )

        channels = [64, 128, 256, 512, 512]

        self.out_channels = [
            channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class Darknet(nn.Module):
    def __init__(self, feature_levels=(3, 4, 5), pretrained=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model("darknet53", pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = nn.Sequential(
            net.init_block,
            net.stage1,
        )
        self.layer2 = net.stage2
        self.layer3 = net.stage3
        self.layer4 = net.stage4
        self.layer5 = net.stage5

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ResNet(nn.Module):
    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.stem
        self.layer2 = nn.Sequential(
            net.init_block.pool,
            net.stage1,
        )
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if hasattr(net, "post_activ"):
            self.layer5 = nn.Sequential(
                net.stage4,
                net.post_activ,
            )
        else:
            self.layer5 = nn.Sequential(
                net.stage4,
            )
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ResNetTV(nn.Module):
    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        from torchvision.models import resnet18
        net = resnet18(pretrained=pretrained)
        del net.fc
        self.layer1 = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
        )
        self.layer2 = net.maxpool
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        self.layer5 = net.layer4

        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class DLA(nn.Module):

    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block
        self.layer2 = net.stage1
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        self.layer5 = net.stage4
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class Backbone(nn.Module):
    r"""
    General backbone network for ResNet-like architecture.
    Supported: ResNet, DenseNet, SENet, PyramidNet
    """

    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.stem
        self.layer2 = nn.Sequential(
            net.init_block.pool,
            net.stage1,
        )
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if hasattr(net, "post_activ"):
            self.layer5 = nn.Sequential(
                net.stage4,
                net.post_activ,
            )
        else:
            self.layer5 = net.stage4
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ESPNetv2(nn.Module):
    r""" ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

    width_mult  Top1    Top5    Params      FLOPs/2
    x0.5        42.32	20.15	1,241,332   35.36M
    x1.0        33.92	13.45	1,670,072   98.09M
    x1.25       32.06	12.18	1,965,440   138.18M
    x1.5        30.83	11.29	2,314,856   185.77M
    x2.0        27.94   9.61    3,498,136   306.93M
    """
    mult2name = {
        0.5: 'espnetv2_wd2',
        1.0: 'espnetv2_w1',
        1.25: 'espnetv2_w5d4',
        1.5: 'espnetv2_w3d2',
        2.0: 'espnetv2_w2',
    }

    def __init__(self, width_mult=2.0, feature_levels=(3, 4, 5), pretrained=True, include_final=False):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        self.include_final = include_final
        name = self.mult2name[float(width_mult)]
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block
        self.layer2 = net.stage1
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if include_final:
            self.layer51 = net.stage4
            self.layer52 = net.final_block
        else:
            self.layer5 = net.stage4
        out_channels = [
            net.stage1[-1].activ.num_parameters,
            net.stage2[-1].activ.num_parameters,
            net.stage3[-1].activ.num_parameters,
            net.final_block.conv2.stem.out_channels if include_final else net.stage4[-1].activ.num_parameters,
        ]
        self.out_channels = [
            out_channels[i-2] for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x, x)
        x = self.layer2(*x)
        if 2 in self.feature_levels:
            outs.append(x[0])
        x = self.layer3(*x)
        if 3 in self.feature_levels:
            outs.append(x[0])
        x = self.layer4(*x)
        if 4 in self.feature_levels:
            outs.append(x[0])
        if 5 in self.forward_levels:
            if self.include_final:
                x = self.layer51(*x)
                x = self.layer52(x[0])
            else:
                x = self.layer5(*x)[0]
            if 5 in self.feature_levels:
                outs.append(x)
        return outs


class MobileNetV3(nn.Module):
    r"""MobileNetV3: Searching for MobileNetV3

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, feature_levels=(3, 4, 5), pretrained=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        backbone = mobilenetv3_large(pretrained=pretrained)
        del backbone.classifier
        features = backbone.features

        self.layer1 = features[:2]
        self.layer2 = nn.Sequential(
            features[2:4],
            features[4].stem[:3],
        )
        self.layer3 = nn.Sequential(
            features[4].stem[3:],
            *features[5:7],
            features[7].stem[:3],
        )
        self.layer4 = nn.Sequential(
            features[7].stem[3:],
            *features[8:13],
            features[13].stem[:3],
        )
        self.layer5 = nn.Sequential(
            features[13].stem[3:],
            *features[14:],
            backbone.stem
        )

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class VoVNet(nn.Module):
    version2name = {
        27: vovnet27_slim,
        39: vovnet39,
        57: vovnet57,
    }

    def __init__(self, version=27, feature_levels=(3, 4, 5), pretrained=True, no_down=0, atrous=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels

        if no_down != 0:
            assert feature_levels == (3, 4) and no_down == -1

        backbone = self.version2name[version](pretrained=pretrained)
        del backbone.classifier
        self.layer1 = backbone.stem[:6]
        self.layer2 = nn.Sequential(
            backbone.stem[6:],
            backbone.stage2,
        )
        self.layer3 = backbone.stage3

        if no_down:
            del backbone.stage5.Pooling
            if atrous:
                conv_to_atrous(backbone.stage5, rate=2)
            self.layer4 = nn.Sequential(
                backbone.stage4,
                backbone.stage5
            )
        else:
            self.layer4 = backbone.stage4
            self.layer5 = backbone.stage5

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class VGG16BN(nn.Module):
    def __init__(self, feature_levels=(3, 4), pretrained=True):
        super().__init__()
        assert pretrained, "Only pretrained VGG16 is provided."
        self.feature_levels = feature_levels
        from torchvision.models import vgg16_bn
        backbone = vgg16_bn(pretrained=True)
        f = backbone.features
        f[6].ceil_mode = True
        f[13].ceil_mode = True
        f[23].ceil_mode = True
        f[33].ceil_mode = True
        f[43].kernel_size = (3, 3)
        f[43].stride = (1, 1)
        f[43].padding = (1, 1)

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        fc6 = backbone.classifier[0]
        fc6_weight = fc6.weight.data.view(4096, 512, 7, 7)
        fc6_bias = fc6.bias.data
        conv6.weight.data = decimate(fc6_weight, m=[4, None, 3, 3])
        conv6.bias.data = decimate(fc6_bias, m=[4])

        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        fc7 = backbone.classifier[3]
        fc7_weight = fc7.weight.data.view(4096, 4096, 1, 1)
        fc7_bias = fc7.bias.data
        conv7.weight.data = decimate(fc7_weight, m=[4, 4, None, None])
        conv7.bias.data = decimate(fc7_bias, m=[4])

        self.stage0 = f[:6]
        self.stage1 = f[6:13]
        self.stage2 = f[13:23]
        self.stage3 = f[23:33]
        self.stage4 = nn.Sequential(
            *f[33:],
            conv6,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            conv7,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [
            get_out_channels(getattr(self, ("stage%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        outs = []

        for i in range(5):
            l = getattr(self, ("stage%d" % i))
            x = l(x)
            if i in self.feature_levels:
                outs.append(x)

        return outs


class VGG16(nn.Module):
    def __init__(self, feature_levels=(3, 4), pretrained=True, l2_norm=True, **kwargs):
        super().__init__()
        assert feature_levels == (3, 4)
        assert pretrained, "Only pretrained VGG16 is provided."
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        from torchvision.models import vgg16
        backbone = vgg16(pretrained=True)
        f = backbone.features
        f[4].ceil_mode = True
        f[9].ceil_mode = True
        f[16].ceil_mode = True
        f[23].ceil_mode = True
        f[30].kernel_size = (3, 3)
        f[30].stride = (1, 1)
        f[30].padding = (1, 1)

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        fc6 = backbone.classifier[0]
        fc6_weight = fc6.weight.data.view(4096, 512, 7, 7)
        fc6_bias = fc6.bias.data
        conv6.weight.data = decimate(fc6_weight, m=[4, None, 3, 3])
        conv6.bias.data = decimate(fc6_bias, m=[4])

        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        fc7 = backbone.classifier[3]
        fc7_weight = fc7.weight.data.view(4096, 4096, 1, 1)
        fc7_bias = fc7.bias.data
        conv7.weight.data = decimate(fc7_weight, m=[4, 4, None, None])
        conv7.bias.data = decimate(fc7_bias, m=[4])

        self.layer1 = f[:9]
        self.layer2 = f[9:16]
        self.layer3 = f[16:23]
        self.l2_norm = L2Norm(512, 20) if l2_norm else nn.Identity()
        self.layer4 = nn.Sequential(
            *f[23:],
            conv6,
            nn.ReLU(inplace=True),
            conv7,
            nn.ReLU(inplace=True),
        )

        self.out_channels = [512, 1024]

    def forward(self, x):
        c3, c4 = backbone_forward(self, x)
        c3 = self.l2_norm(c3)
        return c3, c4


def search(name, n=10, cutoff=0.6):
    from pytorchcv.models.model_store import _model_sha1
    models = _model_sha1.keys()
    return get_close_matches(name, models, n=n, cutoff=cutoff)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m._freezed = True
