import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50, resnet101

from horch.models.mobilenet import mobilenetv2
from horch.models.mobilenetv3 import mobilenetv3
from horch.models.snet import SNet as BSNet
from horch.models.squeezenext import SqueezeNext as BSqueezeNext
from horch.models.utils import get_out_channels
from horch.models.darknet import Darknet as BDarknet


class ShuffleNetV2(nn.Module):
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

    Args:
        mult (number): 0.5, 1, 1.5, 2
            Default: 0.5
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        if pretrained:
            norm_layer = kwargs.get("norm_layer")
            assert norm_layer is None or norm_layer == 'bn', "`gn` can be only set when `pretrained` is False"
            from horch.models.shufflenet import shufflenetv2 as BShuffleNetV2
            net = BShuffleNetV2(mult=mult, pretrained=pretrained, **kwargs)
            channels = net._stage_out_channels
        else:
            from horch.models.re.shufflenet import shufflenet_v2 as BShuffleNetV2
            net = BShuffleNetV2(mult=mult, **kwargs)
            channels = net.out_channels
        del net.fc
        self.layer1 = net.conv1
        self.layer2 = net.maxpool
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        self.layer5 = nn.Sequential(
            net.stage4,
            net.conv5,
        )
        self.feature_levels = feature_levels
        out_channels = [channels[0]] + channels[:3] + [channels[-1]]
        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return outs


class SNet(nn.Module):
    r"""ThunderNet: Towards Real-time Generic Object Detection

    Args:
        version (int): 49, 146, 535
            Default: 49
        feature_levels (sequence of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, version=49, feature_levels=(3, 4, 5), **kwargs):
        super().__init__()
        net = BSNet(num_classes=1, version=version, **kwargs)
        del net.fc
        channels = net.channels
        self.layer1 = net.conv1
        self.layer2 = net.maxpool
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if len(channels) == 5:
            self.layer5 = nn.Sequential(
                net.stage4,
                net.conv5,
            )
        else:
            self.layer5 = net.stage4
        out_channels = [channels[0]] + channels[:3] + [channels[-1]]
        self.feature_levels = feature_levels
        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return tuple(outs)


class ResNet(nn.Module):
    r"""Pretrained ResNet from torchvision

    Args:
        version (int): 18, 50, 101
            Default: 18
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, version=18, feature_levels=(3, 4, 5)):
        super().__init__()
        if version == 18:
            net = resnet18(pretrained=True)
        elif version == 50:
            net = resnet50(pretrained=True)
        elif version == 101:
            net = resnet101(pretrained=True)
        else:
            raise ValueError("version must be one of [18, 50, 101]")
        del net.fc
        self.layer1 = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
        )
        self.layer2 = net.maxpool
        self.layer3 = nn.Sequential(
            net.layer1,
            net.layer2,
        )
        self.layer4 = net.layer3
        self.layer5 = net.layer4
        out_channels = [
            64, 64,
            get_out_channels(self.layer3),
            get_out_channels(self.layer4),
            get_out_channels(self.layer5),
        ]
        self.feature_levels = feature_levels
        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return outs


class MobileNetV2(nn.Module):
    r"""MobileNetV2: Inverted Residuals and Linear Bottlenecks

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        backbone = mobilenetv2(mult=mult, pretrained=pretrained)
        del backbone.classifier
        features = backbone.features

        self.layer1 = features[:2]
        self.layer2 = features[2:4]
        self.layer3 = nn.Sequential(
            *features[4:7],
            features[7].conv[:3],
        )
        self.layer4 = nn.Sequential(
            features[7].conv[3:],
            *features[8:14],
            features[14].conv[:3],
        )
        self.layer5 = nn.Sequential(
            features[14].conv[3:],
            *features[15:],
            backbone.conv,
        )
        self.feature_levels = feature_levels
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return outs


class MobileNetV3(nn.Module):
    r"""MobileNetV3: Searching for MobileNetV3

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), norm_layer='bn'):
        super().__init__()
        backbone = mobilenetv3(mult=mult, num_classes=1, norm_layer=norm_layer)
        del backbone.classifier
        features = backbone.features

        self.layer1 = features[:2]
        self.layer2 = features[2:4]
        self.layer3 = nn.Sequential(
            *features[4:7],
            features[7].conv[:3],
        )
        self.layer4 = nn.Sequential(
            features[7].conv[3:],
            *features[8:14],
            features[14].conv[:3],
        )
        self.layer5 = nn.Sequential(
            features[14].conv[3:],
            *features[15:],
        )
        self.feature_levels = feature_levels
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return outs


class SqueezeNet(nn.Module):
    r"""SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB models size

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, feature_levels=(3, 4, 5)):
        super().__init__()
        from torchvision.models.squeezenet import squeezenet1_1, Fire
        backbone = squeezenet1_1(pretrained=True)
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

        self.feature_levels = feature_levels
        self.out_channels = [
            channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        if 5 in self.feature_levels:
            x = self.layer5(x)
            outs.append(x)
        return outs


class Darknet(nn.Module):
    def __init__(self, feature_levels=(3, 4, 5), f_channels=128):
        super().__init__()
        backbone = BDarknet(num_classes=1, f_channels=f_channels)
        del backbone.fc
        self.f_channels = backbone.f_channels
        self.layer1 = nn.Sequential(
            backbone.conv0,
            backbone.down1,
            backbone.layer1,
        )

        self.layer2 = nn.Sequential(
            backbone.down2,
            backbone.layer2,
        )

        self.layer3 = nn.Sequential(
            backbone.down3,
            backbone.layer3,
        )

        self.layer4 = nn.Sequential(
            backbone.down4,
            backbone.layer4,
        )

        self.layer5 = nn.Sequential(
            backbone.down5,
            backbone.layer5,
        )

        self.feature_levels = feature_levels
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
        return outs

# class SqueezeNext(nn.Module):
#     r"""SqueezeNext: Hardware-Aware Neural Network Design
#
#     Args:
#         feature_levels (list of int): features of which layers to output
#             Default: (3, 4, 5)
#     """
#
#     def __init__(self, feature_levels=(3, 4, 5)):
#         super().__init__()
#         backbone = SqueezeNext(feature_levels)
#
#         self.layer1 = backbone[:2]
#         self.layer2 = backbone[2:5]
#         self.layer3 = backbone[5:8]
#         self.layer4 = backbone[8:]
#
#         if 5 in feature_levels:
#             self.layer5 = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(512, 64, 256, 256),
#             )
#
#         channels = [64, 128, 256, 512, 512]
#
#         self.feature_levels = feature_levels
#         self.out_channels = [
#             channels[i-1] for i in feature_levels
#         ]
#
#     def forward(self, x):
#         outs = []
#         x = self.layer1(x)
#         x = self.layer2(x)
#         if 2 in self.feature_levels:
#             outs.append(x)
#         x = self.layer3(x)
#         if 3 in self.feature_levels:
#             outs.append(x)
#         x = self.layer4(x)
#         if 4 in self.feature_levels:
#             outs.append(x)
#         if 5 in self.feature_levels:
#             x = self.layer5(x)
#             outs.append(x)
#         return outs
