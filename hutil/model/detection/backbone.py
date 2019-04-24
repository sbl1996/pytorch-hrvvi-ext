import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50, resnet101

from hutil.model.mobilenet import mobilenet_v2
from hutil.model.shufflenet import ShuffleNetV2 as BShuffleNetV2
from hutil.model.snet import SNet as BSNet
from hutil.model.utils import get_out_channels


class ShuffleNetV2(nn.Module):
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

    Args:
        mult (number): 0.5, 1, 1.5, 2
            Default: 0.5
        feature_levels (list of int): features of which layers to output
            Default: [3, 4, 5]
    """

    def __init__(self, mult=0.5, feature_levels=[3, 4, 5]):
        super().__init__()
        net = BShuffleNetV2(num_classes=1, mult=mult)
        del net.fc
        channels = net.channels
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
            out_channels[i-1] for i in feature_levels
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
        feature_levels (list of int): features of which layers to output
            Default: [3, 4, 5]
    """

    def __init__(self, version=49, feature_levels=[3, 4, 5], normalization='bn'):
        super().__init__()
        net = BSNet(num_classes=1, version=version,
                    normalization=normalization)
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
            out_channels[i-1] for i in feature_levels
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


class ResNet(nn.Module):
    r"""Pretrained ResNet from torchvision

    Args:
        version (int): 18, 50, 101
            Default: 18
        feature_levels (list of int): features of which layers to output
            Default: [3, 4, 5]
    """

    def __init__(self, version=18, feature_levels=[3, 4, 5]):
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
            out_channels[i-1] for i in feature_levels
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
            Default: [3, 4, 5]
    """

    def __init__(self, feature_levels=[3, 4, 5]):
        super().__init__()
        backbone = mobilenet_v2(num_classes=1)
        del backbone.classifier
        backbone = backbone.features

        self.layer1 = backbone[:2]
        self.layer2 = backbone[2:4]
        self.layer3 = nn.Sequential(
            *backbone[4:7],
            backbone[7].conv[0],
        )
        self.layer4 = nn.Sequential(
            backbone[7].conv[1:],
            *backbone[8:14],
            backbone[14].conv[0]
        )
        self.layer5 = nn.Sequential(
            backbone[14].conv[1:],
            *backbone[15:]
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
