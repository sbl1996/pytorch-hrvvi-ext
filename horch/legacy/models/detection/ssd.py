import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import tuplify
from horch.models.modules import Conv2d
from horch.models.detection.head import SSDHead


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, lite=False):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels // 2, kernel_size=1,
            norm='default', act='default')
        self.conv2 = Conv2d(
            out_channels // 2, out_channels, kernel_size=3, stride=2,
            norm='default', act='default', padding=padding, depthwise_separable=lite)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SSD(nn.Module):
    r"""
    Note: Feature levels of backbone could be (4,5) or (3,4,5).
    """

    def __init__(self, backbone, num_anchors=(4, 6, 6, 6, 6, 4), num_classes=21,
                 f_channels=256, extra_levels=(6, 7, 8),
                 pad_last=False, lite=False):
        super().__init__()
        extra_levels = tuplify(extra_levels)
        feature_levels = backbone.feature_levels + extra_levels
        num_anchors = tuplify(num_anchors, len(feature_levels))
        self.num_classes = num_classes
        self.backbone = backbone
        self.feature_levels = feature_levels
        self.num_anchors = num_anchors
        backbone_channels = backbone.out_channels
        head_in_channels = list(backbone_channels)
        if 6 in extra_levels:
            self.layer6 = DownBlock(backbone_channels[-1], 2 * f_channels, lite=lite)
            head_in_channels.append(2 * f_channels)
        if 7 in extra_levels:
            self.layer7 = DownBlock(2 * f_channels, f_channels, lite=lite)
            head_in_channels.append(f_channels)
        if 8 in extra_levels:
            padding = 1 if pad_last else 0
            self.layer8 = DownBlock(f_channels, f_channels, padding=padding, lite=lite)
            head_in_channels.append(f_channels)
        self.head = SSDHead(num_anchors, num_classes, head_in_channels, lite=lite)

    def forward(self, x):
        cs = list(self.backbone(x))
        if 6 in self.feature_levels:
            c6 = self.layer6(cs[-1])
            cs.append(c6)
        if 7 in self.feature_levels:
            c7 = self.layer7(cs[-1])
            cs.append(c7)
        if 8 in self.feature_levels:
            c8 = self.layer8(cs[-1])
            cs.append(c8)
        loc_p, cls_p = self.head(*cs)
        return loc_p, cls_p
