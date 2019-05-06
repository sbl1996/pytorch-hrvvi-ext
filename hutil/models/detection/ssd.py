from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.models.utils import get_out_channels, get_loc_cls_preds
from hutil.models.modules import Conv2d


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels // 2, kernel_size=1,
            norm_layer='gn', activation='relu')
        self.conv2 = Conv2d(
            out_channels // 2, out_channels, kernel_size=3, stride=2,
            norm_layer='gn', activation='relu', padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SSD(nn.Module):
    r"""
    Note: Feature levels of backbone must be (3, 4, 5).
    """
    def __init__(self, backbone, num_anchors=(4, 6, 6, 6, 6, 4), num_classes=21, f_channels=256, pad_last=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        if isinstance(num_anchors, Number):
            num_anchors = [int(num_anchors)] * 6
        num_anchors = tuple(num_anchors)
        backbone_channels = backbone.out_channels
        self.with_stride8 = 3 in backbone.feature_levels
        if self.with_stride8:
            self.pred3 = Conv2d(
                backbone_channels[-3], num_anchors[0] * (4 + num_classes), kernel_size=3)
        else:
            num_anchors = (0,) + num_anchors
        self.pred4 = Conv2d(
            backbone_channels[-2], num_anchors[1] * (4 + num_classes), kernel_size=3)
        self.pred5 = Conv2d(
            backbone_channels[-1], num_anchors[2] * (4 + num_classes), kernel_size=3)
        if len(num_anchors) > 3:
            self.layer6 = DownBlock(
                backbone_channels[-1], 2 * f_channels)
            self.pred6 = Conv2d(
                get_out_channels(self.layer6), num_anchors[3] * (4 + num_classes), kernel_size=3)
        if len(num_anchors) > 4:
            self.layer7 = DownBlock(2 * f_channels, f_channels)
            self.pred7 = Conv2d(
                get_out_channels(self.layer7), num_anchors[4] * (4 + num_classes), kernel_size=3)
        if len(num_anchors) > 5:
            padding = 1 if pad_last else 0
            self.layer8 = DownBlock(f_channels, f_channels, padding=padding)
            self.pred8 = Conv2d(
                get_out_channels(self.layer8), num_anchors[5] * (4 + num_classes), kernel_size=3)

        self.num_anchors = num_anchors

    def forward(self, x):
        b = x.size(0)
        if self.with_stride8:
            c3, c4, c5 = self.backbone(x)
        else:
            c4, c5 = self.backbone(x)

        p4 = self.pred4(c4)
        p5 = self.pred5(c5)
        ps = [p4, p5]
        if self.with_stride8:
            p3 = self.pred3(c3)
            ps = [p3] + ps
        if len(self.num_anchors) > 3:
            c6 = self.layer6(c5)
            p6 = self.pred6(c6)
            ps.append(p6)
        if len(self.num_anchors) > 4:
            c7 = self.layer7(c6)
            p7 = self.pred7(c7)
            ps.append(p7)
        if len(self.num_anchors) > 5:
            c8 = self.layer8(c7)
            p8 = self.pred8(c8)
            ps.append(p8)

        loc_p, cls_p = get_loc_cls_preds(ps, self.num_classes)
        return loc_p, cls_p

