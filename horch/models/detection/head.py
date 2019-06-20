from math import log

import torch.nn as nn

from horch.models.modules import Conv2d, DWConv2d, get_norm_layer, MBConv
from horch.models.utils import weight_init_normal, bias_init_constant, get_last_conv
from horch.common import tuplify, _concat, inverse_sigmoid


def to_pred(p, c: int):
    p = p.permute(0, 3, 2, 1).contiguous()
    b, w, h = p.size()[:3]
    p = p.view(b, w, h, -1, c).squeeze(3)
    return p


class ThunderRCNNHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """

    def __init__(self, num_classes, f_channels=256):
        super().__init__()
        self.fc1 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer='default', activation='default')
        self.fc2 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer='default', activation='default')
        self.loc_fc = nn.Linear(f_channels, 4)
        self.cls_fc = nn.Linear(f_channels, num_classes)

    def forward(self, p):
        if p.ndimension() == 3:
            n1, n2, c = p.size()
            p = p.view(n1 * n2, c, 1, 1)
            p = self.fc(p).view(n1, n2, -1)
        else:
            n, c = p.size()
            p = self.fc(p.view(n, c, 1, 1)).view(n, -1)
        loc_p = self.loc_fc(p)
        cls_p = self.cls_fc(p)
        return loc_p, cls_p


class SharedDWConvHead(nn.Module):
    r"""
    Light head for RPN, not for R-CNN.
    """

    def __init__(self, num_anchors, num_classes=2, in_channels=245, f_channels=256):
        super().__init__()
        self.num_classes = num_classes
        self.conv = DWConv2d(
            in_channels, f_channels, kernel_size=5,
            norm_layer='default', activation='default')
        self.loc_conv = Conv2d(
            f_channels, num_anchors * 4, kernel_size=1)
        self.cls_conv = Conv2d(
            f_channels, num_anchors * num_classes, kernel_size=1)

        bias_init_constant(self.cls_conv, inverse_sigmoid(0.01))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            p = self.conv(p)
            loc_p = to_pred(self.loc_conv(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_conv(p), self.num_classes)
            cls_preds.append(cls_p)
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p


class RetinaLiteHead(nn.Module):
    r"""
    Head of RetinaNet.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    f_channels : int
        Number of feature channels.
    num_layers : int
        Number of conv layers in each subnet.
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """

    def __init__(self, num_anchors, num_classes, in_channels_list, f_channels=256, expand_ratio=4, num_layers=4):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.projects = nn.ModuleList([
            MBConv(c, c, f_channels, kernel_size=5)
            for c in in_channels_list
        ])
        self.num_classes = num_classes
        self.loc_head = self._make_head(
            num_layers, f_channels, num_anchors * 4)
        self.cls_head = self._make_head(
            num_layers, f_channels, num_anchors * num_classes)

        bias_init_constant(self.cls_head[-1][1], inverse_sigmoid(0.01))

    def _make_head(self, num_layers, f_channels, out_channels):
        layers = []
        for i in range(num_layers):
            layers.append(MBConv(f_channels, f_channels * self.expand_ratio, f_channels, kernel_size=5))
        layers.append(nn.Sequential(
            get_norm_layer('default', f_channels),
            Conv2d(f_channels, out_channels, kernel_size=1),
        ))
        return nn.Sequential(*layers)

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p, proj in zip(ps, self.projects):
            p = proj(p)
            loc_p = to_pred(self.loc_head(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p)
        return loc_preds, cls_preds


def _make_head(f_channels, num_layers, out_channels, lite):
    layers = []
    for i in range(num_layers):
        layers.append(Conv2d(f_channels, f_channels, kernel_size=3,
                             norm_layer='default', activation='default', depthwise_separable=lite))
    layers.append(Conv2d(f_channels, out_channels, kernel_size=3,
                         mid_norm_layer='default', depthwise_separable=lite))
    return nn.Sequential(*layers)


class RetinaHead(nn.Module):
    r"""
    Head of RetinaNet.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    f_channels : int
        Number of feature channels.
    num_layers : int
        Number of conv layers in each subnet.
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """

    def __init__(self, num_anchors, num_classes, f_channels=256, num_layers=4, lite=False):
        super().__init__()
        self.num_classes = num_classes
        self.loc_head = _make_head(
            f_channels, num_layers, num_anchors * 4, lite=lite)
        self.cls_head = _make_head(
            f_channels, num_layers, num_anchors * num_classes, lite=lite)

        weight_init_normal(self.loc_head, 0, 0.01)
        weight_init_normal(self.cls_head, 0, 0.01)

        fc = self.cls_head[-1]
        if lite:
            fc = fc[-1]
        bias_init_constant(fc, inverse_sigmoid(0.01))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            loc_p = to_pred(self.loc_head(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p)
        return loc_preds, cls_preds


class SSDHead(nn.Module):
    r"""
    Head of SSD.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(256,512,1024,256,256,128)``
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """

    def __init__(self, num_anchors, num_classes, in_channels_list, focal_init=False, lite=False, large_kernel=False):
        super().__init__()
        self.num_classes = num_classes
        num_anchors = tuplify(num_anchors, len(in_channels_list))
        kernel_size = 5 if (lite and large_kernel) else 3
        self.loc_heads = nn.ModuleList([
            nn.Sequential(
                get_norm_layer("default", c),
                Conv2d(c, n * 4, kernel_size=kernel_size,
                        depthwise_separable=lite, mid_norm_layer='default')
            )
            for c, n in zip(in_channels_list, num_anchors)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                get_norm_layer("default", c),
                Conv2d(c, n * num_classes, kernel_size=kernel_size,
                        depthwise_separable=lite, mid_norm_layer='default')
            )
            for c, n in zip(in_channels_list, num_anchors)
        ])

        if focal_init:
            for p in self.cls_heads:
                get_last_conv(p).bias.data.fill_(inverse_sigmoid(0.01))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p, loc_head, cls_head in zip(ps, self.loc_heads, self.cls_heads):
            loc_p = to_pred(loc_head(p), 4)
            cls_p = to_pred(cls_head(p), self.num_classes)
            loc_preds.append(loc_p)
            cls_preds.append(cls_p)
        return loc_preds, cls_preds


class ConvHead(nn.Module):
    r"""
    Simplest SSD-style head with just 1x1 convs.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(256,512,1024,256,256,128)``
    focal_init : bool
        Whether to initialize the last layer of cls_head according to Focal Loss.
    """

    def __init__(self, num_anchors, num_classes, in_channels_list, focal_init=False):
        super().__init__()
        self.num_classes = num_classes
        num_anchors = tuplify(num_anchors, len(in_channels_list))
        self.loc_heads = nn.ModuleList([
            nn.Sequential(
                get_norm_layer("default", c),
                Conv2d(c, n * 4, kernel_size=1)
            )
            for c, n in zip(in_channels_list, num_anchors)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                get_norm_layer("default", c),
                Conv2d(c, n * num_classes, kernel_size=1)
            )
            for c, n in zip(in_channels_list, num_anchors)
        ])

        if focal_init:
            for p in self.cls_heads:
                get_last_conv(p).bias.data.fill_(inverse_sigmoid(0.01))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p, loc_head, cls_head in zip(ps, self.loc_heads, self.cls_heads):
            loc_p = to_pred(loc_head(p), 4)
            cls_p = to_pred(cls_head(p), self.num_classes)
            loc_preds.append(loc_p)
            cls_preds.append(cls_p)
        return loc_preds, cls_preds
