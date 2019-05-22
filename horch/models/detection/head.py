from math import log

import torch.nn as nn

from horch.models.modules import Conv2d, DWConv2d, get_norm_layer
from horch.models.utils import weight_init_normal, bias_init_constant, get_last_conv
from horch.common import _tuple, _concat, inverse_sigmoid


def to_pred(p, c: int):
    b = p.size(0)
    p = p.permute(0, 3, 2, 1).contiguous()
    if c == 1:
        p = p.view(b, -1)
    else:
        p = p.view(b, -1, c)
    return p


class ThunderRCNNHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """

    def __init__(self, num_classes, f_channels=256, norm_layer='bn'):
        super().__init__()
        self.fc1 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer=norm_layer, activation='default')
        self.fc2 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer=norm_layer, activation='default')
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

    def __init__(self, num_anchors, num_classes=2, in_channels=245, f_channels=256, norm_layer='bn'):
        super().__init__()
        self.num_classes = num_classes
        self.conv = DWConv2d(
            in_channels, f_channels, kernel_size=5,
            mid_norm_layer=norm_layer, norm_layer=norm_layer,
            activation='default')
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


def _make_head(f_channels, num_layers, out_channels, norm_layer, lite):
    layers = []
    for i in range(num_layers):
        layers.append(Conv2d(f_channels, f_channels, kernel_size=3,
                             norm_layer=norm_layer, activation='default', depthwise_separable=lite))
    layers.append(Conv2d(f_channels, out_channels, kernel_size=3))
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
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: "bn"
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    concat : bool
        Whether to concat predictions in `eval` mode.
    """

    def __init__(self, num_anchors, num_classes, f_channels=256, num_layers=4,
                 norm_layer='bn', lite=False, concat=True, box_std=False):
        super().__init__()
        self.num_classes = num_classes
        self.concat = concat
        self.box_std = box_std
        self.loc_channels = 8 if box_std else 4
        self.loc_head = _make_head(
            f_channels, num_layers, num_anchors * self.loc_channels, norm_layer=norm_layer, lite=lite)
        self.cls_head = _make_head(
            f_channels, num_layers, num_anchors * num_classes, norm_layer=norm_layer, lite=lite)

        # weight_init_normal(self.loc_head[-1][-1], 0, 0.0001)
        bias_init_constant(self.cls_head[-1], inverse_sigmoid(0.01))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        if self.box_std:
            log_var_preds = []
        for p in ps:
            loc_p = to_pred(self.loc_head(p), self.loc_channels)
            if self.box_std:
                log_var_preds.append(loc_p[..., 4:])
            loc_preds.append(loc_p[..., :4])

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p)

        if not self.training and not self.concat:
            if self.box_std:
                return loc_preds, cls_preds, log_var_preds
            else:
                return loc_preds, cls_preds

        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        if self.box_std:
            log_var_p = _concat(log_var_preds, dim=1)
            return loc_p, cls_p, log_var_p
        else:
            return loc_p, cls_p


class SSDHead(nn.Module):
    r"""
    Head of SSD.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    in_channels : sequence of ints
        Number of input channels of every level, e.g., ``(256,512,1024,256,256,128)``
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: "bn"
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    concat : bool
        Whether to concat predictions in `eval` mode.
    """

    def __init__(self, num_anchors, num_classes, in_channels, norm_layer='bn', lite=False, concat=True, box_std=False):
        super().__init__()
        self.num_classes = num_classes
        self.box_std = box_std
        self.concat = concat
        num_anchors = _tuple(num_anchors, len(in_channels))
        self.preds = nn.ModuleList([
            nn.Sequential(
                get_norm_layer(norm_layer, c),
                Conv2d(c, n * (num_classes + 4), kernel_size=3, depthwise_separable=lite, mid_norm_layer=norm_layer)
            )
            for c, n in zip(in_channels, num_anchors)
        ])
        if box_std:
            self.box_std_preds = nn.ModuleList([
                nn.Sequential(
                    get_norm_layer(norm_layer, c),
                    Conv2d(c, n * 4, kernel_size=3, depthwise_separable=lite, mid_norm_layer=norm_layer)
                )
                for c, n in zip(in_channels, num_anchors)
            ])

        for p in self.preds:
            get_last_conv(p).bias.data[4:].fill_(inverse_sigmoid(0.01))
        # if self.box_std:
        #     for p in self.box_std_preds:
        #         weight_init_normal(get_last_conv(p), 0, 0.0001)

    def forward(self, *ps):
        preds = [pred(p) for p, pred in zip(ps, self.preds)]
        loc_preds = []
        cls_preds = []
        if self.box_std:
            log_var_preds = []
        b = ps[0].size(0)
        for p in preds:
            p = p.permute(0, 3, 2, 1).contiguous().view(b, -1, 4 + self.num_classes)
            loc_preds.append(p[..., :4])
            cls_p = p[..., 4:]
            if cls_p.size(-1) == 1:
                cls_p = cls_p[..., 0]
            cls_preds.append(cls_p)

        if self.box_std:
            log_var_preds = [
                pred(p).permute(0, 3, 2, 1).contiguous().view(b, -1, 4)
                for p, pred in zip(ps, self.box_std_preds)
            ]

        if not self.training and not self.concat:
            if self.box_std:
                return loc_preds, cls_preds, log_var_preds
            else:
                return loc_preds, cls_preds
        else:
            loc_p = _concat(loc_preds, dim=1)
            cls_p = _concat(cls_preds, dim=1)
            if self.box_std:
                log_var_p = _concat(log_var_preds, dim=1)
                return loc_p, cls_p, log_var_p
            else:
                return loc_p, cls_p
