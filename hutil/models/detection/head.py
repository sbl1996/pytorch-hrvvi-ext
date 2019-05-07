from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.models.modules import Conv2d, depthwise_seperable_conv3x3
from hutil.models.utils import get_out_channels, get_loc_cls_preds
from hutil.common import _tuple

def to_pred(p, c: int):
    b = p.size(0)
    p = p.permute(0, 3, 2, 1).contiguous().view(b, -1, c)
    return p


def _concat(preds, dim=1):
    if len(preds) == 1:
        return preds
    return torch.cat(preds, dim=dim)


class ThunderRCNNHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """
    def __init__(self, num_classes, in_channels=245, f_channels=256, norm_layer='bn'):
        super().__init__()
        self.fc = Conv2d(in_channels, f_channels, kernel_size=1,
                         norm_layer=norm_layer, activation='relu')
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
    def __init__(self, num_anchors, num_classes=2, in_channels=245, f_channels=256, return_features=False, norm_layer='bn', with_se=False):
        super().__init__()
        self.num_classes = num_classes
        self.return_features = return_features
        self.conv = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels,
                   norm_layer=norm_layer),
            Conv2d(in_channels, f_channels, kernel_size=1,
                   norm_layer=norm_layer, activation='relu', with_se=with_se)
        )
        self.loc_conv = Conv2d(
            f_channels, num_anchors * 4, kernel_size=1)
        self.cls_conv = Conv2d(
            f_channels, num_anchors * num_classes, kernel_size=1)

        self.cls_conv.apply(self._init_final_cls_layer)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            p = self.conv(p)
            loc_p = to_pred(self.loc_conv(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_conv(p), self.num_classes)
            cls_preds.append(cls_p)
            if self.return_features:
                return loc_p, cls_p, p
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p


def _make_head(f_channels, num_layers, out_channels, **kwargs):
    layers = []
    for i in range(num_layers):
        layers.append(Conv2d(f_channels, f_channels, kernel_size=3,
                             activation='relu', **kwargs))
    layers.append(Conv2d(f_channels, out_channels, kernel_size=3))
    return nn.Sequential(*layers)


class ConvHead(nn.Module):
    def __init__(self, num_anchors, num_classes, f_channels=256, num_layers=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.loc_head = _make_head(
            f_channels, num_layers, num_anchors * 4, **kwargs)
        self.cls_head = _make_head(
            f_channels, num_layers, num_anchors * num_classes, **kwargs)

        self.cls_head[-1].apply(self._init_final_cls_layer)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            loc_p = to_pred(self.loc_head(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p)
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p


class SSDHead(nn.Module):
    def __init__(self, num_anchors, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes
        num_anchors = _tuple(num_anchors, len(in_channels))
        self.preds = nn.ModuleList([
            Conv2d(c, n * (num_classes + 4))
            for c, n in zip(in_channels, num_anchors)
        ])

    def forward(self, *ps):
        ps = [pred(p) for p, pred in zip(ps, self.preds)]
        loc_p, cls_p = get_loc_cls_preds(ps, self.num_classes)
        return loc_p, cls_p


class SSDLightHead(nn.Module):
    r"""
    Head of SSDLite.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    num_classes : int
        Number of classes.
    in_channels : tuple of ints
        Number of input channels of every level, e.g., ``(256,512,1024,256,256,128)``
    norm_layer : str
        Type of normalization layer in the middle of depthwise seperable convolution.

    """
    def __init__(self, num_anchors, num_classes, in_channels, norm_layer='bn'):
        super().__init__()
        self.num_classes = num_classes
        num_anchors = _tuple(num_anchors, len(in_channels))
        self.convs = nn.ModuleList([
            depthwise_seperable_conv3x3(c, n * (num_classes + 4), norm_layer=norm_layer)
            for c, n in zip(in_channels, num_anchors)
        ])

    def forward(self, *ps):
        preds = []
        for p, conv in zip(ps, self.convs):
            print(conv)
            preds.append(conv(p))
        # preds = [conv(p) for p, conv in zip(ps, self.convs)]
        print("Sep")
        loc_p, cls_p = get_loc_cls_preds(preds, self.num_classes)
        return loc_p, cls_p
