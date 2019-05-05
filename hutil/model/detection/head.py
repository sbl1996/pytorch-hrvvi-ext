from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import Conv2d


def to_pred(p, c: int):
    b = p.size(0)
    p = p.permute(0, 3, 2, 1).contiguous().view(b, -1, c)
    return p


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


class ThunderRPNHead(nn.Module):
    r"""
    Light head for RPN, not for R-CNN.
    """
    def __init__(self, num_anchors, in_channels=245, f_channels=256, return_features=True, norm_layer='bn', with_se=False):
        super().__init__()
        self.return_features = return_features
        self.conv = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels,
                   norm_layer=norm_layer),
            Conv2d(in_channels, f_channels, kernel_size=1,
                   norm_layer=norm_layer, activation='relu', with_se=with_se)
        )
        self.loc_fc = Conv2d(
            f_channels, num_anchors * 4, kernel_size=1)
        self.cls_fc = Conv2d(
            f_channels, num_anchors * 2, kernel_size=1)

        self.cls_fc.apply(self._init_final_cls_layer)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, p):
        p = self.conv(p)
        loc_p = to_pred(self.loc_fc(p), 4)
        cls_p = to_pred(self.cls_fc(p), 2)
        if self.return_features:
            return loc_p, cls_p, p
        else:
            return loc_p, cls_p


def _make_head(f_channels, num_layers, out_channels, **kwargs):
    layers = []
    for i in range(num_layers):
        layers.append(Conv2d(f_channels, f_channels, kernel_size=3,
                             activation='relu', **kwargs))
    layers.append(Conv2d(f_channels, out_channels, kernel_size=3))
    return nn.Sequential(*layers)


class BoxHead(nn.Module):
    def __init__(self, f_channels, num_anchors, num_classes, num_layers=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.loc_head = _make_head(
            f_channels, num_layers, num_anchors * 4, **kwargs)
        self.cls_head = _make_head(
            f_channels, num_layers, num_anchors * num_classes, **kwargs)

        for m in [self.loc_head, self.cls_head]:
            m.apply(self._init_new_layers)
        self.cls_head[-1].apply(self._init_final_cls_layer)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            loc_p = to_pred(self.loc_head(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p)
        loc_p = torch.cat(loc_preds, dim=1)
        cls_p = torch.cat(cls_preds, dim=1)
        return loc_p, cls_p
