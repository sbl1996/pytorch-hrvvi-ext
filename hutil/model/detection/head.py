from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import get_groups, Conv2d

def to_pred(p, c: int):
    b = p.size(0)
    p = p.permute(0, 3, 2, 1).contiguous().view(b, -1, c)
    return p


class ThunderRCNNHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """
    def __init__(self, num_classes, in_channels=245, f_channels=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, f_channels),
            nn.GroupNorm(get_groups(f_channels), f_channels),
            nn.ReLU(inplace=True)
        )
        self.loc_fc = nn.Linear(f_channels, 4)
        self.cls_fc = nn.Linear(f_channels, num_classes)

    def forward(self, p):
        if p.ndimension() == 3:
            n1, n2, c = p.size()
            p = p.view(n1 * n2, c)
            p = self.fc(p.view(n1 * n2, c)).view(n1, n2, -1)
        else:
            p = self.fc(p)
        loc_p = self.loc_fc(p)
        cls_p = self.cls_fc(p)
        return loc_p, cls_p


class ThunderRPNHead(nn.Module):
    r"""
    Light head for RPN, not for R-CNN.
    """
    def __init__(self, num_anchors, in_channels=245, f_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels,
                   norm_layer='gn'),
            Conv2d(in_channels, f_channels, kernel_size=1,
                   norm_layer='gn', activation='relu')
        )
        self.loc_fc = Conv2d(
            f_channels, num_anchors * 4, kernel_size=1)
        self.cls_fc = Conv2d(
            f_channels, num_anchors * 2, kernel_size=1)


        for m in [self.loc_fc, self.cls_fc]:
            m.apply(self._init_new_layers)
        self.cls_fc.apply(self._init_final_cls_layer)

    def _init_new_layers(self, m):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, p):
        p = self.conv(p)
        loc_p = to_pred(self.loc_fc(p), 4)
        cls_p = to_pred(self.cls_fc(p), 2)
        return loc_p, cls_p