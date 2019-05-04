import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import upsample_add, Conv2d
from hutil.model.detection.head import BoxHead


class TopDown(nn.Module):
    def __init__(self, in_channels, f_channels, norm_layer='gn'):
        super().__init__()
        self.lat = Conv2d(
            in_channels, f_channels, kernel_size=1,
            norm_layer=norm_layer)
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=3,
            norm_layer=norm_layer)

    def forward(self, c, p):
        p = upsample_add(p, self.lat(c))
        p = self.conv(p)
        return p


class RetinaNet(nn.Module):

    def __init__(self, backbone, num_classes=10, num_anchors=3, f_channels=256, norm_layer='gn', num_head_layers=4):
        super().__init__()
        assert tuple(backbone.feature_levels) == (3, 4, 5), "Feature levels of backbone must be (3,4,5)"
        self.num_classes = num_classes
        self.backbone = backbone
        backbone_channels = backbone.out_channels

        self.layer6 = Conv2d(
            backbone_channels[-1], f_channels, kernel_size=3, stride=2,
            norm_layer=norm_layer)
        self.layer7 = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2d(f_channels, f_channels, kernel_size=3, stride=2,
                   norm_layer=norm_layer)
        )

        self.lat5 = Conv2d(
            backbone_channels[-1], f_channels, kernel_size=1, norm_layer=norm_layer)

        self.topdown54 = TopDown(
            backbone_channels[-2], f_channels, norm_layer=norm_layer)
        self.topdown43 = TopDown(
            backbone_channels[-3], f_channels, norm_layer=norm_layer)

        self.head = BoxHead(
            f_channels, num_anchors, num_classes, norm_layer=norm_layer, num_layers=num_head_layers)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)

        p6 = self.layer6(c5)
        p7 = self.layer7(p6)

        p5 = self.lat5(c5)
        p4 = self.topdown54(c4, p5)
        p3 = self.topdown43(c3, p4)
        ps = [p3, p4, p5, p6, p7]

        loc_p, cls_p = self.head(ps)
        return loc_p, cls_p
