import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.models.modules import Conv2d
from hutil.models.detection.head import ConvHead
from hutil.models.detection.fpn import FPN


class RetinaNet(nn.Module):

    def __init__(self, backbone, num_classes=10, num_anchors=3, f_channels=256, extra_layers=(6,7), norm_layer='gn', num_head_layers=4):
        super().__init__()
        assert tuple(backbone.feature_levels) == (3, 4, 5), "Feature levels of backbone must be (3,4,5)"
        self.num_classes = num_classes
        self.backbone = backbone
        backbone_channels = backbone.out_channels
        self.extra_layers = extra_layers
        if 6 in extra_layers:
            self.layer6 = Conv2d(
                backbone_channels[-1], f_channels, kernel_size=3, stride=2,
                norm_layer=norm_layer)
        if 7 in extra_layers:
            assert 6 in extra_layers
            self.layer7 = nn.Sequential(
                nn.ReLU(inplace=True),
                Conv2d(f_channels, f_channels, kernel_size=3, stride=2,
                       norm_layer=norm_layer)
            )

        self.fpn = FPN(backbone, f_channels, norm_layer=norm_layer)

        self.head = ConvHead(
            f_channels, num_anchors, num_classes, norm_layer=norm_layer, num_layers=num_head_layers)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)

        p3, p4, p5 = self.fpn(c3, c4, c5)
        ps = [p3, p4, 5]
        if 6 in self.extra_layers:
            p6 = self.layer6(c5)
            ps.append(p6)
        if 7 in self.extra_layers:
            p7 = self.layer7(p6)
            ps.append(p7)

        loc_p, cls_p = self.head(ps)
        return loc_p, cls_p
