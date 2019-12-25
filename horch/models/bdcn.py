import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d


class BDCN(nn.Module):
    def __init__(self, backbone, side_in_channels):
        super().__init__()
        self.backbone = backbone
        self.side_in_channels = side_in_channels
        self.preds = nn.ModuleList([
            Conv2d(c, 1, 1, norm_layer='default', activation='default')
            for i, c in enumerate(side_in_channels)
        ])

    def get_param_groups(self):
        group1 = self.backbone.parameters()
        layers = self.preds
        group2 = [
            p
            for l in layers
            for p in l.parameters()
        ]
        return [group1, group2]

    def forward(self, x):
        size = x.shape[2:4]
        cs = self.backbone(x)
        p = self.preds[0](cs[0])
        ps = [p]
        for c, pred in zip(cs[1:], self.preds[1:]):
            p = pred(c)
            p = F.interpolate(p, size, mode='bilinear', align_corners=False)
            p = p + ps[-1]
            ps.append(p)
        p = torch.cat(ps, dim=1)
        return p
