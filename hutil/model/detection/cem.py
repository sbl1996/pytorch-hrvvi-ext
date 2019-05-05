import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import Conv2d, upsample_add


class ContextEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer='bn'):
        super().__init__()
        self.lats = nn.ModuleList([
            Conv2d(c, out_channels, kernel_size=1, norm_layer=norm_layer)
            for c in in_channels
        ])
        self.lat_glb = Conv2d(in_channels[-1], out_channels, kernel_size=1,
                              norm_layer=norm_layer)

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        p = self.lats[0](cs[0])
        for c, lat in zip(cs[1:], self.lats[1:]):
            p += F.interpolate(lat(c), size=size, mode='bilinear', align_corners=False)
        c_glb = F.adaptive_avg_pool2d(cs[-1], 1)
        p_glb = self.lat_glb(c_glb)
        p += p_glb
        return p