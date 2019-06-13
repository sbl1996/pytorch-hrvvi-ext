import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import upsample_add, get_activation, Conv2d, get_norm_layer


def global_pool(x1, x2):
    ctx = F.adaptive_avg_pool2d(x1, 1).sigmoid()
    x = upsample_add(x1, x2 * ctx)
    return x


class GlobalPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return global_pool(x1, x2)


class ReLUConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, lite=False):
        super().__init__()
        self.conv = nn.Sequential(
            get_activation("default"),
            *Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    norm_layer='default', depthwise_separable=lite),
        )

    def forward(self, x):
        return self.conv(x)


class NASFPN(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.gp64 = GlobalPool()
        self.gp64_rcb = ReLUConvBN(f_channels, f_channels)
        self.gp64_rcb2 = ReLUConvBN(f_channels, f_channels)
        self.rcb3 = ReLUConvBN(f_channels, f_channels)
        self.rcb4 = ReLUConvBN(f_channels, f_channels)
        self.gp43 = GlobalPool()
        self.rcb5 = ReLUConvBN(f_channels, f_channels)
        self.gp54 = GlobalPool()
        self.rcb7 = ReLUConvBN(f_channels, f_channels)
        self.rcb6 = ReLUConvBN(f_channels, f_channels)
        self.gp56 = GlobalPool()

    def forward(self, c3, c4, c5, c6, c7):
        p4 = self.gp64(c6, c4)
        p4 = self.gp64_rcb(p4) + c4
        p4 = self.gp64_rcb2(p4)
        p3o = self.rcb3(upsample_add(p4, c3))
        p4o = self.rcb4(upsample_add(p3o, p4))
        p5o = self.rcb5(upsample_add(self.gp43(p4o, p3o), c5))
        p7o = self.rcb7(upsample_add(self.gp54(p5o, p4), c7))
        h, w = c6.size()[2:4]
        p6 = F.interpolate(p7o, (h, w), mode='bilinear', align_corners=False)
        p6o = self.rcb6(self.gp56(p5o, p6))
        return p3o, p4o, p5o, p6o, p7o
