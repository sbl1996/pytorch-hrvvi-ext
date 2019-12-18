import torch
from horch.models.modules import Conv2d, HardSigmoid, Identity
from torch import nn as nn
from torch.nn import functional as F


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.pool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


class CBAMChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        aa = F.adaptive_avg_pool2d(x, 1).view(b, c)
        aa = self.mlp(aa)
        am = F.adaptive_max_pool2d(x, 1).view(b, c)
        am = self.mlp(am)
        a = torch.sigmoid(aa + am).view(b, c, 1, 1)
        return x * a


class CBAMSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(2, 1, kernel_size=7, norm_layer='bn')

    def forward(self, x):
        aa = x.mean(dim=1, keepdim=True)
        am = x.max(dim=1, keepdim=True)[0]
        a = torch.cat([aa, am], dim=1)
        a = torch.sigmoid(self.conv(a))
        return x * a


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.channel = CBAMChannelAttention(in_channels, reduction)
        self.spatial = CBAMSpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class SELayerM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        channels = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU6(True),
            nn.Linear(channels, in_channels),
            HardSigmoid(True),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.avgpool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


def get_attention(name, **kwargs):
    if not name:
        return Identity()
    name = name.lower()
    if name == 'se':
        return SEModule(**kwargs)
    elif name == 'sem':
        return SELayerM(**kwargs)
    elif name == 'cbam':
        return CBAM(**kwargs)
    else:
        raise NotImplementedError("No attention module named %s" % name)