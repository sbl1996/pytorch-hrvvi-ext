import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.gan.common import ResBlock
from horch.models.modules import SelfAttention, seq, SelfAttention2
from torch.nn.utils import spectral_norm


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, channels=64, out_channels=3, use_sn=True, non_local=True):
        super().__init__()
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, 6 * 6 * channels * 8)
        self.conv = seq(
            ("block1", ResBlock(channels * 8, channels * 4, 'up', use_sn=use_sn)),
            ("block2", ResBlock(channels * 4, channels * 2, 'up', use_sn=use_sn)),
            ("attn", SelfAttention2(channels * 2) if non_local else None),
            ("block3", ResBlock(channels * 2, channels * 1, 'up', use_sn=use_sn)),
            ("bn", nn.BatchNorm2d(channels * 1)),
            ("relu", nn.ReLU(True)),
            ("conv", nn.Conv2d(channels * 1, out_channels, kernel_size=3, padding=1)),
            ("tanh", nn.Tanh()),
        )

        if use_sn:
            spectral_norm(self.dense)
            spectral_norm(self.conv.conv)

    def forward(self, x):
        x = self.dense(x).view(x.size(0), -1, 6, 6)
        x = self.conv(x)
        return x


class ResNetDiscriminator(nn.Module):

    def __init__(self, in_channels=3, channels=64, out_channels=1, use_sn=True, use_bn=False):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            ResBlock(in_channels, channels * 1, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 1, channels * 2, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 2, channels * 4, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 4, channels * 8, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 8, channels * 16, None, use_sn=use_sn, use_bn=use_bn),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 16, out_channels, kernel_size=1),
        )

        if use_sn:
            spectral_norm(self.conv[-1])

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return x
