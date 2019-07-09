import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from horch.models.modules import seq, SelfAttention2, ConditionalBatchNorm2d
from horch.models.gan.stl import ResNetDiscriminator


class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, scale=None, use_sn=False):
        assert scale in ['down', 'up', None]
        super().__init__()
        self.scale = scale
        shortcut = nn.Sequential()
        if scale == 'up':
            # shortcut.scale = MaxUnpool2d(True)
            shortcut.scale = nn.UpsamplingNearest2d(scale_factor=2)
        shortcut.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if scale == 'down':
            shortcut.scale = nn.AvgPool2d(2, 2)
        self.shortcut = shortcut

        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu1 = nn.ReLU(True)

        if scale == 'up':
            self.scale = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if scale == 'down':
            self.scale = nn.AvgPool2d(2, 2)

        if use_sn:
            spectral_norm(self.shortcut.conv)
            spectral_norm(self.conv1)
            spectral_norm(self.conv2)

    def forward(self, x, y):
        identity = x

        x = self.bn1(x)
        x = self.relu1(x)
        if self.scale == 'up':
            x = self.scale(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        if self.scale == 'down':
            x = self.scale(x)
        return x + self.shortcut(identity)


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, channels=64, out_channels=3, num_classes=10, use_sn=True, non_local=True):
        super().__init__()
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, 6 * 6 * channels * 8)
        self.conv = seq(
            ("block1", GenResBlock(channels * 8, channels * 4, num_classes, 'up', use_sn=use_sn)),
            ("block2", GenResBlock(channels * 4, channels * 2, num_classes, 'up', use_sn=use_sn)),
            ("attn", SelfAttention2(channels * 2) if non_local else None),
            ("block3", GenResBlock(channels * 2, channels * 1, num_classes, 'up', use_sn=use_sn)),
            ("bn", ConditionalBatchNorm2d(channels * 1, num_classes)),
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