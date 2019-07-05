import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.gan.common import ResBlock
from torch.nn.utils import spectral_norm


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, 6 * 6 * channels * 8)
        self.conv = nn.Sequential(
            ResBlock(channels * 8, channels * 4, 'upscale'),
            ResBlock(channels * 4, channels * 4, 'upscale'),
            ResBlock(channels * 4, channels * 2, 'upscale'),
            ResBlock(channels * 2, channels * 1, 'upscale'),
            nn.BatchNorm2d(channels * 1),
            nn.ReLU(True),
            nn.Conv2d(channels * 1, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        spectral_norm(self.dense)
        spectral_norm(self.conv[-2])

    def forward(self, x):
        x = self.dense(x).view(x.size(0), -1, 6, 6)
        x = self.conv(x)
        return x


class ResNetDiscriminator(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            ResBlock(in_channels, channels * 1, 'downscale'),
            ResBlock(channels * 1, channels * 2, 'downscale'),
            ResBlock(channels * 2, channels * 4, 'downscale'),
            ResBlock(channels * 4, channels * 8, 'downscale'),
            ResBlock(channels * 8, channels * 16, None),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 16, 1, kernel_size=1),
        )

        spectral_norm(self.conv[-1])

    def forward(self, x):
        x = self.conv(x).view(-1)
        return x
