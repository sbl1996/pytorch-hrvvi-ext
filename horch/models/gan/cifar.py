import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.gan.common import ResBlock
from torch.nn.utils import spectral_norm


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, channels=64, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, 4 * 4 * channels * 4)
        self.conv = nn.Sequential(
            ResBlock(channels * 4, channels * 4, 'upscale'),
            ResBlock(channels * 4, channels * 4, 'upscale'),
            ResBlock(channels * 4, channels * 4, 'upscale'),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            nn.Conv2d(channels * 4, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        spectral_norm(self.dense)
        spectral_norm(self.conv[-2])

    def forward(self, x):
        x = self.dense(x).view(x.size(0), -1, 4, 4)
        x = self.conv(x)
        return x


class ResNetDiscriminator(nn.Module):

    def __init__(self, in_channels=3, channels=64, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            ResBlock(in_channels, channels * 2, 'downscale'),
            ResBlock(channels * 2, channels * 2, 'downscale'),
            ResBlock(channels * 2, channels * 2, None),
            ResBlock(channels * 2, channels * 2, None),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, out_channels, kernel_size=1),
        )

        spectral_norm(self.conv[-1])

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return x
