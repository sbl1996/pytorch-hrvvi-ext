import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.nn import MaxUnpool2d

from torch.nn.utils import spectral_norm


class Generator(nn.Module):

    def __init__(self, in_channels, channels, out_channels, size=(32, 32)):
        super().__init__()
        self.h, self.w = size
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, (self.h // 8) * (self.w // 8) * channels * 8)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, channels * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.dense(x).view(x.size(0), -1, self.h // 8, self.w // 8)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, channels=64, out_channels=1, size=(32, 32), leaky_slope=0.1, use_sn=True):
        super().__init__()
        self.h, self.w = size
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
        )
        self.dense = nn.Linear((self.h // 8) * (self.w // 8) * channels * 8, out_channels)

        if use_sn:
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
            spectral_norm(self.dense)

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=None, use_sn=True):
        assert scale in ['down', 'up', None]
        super().__init__()
        shortcut = nn.Sequential()
        if scale == 'up':
            shortcut.scale = MaxUnpool2d(True)
        shortcut.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if scale == 'down':
            shortcut.scale = nn.AvgPool2d(2, 2)
        self.shortcut = shortcut

        conv = nn.Sequential()
        conv.bn1 = nn.BatchNorm2d(in_channels)
        conv.relu1 = nn.ReLU(True)

        if scale == 'up':
            conv.scale = MaxUnpool2d(True)
        conv.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        conv.bn2 = nn.BatchNorm2d(out_channels)
        conv.relu2 = nn.ReLU(inplace=True)
        conv.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if scale == 'down':
            conv.scale = nn.AvgPool2d(2, 2)
        self.conv = conv

        if use_sn:
            spectral_norm(self.shortcut.conv)
            spectral_norm(self.conv.conv1)
            spectral_norm(self.conv.conv2)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
