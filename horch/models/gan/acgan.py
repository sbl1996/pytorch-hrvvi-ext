import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


class Generator(nn.Module):

    def __init__(self, in_channels, channels=48, out_channels=3, size=(32, 32)):
        super().__init__()
        self.h, self.w = size
        self.in_channels = in_channels
        self.dense = nn.Sequential(
            nn.Linear(in_channels, (self.h // 8) * (self.w // 8) * channels * 8),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.dense(x).view(x.size(0), -1, self.h // 8, self.w // 8)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, channels=64, out_channels=1, size=(32, 32), dropout=0.5, use_sn=False):
        super().__init__()
        self.h, self.w = size
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels // 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
        )
        self.dense = nn.Linear((self.h // 8) * (self.w // 8) * channels * 8, out_channels)

        if use_sn:
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
            spectral_norm(self.dense)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        p = self.dense(x).view(x.size(0), -1)
        return p
