import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d


class DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels * 2, out_channels, kernel_size=3,
            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')

    def forward(self, x, p):
        x = torch.cat((x, p), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.down1 = DownBlock(in_channels, channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = DownBlock(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.down3 = DownBlock(channels * 2, channels * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.down4 = DownBlock(channels * 4, channels * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.down5 = DownBlock(channels * 8, channels * 16)

        self.deconv4 = Conv2d(channels * 16, channels * 8, 2, 2, transposed=True)
        self.up4 = UpBlock(channels * 8, channels * 8)

        self.deconv3 = Conv2d(channels * 8, channels * 4, 2, 2, transposed=True)
        self.up3 = UpBlock(channels * 4, channels * 4)

        self.deconv2 = Conv2d(channels * 4, channels * 2, 2, 2, transposed=True)
        self.up2 = UpBlock(channels * 2, channels * 2)

        self.deconv1 = Conv2d(channels * 2, channels, 2, 2, transposed=True)
        self.up1 = UpBlock(channels, channels)

        self.pred = Conv2d(channels, out_channels, 1)

    def forward(self, x):
        c0 = self.down1(x)
        c1 = self.pool1(c0)

        c1 = self.down2(c1)
        c2 = self.pool2(c1)

        c2 = self.down3(c2)
        c3 = self.pool3(c2)

        c3 = self.down4(c3)
        c4 = self.pool4(c3)

        c4 = self.down5(c4)

        d3 = self.up4(self.deconv4(c4), c3)
        d2 = self.up3(self.deconv3(d3), c2)
        d1 = self.up2(self.deconv2(d2), c1)
        d0 = self.up1(self.deconv1(d1), c0)

        p = self.pred(d0)
        return p
