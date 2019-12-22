import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d
from horch.models.drop import DropConnect
from horch.ops import inverse_sigmoid

class DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_connect):
        super().__init__()
        self.conv1 = Conv2d(in_channels * 2, out_channels, kernel_size=3,
            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
            norm_layer='default', activation='default')
        self.drop_connect = DropConnect(drop_connect)

    def forward(self, x, p):
        x = self.drop_connect(x)
        x = torch.cat((x, p), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='deconv'):
        super().__init__()
        assert mode in ['deconv', 'interp']
        self.mode = mode
        if mode == 'deconv':
            self.conv = Conv2d(in_channels, out_channels, 2, 2, transposed=True,
                               norm_layer='default', activation='relu')
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                Conv2d(in_channels, out_channels, 1,
                       norm_layer='default', activation='relu'),
            )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels, up_mode='deconv', drop_connect=0):
        super().__init__()
        drop_rates = np.linspace(0, drop_connect, 5)
        self.down_conv1 = DownBlock(in_channels, channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down_conv2 = DownBlock(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.down_conv3 = DownBlock(channels * 2, channels * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.down_conv4 = DownBlock(channels * 4, channels * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.down_conv5 = DownBlock(channels * 8, channels * 16)

        self.up4 = Upsample(channels * 16, channels * 8, mode=up_mode)
        self.up_conv4 = UpBlock(channels * 8, channels * 8, drop_connect=drop_rates[3])

        self.up3 = Upsample(channels * 8, channels * 4, mode=up_mode)
        self.up_conv3 = UpBlock(channels * 4, channels * 4, drop_connect=drop_rates[2])

        self.up2 = Upsample(channels * 4, channels * 2, mode=up_mode)
        self.up_conv2 = UpBlock(channels * 2, channels * 2, drop_connect=drop_rates[1])

        self.up1 = Upsample(channels * 2, channels, mode=up_mode)
        self.up_conv1 = UpBlock(channels, channels, drop_connect=drop_rates[0])

        self.pred = Conv2d(channels, out_channels, 1)

    def forward(self, x):
        c0 = self.down_conv1(x)
        c1 = self.pool1(c0)

        c1 = self.down_conv2(c1)
        c2 = self.pool2(c1)

        c2 = self.down_conv3(c2)
        c3 = self.pool3(c2)

        c3 = self.down_conv4(c3)
        c4 = self.pool4(c3)

        c4 = self.down_conv5(c4)

        d3 = self.up_conv4(self.up4(c4), c3)
        d2 = self.up_conv3(self.up3(d3), c2)
        d1 = self.up_conv2(self.up2(d2), c1)
        d0 = self.up_conv1(self.up1(d1), c0)

        p = self.pred(d0)
        return p