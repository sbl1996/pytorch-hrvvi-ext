import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer='default', activation='default')


class DecoderDeconvBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.deconv = Conv2d(in_channels2, out_channels, 4, 2, transposed=True,
                             norm_layer='default', activation='relu')
        self.conv = Conv2d(out_channels + in_channels1, out_channels, kernel_size=3,
                           norm_layer='default', activation='default')

    def forward(self, c, x):
        x = torch.cat([c, self.deconv(x)], dim=1)
        x = self.conv(x)
        return x


class DecoderUpsamplingBlock(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3,
                            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer='default', activation='default')

    def forward(self, c, x):
        x = torch.cat([c, self.upsample(x)], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, backbone, in_channels_list, channels, num_classes, up_mode='deconv'):
        super().__init__()

        if up_mode == 'deconv':
            decoder_block = DecoderDeconvBlock
        elif up_mode == 'upsample':
            decoder_block = DecoderUpsamplingBlock

        n_upsample_blocks = len(in_channels_list)

        channels = in_channels_list[-1]
        for i in range(n_upsample_blocks)
            decode_stage = decoder_block(in_channels_list[-(i+2), ])
        self.down_conv1 = ConvBlock(in_channels, channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down_conv2 = ConvBlock(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.down_conv3 = ConvBlock(channels * 2, channels * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.down_conv4 = ConvBlock(channels * 4, channels * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.down_conv5 = ConvBlock(channels * 8, channels * 16)

        self.up4 = Upsample(channels * 16, channels * 8, mode=up_mode)
        self.up_conv4 = ConvBlock(channels * 16, channels * 8)

        self.up3 = Upsample(channels * 8, channels * 4, mode=up_mode)
        self.up_conv3 = ConvBlock(channels * 8, channels * 4)

        self.up2 = Upsample(channels * 4, channels * 2, mode=up_mode)
        self.up_conv2 = ConvBlock(channels * 4, channels * 2)

        self.up1 = Upsample(channels * 2, channels, mode=up_mode)
        self.up_conv1 = ConvBlock(channels * 2, channels)

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

        c4 = torch.cat((self.up4(c4), c3), dim=1)
        c3 = self.up_conv4(c4)

        c3 = torch.cat((self.up3(c3), c2), dim=1)
        c2 = self.up_conv3(c3)

        c2 = torch.cat((self.up2(c2), c1), dim=1)
        c1 = self.up_conv2(c2)

        c1 = torch.cat((self.up1(c1), c0), dim=1)
        c0 = self.up_conv1(c1)

        p = self.pred(c0)
        return p
