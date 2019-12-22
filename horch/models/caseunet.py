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


class CASEUNet(nn.Module):
    def __init__(self, backbone, in_channels_list, num_classes, up_mode='deconv'):
        super().__init__()
        self.backbone = backbone
        if up_mode == 'deconv':
            decoder_block = DecoderDeconvBlock
        elif up_mode == 'upsample':
            decoder_block = DecoderUpsamplingBlock
        else:
            raise ValueError

        self.block2 = decoder_block(in_channels_list[2], in_channels_list[3], in_channels_list[2])
        self.block1 = decoder_block(in_channels_list[1], in_channels_list[2], in_channels_list[1])
        self.block0 = decoder_block(in_channels_list[0], in_channels_list[1], in_channels_list[0])

        self.side0 = Conv2d(in_channels_list[0], 1, 1,
                            norm_layer='default', activation='default')
        self.side1 = Conv2d(in_channels_list[1], 1, 1,
                            norm_layer='default', activation='default')
        self.side2 = Conv2d(in_channels_list[2], 1, 1,
                            norm_layer='default', activation='default')
        self.side3 = Conv2d(in_channels_list[3], num_classes, 1)

        self.conv = nn.Conv2d(4 * num_classes, num_classes, 1, groups=num_classes)

    def forward(self, x):
        size = x.shape[2:4]
        c0, c1, c2, _, c3 = self.backbone(x)
        c2 = self.block2(c2, c3)
        c1 = self.block1(c1, c2)
        c0 = self.block0(c0, c1)

        s0 = self.side0(c0)
        s0 = F.interpolate(s0, size, mode='bilinear', align_corners=False)
        s1 = self.side1(c1)
        s1 = F.interpolate(s1, size, mode='bilinear', align_corners=False)
        s2 = self.side2(c2)
        s2 = F.interpolate(s2, size, mode='bilinear', align_corners=False)
        s3 = self.side3(c3)
        s3 = F.interpolate(s3, size, mode='bilinear', align_corners=False)

        xs = []
        for i in range(s3.size(1)):
            xs.append(s3[:, [i], :, :])
            xs.extend([s0, s1, s2])
        x = torch.cat(xs, dim=1)
        x = self.conv(x)
        return x
