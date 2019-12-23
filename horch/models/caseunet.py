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
    def __init__(self, backbone, in_channels_list, num_classes, f_channels_list=None, up_mode='deconv', dropout=0.2):
        super().__init__()
        self.backbone = backbone
        if up_mode == 'deconv':
            decoder_block = DecoderDeconvBlock
        elif up_mode == 'upsample':
            decoder_block = DecoderUpsamplingBlock
        else:
            raise ValueError

        if f_channels_list is None:
            f_channels_list = in_channels_list[:3]

        self.block2 = decoder_block(in_channels_list[2], in_channels_list[3], f_channels_list[2])
        self.block1 = decoder_block(in_channels_list[1], in_channels_list[2], f_channels_list[1])
        self.block0 = decoder_block(in_channels_list[0], in_channels_list[1], f_channels_list[0])

        self.side0 = Conv2d(f_channels_list[0], 1, 1,
                            norm_layer='default', activation='default')
        self.side1 = Conv2d(f_channels_list[1], 1, 1,
                            norm_layer='default', activation='default')
        self.side2 = Conv2d(f_channels_list[2], 1, 1,
                            norm_layer='default', activation='default')
        self.side3 = Conv2d(in_channels_list[3], num_classes, 1)

        self.dropout = nn.Dropout2d(dropout)
        self.pred = nn.Conv2d(4 * num_classes, num_classes, 1, groups=num_classes)

    def get_parameters(self):
        layers = [
            self.block0, self.block1, self.block2,
            self.side0, self.side1, self.side2, self.side3,
            self.pred
        ]
        params = [
            p
            for l in layers
            for p in l.parameters()
        ]
        return params

    def forward(self, x):
        size = x.shape[2:4]
        c0, c1, c2, _, c3 = self.backbone(x)

        c0 = self.dropout(c0)
        c1 = self.dropout(c1)
        c2 = self.dropout(c2)
        c3 = self.dropout(c3)

        c2 = self.block2(c2, c3)
        c1 = self.block1(c1, c2)
        c0 = self.block0(c0, c1)

        c2 = self.dropout(c2)
        c1 = self.dropout(c1)
        c0 = self.dropout(c0)

        c0 = self.side0(c0)
        c0 = F.interpolate(c0, size, mode='bilinear', align_corners=False)
        c1 = self.side1(c1)
        c1 = F.interpolate(c1, size, mode='bilinear', align_corners=False)
        c2 = self.side2(c2)
        c2 = F.interpolate(c2, size, mode='bilinear', align_corners=False)
        c3 = self.side3(c3)
        c3 = F.interpolate(c3, size, mode='bilinear', align_corners=False)

        xs = []
        for i in range(c3.size(1)):
            xs.append(c3[:, [i], :, :])
            xs.extend([c0, c1, c2])
        x = torch.cat(xs, dim=1)

        x = self.dropout(x)
        x = self.pred(x)
        return x
