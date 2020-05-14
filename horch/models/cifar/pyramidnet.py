import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d, get_activation, get_norm_layer


class PadChannel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels >= 0
        self.c = channels

    def forward(self, x):
        return F.pad(x, [0, 0, 0, 0, 0, self.c])


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = []
        if stride == 2:
            layers.append(nn.AvgPool2d(2, 2))
        if in_channels != out_channels:
            layers.append(PadChannel(out_channels - in_channels))
        self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        return self.shortcut(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            get_norm_layer(in_channels),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False),
            get_norm_layer(out_channels),
            get_activation(),
            Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            get_norm_layer(out_channels),
        )
        self.shortcut = Shortcut(in_channels, out_channels, stride)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv = nn.Sequential(
            get_norm_layer(in_channels),
            Conv2d(in_channels, channels, kernel_size=1, bias=False),
            get_norm_layer(channels),
            get_activation(),
            Conv2d(channels, channels, kernel_size=3, stride=stride, bias=False),
            get_norm_layer(channels),
            get_activation(),
            Conv2d(channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(out_channels),
        )
        self.shortcut = Shortcut(in_channels, out_channels, stride)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def rd(c):
    return int(round(c, 2))


class PyramidNet(nn.Module):
    def __init__(self,
                 start_channels,
                 num_classes,
                 block,
                 widening_fractor,
                 depth):
        super().__init__()

        if block == 'basic':
            block = BasicBlock
            num_layers = [(depth - 2) // 6] * 3
        elif block == 'bottleneck':
            block = Bottleneck
            num_layers = [(depth - 2) // 9] * 3
        else:
            raise ValueError("invalid block type: %s" % block)

        strides = [1, 2, 2]

        self.add_channel = widening_fractor / sum(num_layers)
        self.in_channels = start_channels
        self.channels = start_channels

        layers = [Conv2d(3, start_channels, kernel_size=3, norm_layer='default')]

        for n, s in zip(num_layers, strides):
            layers.append(self._make_layer(block, n, stride=s))

        self.features = nn.Sequential(*layers)
        assert (start_channels + widening_fractor) * block.expansion == self.in_channels
        self.post_activ = nn.Sequential(
            get_norm_layer(self.in_channels),
            get_activation('default'),
        )
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, num_layers, stride):
        self.channels = self.channels + self.add_channel
        layers = [block(self.in_channels, rd(self.channels), stride=stride)]
        self.in_channels = rd(self.channels) * block.expansion
        for i in range(1, num_layers):
            self.channels = self.channels + self.add_channel
            layers.append(block(self.in_channels, rd(self.channels)))
            self.in_channels = rd(self.channels) * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.post_activ(x)
        x = self.final_pool(x).view(x.size(0), -1)
        x = self.output(x)
        return x


def pyramidnet110_a48(num_classes):
    return PyramidNet(16, num_classes, 'basic', 48, 110)


def pyramidnet110_a84(num_classes):
    return PyramidNet(16, num_classes, 'basic', 84, 110)


def pyramidnet110_a270(num_classes):
    return PyramidNet(16, num_classes, 'basic', 270, 110)


def pyramidnet164_a270(num_classes):
    return PyramidNet(16, num_classes, 'bottleneck', 270, 164)


def pyramidnet200_a240(num_classes):
    return PyramidNet(16, num_classes, 'bottleneck', 240, 200)


def pyramidnet236_a220(num_classes):
    return PyramidNet(16, num_classes, 'bottleneck', 220, 236)


def pyramidnet272_a200(num_classes):
    return PyramidNet(16, num_classes, 'bottleneck', 200, 272)
