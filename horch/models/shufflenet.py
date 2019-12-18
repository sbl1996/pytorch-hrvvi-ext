import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d, get_activation
from horch.models.attention import SEModule


def channel_shuffle(x, g):
    n, c, h, w = x.size()
    x = x.view(n, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, g=self.groups)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2):
        super().__init__()
        channels = in_channels // 2
        self.conv1 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=3, groups=channels,
            norm_layer='default',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2):
        super().__init__()
        channels = out_channels - in_channels // 2
        self.conv1 = Conv2d(
            in_channels // 2, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=3, groups=channels,
            norm_layer='default',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default',
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels // 2, channels, kernel_size=1,
                                   norm_layer='default')
        self.relu = get_activation('default')
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        identity = x2
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = x2 + self.shortcut(identity)
        x2 = self.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2):
        super().__init__()
        channels = in_channels // 2
        self.conv1 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=3, groups=channels,
            norm_layer='default',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.se = SEModule(channels, reduction=2)
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.se(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2):
        super().__init__()
        channels = out_channels - in_channels // 2
        self.conv1 = Conv2d(
            in_channels // 2, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=3, groups=channels,
            norm_layer='default',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default',
        )
        self.shortcut = nn.Sequential()
        self.se = SEModule(channels, reduction=2)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels // 2, channels, kernel_size=1,
                                   norm_layer='default')
        self.relu = get_activation('default')
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        identity = x2
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.se(x2)
        x2 = x2 + self.shortcut(identity)
        x2 = self.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2):
        super().__init__()
        channels = out_channels // 2
        self.conv11 = Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels,
            norm_layer='default',
        )
        self.conv12 = Conv2d(
            in_channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv21 = Conv2d(
            in_channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv22 = Conv2d(
            channels, channels, kernel_size=3, stride=2, groups=channels,
            norm_layer='default',
        )
        self.conv23 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class ShuffleNetV2_L(nn.Module):
    cfg = {
        50: [
            [64, 244, 488, 976, 1952, 2048],
            [3, 4, 6, 3],
        ],
        164: [
            [64, 340, 680, 1360, 2720, 2048],
            [10, 10, 23, 10],
        ]

    }

    def __init__(self, version, num_classes=1000, with_se=False):
        super().__init__()
        channels = self.cfg[version][0]
        self.channels = channels
        num_layers = self.cfg[version][1]
        self.num_layers = num_layers
        block = SEResBlock if with_se else ResBlock

        self.conv1 = Conv2d(
            3, channels[0], kernel_size=3, stride=2,
            norm_layer='default', activation='default'
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )
        self.stage2 = self._make_layer(block, num_layers[0], channels[0], channels[1], stride=1)
        self.stage3 = self._make_layer(block, num_layers[1], channels[1], channels[2])
        self.stage4 = self._make_layer(block, num_layers[2], channels[2], channels[3])
        self.stage5 = self._make_layer(block, num_layers[3], channels[3], channels[4])
        self.conv6 = Conv2d(channels[4], channels[5], kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[5], num_classes)

    def _make_layer(self, block, num_layers, in_channels, out_channels, stride=2):
        layers = []
        if stride == 2:
            layers.append(DownBlock(in_channels, out_channels))
        else:
            layers.append(block(in_channels, out_channels))
        for i in range(num_layers - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ShuffleNetV2(nn.Module):
    cfg = {
        0.5: [24, 48, 96, 192, 1024],
        1: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2: [24, 244, 488, 976, 2048],
    }

    def __init__(self, mult=0.5, num_classes=1000, with_se=False):
        super().__init__()
        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[mult]
        self.out_channels = channels
        block = SEBlock if with_se else BasicBlock

        self.conv1 = Conv2d(
            3, channels[0], kernel_size=3, stride=2,
            norm_layer='default', activation='default'
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )
        self.stage2 = self._make_layer(block, num_layers[0], channels[0], channels[1])
        self.stage3 = self._make_layer(block, num_layers[1], channels[1], channels[2])
        self.stage4 = self._make_layer(block, num_layers[2], channels[2], channels[3])
        self.conv5 = Conv2d(channels[3], channels[4], kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[4], num_classes)

    def _make_layer(self, block, num_layers, in_channels, out_channels):
        layers = [DownBlock(in_channels, out_channels)]
        for i in range(num_layers - 1):
            layers.append(block(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def shufflenet_v2(mult=0.5, **kwargs):
    return ShuffleNetV2(mult, **kwargs)


def shufflenet_v2_large(version=50, **kwargs):
    return ShuffleNetV2_L(version, **kwargs)
