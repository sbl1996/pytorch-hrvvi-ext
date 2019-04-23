import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import Conv2d


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
            normalization='bn', activation='relu',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=3, groups=channels,
            normalization='bn',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            normalization='bn', activation='relu',
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.shuffle(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2):
        super().__init__()
        channels = out_channels // 2
        self.conv11 = Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels,
            normalization='bn',
        )
        self.conv12 = Conv2d(
            in_channels, channels, kernel_size=1,
            normalization='bn', activation='relu',
        )
        self.conv21 = Conv2d(
            in_channels, channels, kernel_size=1,
            normalization='bn', activation='relu',
        )
        self.conv22 = Conv2d(
            channels, channels, kernel_size=3, stride=2, groups=channels,
            normalization='bn',
        )
        self.conv23 = Conv2d(
            channels, channels, kernel_size=1,
            normalization='bn', activation='relu',
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.shuffle(x)
        return x


class ShuffleNetV2(nn.Module):
    cfg = {
        0.5: [24, 48, 96, 192, 1024],
        1: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2: [24, 244, 488, 976, 2048],
    }

    def __init__(self, num_classes=1000, mult=0.5):
        super().__init__()
        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[mult]
        self.channels = channels

        self.conv1 = Conv2d(
            3, channels[0], kernel_size=3, stride=2,
            normalization='bn', activation='relu'
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )
        self.stage2 = self._make_layer(num_layers[0], channels[0], channels[1])
        self.stage3 = self._make_layer(num_layers[1], channels[1], channels[2])
        self.stage4 = self._make_layer(num_layers[2], channels[2], channels[3])
        self.conv5 = Conv2d(channels[3], channels[4], kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[4], num_classes)

    def _make_layer(self, num_layers, in_channels, out_channels):
        layers = []
        layers.append(DownBlock(in_channels, out_channels))
        for i in range(num_layers - 1):
            layers.append(BasicBlock(out_channels))
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
