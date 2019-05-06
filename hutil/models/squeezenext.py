import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.models.modules import Conv2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(
                in_channels, out_channels // 2, kernel_size=1,
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                out_channels // 2, out_channels // 4, kernel_size=1,
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                out_channels // 4, out_channels // 2, kernel_size=(3, 1),
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                out_channels // 2, out_channels // 2, kernel_size=(1, 3),
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                out_channels // 2, out_channels, kernel_size=1,
                norm_layer=norm_layer, activation='relu'
            )
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels, out_channels, kernel_size=1,
                norm_layer=norm_layer,
            )

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, channels, norm_layer):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(
                channels, channels, kernel_size=1, stride=2,
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                channels, channels // 2, kernel_size=1,
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                channels // 2, channels, kernel_size=(3, 1),
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                channels, channels, kernel_size=(1, 3),
                norm_layer=norm_layer, activation='relu'
            ),
            Conv2d(
                channels, channels * 2, kernel_size=1,
                norm_layer=norm_layer, activation='relu'
            )
        )
        self.shortcut = Conv2d(
            channels, channels * 2, kernel_size=1, stride=2,
            norm_layer=norm_layer
        )

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x


class SqueezeNext(nn.Module):
    cfg = {
        0.5: [24, 48, 96, 192, 1024],
        1: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2: [24, 244, 488, 976, 2048],
    }

    def __init__(self, num_classes=1000, norm_layer='bn'):
        super().__init__()
        num_layers = [5, 5, 7]
        self.num_layers = num_layers
        channels = [64, 32, 64, 128, 256, 128]
        self.channels = channels
        self.norm_layer = norm_layer

        self.conv1 = Conv2d(
            3, channels[0], kernel_size=3, stride=2,
            norm_layer=norm_layer, activation='relu'
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )
        self.stage21 = BasicBlock(channels[0], channels[1], norm_layer)
        self.stage22 = self._make_layer(num_layers[0], channels[1])
        self.stage31 = DownBlock(channels[1], norm_layer)
        self.stage32 = self._make_layer(num_layers[1], channels[2])
        self.stage41 = DownBlock(channels[2], norm_layer)
        self.stage42 = self._make_layer(num_layers[2], channels[3])
        self.stage51 = DownBlock(channels[3], norm_layer)
        self.stage52 = Conv2d(
            channels[4], channels[5], kernel_size=1,
            norm_layer=norm_layer, activation='relu'
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[5], num_classes)

    def _make_layer(self, num_layers, channels):
        layers = []
        for i in range(num_layers):
            layers.append(BasicBlock(
                channels, channels, self.norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage21(x)
        x = self.stage22(x)
        x = self.stage31(x)
        x = self.stage32(x)
        x = self.stage41(x)
        x = self.stage42(x)
        x = self.stage51(x)
        x = self.stage52(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
