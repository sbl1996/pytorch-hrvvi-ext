import torch
import torch.nn as nn
import torch.nn.functional as F

from hutil.model.modules import Conv2d


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True, norm_layer='bn'):
        super().__init__()
        self.residual = residual
        self.conv1 = Conv2d(in_channels, out_channels // 2, kernel_size=1,
                            norm_layer=norm_layer, activation='leaky_relu')
        self.conv2 = Conv2d(out_channels // 2, out_channels, kernel_size=3,
                            norm_layer = norm_layer, activation='leaky_relu')

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity if self.residual else x


def _make_layer(num_layers, in_channels, out_channels, **kwargs):
    layers = [Bottleneck(in_channels, out_channels, **kwargs)]
    for _ in range(num_layers - 1):
        layers.append(Bottleneck(out_channels, out_channels, **kwargs))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_layer='bn'):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            in_channels = in_channels // 2
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                           norm_layer=norm_layer, activation='leaky_relu')

    def forward(self, x):
        return self.conv(x)


class Darknet(nn.Module):
    def __init__(self, num_classes=1000, num_layers=(1, 2, 8, 8, 4), f_channels=128, norm_layer='bn'):
        super().__init__()
        self.f_channels = f_channels
        self.conv0 = Conv2d(3, 32, kernel_size=3,
                            norm_layer=norm_layer, activation='leaky_relu')
        self.down1 = DownBlock(64, norm_layer=norm_layer)
        self.layer1 = _make_layer(num_layers[0], 64, 64)

        self.down2 = DownBlock(64, f_channels * 1, norm_layer=norm_layer)
        self.layer2 = _make_layer(num_layers[1], f_channels * 1, f_channels * 1, norm_layer=norm_layer)

        self.down3 = DownBlock(f_channels * 2, norm_layer=norm_layer)
        self.layer3 = _make_layer(num_layers[2], f_channels * 2, f_channels * 2, norm_layer=norm_layer)

        self.down4 = DownBlock(f_channels * 4, norm_layer=norm_layer)
        self.layer4 = _make_layer(num_layers[3], f_channels * 4, f_channels * 4, norm_layer=norm_layer)

        self.down5 = DownBlock(f_channels * 8, norm_layer=norm_layer)
        self.layer5 = _make_layer(num_layers[4], f_channels * 8, f_channels * 8, norm_layer=norm_layer)

        self.fc = nn.Linear(f_channels * 8, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.conv0(x)
        x = self.down1(x)
        x = self.layer1(x)
        x = self.down2(x)
        x = self.layer2(x)
        x = self.down3(x)
        x = self.layer3(x)
        x = self.down4(x)
        x = self.layer4(x)
        x = self.down5(x)
        x = self.layer5(x)
        x = F.adaptive_avg_pool2d(x, 1).view(b, -1)
        x = self.fc(x)
        return x

