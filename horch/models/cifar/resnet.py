import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import get_activation, Conv2d, Identity
from horch.models.attention import SEModule


class PreActDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nl1 = get_activation("default")
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nl2 = get_activation("default")
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        self.se = SEModule(out_channels, reduction=8) if se else Identity()

        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.bn1(x)
        x = self.nl1(x)
        identity = x
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.nl2(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + self.shortcut(identity)


class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nl1 = get_activation("default")
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nl2 = get_activation("default")
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        self.se = SEModule(out_channels, reduction=8) if se else Identity()

    def forward(self, x):
        identity = x
        x = self.bn1(x)
        x = self.nl1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.nl2(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + identity


class PreActResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, layers, k=4, num_classes=10, **kwargs):
        super().__init__()
        self.block_kwargs = kwargs
        self.conv = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, layers[2], stride=2, **kwargs)

        self.bn = nn.BatchNorm2d(self.stages[3] * k)
        self.nl = get_activation('default')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, **kwargs):
        layers = [PreActDownBlock(in_channels, out_channels, stride=stride, **kwargs)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.nl(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


