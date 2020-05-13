import torch.nn as nn

from horch.models.drop import DropPath
from horch.models.modules import get_activation, Conv2d, get_norm_layer
from horch.models.attention import SEModule


class PreActDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.bn1 = get_norm_layer(in_channels)
        self.nl1 = get_activation("default")
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn2 = get_norm_layer(out_channels)
        self.nl2 = get_activation("default")
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        if self.use_se:
            self.se = SEModule(out_channels, reduction=8)

        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.bn1(x)
        x = self.nl1(x)
        identity = x
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.nl2(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        return x + self.shortcut(identity)


class PreActResBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_se, drop_path):
        super().__init__()
        self.bn1 = get_norm_layer(in_channels)
        self.nl1 = get_activation("default")
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = get_norm_layer(out_channels)
        self.nl2 = get_activation("default")
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        if use_se:
            self.se = SEModule(out_channels, reduction=8)
        if drop_path:
            self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = super().forward(x)
        return x + identity


class PreActResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, num_classes=10, use_se=False, drop_path=0):
        super().__init__()
        num_blocks = (depth - 4) // 6
        self.conv = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, num_blocks, stride=1, use_se=use_se, drop_path=drop_path)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, num_blocks, stride=2, use_se=use_se, drop_path=drop_path)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, num_blocks, stride=2, use_se=use_se, drop_path=drop_path)

        self.bn = get_norm_layer(self.stages[3] * k)
        self.nl = get_activation('default')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, use_se, drop_path):
        layers = [PreActDownBlock(in_channels, out_channels, stride=stride, use_se=use_se)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels,
                               use_se=use_se, drop_path=drop_path))
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


