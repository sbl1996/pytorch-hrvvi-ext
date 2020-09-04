import torch.nn as nn

from horch.models.modules import Act, Conv2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='default', act='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='default')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='default') if stride != 1 else nn.Identity()
        self.relu = Act('default')

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        channels = out_channels // self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='default', act='default')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='default', act='default')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='default')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='default') if stride != 1 else nn.Identity()
        self.relu = Act('default')

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, num_classes=10, block='basic'):
        super().__init__()
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.conv = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            block, self.stages[0], self.stages[1], layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, self.stages[1], self.stages[2], layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, self.stages[2], self.stages[3], layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3], num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = [block(in_channels, out_channels, stride=stride)]
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


