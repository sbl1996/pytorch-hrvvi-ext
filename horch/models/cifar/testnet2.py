import torch.nn as nn
from horch.models.layers import Conv2d, Norm, Act, Linear
from horch.nn import GlobalAvgPool


class PreActResBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, depthwise=True):
        layers = [
            Norm(in_channels),
            Act(),
            Conv2d(in_channels, out_channels, kernel_size=1),
            Norm(out_channels),
            Act(),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                   groups=out_channels if depthwise else 1),
            Norm(out_channels),
            Act(),
            Conv2d(out_channels, out_channels, kernel_size=1),
        ]
        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        super().__init__(*layers)

    def forward(self, x):
        return self.shortcut(x) + super().forward(x)


class ResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, num_classes=10, depthwise=True):
        super().__init__()
        num_blocks = (depth - 4) // 6
        self.stem = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, num_blocks, stride=1, depthwise=depthwise)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, num_blocks, stride=2, depthwise=depthwise)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, num_blocks, stride=2, depthwise=depthwise)

        self.norm = Norm(self.stages[3] * k)
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, depthwise):
        layers = [PreActResBlock(in_channels, out_channels, stride=stride, depthwise=depthwise)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, depthwise=depthwise))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x