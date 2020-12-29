import math
from torch.nn import Module

from horch.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Sequential


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride, cardinality, base_width):
        super().__init__()
        out_channels = channels * self.expansion

        D = math.floor(channels * (base_width / 64))
        C = cardinality

        self.conv1 = Conv2d(in_channels, D * C, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(D * C, D * C, kernel_size=3, stride=stride, groups=cardinality,
                            norm='def', act='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1,
                            norm='def')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='def') if in_channels != out_channels else Identity()
        self.act = Act()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNeXt(Module):

    def __init__(self, depth, cardinality, base_width, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3,
                           norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            cardinality=cardinality, base_width=base_width)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                **kwargs))
        return Sequential(layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


