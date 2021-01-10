from torch.nn import Module, Dropout

from horch.nn import DropPath
from horch.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Norm, Pool2d, Sequential


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride, dropout, drop_path,
                 start_block=False, end_block=False, exclude_bn0=False):
        super().__init__()
        # For torch.jit.script
        self.bn0 = Identity()
        self.act0 = Identity()
        self.act2 = Identity()
        self.bn2 = Identity()

        out_channels = channels * self.expansion
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)

        if not start_block:
            self.act0 = Act()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn1 = Norm(out_channels)
        self.act1 = Act()
        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if start_block:
            self.bn2 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.bn2 = Norm(out_channels)
            self.act2 = Act()

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()
        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = self.shortcut(x)

        if self.start_block:
            x = self.conv1(x)
        else:
            if not self.exclude_bn0:
                x = self.bn0(x)
            x = self.act0(x)
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.start_block:
            x = self.bn2(x)
        x = self.drop_path(x)
        x = x + identity
        if self.end_block:
            x = self.bn2(x)
            x = self.act2(x)
        return x


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride, dropout, drop_path,
                 start_block=False, end_block=False, exclude_bn0=False):
        super().__init__()
        # For torch.jit.script
        self.bn0 = Identity()
        self.act0 = Identity()
        self.bn3 = Identity()
        self.act3 = Identity()

        out_channels = channels * self.expansion
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)

        if not start_block:
            self.act0 = Act()
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1)
        self.bn1 = Norm(channels)
        self.act1 = Act()
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride)
        self.bn2 = Norm(channels)
        self.act2 = Act()
        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.bn3 = Norm(out_channels)
            self.act3 = Act()

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()
        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = self.shortcut(x)
        if self.start_block:
            x = self.conv1(x)
        else:
            if not self.exclude_bn0:
                x = self.bn0(x)
            x = self.act0(x)
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        if self.start_block:
            x = self.bn3(x)
        x = self.drop_path(x)
        x = x + identity
        if self.end_block:
            x = self.bn3(x)
            x = self.act3(x)
        return x


class ResNet(Module):

    def __init__(self, depth, block='basic', dropout=0, drop_path=0, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.conv = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            dropout=dropout, drop_path=drop_path)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            dropout=dropout, drop_path=drop_path)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            dropout=dropout, drop_path=drop_path)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride,
                    dropout, drop_path):
        layers = [block(self.in_channels, channels, stride=stride, start_block=True,
                        dropout=dropout, drop_path=drop_path)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                dropout=dropout, drop_path=drop_path))
        return Sequential(layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

