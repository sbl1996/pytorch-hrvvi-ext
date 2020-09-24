import torch.nn as nn

from horch.nn import DropPath, GlobalAvgPool
from horch.models.attention import SEModule
from horch.models.layers import Act, Conv2d, Norm, Linear


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0, use_se=False):
        super().__init__()
        self.use_se = use_se
        self._dropout = dropout

        self.norm1 = Norm(in_channels)
        self.act1 = Act()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm2 = Norm(out_channels)
        self.act2 = Act()
        if self._dropout:
            self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        if self.use_se:
            self.se = SEModule(out_channels, reduction=8)

        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.norm1(x)
        x = self.act1(x)
        identity = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        if self._dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        return x + self.shortcut(identity)


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout, use_se, drop_path):
        super().__init__()
        self.norm1 = Norm(in_channels)
        self.act1 = Act()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.norm2 = Norm(out_channels)
        self.act2 = Act()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        if use_se:
            self.se = SEModule(out_channels, reduction=8)
        if drop_path:
            self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + super().forward(x)


class ResNet(nn.Module):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, num_classes=10, dropout=0, use_se=False, drop_path=0):
        super().__init__()
        num_blocks = (depth - 4) // 6
        self.stem = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, num_blocks, stride=1,
            dropout=dropout, use_se=use_se, drop_path=drop_path)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, num_blocks, stride=2,
            dropout=dropout, use_se=use_se, drop_path=drop_path)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, num_blocks, stride=2,
            dropout=dropout, use_se=use_se, drop_path=drop_path)

        self.post_activ = nn.Sequential(
            Norm(self.stages[3] * k),
            Act()
        )

        self.classifier = nn.Sequential(
            GlobalAvgPool(),
            Linear(self.stages[3] * k, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride,
                    dropout, use_se, drop_path):
        layers = [DownBlock(in_channels, out_channels, stride=stride,
                            dropout=dropout, use_se=use_se)]
        for i in range(1, blocks):
            layers.append(
                BasicBlock(out_channels, out_channels,
                           dropout=dropout, use_se=use_se, drop_path=drop_path))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.post_activ(x)

        x = self.classifier(x)
        return x


