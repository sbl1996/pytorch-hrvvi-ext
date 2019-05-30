import torch
import torch.nn as nn

from horch.models.modules import Conv2d, Identity
from horch.models.efficientnet import round_channels, MBConv


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0, drop_connect=0.2):
        super().__init__()
        in_channels = 32
        last_channels = 1280
        setting = [
            # r, k, s, e, i, o, se,
            [1, 3, 1, 1, 32, 16, 0.25],
            [2, 3, 1, 6, 16, 24, 0.25],
            [2, 5, 1, 6, 24, 40, 0.25],
            [3, 3, 2, 6, 40, 80, 0.25],
            [3, 5, 1, 6, 80, 112, 0.25],
            [4, 5, 2, 6, 112, 192, 0.25],
            [1, 3, 1, 6, 192, 320, 0.25],
        ]

        last_channels = round_channels(last_channels, width_mult) if width_mult > 1.0 else last_channels

        # building stem
        features = [Conv2d(3, in_channels, kernel_size=3, stride=1,
                           norm_layer='default', activation='swish')]
        # building inverted residual blocks

        for idx, (r, k, s, e, i, o, se) in enumerate(setting):
            drop_rate = drop_connect * (float(idx) / len(setting))
            in_channels = round_channels(i, width_mult)
            out_channels = round_channels(o, width_mult)
            features.append(MBConv(
                in_channels, out_channels, k, s, e, se, drop_connect=drop_rate))
            for _ in range(r-1):
                features.append(MBConv(
                    out_channels, out_channels, k, 1, e, se, drop_connect=drop_rate))

        self.features = nn.Sequential(*features)

        dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else Identity()

        # building classifier
        self.classifier = nn.Sequential(
            Conv2d(out_channels, last_channels, kernel_size=1,
                   norm_layer='default', activation='swish'),
            nn.AdaptiveAvgPool2d(1),
            dropout,
            Conv2d(last_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size()[:2])
        return x


def efficientnet_b0(**kwargs):
    return EfficientNet(width_mult=1.0, **kwargs)
