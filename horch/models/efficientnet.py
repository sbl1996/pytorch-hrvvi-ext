import math

import torch
import torch.nn as nn

from horch.models.modules import Conv2d, Identity, seq
from horch.models.drop import DropConnect


def round_channels(channels, multiplier=None, divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""

    if not multiplier:
        return channels

    channels *= multiplier
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def round_repeats(repeats, multiplier=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEModule(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[:2]
        s = self.pool(x).view(b, c)
        s = self.layers(s).view(b, c, 1, 1)
        return x * s


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_connect=0.2):
        super().__init__()

        channels = in_channels * expand_ratio
        use_se = se_ratio is not None and 0 < se_ratio < 1
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = nn.Sequential()
        if expand_ratio != 1:
            layers.add_module(
                "expand", Conv2d(in_channels, channels, kernel_size=1,
                                 norm_layer='default', activation='swish'))

        layers.add_module(
            "dwconv", Conv2d(channels, channels, kernel_size, stride, groups=channels,
                             norm_layer='default', activation='swish'))

        if use_se:
            layers.add_module(
                "se", SEModule(channels, int(in_channels * se_ratio)))

        layers.add_module(
            "project", Conv2d(channels, out_channels, kernel_size=1,
                              norm_layer='default'))

        self.layers = layers
        if self.use_res_connect:
            self.drop_connect = DropConnect(drop_connect)

    def forward(self, x):
        out = self.layers(x)
        if self.use_res_connect:
            out = self.drop_connect(out)
            out += x
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, depth_coef=1.0, dropout=0.2, drop_connect=0.2):
        super().__init__()
        in_channels = 32
        last_channels = 1280
        setting = [
            # r, k, s, e, i, o, se,
            [1, 3, 1, 1, 32, 16, 0.25],
            [2, 3, 2, 6, 16, 24, 0.25],
            [2, 5, 2, 6, 24, 40, 0.25],
            [3, 3, 2, 6, 40, 80, 0.25],
            [3, 5, 1, 6, 80, 112, 0.25],
            [4, 5, 2, 6, 112, 192, 0.25],
            [1, 3, 1, 6, 192, 320, 0.25],
        ]

        in_channels = round_channels(in_channels, width_mult)
        last_channels = round_channels(last_channels, width_mult)

        # building stem
        self.features = nn.Sequential()
        self.features.init_block = Conv2d(3, in_channels, kernel_size=3, stride=2,
                                          norm_layer='default', activation='swish')
        si = 1
        j = 1
        stage = nn.Sequential()
        # building inverted residual blocks
        for idx, (r, k, s, e, i, o, se) in enumerate(setting):
            drop_rate = drop_connect * (float(idx) / len(setting))
            if s == 2:
                self.features.add_module("stage%d" % si, stage)
                si += 1
                j = 1
                stage = nn.Sequential()
            in_channels = round_channels(i, width_mult)
            out_channels = round_channels(o, width_mult)
            stage.add_module("unit%d" % j, MBConv(
                in_channels, out_channels, k, s, e, se, drop_connect=drop_rate))
            j += 1
            for _ in range(round_repeats(r, depth_coef) - 1):
                stage.add_module("unit%d" % j, MBConv(
                    out_channels, out_channels, k, 1, e, se, drop_connect=drop_rate))
                j += 1
        self.features.add_module("stage%d" % si, stage)
        self.features.add_module("final_block",
                                 Conv2d(out_channels, last_channels, kernel_size=1,
                                        norm_layer='default', activation='swish'))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout),
            Conv2d(last_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size()[:2])
        return x


efficient_params = {
    # width_coef, depth_coef, resolution, dropout_rate
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
}


def efficientnet(version='b0', num_classes=10, **kwargs):
    name = 'efficientnet-%s' % version
    assert name in efficient_params, "%s is invalid." % name
    width_mult, depth_coef, resolution, dropout = efficient_params[name]
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs,
    )


def efficientnet_b0(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b0']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b1(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b1']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b2(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b2']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b3(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b3']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b4(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b4']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes
                    ** kwargs,
    )


def efficientnet_b5(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b5']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b6(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b6']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )


def efficientnet_b7(num_classes, **kwargs):
    width_mult, depth_coef, resolution, dropout = efficient_params['efficientnet-b7']
    return EfficientNet(
        width_mult=width_mult,
        depth_coef=depth_coef,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )
