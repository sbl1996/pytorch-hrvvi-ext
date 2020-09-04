import torch.nn as nn

from horch.models.attention import SELayerM
from horch.models.modules import Conv2d, Identity


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, channels, out_channels, kernel_size, stride, activation='relu6', with_se=True):
        super().__init__()
        self.with_se = with_se
        if in_channels != channels:
            self.expand = Conv2d(in_channels, channels, kernel_size=1,
                                 norm='default', act=activation)
        else:
            self.expand = Identity()

        self.dwconv = Conv2d(channels, channels, kernel_size, stride, groups=channels,
                             norm='default',
                             act=activation)

        if self.with_se:
            self.se = SELayerM(channels, 4)

        self.project = Conv2d(channels, out_channels, kernel_size=1,
                              norm='default')
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.dwconv(x)
        if self.with_se:
            x = self.se(x)
        x = self.project(x)
        if self.use_res_connect:
            x += identity
        return x


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        in_channels = 16
        last_channels = 1280
        inverted_residual_setting = [
            # k, e, o,  se,     nl,  s,
            [3, 16, 16, False, 'relu6', 1],
            [3, 64, 24, False, 'relu6', 1],
            [3, 72, 24, False, 'relu6', 1],
            [5, 72, 40, True, 'relu6', 1],
            [5, 120, 40, True, 'relu6', 1],
            [5, 120, 40, True, 'relu6', 1],
            [3, 240, 80, False, 'hswish', 2],
            [3, 200, 80, False, 'hswish', 1],
            [3, 184, 80, False, 'hswish', 1],
            [3, 184, 80, False, 'hswish', 1],
            [3, 480, 112, True, 'hswish', 1],
            [3, 672, 112, True, 'hswish', 1],
            [5, 672, 160, True, 'hswish', 2],
            [5, 960, 160, True, 'hswish', 1],
            [5, 960, 160, True, 'hswish', 1],
        ]

        last_channels = _make_divisible(last_channels * width_mult) if width_mult > 1.0 else last_channels

        # building first layer
        features = [Conv2d(3, in_channels, kernel_size=3, stride=1,
                           norm='default', act='hswish')]
        # building inverted residual blocks
        for k, exp, c, se, nl, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult)
            exp_channels = _make_divisible(exp * width_mult)
            features.append(block(
                in_channels, exp_channels, out_channels, k, s, nl, se))
            in_channels = out_channels
        # building last several layers
        features.extend([
            Conv2d(in_channels, exp_channels, kernel_size=1,
                   norm='default', act='hswish'),
        ])
        in_channels = exp_channels
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_channels, last_channels, kernel_size=1,
                   act='hswish'),
            Conv2d(last_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size()[:2])
        return x


def mobilenetv3(mult=1.0, **kwargs):
    return MobileNetV3(width_mult=mult, **kwargs)
