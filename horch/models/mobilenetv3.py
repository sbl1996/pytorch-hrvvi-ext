import torch
import torch.nn as nn

from horch.models.modules import Conv2d, SELayerM, get_activation


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
    def __init__(self, in_channels, exp, out_channels, kernel_size, stride, activation='relu6', with_se=True,
                 norm_layer='bn'):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = [
            *Conv2d(in_channels, exp, kernel_size=1,
                   norm_layer=norm_layer, activation=activation),
            *Conv2d(exp, exp, kernel_size=kernel_size, stride=stride, groups=exp,
                   norm_layer=norm_layer),
        ]
        if with_se:
            layers.append(SELayerM(exp))
        layers.extend([
            get_activation(activation),
            *Conv2d(exp, out_channels, kernel_size=1,
                   norm_layer=norm_layer)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, norm_layer='bn'):
        super().__init__()
        block = InvertedResidual
        in_channels = 16
        last_channels = 1280
        inverted_residual_setting = [
            # in, k, exp, out, se, nl, s
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, False, 'relu6', 1],
            [3, 64, 24, False, 'relu6', 2],
            [3, 72, 24, False, 'relu6', 1],
            [5, 72, 40, True, 'relu6', 2],
            [5, 120, 40, True, 'relu6', 1],
            [5, 120, 40, True, 'relu6', 1],
            [3, 240, 80, False, 'hswish', 2],
            [3, 200, 80, False, 'hswish', 1],
            [3, 184, 80, False, 'hswish', 1],
            [3, 184, 80, False, 'hswish', 1],
            [3, 480, 112, True, 'hswish', 1],
            [3, 672, 112, True, 'hswish', 1],
            [5, 672, 112, True, 'hswish', 1],
            [5, 672, 160, True, 'hswish', 2],
            [5, 960, 160, True, 'hswish', 1],
        ]

        last_channels = _make_divisible(last_channels * width_mult) if width_mult > 1.0 else last_channels

        # building first layer
        features = [Conv2d(3, in_channels, kernel_size=3, stride=2,
                           norm_layer=norm_layer, activation='hswish')]
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
                   norm_layer=norm_layer, activation='hswish'),
        ])
        in_channels = exp_channels
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            get_activation('hswish'),
            Conv2d(in_channels, last_channels, kernel_size=1,
                   activation='hswish'),
            Conv2d(last_channels, num_classes, kernel_size=1)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size()[:2])
        return x


def mobilenetv3(mult=1.0, **kwargs):
    return MobileNetV3(width_mult=mult, **kwargs)