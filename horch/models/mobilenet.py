"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2 by d-li14
import from https://github.com/d-li14/mobilenetv2.pytorch by HrvvI
"""
import math

import torch
import torch.nn as nn

from horch.models.utils import load_state_dict_from_google_drive

__all__ = ['mobilenetv2']

model_urls = {
    1.0: {
        "file_id": "1VKLpm60yPawzfxQAZVza8pGbVamWKlL2",
        "filename": "mobilenetv2-0c6065bc.pth",
        "md5": "9c6d85d239e4e5ff8ec2868e2f521ed8",
    },
    0.5: {
        "file_id": "1Jrnc200t25931pf4jJtOKovpAkuzm9aI",
        "filename": "mobilenetv2_0.5-eaa6f9ad.pth",
        "md5": "a568ee506b47358a752154f42e85d54d",
    },
    0.25: {
        "file_id": "1BRxPgOtjQuC5EUJX7ed808MMUZU4fEl_",
        "filename": "mobilenetv2_0.25-b61d2159.pth",
        "md5": "70ccfb1d1000c025ef03648b51afd791",
    },
    0.1: {
        "file_id": "1WxSUxOLQwy6EVee15mCsKATRQYPz99Eo",
        "filename": "mobilenetv2_0.1-7d1d638a.pth",
        "md5": "3194ca7431168a3fb66eae30db600ca4",
    }
}


def _make_divisible(v, divisor, min_value=None):
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


def ConvBNReLU(inp, oup, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        if 32 * width_mult <= 4:
            divisor = 4
        else:
            divisor = 8
        # building first layer
        input_channel = _make_divisible(32 * width_mult, divisor)
        layers = [ConvBNReLU(3, input_channel, kernel_size=3, stride=2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, divisor)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, divisor) if width_mult > 1.0 else 1280
        self.conv = ConvBNReLU(input_channel, output_channel, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(mult=1.0, pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2(width_mult=mult, **kwargs)
    if pretrained:
        supported = list(model_urls.keys())
        assert mult in supported, "Only mult in %s has pretrained model." % supported
        info = model_urls[mult]
        model.load_state_dict(
            load_state_dict_from_google_drive(
                info['file_id'], info['filename'], info['md5'], map_location='cpu'))
    return model
