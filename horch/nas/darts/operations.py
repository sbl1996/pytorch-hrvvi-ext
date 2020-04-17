import torch
import torch.nn as nn

from horch.models.modules import Conv2d, get_activation, get_norm_layer

OPS = {
    'none': lambda C, stride: Zero(stride),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride, 3),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 4),
    'conv_7x1_1x7': lambda C, stride: nn.Sequential(
        get_activation(),
        Conv2d(C, C, (1, 7), stride=(1, stride), bias=False),
        Conv2d(C, C, (7, 1), stride=(stride, 1), bias=False),
        get_norm_layer(C),
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size):
        super().__init__()
        self.op = nn.Sequential(
            get_activation(),
            Conv2d(C_in, C_out, kernel_size, bias=False),
            get_norm_layer(C_out),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, dilation):
        super().__init__()
        self.op = nn.Sequential(
            get_activation(),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, bias=False),
            get_norm_layer(C_out),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            get_activation(),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, groups=C_in, bias=False),
            Conv2d(C_in, C_in, kernel_size=1, bias=False),
            get_norm_layer(C_in),
            get_activation(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            get_norm_layer(C_out),
        )

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = get_norm_layer(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
