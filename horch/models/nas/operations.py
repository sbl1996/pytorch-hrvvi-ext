import torch
import torch.nn as nn

from horch.models.layers import Conv2d, Act, Norm

OPS = {
    'none': lambda C, stride: Zero(stride),
    'avg_pool_3x3': lambda C, stride: nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        Norm(C),
    ),
    'max_pool_3x3': lambda C, stride: nn.Sequential(
        nn.MaxPool2d(3, stride=stride, padding=1),
        Norm(C),
    ),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride),
    'nor_conv_1x1': lambda C, stride: ReLUConvBN(C, C, 1, stride),
    'nor_conv_3x3': lambda C, stride: ReLUConvBN(C, C, 3, stride),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 2),
    'conv_7x1_1x7': lambda C, stride: nn.Sequential(
        Act('relu', inplace=False),
        Conv2d(C, C, (1, 7), stride=(1, stride), bias=False),
        Conv2d(C, C, (7, 1), stride=(stride, 1), bias=False),
        Norm(C),
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            Act('relu', inplace=False),
            Conv2d(C_in, C_out, kernel_size, bias=False, stride=stride),
            Norm(C_out),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, dilation):
        super().__init__()
        self.op = nn.Sequential(
            Act('relu', inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, bias=False),
            Norm(C_out),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride):
        super().__init__()
        self.op = nn.Sequential(
            Act('relu', inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, groups=C_in, bias=False),
            Conv2d(C_in, C_in, kernel_size=1, bias=False),
            Norm(C_in),
            Act('relu', inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, bias=False),
            Norm(C_out),
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
        self.act = Act('relu', inplace=False)
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = Norm(C_out)

    def forward(self, x):
        x = self.act(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
