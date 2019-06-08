import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d, get_norm_layer, get_activation, Pool

from pytorchcv.models.nasnet import nasnet_dual_path_sequential


class NasConv(nn.Module):
    """
    NASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(NasConv, self).__init__()
        self.activ = get_activation('default')
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.bn = get_norm_layer('default', out_channels)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class NasDWConv(nn.Module):
    """
    NASNet specific depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.activ = get_activation("default")
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            depthwise_separable=True)
        self.bn = get_norm_layer('bn', out_channels)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DWBranch(nn.Module):
    """
    NASNet specific block with depthwise separable convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv1 = NasDWConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride)
        self.conv2 = NasDWConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ReductionCell(nn.Module):
    """
    NASNet Reduction base unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, prev_in_channels, out_channels):
        super().__init__()
        self.skip_input = True
        mid_channels = out_channels // 4

        self.conv1x1_prev = NasConv(prev_in_channels, mid_channels, 1)
        self.conv1x1 = NasConv(in_channels, mid_channels, 1)

        self.comb0_left = DWBranch(mid_channels, mid_channels, 5, 2)
        self.comb0_right = DWBranch(mid_channels, mid_channels, 7, 2)

        self.comb1_left = Pool('max', 3, 2)
        self.comb1_right = DWBranch(mid_channels, mid_channels, 7, 2)

        self.comb2_left = Pool('avg', 3, 2)
        self.comb2_right = DWBranch(mid_channels, mid_channels, 5, 2)

        self.comb3_right = Pool('avg', 3, 1)

        self.comb4_left = DWBranch(mid_channels, mid_channels, 3, 1)
        self.comb4_right = Pool('max', 3, 2)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out


class NasPathBranch(nn.Module):
    """
    NASNet specific `path` branch (auxiliary block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels, extra_padding=False):
        super().__init__()
        self.extra_padding = extra_padding
        self.avgpool = Pool('avg', 1, 2)
        self.conv = Conv2d(in_channels, out_channels, 1)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
            x = x[:, :, 1:, 1:].contiguous()
        x = self.avgpool(x)
        x = self.conv(x)
        return x


class NasPathBlock(nn.Module):
    """
    NASNet specific `path` block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2

        self.activ = get_activation('default')
        self.path1 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.path2 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            extra_padding=True)
        self.bn = get_norm_layer('default', out_channels)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


class AdjustUnit(nn.Module):
    def __init__(self, prev_in_channels, mid_channels, reduce=False):
        super().__init__()
        if prev_in_channels is None:
            self.path = None
        elif reduce:
            self.path = NasPathBlock(prev_in_channels, mid_channels)
        else:
            self.path = NasConv(prev_in_channels, mid_channels, 1)

    def forward(self, x, x_prev):
        if self.path is None:
            x_prev = x
        else:
            x_prev = self.path(x_prev)
        return x_prev


class NormalUnit(nn.Module):
    """
    NASNet Normal unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 reduce=False):
        super(NormalUnit, self).__init__()
        mid_channels = out_channels // 6

        self.conv1x1 = NasConv(in_channels, mid_channels, 1)

        self.adjust = AdjustUnit(prev_in_channels, mid_channels, reduce)

        self.comb0_left = DWBranch(mid_channels, mid_channels, 5)
        channels = in_channels if prev_in_channels is None else mid_channels
        self.comb0_right = DWBranch(channels, mid_channels, 3)

        self.comb1_left = DWBranch(mid_channels, mid_channels, 5)
        self.comb1_right = DWBranch(mid_channels, mid_channels, 3)

        self.comb2_left = Pool('avg', 3)

        self.comb3_left = Pool('avg', 3)
        self.comb3_right = Pool('avg', 3)

        self.comb4_left = DWBranch(mid_channels, mid_channels, 3)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.adjust(x, x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        # x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x3 = self.comb3_left(x_right) * 2
        x4 = self.comb4_left(x_left) + x_left

        x_out = torch.cat((x_right, x0, x1, x2, x3, x4), dim=1)
        return x_out


class NASNet(nn.Module):
    """
    NASNet-A model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=2)
        self.features.add_module("init_block",
                                 Conv2d(in_channels, init_block_channels, kernel_size=3,
                                        norm_layer='default'))
        self.prev_in_channels = None
        self.in_channels = init_block_channels

        strides = [1, 2, 2]
        for i, stride, channels_per_stage in zip(range(len(channels)), strides, channels):
            self.features.add_module("stage%d" % (i+1),
                                     self._make_layer(channels_per_stage, stride))

        self.features.add_module("activ", get_activation('default'))
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))

        self.output = nn.Sequential()
        self.output.add_module('fc', nn.Linear(
            in_features=self.in_channels,
            out_features=num_classes))

    def _make_layer(self, channels, stride):
        stage = nasnet_dual_path_sequential()
        j = 1
        if stride == 2:
            out_channels = channels[0]
            stage.add_module("unit1", ReductionCell(
                in_channels=self.in_channels,
                prev_in_channels=self.prev_in_channels,
                out_channels=out_channels))
            self.prev_in_channels = self.in_channels
            self.in_channels = out_channels
            j += 1
            channels = channels[1:]

        for out_channels in channels:
            stage.add_module("unit%d" % j, NormalUnit(
                in_channels=self.in_channels,
                prev_in_channels=self.prev_in_channels,
                out_channels=out_channels,
                reduce=(stride == 2 and j == 2)))
            self.prev_in_channels = self.in_channels
            self.in_channels = out_channels
            j += 1
        return stage

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def nasnet_6a768(num_classes=10, in_channels=3):
    """
    NASNet-A 6@4032 (NASNet-A-Large) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return NASNet(
        [[192, 192, 192, 192, 192, 192],
         [256, 384, 384, 384, 384, 384, 384],
         [512, 768, 768, 768, 768, 768, 768]],
        32,
        in_channels=in_channels,
        num_classes=num_classes,
    )


def nasnet_7a2304(num_classes=10, in_channels=3):
    """
    NASNet-A 6@4032 (NASNet-A-Large) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return NASNet(
        [[576, 576, 576, 576, 576, 576],
         [768, 1152, 1152, 1152, 1152, 1152, 1152],
         [1536, 2304, 2304, 2304, 2304, 2304, 2304]],
        96,
        in_channels=in_channels,
        num_classes=num_classes,
    )
