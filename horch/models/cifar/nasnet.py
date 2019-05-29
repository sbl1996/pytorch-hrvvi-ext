import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from pytorchcv.models.nasnet import Reduction1Unit, Reduction2Unit, nasnet_dual_path_sequential, NormalUnit, \
    nasnet_batch_norm, nas_conv1x1, NasPathBlock, dws_branch_k5_s1_p2, dws_branch_k3_s1_p1, nasnet_avgpool3x3_s1


class NASNetCIFARInitBlock(nn.Module):
    """
    NASNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FirstUnit(nn.Module):
    """
    NASNet First unit.

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
                 out_channels):
        super(FirstUnit, self).__init__()
        mid_channels = out_channels // 6

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        if prev_in_channels is None:
            self.path = None
            self.comb0_left = dws_branch_k5_s1_p2(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.comb0_right = dws_branch_k3_s1_p1(
                in_channels=in_channels,
                out_channels=mid_channels)
        else:
            self.path = NasPathBlock(
                in_channels=prev_in_channels,
                out_channels=mid_channels)

            self.comb0_left = dws_branch_k5_s1_p2(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.comb0_right = dws_branch_k3_s1_p1(
                in_channels=mid_channels,
                out_channels=mid_channels)

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb2_left = nasnet_avgpool3x3_s1()

        self.comb3_left = nasnet_avgpool3x3_s1()
        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev) if self.path else x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
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
    stem_blocks_channels : list of 2 int
        Number of output channels for the Stem units.
    final_pool_size : int
        Size of the pooling windows for final pool.
    extra_padding : bool
        Whether to use extra padding.
    skip_reduction_layer_input : bool
        Whether to skip the reduction layers when calculating the previous layer to connect to.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
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
        reduction_units = [Reduction1Unit, Reduction2Unit]
        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=2)
        self.features.add_module("init_block", NASNetCIFARInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        prev_in_channels = None
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                    unit = FirstUnit
                else:
                    unit = NormalUnit
                if unit == Reduction2Unit:
                    stage.add_module("unit{}".format(j + 1), Reduction2Unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels,
                        extra_padding=False))
                else:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("activ", nn.ReLU())
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))

        self.output = nn.Sequential()
        self.output.add_module('fc', nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

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
        [[192, 192, 192, 192, 192, 192], [256, 384, 384, 384, 384, 384, 384], [512, 768, 768, 768, 768, 768, 768]],
        32,
        in_channels=in_channels,
        num_classes=num_classes,
    )