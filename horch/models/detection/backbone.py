from difflib import get_close_matches

import torch.nn as nn

from pytorchcv.model_provider import get_model as ptcv_get_model

from horch.models.mobilenet import mobilenetv2
from horch.models.efficientnet import efficientnet
from horch.models.mobilenetv3 import mobilenetv3
from horch.models.snet import SNet as BSNet
from horch.models.utils import get_out_channels, calc_out_channels
from horch.models.darknet import Darknet as BDarknet



def _check_levels(levels):
    assert tuple(range(levels[0], levels[-1] + 1)) == tuple(levels), "Feature levels must in ascending order."


def backbone_forward(self, x):
    outs = []
    x = self.layer1(x)
    x = self.layer2(x)
    if 2 in self.out_levels:
        outs.append(x)
    x = self.layer3(x)
    if 3 in self.out_levels:
        outs.append(x)
    x = self.layer4(x)
    if 4 in self.out_levels:
        outs.append(x)
    if 5 in self.forward_levels:
        x = self.layer5(x)
        if 5 in self.out_levels:
            outs.append(x)
    return outs


class ShuffleNetV2(nn.Module):
    mult2name = {
        0.5: "shufflenetv2_wd2",
        1.0: "shufflenetv2_w1",
        1.5: "shufflenetv2_w3d2",
        2.0: "shufflenetv2_w2",
    }
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

    Parameters
    ----------
    mult: float
        Width multiplier which could one of [ 0.5, 1.0, 1.5, 2.0 ]
        Default: 0.5
    feature_levels : sequence of int
        Feature levels to output.
        Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        if pretrained:
            norm_layer = kwargs.get("norm_layer")
            assert norm_layer is None or norm_layer == 'bn', "`gn` can be only set when `pretrained` is False"
            net = ptcv_get_model(self.mult2name[mult], pretrained=True)
            del net.output
            net = net.features
            self.layer1 = net.init_block.conv
            self.layer2 = net.init_block.pool
            self.layer3 = net.stage1
            self.layer4 = net.stage2
            self.layer5 = nn.Sequential(
                net.stage3,
                net.final_block,
            )
            out_channels = [
                get_out_channels(self.layer1),
                get_out_channels(self.layer1),
                calc_out_channels(self.layer3),
                calc_out_channels(self.layer4),
                calc_out_channels(self.layer5),
            ]
        else:
            from horch.models.re.shufflenet import shufflenet_v2 as BShuffleNetV2
            net = BShuffleNetV2(mult=mult, **kwargs)
            channels = net.out_channels
            del net.fc
            self.layer1 = net.conv1
            self.layer2 = net.maxpool
            self.layer3 = net.stage2
            self.layer4 = net.stage3
            self.layer5 = nn.Sequential(
                net.stage4,
                net.conv5,
            )
            out_channels = [channels[0]] + channels[:3] + [channels[-1]]
        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class SNet(nn.Module):
    r"""ThunderNet: Towards Real-time Generic Object Detection

    Args:
        version (int): 49, 146, 535
            Default: 49
        feature_levels (sequence of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, version=49, feature_levels=(3, 4, 5), **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels

        net = BSNet(num_classes=1, version=version, **kwargs)
        del net.fc
        channels = net.channels
        self.layer1 = net.conv1
        self.layer2 = net.maxpool
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if len(channels) == 5:
            self.layer5 = nn.Sequential(
                net.stage4,
                net.conv5,
            )
        else:
            self.layer5 = net.stage4
        out_channels = [channels[0]] + channels[:3] + [channels[-1]]
        self.out_channels = [
            out_channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class MobileNetV2(nn.Module):
    r"""MobileNetV2: Inverted Residuals and Linear Bottlenecks

    Args:
        feature_levels (sequence of int): features of which layers to output
            Default: (3, 4, 5)
    """

    mult2name = {
        0.25: "mobilenetv2_wd4",
        0.5: "mobilenetv2_wd2",
        0.75: "mobilenetv2_w3d4",
        1.0: "mobilenetv2_w1",
    }

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=True, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels

        if pretrained:
            norm_layer = kwargs.get("norm_layer")
            assert norm_layer is None or norm_layer == 'bn', "`gn` can be only set when `pretrained` is False"
            net = ptcv_get_model(self.mult2name[mult], pretrained=True)
            del net.output
            net = net.features
            self.layer1 = nn.Sequential(
                net.init_block,
                net.stage1,
                net.stage2.unit1.conv1,
            )
            self.layer2 = nn.Sequential(
                net.stage2.unit1.conv2,
                net.stage2.unit1.conv3,
                net.stage2.unit2,
                net.stage3.unit1.conv1,
            )
            self.layer3 = nn.Sequential(
                net.stage3.unit1.conv2,
                net.stage3.unit1.conv3,
                *net.stage3[1:],
                net.stage4.unit1.conv1,
            )
            self.layer4 = nn.Sequential(
                net.stage4.unit1.conv2,
                net.stage4.unit1.conv3,
                *net.stage4[1:],
                net.stage5.unit1.conv1,
            )
            self.layer5 = nn.Sequential(
                net.stage5.unit1.conv2,
                net.stage5.unit1.conv3,
                *net.stage5[1:],
                net.final_block,
            )
        else:
            backbone = mobilenetv2(mult=mult, pretrained=pretrained)
            del backbone.classifier
            features = backbone.features

            self.layer1 = features[:2]
            self.layer2 = features[2:4]
            self.layer3 = nn.Sequential(
                *features[4:7],
                features[7].conv[:3],
            )
            self.layer4 = nn.Sequential(
                features[7].conv[3:],
                *features[8:14],
                features[14].conv[:3],
            )
            self.layer5 = nn.Sequential(
                features[14].conv[3:],
                *features[15:],
                backbone.conv,
            )
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class MobileNetV3(nn.Module):
    r"""MobileNetV3: Searching for MobileNetV3

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=False, norm_layer='bn'):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        assert not pretrained, "Pretrained models are not avaliable now."
        backbone = mobilenetv3(mult=mult, num_classes=1, norm_layer=norm_layer)
        del backbone.classifier
        features = backbone.features

        self.layer1 = features[:2]
        self.layer2 = nn.Sequential(
            features[2:4],
            features[4].expand,
        )
        self.layer3 = nn.Sequential(
            features[4].dwconv,
            features[4].se,
            features[4].project,
            *features[5:7],
            features[7].expand,
        )
        self.layer4 = nn.Sequential(
            features[7].dwconv,
            features[7].project,
            *features[8:13],
            features[13].expand,
        )
        self.layer5 = nn.Sequential(
            features[13].dwconv,
            features[13].se,
            features[13].project,
            *features[14:],
        )

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class EfficientNet(nn.Module):
    r"""EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

    Parameters
    ----------
    version : str
        b0, b1, b2, b3, b4, b5, b6, b7 are avaliable.
        Default: b0
    feature_levels (sequence of int): features of which layers to output
        Default: (3, 4, 5)
    """

    def __init__(self, version='b0', feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels

        if pretrained:
            raise NotImplementedError("Pretrained models for EfficientNet are not provided.")
        else:
            net = efficientnet(version, num_classes=1)
            del net.classifier
            net = net.features
            self.layer1 = nn.Sequential(
                net.init_block,
                net.stage1,
                net.stage2.unit1.expand,
            )
            self.layer2 = nn.Sequential(
                net.stage2.unit1.dwconv,
                net.stage2.unit1.se,
                net.stage2.unit1.project,
                net.stage2.unit2,
                net.stage3.unit1.expand,
            )
            self.layer3 = nn.Sequential(
                net.stage3.unit1.dwconv,
                net.stage3.unit1.se,
                net.stage3.unit1.project,
                *net.stage3[1:],
                net.stage4.unit1.expand,
            )
            self.layer4 = nn.Sequential(
                net.stage4.unit1.dwconv,
                net.stage4.unit1.se,
                net.stage4.unit1.project,
                *net.stage4[1:],
                net.stage5.unit1.expand,
            )
            self.layer5 = nn.Sequential(
                net.stage5.unit1.dwconv,
                net.stage5.unit1.se,
                net.stage5.unit1.project,
                *net.stage5[1:],
                net.final_block,
            )
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class SqueezeNet(nn.Module):
    r"""SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB models size

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        from torchvision.models.squeezenet import squeezenet1_1, Fire
        backbone = squeezenet1_1(pretrained=pretrained)
        del backbone.classifier
        backbone = backbone.features
        backbone[0].padding = (1, 1)

        self.layer1 = backbone[:2]
        self.layer2 = backbone[2:5]
        self.layer3 = backbone[5:8]
        self.layer4 = backbone[8:]

        if 5 in feature_levels:
            self.layer5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )

        channels = [64, 128, 256, 512, 512]

        self.out_channels = [
            channels[i - 1] for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)



class Darknet(nn.Module):
    def __init__(self, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        if pretrained:
            norm_layer = kwargs.get("norm_layer")
            assert norm_layer is None or norm_layer == 'bn', "`gn` can be only set when `pretrained` is False"
            net = ptcv_get_model("darknet53", pretrained=True)
            del net.output
            net = net.features
            self.layer1 = nn.Sequential(
                net.init_block,
                net.stage1,
            )
            self.layer2 = net.stage2
            self.layer3 = net.stage3
            self.layer4 = net.stage4
            self.layer5 = net.stage5
        else:
            backbone = BDarknet(num_classes=1, **kwargs)
            del backbone.fc
            self.layer1 = nn.Sequential(
                backbone.conv0,
                backbone.down1,
                backbone.layer1,
            )

            self.layer2 = nn.Sequential(
                backbone.down2,
                backbone.layer2,
            )

            self.layer3 = nn.Sequential(
                backbone.down3,
                backbone.layer3,
            )

            self.layer4 = nn.Sequential(
                backbone.down4,
                backbone.layer4,
            )

            self.layer5 = nn.Sequential(
                backbone.down5,
                backbone.layer5,
            )
        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class ResNet(nn.Module):
    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.conv
        self.layer2 = nn.Sequential(
            net.init_block.pool,
            net.stage1,
        )
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if hasattr(net, "post_activ"):
            self.layer5 = nn.Sequential(
                net.stage4,
                net.post_activ,
            )
        else:
            self.layer5 = nn.Sequential(
                net.stage4,
            )
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class DLA(nn.Module):

    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block
        self.layer2 = net.stage1
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        self.layer5 = net.stage4
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


class Backbone(nn.Module):
    r"""
    General backbone network for ResNet-like architecture.
    Supported: ResNet, DenseNet, SENet, PyramidNet
    """
    def __init__(self, name, feature_levels=(3, 4, 5), pretrained=True):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.out_levels = feature_levels
        net = ptcv_get_model(name, pretrained=pretrained)
        del net.output
        net = net.features
        self.layer1 = net.init_block.conv
        self.layer2 = nn.Sequential(
            net.init_block.pool,
            net.stage1,
        )
        self.layer3 = net.stage2
        self.layer4 = net.stage3
        if hasattr(net, "post_activ"):
            self.layer5 = nn.Sequential(
                net.stage4,
                net.post_activ,
            )
        else:
            self.layer5 = net.stage4
        self.out_channels = [
            calc_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)


def search(name, n=10, cutoff=0.6):
    from pytorchcv.models.model_store import _model_sha1
    models = _model_sha1.keys()
    return get_close_matches(name, models, n=n, cutoff=cutoff)