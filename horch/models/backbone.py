import torch.nn as nn

from horch.models.efficientnet import efficientnet, EfficientNet as BEfficientNet
from horch.models.mobilenetv3 import mobilenetv3
from horch.models.snet import SNet as BSNet
from horch.models.utils import get_out_channels
from horch.models.darknet import Darknet as BDarknet
from horch.models.vovnet import get_vovnet


def _check_levels(levels):
    assert tuple(range(levels[0], levels[-1] + 1)) == tuple(levels), "Feature levels must in ascending order."


def backbone_forward(self, x):
    outs = []
    x = self.layer1(x)
    x = self.layer2(x)
    if 2 in self.feature_levels:
        outs.append(x)
    x = self.layer3(x)
    if 3 in self.feature_levels:
        outs.append(x)
    x = self.layer4(x)
    if 4 in self.feature_levels:
        outs.append(x)
    if 5 in self.forward_levels:
        x = self.layer5(x)
        if 5 in self.feature_levels:
            outs.append(x)
    return outs


class ShuffleNetV2(nn.Module):
    r"""ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

    Parameters
    ----------
    mult: float
        Width multiplier which could one of [ 0.5, 1.0, 1.5, 2.0 ]
        Default: 1.0
    feature_levels : sequence of int
        Feature levels to output.
        Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        assert not pretrained, "Pretrained models are in horch.models.pretrained.backbone."
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels

        from horch.models.shufflenet import shufflenet_v2 as BShuffleNetV2
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

    def __init__(self, version=49, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        assert not pretrained, "No pretrained models for SNet, please use ShuffleNetV2 instead."
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels

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


class MobileNetV3(nn.Module):
    r"""MobileNetV3: Searching for MobileNetV3

    Args:
        feature_levels (list of int): features of which layers to output
            Default: (3, 4, 5)
    """

    def __init__(self, mult=1.0, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        assert not pretrained, "Pretrained models are not avaliable now."
        backbone = mobilenetv3(mult=mult, num_classes=1, **kwargs)
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
        self.feature_levels = feature_levels

        assert not pretrained, "Pretrained models are not avaliable now."

        net = efficientnet(version, num_classes=1, drop_connect=0, **kwargs)
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


class EfficientNetC(nn.Module):
    r"""EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

    Parameters
    ----------
    version : str
        b0, b1, b2, b3, b4, b5, b6, b7 are avaliable.
        Default: b0
    feature_levels (sequence of int): features of which layers to output
        Default: (3, 4, 5)
    """

    def __init__(self, width_mult=1.0, depth_coef=1.0, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        assert not pretrained, "Pretrained models are not avaliable now."

        net = BEfficientNet(width_mult=width_mult, depth_coef=depth_coef, drop_connect=0, num_classes=1, **kwargs)
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


class Darknet(nn.Module):
    def __init__(self, feature_levels=(3, 4, 5), pretrained=False, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        assert not pretrained, "Pretrained models are in horch.models.pretrained.backbone."

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


class VoVNet(nn.Module):

    def __init__(self, version=27, feature_levels=(3, 4, 5), pretrained=False, no_down=0, **kwargs):
        super().__init__()
        _check_levels(feature_levels)
        self.forward_levels = tuple(range(1, feature_levels[-1] + 1))
        self.feature_levels = feature_levels
        assert not pretrained, "Pretrained models are not avaliable now."

        if no_down != 0:
            assert feature_levels == (3, 4) and no_down == -1
        backbone = get_vovnet(version)
        del backbone.output
        f = backbone.features
        self.layer1 = f.init_block
        self.layer2 = f.stage1
        self.layer3 = f.stage2
        if no_down == 0:
            self.layer4 = f.stage3
            self.layer5 = nn.Sequential(
                f.stage4,
                f.post_activ
            )
        else:
            del f.stage4.pool
            self.layer4 = nn.Sequential(
                f.stage3,
                f.stage4,
                f.post_activ
            )

        self.out_channels = [
            get_out_channels(getattr(self, ("layer%d" % i)))
            for i in feature_levels
        ]

    def forward(self, x):
        return backbone_forward(self, x)