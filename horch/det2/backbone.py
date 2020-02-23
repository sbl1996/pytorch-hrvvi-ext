import torch.nn as nn

from detectron2.modeling import Backbone, ShapeSpec, BACKBONE_REGISTRY

from horch.models.pretrained.backbone import EfficientNet
from horch.models.bifpn import BiFPN
from horch.models.detection.enhance import FPNExtraLayers


@BACKBONE_REGISTRY.register()
class EfficientDet(Backbone):

    def __init__(self, cfg, input_shape):
        super().__init__()
        version = cfg.MODEL.EFFICIENTNET.VERSION
        pretrained = cfg.MODEL.EFFICIENTNET.PRETRAINED
        f_channels = cfg.MODEL.BIFPN.F_CHANNELS
        num_fpn_layers = cfg.MODEL.BIFPN.NUM_LAYERS

        self.backbone = EfficientNet(version, feature_levels=(3, 4, 5), pretrained=pretrained)
        out_channels = self.backbone.out_channels

        self.fpn1 = FPNExtraLayers(out_channels[-1], extra_layers=(6, 7), f_channels=f_channels)

        out_channels = [*out_channels, f_channels, f_channels]

        self.fpns = nn.ModuleList([
            BiFPN(out_channels, f_channels),
            *[BiFPN([f_channels] * 5, f_channels)
              for _ in range(num_fpn_layers - 1)]
        ])

        self._out_features = ["p3", "p4", "p5", "p6", "p7"]
        self._out_channels = [f_channels] * 5
        self._out_strides = [2 ** s for s in range(3, 9)]

    def forward(self, x):
        cs = self.backbone(x)
        ps = cs + self.fpn1(cs[-1])
        for l in self.fpns:
            ps = l(*ps)
        return dict(zip(self._out_features, ps))

    def output_shape(self):
        return {
            name: ShapeSpec(channels=c, stride=s)
            for name, c, s in zip(self._out_features, self._out_channels, self._out_strides)
        }
