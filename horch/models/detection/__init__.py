import torch
import torch.nn as nn

from horch.models.detection.ssd import SSD
from horch.models.detection.light.ssdlite import SSDLite
from horch.models.detection.retinanet import RetinaNet


from horch.common import _tuple


class OneStageDetector(nn.Module):
    r"""
    A simple composation of backbone, head, inference and optional fpn.

    Parameters
    ----------
    backbone : nn.Module
        Backbone network from `horch.models.detection.backbone`.
    head : nn.Module
        Head of the detector from `horch.models.detection.head`.
    inference
        A function or callable to inference on the outputs of the `head`.
        For most cases, use `horch.detection.one.AnchorBasedInference`.
    fpn : nn.Module
        Optional feature enhance module from `horch.models.detection.enhance`.
    """
    def __init__(self, backbone, head, inference, fpn=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._inference = inference
        self.fpn = fpn

    def forward(self, x):
        cs = self.backbone(x)
        if self.fpn is not None:
            cs = self.fpn(*cs)
        return self.head(*_tuple(cs))

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        dets = self._inference(*_tuple(preds))
        self.train()
        return dets


def split_levels(levels, split_at=5):
    levels = _tuple(levels)
    lo = levels[0]
    hi = levels[-1]
    assert levels == tuple(range(lo, hi + 1))
    basic_levels = tuple(range(lo, min(hi, split_at) + 1))
    extra_levels = tuple(range(max(lo, split_at + 1), hi + 1))
    return basic_levels, extra_levels
