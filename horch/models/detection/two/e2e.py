import torch
import torch.nn as nn

from horch.common import detach, _tuple
from horch.models.modules import Sequential


class RPN(Sequential):
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

    def __init__(self, backbone, fpn, head, inference=None):
        super().__init__(inference=inference)
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def region_proposal(self, x):
        cs = self.backbone(x)
        ps = self.fpn(*cs)
        loc_p, cls_p = self.head(*_tuple(ps))
        image_dets = self._inference(detach(loc_p), detach(cls_p))
        return ps, loc_p, cls_p, image_dets


def dets_to_rois(image_dets, ref):
    rois = [[[i, *d['bbox']] for d in idets]
            for i, idets in enumerate(image_dets)]
    rois = ref.new_tensor(rois)

    # LTWH to LTRB
    rois[..., 3] += rois[..., 1]
    rois[..., 4] += rois[..., 2]
    return rois


class RCNN(nn.Module):
    def __init__(self, rpn, roi_match, roi_pool, head, inference):
        super().__init__()
        self.rpn = rpn
        self.roi_match = roi_match
        self.roi_pool = roi_pool
        self.ps = 'PS' in type(roi_pool).__name__
        self.head = head
        self._inference = inference

    def forward(self, x, image_gts=None):
        ps, rpn_loc_p, rpn_cls_p, image_dets = self.rpn.region_proposal(x)
        if torch.is_tensor(ps):
            ps = [ps]
        rois = dets_to_rois(image_dets, x)

        if self.training:
            loc_t, cls_t, rois = self.roi_match(rois, image_gts)

        ps = [self.roi_pool(p, rois) for p in ps]
        if self.ps:
            ps = [p.view(p.size(0), -1, 1, 1) for p in ps]
        loc_p, cls_p = self.head(*ps)

        if self.training:
            return loc_p, cls_p, loc_t, cls_t, rpn_loc_p, rpn_cls_p
        else:
            return rois[..., 1:], loc_p, cls_p

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        image_dets = self._inference(*_tuple(preds))
        self.train()
        return image_dets
