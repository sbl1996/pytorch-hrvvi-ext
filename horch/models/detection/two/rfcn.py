from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import detach, tuplify, _concat, inverse_sigmoid
from horch.models.utils import bias_init_constant, weight_init_normal
from horch.models.modules import Sequential, Conv2d
from horch.models.detection.head import to_pred
from horch.nn import PSRoIAlign
from horch.train.trainer import set_training


class RPNHead(nn.Module):
    r"""
    RPN Head of Faster R-CNN.

    Parameters
    ----------
    num_anchors : int or tuple of ints
        Number of anchors of every level, e.g., ``(4,6,6,6,6,4)`` or ``6``
    in_channels : int
        Number of input channels.
    f_channels : int
        Number of feature channels.
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """

    def __init__(self, num_anchors, in_channels, f_channels=256, lite=False):
        super().__init__()
        kernel_size = 5 if lite else 3
        self.conv = Conv2d(
            in_channels, f_channels, kernel_size=kernel_size,
            norm_layer='default', activation='default', depthwise_separable=lite)
        self.loc_conv = Conv2d(f_channels, num_anchors * 4, kernel_size=1)
        self.cls_conv = Conv2d(f_channels, num_anchors * 2, kernel_size=1)

        bias_init_constant(self.cls_conv, inverse_sigmoid(0.01))

    def forward(self, p):
        p = self.conv(p)
        loc_p = to_pred(self.loc_conv(p), 4)
        cls_p = to_pred(self.cls_conv(p), 2)
        return loc_p, cls_p


class RFCNHead(nn.Module):

    def __init__(self, num_classes, in_channels, roi_size=(7, 7)):
        super().__init__()
        self.num_classes = num_classes

        rc = roi_size[0] * roi_size[1]
        self.loc_fc = Conv2d(in_channels, 4 * rc, kernel_size=1)
        self.loc_pool = PSRoIAlign(4, roi_size)

        self.cls_fc = Conv2d(in_channels, num_classes * rc, kernel_size=1)
        self.cls_pool = PSRoIAlign(num_classes, roi_size)

        weight_init_normal(self.loc_fc, 0, 0.001)
        weight_init_normal(self.cls_fc, 0, 0.01)

    def forward(self, p, rois):
        r"""
        p : torch.Tensor
            (batch_size, in_channels, h, w)

        Outputs:
        loc_p : (batch_size, #proposals, 4)
        cls_p : (batch_size, #proposals, C)
        """
        batch_size = p.size(0)
        loc_p = self.loc_pool(self.loc_fc(p))
        loc_p = loc_p.mean(dim=(2, 3)).view(batch_size, -1, 4)

        cls_p = self.cls_pool(self.cls_fc(p))
        cls_p = cls_p.mean(dim=(2, 3)).view(batch_size, -1, self.num_classes)
        return loc_p, cls_p


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
        self._inference = inference

    def forward(self, inputs):
        c = self.backbone(inputs)
        loc_p, cls_p = self.head(c)
        return loc_p, cls_p

    def region_proposal(self, inptus):
        c = self.backbone(inptus)
        loc_p, cls_p = self.head(c)
        rois = self._inference(detach(loc_p), detach(cls_p))
        if self.training and self._e2e:
            return c, rois, loc_p, cls_p
        else:
            return c, rois


class RFCN(nn.Module):
    def __init__(self, rpn, box_head, inference):
        super().__init__()
        self.rpn = rpn
        self.box_head = box_head
        self._inference = inference

    def forward(self, x, image_gts=None):
        ps, rois, rpn_loc_p, rpn_cls_p = \
            self.rpn.region_proposal(x)

        ps = [self.roi_pool(p, rois) for p in ps]
        # if self._position_sensitive:
        #     ps = [p.view(p.size(0), -1, 1, 1) for p in ps]
        preds = self.box_head(*ps)
        return preds + (loc_t, cls_t, rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore)

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            ps, rois = self.rpn.region_proposal(x)
            ps = [self.roi_pool(p, rois) for p in ps]
            # if self._position_sensitive:
            #     ps = [p.view(p.size(0), -1, 1, 1) for p in ps]
            preds = self.box_head(*ps)
        image_dets = self._inference(rois[..., 1:], *preds)
        set_training(self)
        return image_dets
