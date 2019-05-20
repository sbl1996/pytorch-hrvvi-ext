from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import detach, _tuple, _concat
from horch.models.modules import Sequential, Conv2d, DeConv2d
from horch.models.detection.head import to_pred


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
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: "bn"
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """

    def __init__(self, num_anchors, in_channels, f_channels=256, norm_layer='bn', lite=False):
        super().__init__()
        kernel_size = 5 if lite else 3
        self.conv = Conv2d(
            in_channels, f_channels, kernel_size=kernel_size,
            norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.loc_conv = Conv2d(f_channels, num_anchors * 4, kernel_size=1)
        self.cls_conv = Conv2d(f_channels, num_anchors * 2, kernel_size=1)

        self.cls_conv.apply(self._init_final_cls_layer)

    def _init_final_cls_layer(self, m, p=0.01):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.constant_(m.bias, -log((1 - p) / p))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        for p in ps:
            p = self.conv(p)
            loc_p = to_pred(self.loc_conv(p), 4)
            loc_preds.append(loc_p)

            cls_p = to_pred(self.cls_conv(p), 2)
            cls_preds.append(cls_p)
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p


class BoxHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """

    def __init__(self, num_classes, f_channels=256, norm_layer='bn'):
        super().__init__()
        self.fc1 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer=norm_layer, activation='default')
        self.fc2 = Conv2d(f_channels, f_channels, kernel_size=1,
                          norm_layer=norm_layer, activation='default')
        self.loc_fc = Conv2d(f_channels, 4, kernel_size=1)
        self.cls_fc = Conv2d(f_channels, num_classes, kernel_size=1)

    def forward(self, ps):
        r"""
        p : torch.Tensor
            (batch_size * num_rois, C, 14, 14)
        """
        ps = [self.fc1(F.adaptive_avg_pool2d(p, 1)) for p in ps]
        p = torch.stack(ps).max(dim=0)[0]
        p = self.fc2(p)
        loc_p = self.loc_fc(p).squeeze()
        cls_p = self.cls_fc(p).squeeze()
        return loc_p, cls_p


class MaskHead(nn.Module):
    r"""
    Light head only for R-CNN, not for one-stage detector.
    """

    def __init__(self, num_classes, f_channels=256, norm_layer='bn', lite=False):
        super().__init__()
        self.conv1 = Conv2d(f_channels, f_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.conv2 = Conv2d(f_channels, f_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.conv3 = Conv2d(f_channels, f_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.conv4 = Conv2d(f_channels, f_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.deconv = DeConv2d(f_channels, f_channels, kernel_size=4, stride=2,
                               norm_layer=norm_layer, activation='default', depthwise_separable=lite)
        self.mask_fc = Conv2d(f_channels, num_classes, kernel_size=1)

    def forward(self, ps):
        r"""
        p : torch.Tensor
            (batch_size * num_rois, C, 7, 7)
        """
        ps = [self.conv1(p) for p in ps]
        p = torch.stack(ps).max(dim=0)[0]
        p = self.conv2(p)
        p = self.conv3(p)
        p = self.conv4(p)
        p = self.deconv(p)
        mask_p = self.mask_fc(p)
        return mask_p


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

    def __init__(self, backbone, fpn, head, inference=None, match_anchors=None):
        super().__init__(inference=inference)
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.match_anchors = match_anchors

    def region_proposal(self, x, image_gts=None):
        cs = self.backbone(x)
        ps = self.fpn(*cs)
        ps = _tuple(ps)
        loc_p, cls_p = self.head(*_tuple(ps))
        rois = self._inference(detach(loc_p), detach(cls_p))
        if self.training:
            loc_t, cls_t, ignore = self.match_anchors(image_gts)
            return ps, rois, loc_p, cls_p, loc_t, cls_t, ignore
        else:
            return ps, rois, loc_p, cls_p


class MaskRCNN(nn.Module):
    def __init__(self, rpn, roi_match, roi_pool, box_head, mask_head, inference):
        super().__init__()
        self.rpn = rpn
        self.roi_match = roi_match
        self.roi_pool = roi_pool
        self.position_sensitive = 'PS' in type(roi_pool).__name__
        self.box_head = box_head
        self.mask_head = mask_head
        self._inference = inference

    def forward(self, x, image_gts=None):
        b = x.size(0)
        if self.training:
            ps, rois, rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore = self.rpn.region_proposal(x, image_gts)
            loc_t, cls_t, mask_t, rois = self.roi_match(rois, image_gts)
        else:
            ps, rois, loc_p, cls_p = self.rpn.region_proposal(x)

        ps = [self.roi_pool(p, rois) for p in ps]
        if self.position_sensitive:
            ps = [p.view(p.size(0), -1, 1, 1) for p in ps]
        loc_p, cls_p = self.box_head(ps)

        if self.training:
            pos = cls_t != 0
            mask_p = self.mask_head([p[pos] for p in ps])
            return loc_p, cls_p, mask_p, loc_t, cls_t, mask_t, rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore
        else:
            ps = [p.view(b, -1, *p.size()[1:]) for p in ps]

            def predict_mask(i, indices):
                return self.mask_head([p[i][indices] for p in ps])

            return rois[..., 1:], loc_p, cls_p, predict_mask

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rois, loc_p, cls_p, predict_mask = self.forward(x)
        image_dets = self._inference(rois, loc_p, cls_p, predict_mask)
        self.train()
        return image_dets
