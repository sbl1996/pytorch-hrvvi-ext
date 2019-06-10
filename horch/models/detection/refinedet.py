import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import _tuple, inverse_sigmoid
from horch.detection.one import MultiBoxLoss, flatten_preds, anchor_based_inference
from horch.models.detection.head import SSDHead
from horch.models.modules import Conv2d, get_activation, get_attention


class TransferConnection(nn.Module):
    def __init__(self, in_channels, f_channels, last=False, lite=False):
        super().__init__()
        kernel_size = 5 if lite else 3
        self.last = last
        self.conv1 = nn.Sequential(
            Conv2d(in_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default', activation='default', depthwise_separable=lite),
            Conv2d(f_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default', depthwise_separable=lite),
            # get_attention(attention, in_channels=f_channels, reduction=4),
        )
        if not last:
            self.deconv = Conv2d(f_channels, f_channels, kernel_size=4, stride=2,
                                 norm_layer='default', transposed=True, depthwise_separable=lite)
        self.conv2 = nn.Sequential(
            get_activation('default'),
            Conv2d(f_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default', activation='default', depthwise_separable=lite)
        )

    def forward(self, x, x_next=None):
        x = self.conv1(x)
        if not self.last:
            x = x + self.deconv(x_next)
        x = self.conv2(x)
        return x


class RefineDet(nn.Module):
    def __init__(self, num_anchors, num_classes, in_channels_list, f_channels, lite=False):
        super().__init__()

        self.r_head = SSDHead(num_anchors, 1, in_channels_list, lite=lite)
        self.tcbs = nn.ModuleList([
            TransferConnection(c, f_channels, lite=lite)
            for c in in_channels_list[:-1]
        ])
        self.tcbs.append(
            TransferConnection(in_channels_list[-1], f_channels, last=True, lite=lite)
        )
        self.d_head = SSDHead(num_anchors, num_classes, [f_channels] * len(in_channels_list), lite=lite)

    def forward(self, *cs):
        r_loc_preds, r_cls_preds = self.r_head(*cs)
        dcs = [self.tcbs[-1](cs[-1])]
        for c, tcb in zip(reversed(cs[:-1]), reversed(self.tcbs[:-1])):
            dcs.append(tcb(c, dcs[-1]))
        dcs.reverse()
        d_loc_preds, d_cls_preds = self.d_head(*dcs)
        return r_loc_preds, r_cls_preds, d_loc_preds, d_cls_preds


class RefineLoss(nn.Module):
    def __init__(self, neg_threshold=0.01, refine_cls_loss='focal', detect_cls_loss='focal', detach=True, p=0.01):
        super().__init__()
        self.neg_threshold = neg_threshold
        self.detach = detach
        self.r_loss = MultiBoxLoss(p=p, cls_loss=refine_cls_loss, prefix='refine')
        if detect_cls_loss == 'ce':
            self.d_loss = MultiBoxLoss(pos_neg_ratio=1 / 3, p=p, cls_loss='ce', prefix='detect')
        elif detect_cls_loss == 'focal':
            self.d_loss = MultiBoxLoss(p=p, cls_loss='focal', prefix='detect')
        self.p = p

    @property
    def p(self):
        return self.r_loss.p

    @p.setter
    def p(self, new_p):
        self.r_loss.p = new_p
        self.d_loss.p = new_p

    def forward(self, r_loc_p, r_cls_p, d_loc_p, d_cls_p, loc_t, cls_t):
        r_loc_p, r_cls_p, d_loc_p, d_cls_p = flatten_preds(r_loc_p, r_cls_p, d_loc_p, d_cls_p)

        pos = cls_t != 0

        r_loss = self.r_loss(r_loc_p, r_cls_p, loc_t, pos.long())
        if self.detach:
            r_loc_p = r_loc_p.detach()
        d_loc_t = loc_t - r_loc_p
        d_loc_t[..., :2].div_(r_loc_p[..., 2:].exp_())

        d_loss = self.d_loss(d_loc_p, d_cls_p, d_loc_t, cls_t, r_cls_p <= inverse_sigmoid(self.neg_threshold))

        loss = r_loss + d_loss
        return loss


def anchor_refine_inference(
        r_loc_p, r_cls_p, d_loc_p, d_cls_p, anchors,
        neg_threshold, iou_threshold,
        refine_topk, detect_topk, detect_conf_strategy,
        detect_conf_threshold, detect_nms):
    pos = r_cls_p > inverse_sigmoid(neg_threshold)
    r_loc_p = r_loc_p[pos]
    r_cls_p = r_cls_p[pos]
    anchors = anchors[pos]
    d_loc_p = d_loc_p[pos]
    d_cls_p = d_cls_p[pos]

    r_loc_p[:, :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    r_loc_p[:, 2:].exp_().mul_(anchors[:, 2:])

    if len(r_cls_p) > refine_topk:
        indices = r_cls_p.topk(refine_topk)[1]
        r_loc_p = r_loc_p[indices]
        d_loc_p = d_loc_p[indices]
        d_cls_p = d_cls_p[indices]

    return anchor_based_inference(
            d_loc_p, d_cls_p, r_loc_p,
            conf_threshold=detect_conf_threshold, iou_threshold=iou_threshold,
            topk=detect_topk, conf_strategy=detect_conf_strategy, nms_method=detect_nms)


class AnchorRefineInference:

    def __init__(self, generator, neg_threshold=0.01,
                 iou_threshold=0.5, refine_topk=400, detect_topk=200,
                 detect_conf_strategy='softmax', detect_conf_threshold=0.01, detect_nms='soft'):
        assert generator.flatten
        assert generator.with_ltrb
        self.generator = generator
        self.neg_threshold = neg_threshold
        self.iou_threshold = iou_threshold
        self.refine_topk = refine_topk
        self.detect_topk = detect_topk
        self.detect_conf_strategy = detect_conf_strategy
        self.detect_conf_threshold = detect_conf_threshold
        self.detect_nms = detect_nms

    def __call__(self, r_loc_preds, r_cls_preds, d_loc_preds, d_cls_preds, *args):
        grid_sizes = [p.size()[1:3] for p in r_loc_preds]
        anchors = self.generator(grid_sizes, r_loc_preds[0].device, r_loc_preds[0].dtype)[0]
        r_loc_p, r_cls_p, d_loc_p, d_cls_p = flatten_preds(
            r_loc_preds, r_cls_preds, d_loc_preds, d_cls_preds)
        batch_size = r_loc_p.size(0)
        image_dets = []
        for i in range(batch_size):
            dets = self.inference_single(
                r_loc_p[i], r_cls_p[i], d_loc_p[i], d_cls_p[i], anchors)
            image_dets.append(dets)
        return image_dets

    def inference_single(self, r_loc_p, r_cls_p, d_loc_p, d_cls_p, anchors):
        return anchor_refine_inference(
            r_loc_p, r_cls_p, d_loc_p, d_cls_p, anchors,
            self.neg_threshold, self.iou_threshold,
            self.refine_topk, self.detect_topk, self.detect_conf_strategy,
            self.detect_conf_threshold, self.detect_nms,
        )
