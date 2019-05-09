import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import _tuple
from horch.models.utils import get_loc_cls_preds
from horch.models.modules import Conv2d, get_norm_layer, get_activation

from horch.detection.one import MultiBoxLoss, anchor_based_inference, flatten


def inverse_sigmoid(x):
    return math.log(x / (1 - x))


class RefineLoss:

    def __init__(self, neg_threshold=0.01, p=0.01, refine_cls_loss='focal'):
        super().__init__()
        self.r_loss = MultiBoxLoss(criterion=refine_cls_loss, prefix='refine', p=p)
        self.d_loss = MultiBoxLoss(criterion='softmax', pos_neg_ratio=1 / 3, prefix='detect', p=p)
        self.neg_threshold = inverse_sigmoid(neg_threshold)
        self.p = p

    def __call__(self, r_loc_p, r_cls_p, d_loc_p, d_cls_p, loc_t, cls_t):
        pos = cls_t != 0

        r_loss = self.r_loss(r_loc_p, r_cls_p, loc_t, pos.long())

        r_loc_p = r_loc_p.clone().detach()
        r_cls_p = r_cls_p.detach()
        d_loc_t = loc_t - r_loc_p
        d_loc_t[..., :2].div_(r_loc_p[..., 2:].exp_())

        d_loss = self.d_loss(d_loc_p, d_cls_p, d_loc_t, cls_t, r_cls_p <= self.neg_threshold)

        loss = r_loss + d_loss
        return loss


def anchor_refine_inference(
        r_loc_p, r_cls_p, d_loc_p, d_cls_p, anchors,
        neg_threshold=inverse_sigmoid(0.01), iou_threshold=0.5, r_topk=400, d_topk=200, detect_conf_strategy='softmax'):
    neg = r_cls_p > neg_threshold
    r_loc_p = r_loc_p[neg]
    r_cls_p = r_cls_p[neg]
    anchors = anchors[neg]
    d_loc_p = d_loc_p[neg]
    d_cls_p = d_cls_p[neg]

    r_loc_p[:, :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    r_loc_p[:, 2:].exp_().mul_(anchors[:, 2:])

    if len(r_cls_p) > r_topk:
        indices = r_cls_p.topk(r_topk)[1]
        r_loc_p = r_loc_p[indices]
        d_loc_p = d_loc_p[indices]
        d_cls_p = d_cls_p[indices]

    return anchor_based_inference(
        d_loc_p, d_cls_p, r_loc_p,
        conf_threshold=0, iou_threshold=iou_threshold,
        topk=d_topk, conf_strategy=detect_conf_strategy)


class AnchorRefineInference:

    def __init__(self, anchors, neg_threshold=0.01,
                 iou_threshold=0.5, r_topk=400, d_topk=200, detect_conf_strategy='softmax'):
        self.neg_threshold = inverse_sigmoid(neg_threshold)
        self.anchors = flatten(anchors)
        self.iou_threshold = iou_threshold
        self.r_topk = r_topk
        self.d_topk = d_topk
        self.detect_conf_strategy = detect_conf_strategy

    def __call__(self, r_loc_p, r_cls_p, d_loc_p, d_cls_p, *args):
        image_dets = []
        batch_size = r_loc_p.size(0)
        for i in range(batch_size):
            dets = anchor_refine_inference(
                r_loc_p[i], r_cls_p[i], d_loc_p[i], d_cls_p[i], self.anchors,
                self.neg_threshold, self.iou_threshold,
                self.r_topk, self.d_topk, self.detect_conf_strategy
            )
            image_dets.append(dets)
        return image_dets


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion=4, norm_layer='bn'):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = out_channels // expansion

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm_layer=norm_layer, activation='default')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm_layer=norm_layer, activation='default')

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm_layer=norm_layer)
        self.relu3 = get_activation('default')

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                     norm_layer=norm_layer)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class TransferConnection(nn.Module):
    def __init__(self, in_channels, out_channels, last=False, norm_layer='bn'):
        super().__init__()
        self.last = last
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer)
        if not last:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels, out_channels, 4, stride=2, padding=1),
                get_norm_layer(norm_layer, out_channels),
            )
        self.relu2 = get_activation('default')
        self.conv3 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default')

    def forward(self, x, x_next=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if not self.last:
            x = x + self.deconv1(x_next)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class TransferConnectionLite(nn.Module):
    def __init__(self, in_channels, out_channels, last=False, norm_layer='bn'):
        super().__init__()
        self.last = last
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer)
        if not last:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels, out_channels, 4, stride=2, padding=1),
                get_norm_layer(norm_layer, out_channels),
            )
        self.relu2 = get_activation('default')
        self.conv3 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='default')

    def forward(self, x, x_next=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if not self.last:
            x = x + self.deconv1(x_next)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class RefineDet(nn.Module):
    def __init__(self, backbone, num_anchors, num_classes, f_channels, inference, norm_layer='bn', extra_levels=(6,)):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self._inference = inference

        stages = backbone.out_channels

        self.extra_levels = _tuple(extra_levels)
        self.extra_layers = nn.ModuleList([])
        for l in self.extra_levels:
            self.extra_layers.append(
                Bottleneck(stages[-1], f_channels, stride=2)
            )
            stages.append(f_channels)

        self.rps = nn.ModuleList([
            Conv2d(c, num_anchors * (4 + 1), kernel_size=3)
            for c in stages
        ])

        self.tcbs = nn.ModuleList([
            TransferConnection(stages[-1], f_channels, norm_layer=norm_layer, last=True)])
        for c in reversed(stages[:-1]):
            self.tcbs.append(
                TransferConnection(c, f_channels, norm_layer=norm_layer)
            )

        self.dps = nn.ModuleList([
            Conv2d(f_channels, num_anchors * (4 + num_classes), kernel_size=3)
            for _ in stages
        ])

    def forward(self, x):
        cs = self.backbone(x)
        cs = [cs] if torch.is_tensor(cs) else list(cs)
        for l in self.extra_layers:
            cs.append(l(cs[-1]))

        rfs = [
            rp(c) for c, rp in zip(cs, self.rps)
        ]

        dcs = [self.tcbs[0](cs[-1])]
        for c, tcb in zip(reversed(cs[:-1]), self.tcbs[1:]):
            dcs.append(tcb(c, dcs[-1]))

        dfs = [
            dp(dc) for dp, dc in zip(self.dps, reversed(dcs))
        ]

        r_loc_p, r_cls_p = get_loc_cls_preds(rfs, 1)
        d_loc_p, d_cls_p = get_loc_cls_preds(dfs, self.num_classes)

        return r_loc_p, r_cls_p, d_loc_p, d_cls_p

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        dets = self._inference(*_tuple(preds))
        self.train()
        return dets