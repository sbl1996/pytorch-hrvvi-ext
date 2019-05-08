import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch import one_hot
from horch.models.modules import upsample_concat, Conv2d
from horch.models.detection.backbone import Darknet

from horch.detection import BBox, iou_1m, soft_nms_cpu


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer='bn'):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels // 2, kernel_size=1,
                            norm_layer=norm_layer, activation='leaky_relu')
        self.conv2 = Conv2d(out_channels // 2, out_channels, kernel_size=3,
                            norm_layer=norm_layer, activation='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class YOLOv3(nn.Module):
    def __init__(self, backbone, num_anchors=3, num_classes=80, norm_layer='bn'):
        super().__init__()
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)
        self.backbone = backbone
        # self.backbone = Darknet(f_channels=f_channels)
        channels = backbone.out_channels
        self.conv51 = nn.Sequential(
            Bottleneck(channels[-1], channels[-1]),
            Bottleneck(channels[-1], channels[-1]),
            Conv2d(channels[-1], channels[-1] // 2, kernel_size=1,
                   norm_layer=norm_layer, activation='leaky_relu'),
        )
        self.conv52 = Conv2d(channels[-1] // 2, channels[-1], kernel_size=3,
                             norm_layer=norm_layer, activation='leaky_relu')
        self.pred5 = Conv2d(channels[-1], out_channels, kernel_size=1)

        self.lat5 = Conv2d(channels[-1] // 2, channels[-1] // 4, kernel_size=1)

        self.conv41 = nn.Sequential(
            Bottleneck(channels[-2] + channels[-1] // 4, channels[-2]),
            Bottleneck(channels[-2], channels[-2]),
            Conv2d(channels[-2], channels[-2] // 2, kernel_size=1,
                   norm_layer=norm_layer, activation='leaky_relu'),
        )
        self.conv42 = Conv2d(channels[-2] // 2, channels[-2], kernel_size=3,
                             norm_layer=norm_layer, activation='leaky_relu')
        self.pred4 = Conv2d(channels[-2], out_channels, kernel_size=1)

        self.lat4 = Conv2d(channels[-2] // 2, channels[-2] // 4, kernel_size=1)

        self.conv31 = nn.Sequential(
            Bottleneck(channels[-3] + channels[-2] // 4, channels[-3]),
            Bottleneck(channels[-3], channels[-3]),
            Conv2d(channels[-3], channels[-3] // 2, kernel_size=1,
                   norm_layer=norm_layer, activation='leaky_relu'),
        )
        self.conv32 = Conv2d(channels[-3] // 2, channels[-3], kernel_size=3,
                             norm_layer=norm_layer, activation='leaky_relu')
        self.pred3 = Conv2d(channels[-3], out_channels, kernel_size=1)

    def forward(self, x):
        b = x.size(0)
        c3, c4, c5 = self.backbone(x)

        p51 = self.conv51(c5)
        p52 = self.conv52(p51)
        p5 = self.pred5(p52)

        p41 = upsample_concat(self.lat5(p51), c4)
        p42 = self.conv41(p41)
        p43 = self.conv42(p42)
        p4 = self.pred4(p43)

        p31 = upsample_concat(self.lat4(p42), c3)
        p32 = self.conv31(p31)
        p33 = self.conv32(p32)
        p3 = self.pred3(p33)

        preds = [p3, p4, p5]

        loc_preds = []
        obj_preds = []
        cls_preds = []
        for p in preds:
            p = p.permute(0, 3, 2, 1).contiguous().view(b, -1, 5 + self.num_classes)
            loc_preds.append(p[..., :4])
            obj_preds.append(p[..., 5])
            cls_preds.append(p[..., 5:])
        return loc_preds, obj_preds, cls_preds


def inverse_sigmoid(x):
    x = min(max(x, .01), .99)
    return math.log(x / (1 - x))


def match_anchors(anns, anchors_xywh_of_levels, anchors_ltrb_of_levels, ignore_thresh=None,
                  get_label=lambda x: x['category_id'], debug=False):
    loc_targets = []
    cls_targets = []
    ignores = []
    locations = []
    as_xywh = []
    as_ltrb = []
    for a_xywh, a_ltrb in zip(anchors_xywh_of_levels, anchors_ltrb_of_levels):
        lx, ly, num_priors = a_xywh.shape[:3]
        num_anchors = lx * ly * num_priors
        loc_targets.append(torch.zeros(num_anchors, 4))
        cls_targets.append(torch.zeros(num_anchors, dtype=torch.long))
        ignores.append(torch.zeros(num_anchors, dtype=torch.uint8))
        locations.append((lx, ly))
        as_xywh.append(a_xywh.view(-1, 4))
        as_ltrb.append(a_ltrb.view(-1, 4))

    for ann in anns:
        label = get_label(ann)
        l, t, w, h = ann['bbox']
        x = l + w / 2
        y = t + h / 2
        bbox = torch.tensor([x, y, w, h])
        bbox_ltrb = BBox.convert(bbox, BBox.XYWH, BBox.LTRB)

        max_ious = []
        for a_ltrb, loc_t, cls_t, ignore in zip(as_ltrb, loc_targets, cls_targets, ignores):
            ious = iou_1m(bbox_ltrb, a_ltrb)
            max_ious.append(ious.max(dim=0))

            ignore |= ious > ignore_thresh

        f_i, (max_iou, i) = max(
            enumerate(max_ious), key=lambda x: x[1][0])
        if debug:
            print("%d: %f" % (f_i, max_iou))
        lx, ly = locations[f_i]
        loc_targets[f_i][i, 0] = inverse_sigmoid(x * lx % 1)
        loc_targets[f_i][i, 1] = inverse_sigmoid(y * ly % 1)
        loc_targets[f_i][i, 2:] = (bbox[2:] / as_xywh[f_i][i, 2:]).log()
        cls_targets[f_i][i] = label

    loc_t = torch.cat(loc_targets, dim=0)
    cls_t = torch.cat(cls_targets, dim=0)
    ignores = torch.cat(ignores, dim=0)
    return loc_t, cls_t, ignores


class YOLOTransform:

    def __init__(self, anchors_of_level, ignore_threshold=0.5, get_label=lambda x: x["category_id"], debug=False):
        self.anchors_xywh_of_levels = anchors_of_level
        self.anchors_ltrb_of_levels = [
            BBox.convert(anchors, BBox.XYWH, BBox.LTRB) for anchors in anchors_of_level
        ]
        self.ignore_threshold = ignore_threshold
        self.get_label = get_label
        self.debug = debug

    def __call__(self, img, anns):
        target = match_anchors(
            anns, self.anchors_xywh_of_levels, self.anchors_ltrb_of_levels,
            self.ignore_threshold, self.get_label, self.debug)
        return img, target


class YOLOLoss(nn.Module):
    def __init__(self, p=0.01):
        super().__init__()
        self.p = p

    def forward(self, loc_preds, obj_preds, cls_preds, loc_t, cls_t, ignore):
        b = loc_preds[0].size(0)
        num_classes = cls_preds[0].size(-1)
        loc_p = torch.cat([ p.view(b, -1, 4) for p in loc_preds ], dim=1)
        obj_p = torch.cat([p.view(b, -1) for p in obj_preds], dim=1)
        cls_p = torch.cat([p.view(b, -1, num_classes) for p in cls_preds], dim=1)

        pos = cls_t != 0
        num_pos = pos.sum().item()

        obj_p_pos = obj_p[pos]
        obj_loss_pos = F.binary_cross_entropy_with_logits(
            obj_p_pos, torch.ones_like(obj_p_pos), reduction='sum'
        )
        obj_p_neg = obj_p[~pos & ~ignore]
        obj_loss_neg = F.binary_cross_entropy_with_logits(
            obj_p_neg, torch.zeros_like(obj_p_neg), reduction='sum'
        )

        loc_loss = F.mse_loss(
            loc_p[pos], loc_t[pos], reduction='sum')

        cls_t = one_hot(cls_t, num_classes + 1)[..., 1:]
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_p[pos], cls_t[pos], reduction='sum')

        obj_loss_neg = 0.5 * obj_loss_neg
        # loc_loss = 5 * loc_loss
        loss = (obj_loss_pos + obj_loss_neg + loc_loss + cls_loss) / num_pos
        if random.random() < self.p:
            print("pos: %.4f | neg: %.4f | loc: %.4f | cls: %.4f" %
                  (obj_loss_pos.item() / num_pos,
                   obj_loss_neg.item() / num_pos,
                   loc_loss.item() / num_pos,
                   cls_loss.item() / num_pos))
        return loss


class YOLOInference:

    def __init__(self, anchors_of_level, conf_threshold=0.5, iou_threshold=0.5):
        self.anchors_of_level = anchors_of_level
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, loc_preds, obj_preds, cls_preds):
        dets = []

        scores = []
        boxes = []
        labels = []
        for loc_p, obj_p, cls_p, anchors in zip(loc_preds, obj_preds, cls_preds, self.anchors_of_level):
            lx, ly, num_priors = anchors.shape
            score = loc_p.sigmoid_()
            mask = score > self.conf_threshold
            score = score[mask]
            box = loc_p[mask]
            label = cls_p[mask].argmax(dim=1)
            anchors = anchors.view(-1, 4)[mask]

            box[..., :2].sigmoid_().div_(box.new_tensor([lx, ly])).add_(anchors[:, :2])
            box[..., 2:].exp_().mul_(anchors[:, 2:])

            scores.append(score)
            boxes.append(box)
            labels.append(label)

        scores = torch.cat(scores, dim=0)
        boxes = torch.cat(boxes, dim=0)
        labels = torch.cat(labels, dim=0)

        boxes = BBox.convert(boxes, BBox.XYWH, BBox.LTRB, inplace=True)
        indices = soft_nms_cpu(boxes, scores, self.iou_threshold)
        boxes = BBox.convert(
            boxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)
        for ind in indices:
            det = {
                'image_id': -1,
                'category_id': labels[ind].item() + 1,
                'bbox': boxes[ind].tolist(),
                'score': scores[ind].item(),
            }
            dets.append(det)
        return dets