from math import log
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import one_hot, inverse_sigmoid
from horch.detection.anchor.generator import AnchorGeneratorBase
from horch.detection.one import flatten, flatten_preds
from horch.models.detection.head import to_pred
from horch.models.modules import upsample_concat, Conv2d
from horch.models.utils import get_last_conv, bias_init_constant
from horch.nn.loss import focal_loss2

from horch.detection import BBox, soft_nms_cpu, nms, generate_mlvl_anchors, calc_grid_sizes
from toolz.curried import get


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lite=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels // 2, kernel_size=1,
                            norm_layer='default', activation='default')
        self.conv2 = Conv2d(out_channels // 2, out_channels, kernel_size=3,
                            norm_layer='default', activation='default', depthwise_separable=lite)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class YOLOv3(nn.Module):
    def __init__(self, in_channels, num_anchors=3, num_classes=80, lite=False):
        super().__init__()
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)
        channels = in_channels
        self.conv51 = nn.Sequential(
            BasicBlock(channels[-1], channels[-1], lite=lite),
            BasicBlock(channels[-1], channels[-1], lite=lite),
            Conv2d(channels[-1], channels[-1] // 2, kernel_size=1,
                   norm_layer='default', activation='default'),
        )
        self.conv52 = Conv2d(channels[-1] // 2, channels[-1], kernel_size=3,
                             norm_layer='default', activation='default', depthwise_separable=lite)
        self.pred5 = Conv2d(channels[-1], out_channels, kernel_size=1)

        self.lat5 = Conv2d(channels[-1] // 2, channels[-1] // 4, kernel_size=1,
                           norm_layer='default')

        self.conv41 = nn.Sequential(
            BasicBlock(channels[-2] + channels[-1] // 4, channels[-2], lite=lite),
            BasicBlock(channels[-2], channels[-2], lite=lite),
            Conv2d(channels[-2], channels[-2] // 2, kernel_size=1,
                   norm_layer='default', activation='default'),
        )
        self.conv42 = Conv2d(channels[-2] // 2, channels[-2], kernel_size=3,
                             norm_layer='default', activation='default', depthwise_separable=lite)
        self.pred4 = Conv2d(channels[-2], out_channels, kernel_size=1)

        self.lat4 = Conv2d(channels[-2] // 2, channels[-2] // 4, kernel_size=1,
                           norm_layer='default')

        self.conv31 = nn.Sequential(
            BasicBlock(channels[-3] + channels[-2] // 4, channels[-3], lite=lite),
            BasicBlock(channels[-3], channels[-3], lite=lite),
            Conv2d(channels[-3], channels[-3] // 2, kernel_size=1,
                   norm_layer='default', activation='default'),
        )
        self.conv32 = Conv2d(channels[-3] // 2, channels[-3], kernel_size=3,
                             norm_layer='default', activation='default', depthwise_separable=lite)
        self.pred3 = Conv2d(channels[-3], out_channels, kernel_size=1)

        get_last_conv(self.pred3).bias.data[4].fill_(inverse_sigmoid(0.01))
        get_last_conv(self.pred3).bias.data[4].fill_(inverse_sigmoid(0.01))
        get_last_conv(self.pred3).bias.data[4].fill_(inverse_sigmoid(0.01))

    def forward(self, c3, c4, c5):
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
            p = to_pred(p, 5 + self.num_classes)
            loc_preds.append(p[..., :4])
            obj_preds.append(p[..., 4:5])
            cls_preds.append(p[..., 5:])
        return loc_preds, obj_preds, cls_preds


def iou_1m_with_size(box, boxes):
    box = box.expand_as(boxes)
    inter_sizes = torch.min(box, boxes)
    inter_areas = inter_sizes[..., 0] * inter_sizes[..., 1]
    areas1 = box[..., 0] * box[..., 1]
    areas2 = boxes[..., 0] * boxes[..., 1]
    union_areas = areas1 + areas2 - inter_areas
    ious = inter_areas / union_areas
    return ious


def match_anchors(anns, mlvl_priors, grid_sizes, ignore_thresh=None,
                  get_label=lambda x: x['category_id']):
    loc_targets = []
    cls_targets = []
    ignores = []
    num_levels, priors_per_level = mlvl_priors.size()[:2]
    for (lx, ly), priors in zip(grid_sizes, mlvl_priors):
        loc_targets.append(
            priors.new_zeros((lx, ly, priors_per_level, 4)))
        cls_targets.append(
            priors.new_zeros((lx, ly, priors_per_level), dtype=torch.long))
        ignores.append(
            priors.new_zeros((lx, ly, priors_per_level), dtype=torch.uint8))

    for ann in anns:
        l, t, w, h = ann['bbox']
        x = l + w / 2
        y = t + h / 2
        size = mlvl_priors.new_tensor([w, h])
        ious = iou_1m_with_size(size, mlvl_priors)
        max_iou, max_ind = ious.view(-1).max(dim=0)
        level, i = divmod(max_ind.item(), priors_per_level)
        # if debug:
        #     print("[%d,%d]: %.4f" % (level, i, max_iou.item()))
        lx, ly = grid_sizes[level]
        pw, ph = mlvl_priors[level, i]
        cx, offset_x = divmod(x * lx, 1)
        cx = int(cx)
        cy, offset_y = divmod(y * ly, 1)
        cy = int(cy)
        tx = inverse_sigmoid(offset_x)
        ty = inverse_sigmoid(offset_y)
        tw = log(w / pw)
        th = log(h / ph)
        loc_targets[level][cx, cy, i] = mlvl_priors.new_tensor([tx, ty, tw, th])
        cls_targets[level][cx, cy, i] = get_label(ann)
        ignore = ious > ignore_thresh
        for level, i in torch.nonzero(ignore):
            lx, ly = grid_sizes[level]
            cx, offset_x = divmod(x * lx, 1)
            cx = int(cx)
            cy, offset_y = divmod(y * ly, 1)
            cy = int(cy)
            ignores[level][cx, cy, i] = 1

    loc_t = torch.cat([t.view(-1, 4) for t in loc_targets], dim=0)
    cls_t = torch.cat([t.view(-1) for t in cls_targets], dim=0)
    ignore = torch.cat([t.view(-1) for t in ignores], dim=0)
    return loc_t, cls_t, ignore


class YOLOMatchAnchors:

    def __init__(self, mlvl_priors, levels, ignore_thresh=0.5, get_label=lambda x: x["category_id"]):
        self.mlvl_priors = mlvl_priors
        self.levels = levels
        self.strides = [2 ** l for l in levels]
        self.ignore_thresh = ignore_thresh
        self.get_label = get_label

    def __call__(self, x, anns):
        height, width = x.shape[1:3]
        grid_sizes = calc_grid_sizes((width, height), self.strides)
        grid_sizes = [torch.Size(s) for s in grid_sizes]
        targets = match_anchors(
            anns, self.mlvl_priors, grid_sizes, self.ignore_thresh, self.get_label)
        return x, targets


class YOLOAnchorMatcher:

    def __init__(self, mlvl_priors, ignore_thresh=0.5, get_label=lambda x: x["category_id"]):
        self.mlvl_priors = mlvl_priors
        self.ignore_thresh = ignore_thresh
        self.get_label = get_label

    def __call__(self, features, targets):
        batch_size = len(targets)
        grid_sizes = [f.size()[-2:][::-1] for f in features]
        loc_targets = []
        cls_targets = []
        ignores = []
        for i in range(batch_size):
            loc_t, cls_t, ignore = self.match_single(targets[i], grid_sizes)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            ignores.append(ignore)
        loc_t = torch.stack(loc_targets, dim=0)
        cls_t = torch.stack(cls_targets, dim=0)
        ignore = torch.stack(ignores, dim=0)
        return loc_t, cls_t, ignore

    def match_single(self, anns, grid_sizes):
        return match_anchors(anns, self.mlvl_priors, grid_sizes, self.ignore_thresh, self.get_label)


class YOLOLoss(nn.Module):
    def __init__(self, p=0.01, obj_loss='bce', neg_gain=1, loc_gain=0.5):
        super().__init__()
        self.p = p
        self.obj_loss = obj_loss
        self.neg_gain = neg_gain
        self.loc_gain = loc_gain

    def forward(self, loc_p, obj_p, cls_p, loc_t, cls_t, ignore):
        loc_p, obj_p, cls_p = flatten_preds(loc_p, obj_p, cls_p)

        pos = cls_t != 0
        num_pos = pos.sum().item()

        criterion = focal_loss2 if self.obj_loss == 'focal' else F.binary_cross_entropy_with_logits
        neg_gain = 1 if self.obj_loss == 'focal' else self.neg_gain

        obj_p_pos = obj_p[pos]
        obj_loss_pos = criterion(
            obj_p_pos, torch.ones_like(obj_p_pos), reduction='sum'
        ) / num_pos
        obj_p_neg = obj_p[~pos & ~ignore]
        obj_loss_neg = neg_gain * criterion(
            obj_p_neg, torch.zeros_like(obj_p_neg), reduction='sum'
        ) / num_pos

        obj_loss = obj_loss_pos + obj_loss_neg

        loc_loss = self.loc_gain * F.mse_loss(
            loc_p[pos], loc_t[pos], reduction='sum') / num_pos

        cls_t = one_hot(cls_t, cls_p.size(-1) + 1)[..., 1:]
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_p[pos], cls_t[pos], reduction='sum') / num_pos

        loss = obj_loss + loc_loss + cls_loss
        if random.random() < self.p:
            print("pos: %.4f | neg: %.4f | loc: %.4f | cls: %.4f" %
                  (obj_loss_pos.item(), obj_loss_neg.item(), loc_loss.item(), cls_loss.item()))
        return loss


def get_locations(mlvl_anchors):
    mlvl_locations = []
    for anchors in mlvl_anchors:
        lx, ly, num_priors = anchors.size()[:3]
        num_anchors = lx * ly * num_priors
        locations = anchors.new_empty((num_anchors, 2))
        locations[:, 0] = lx
        locations[:, 1] = ly
        mlvl_locations.append(locations)
    locations = torch.cat(mlvl_locations, dim=0)
    return locations


class YOLOAnchorGenerator(AnchorGeneratorBase):
    def __init__(self, anchor_sizes, cache=True):
        super().__init__(cache)
        self.anchor_sizes = anchor_sizes

    def calculate(self, grid_sizes, device, dtype):
        mlvl_anchors = generate_mlvl_anchors(
            grid_sizes, self.anchor_sizes, device, dtype)
        locations = get_locations(mlvl_anchors)
        anchors = flatten(mlvl_anchors)
        ret = {
            "anchors": anchors,
            "locations": locations
        }
        return ret


def yolo_inference(
        loc_p, obj_p, cls_p, anchors, locations, conf_threshold=0.01,
        iou_threshold=0.5, topk=100, nms_method='soft'):
    bboxes = loc_p.view(-1, 4)
    objectness = obj_p.view(-1).sigmoid_()
    logits = cls_p.view(-1, cls_p.size(-1))
    scores, labels = torch.sigmoid_(logits).max(dim=1)
    scores *= objectness

    if conf_threshold > 0:
        pos = scores > conf_threshold
        scores = scores[pos]
        labels = labels[pos]
        bboxes = bboxes[pos]
        anchors = anchors[pos]
        locations = locations[pos]

    bboxes[..., :2].sigmoid_().sub_(0.5).div_(locations).add_(anchors[:, :2])
    bboxes[..., 2:].exp_().mul_(anchors[:, 2:])

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms_method == 'nms':
        indices = nms(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk:
            indices = scores.topk(topk)[1]
        else:
            indices = range(scores.size(0))
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, min_score=0.01)
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)
    dets = []
    for ind in indices:
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
        }
        dets.append(det)
    return dets


class YOLOInference:

    def __init__(self, generator, conf_threshold=0.5,
                 iou_threshold=0.5, topk=100, nms='soft'):
        assert isinstance(generator, YOLOAnchorGenerator), \
            "Generator must be YOLOAnchorGenerator, got %s" % type(generator)
        self.generator = generator
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.nms = nms

    def __call__(self, loc_preds, obj_preds, cls_preds):
        grid_sizes = [p.size()[1:3] for p in loc_preds]
        anchors, locations = get(
            ["anchors", "locations"],
            self.generator(grid_sizes, loc_preds[0].device, loc_preds[0].dtype))
        loc_p, obj_p, cls_p = flatten_preds(loc_preds, obj_preds, cls_preds)
        batch_size = loc_p.size(0)
        image_dets = [
            self.inference_single(loc_p[i], obj_p[i], cls_p[i], anchors, locations)
            for i in range(batch_size)
        ]
        return image_dets

    def inference_single(self, loc_p, obj_p, cls_p, anchors, locations):
        return yolo_inference(
            loc_p, obj_p, cls_p, anchors, locations,
            self.conf_threshold, self.iou_threshold,
            self.topk, self.nms,
        )
