import random
from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import one_hot, _tuple
from horch.detection import soft_nms_cpu, BBox, nms_cpu
from horch.transforms.detection.functional import to_percent_coords
from horch.nn.loss import focal_loss2, iou_loss
from horch.models.detection.head import FCOSHead
from horch.models.detection import OneStageDetector

__all__ = [
    "get_mlvl_centers", "FCOSTransform", "FCOSHead", "FCOSInference", "FCOS", "FCOSLoss"
]


def get_mlvl_centers(locations, strides):
    mlvl_centers = []
    for location, stride in zip(locations, strides):
        lx, ly = _tuple(location, 2)
        sw, sh = _tuple(stride, 2)
        centers = torch.zeros(lx, ly, 2)
        centers[..., 0] = (torch.arange(
            lx, dtype=torch.float).view(lx, 1).expand(lx, ly) * sw + sw // 2)
        centers[..., 1] = (torch.arange(
            ly, dtype=torch.float).view(1, ly).expand(lx, ly) * sh + sh // 2)
        mlvl_centers.append(centers)
    return mlvl_centers


def centerness(loc_t):
    l = loc_t[..., 0]
    t = loc_t[..., 1]
    r = loc_t[..., 2]
    b = loc_t[..., 3]
    c = (torch.min(l, r) / torch.max(l, r)) * \
        (torch.min(t, b) / torch.max(t, b))
    c = c.sqrt_()
    return c


class FCOSTransform:

    def __init__(self, mlvl_centers, thresholds=(0, 64, 128, 256, 512, inf), get_label=lambda x: x["category_id"]):
        self.mlvl_centers = mlvl_centers
        self.thresholds = list(zip(thresholds[:-1], thresholds[1:]))
        self.get_label = get_label

    def __call__(self, img, anns):
        loc_targets = []
        cls_targets = []
        ctn_targets = []
        for centers in self.mlvl_centers:
            lx, ly = centers.size()[:2]
            loc_targets.append(torch.zeros(lx, ly, 4))
            cls_targets.append(torch.zeros(lx, ly, dtype=torch.long))
            ctn_targets.append(torch.zeros(lx, ly))

        for ann in anns:
            label = self.get_label(ann)
            l, t, w, h = ann['bbox']
            r = l + w
            b = t + h
            for centers, threshold, loc_t, cls_t, ctn_t in zip(self.mlvl_centers, self.thresholds, loc_targets,
                                                               cls_targets, ctn_targets):
                lo, hi = threshold
                cx = centers[..., 0]
                cy = centers[..., 1]
                mask = (l < cx) & (cx < r) & (t < cy) & (cy < b)
                ts = torch.stack([cx - l, cy - t, r - cx, b - cy], dim=-1)
                max_t = ts.max(dim=-1)[0]
                mask &= (max_t >= lo) & (max_t < hi)
                if not mask.any():
                    continue
                loc_t[mask] = ts[mask]
                cls_t[mask] = label
                ctn_t[mask] = centerness(loc_t[mask])

        loc_t = torch.cat([t.view(-1, 4) for t in loc_targets], dim=0)
        cls_t = torch.cat([t.view(-1) for t in cls_targets], dim=0)
        ctn_t = torch.cat([t.view(-1) for t in ctn_targets], dim=0)

        return img, [loc_t, cls_t, ctn_t]


class FCOSLoss(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, loc_p, cls_p, ctn_p, loc_t, cls_t, ctn_t):
        loc_p = loc_p.exp()
        pos = cls_t != 0
        num_pos = pos.sum().item()
        if num_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)
        loc_loss = iou_loss(loc_p[pos], loc_t[pos], reduction='sum') / num_pos
        cls_t = one_hot(cls_t, C=cls_p.size(-1))
        cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / num_pos
        ctn_loss = F.binary_cross_entropy_with_logits(ctn_p[pos], ctn_t[pos], reduction='sum') / num_pos
        loss = loc_loss + cls_loss + ctn_loss
        if random.random() < self.p:
            print("loc: %.4f | cls: %.4f | ctn: %.4f" %
                  (loc_loss.item(), cls_loss.item(), ctn_loss.item()))
        return loss


def center_based_inference(
        size, loc_p, cls_p, ctn_p, centers, conf_threshold=0.01,
        iou_threshold=0.5, topk=100, nms='soft_nms', soft_nms_threshold=None, use_centerness=True):
    dets = []
    bboxes = loc_p.exp_()
    scores, labels = cls_p[:, 1:].max(dim=-1)
    scores = scores.sigmoid_()
    if use_centerness:
        centerness = ctn_p.sigmoid_()
        scores = scores.mul_(centerness)

    if conf_threshold > 0:
        mask = scores > conf_threshold
        scores = scores[mask]
        labels = labels[mask]
        bboxes = bboxes[mask]
        centers = centers[mask]

    cx = centers[:, 0]
    cy = centers[:, 1]

    bboxes[:, 0] = cx - bboxes[:, 0]
    bboxes[:, 1] = cy - bboxes[:, 1]
    bboxes[:, 2] = cx + bboxes[:, 2]
    bboxes[:, 3] = cy + bboxes[:, 3]

    scores = scores.cpu()
    bboxes = bboxes.cpu()

    if nms == 'nms':
        indices = nms_cpu(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk:
            indices = scores.topk(topk)[1]
        else:
            indices = range(scores.size(0))
    else:
        if soft_nms_threshold is None:
            soft_nms_threshold = conf_threshold
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, conf_threshold=soft_nms_threshold)
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)
    for ind in indices:
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
            'absolute': True,
        }
        dets.append(det)
    dets = to_percent_coords(dets, size)
    return dets


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return torch.cat(xs, dim=0)


class FCOSInference:

    def __init__(self, size, mlvl_centers, conf_threshold=0.05, iou_threshold=0.5, topk=100, nms='nms',
                 soft_nms_threshold=None, use_centerness=True):
        self.size = size
        self.centers = flatten(mlvl_centers)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.nms = nms
        self.soft_nms_threshold = soft_nms_threshold
        self.use_centerness = use_centerness

    def __call__(self, loc_p, cls_p, ctn_p):
        image_dets = []
        batch_size = loc_p.size(0)
        for i in range(batch_size):
            dets = center_based_inference(
                self.size, loc_p[i], cls_p[i], ctn_p[i], self.centers,
                self.conf_threshold, self.iou_threshold,
                self.topk, self.nms, self.soft_nms_threshold, self.use_centerness
            )
            image_dets.append(dets)
        return image_dets


class FCOS(OneStageDetector):

    def forward(self, x):
        loc_p, cls_p = super().forward(x)
        ctn_p = cls_p[..., -1]
        cls_p = cls_p[..., :-1]
        return loc_p, cls_p, ctn_p
