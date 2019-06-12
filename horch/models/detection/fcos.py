import random
from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import one_hot, _tuple
from horch.detection import soft_nms_cpu, BBox, nms, calc_grid_sizes
from horch.detection.anchor.generator import AnchorGeneratorBase
from horch.detection.one import flatten, flatten_preds
from horch.models.detection.head import RetinaHead, to_pred
from horch.transforms.detection.functional import to_percent_coords
from horch.nn.loss import focal_loss2, iou_loss

__all__ = ["FCOSMatchAnchors", "FCOSAnchorGenerator", "FCOSHead", "FCOSMatcher", "FCOSLoss"]


class FCOSHead(RetinaHead):
    def __init__(self, num_levels, num_classes, f_channels=256, num_layers=4, lite=False):
        super().__init__(1, num_classes + 1, f_channels, num_layers, lite)
        start = 1 - (num_levels - 1) * 0.1 / 2
        scales = [start + i / 10 for i in range(num_levels)]
        self.scales = nn.Parameter(torch.tensor(scales))

    def forward(self, *ps):
        loc_preds = []
        cls_preds = []
        ctn_preds = []
        for i, p in enumerate(ps):
            loc_p = to_pred(self.loc_head(p), 4)
            loc_preds.append(loc_p * self.scales[i])

            cls_p = to_pred(self.cls_head(p), self.num_classes)
            cls_preds.append(cls_p[..., :-1])
            ctn_preds.append(cls_p[..., -1:])
        return loc_preds, cls_preds, ctn_preds


__all__ = [
    "get_mlvl_centers", "FCOSInference", "FCOSLoss"
]


def get_mlvl_centers(grid_sizes, strides, device='cpu', dtype=torch.float32):
    mlvl_centers = []
    for (lx, ly), stride in zip(grid_sizes, strides):
        sw, sh = _tuple(stride, 2)
        centers = torch.zeros(lx, ly, 2, device=device, dtype=dtype)
        centers[..., 0] = (torch.arange(
            lx, device=device, dtype=dtype).view(lx, 1).expand(lx, ly) * sw + sw // 2)
        centers[..., 1] = (torch.arange(
            ly, device=device, dtype=dtype).view(1, ly).expand(lx, ly) * sh + sh // 2)
        mlvl_centers.append(centers)
    return mlvl_centers


class FCOSAnchorGenerator(AnchorGeneratorBase):
    def __init__(self, levels, cache=True):
        super().__init__(cache)
        self.levels = levels
        self.strides = [2 ** l for l in levels]

    def calculate(self, grid_sizes, device, dtype):
        mlvl_centers = get_mlvl_centers(grid_sizes, self.strides, device, dtype)
        centers_flat = flatten(mlvl_centers)
        ret = {
            "centers": mlvl_centers,
            "flat": centers_flat,
        }
        return ret


def centerness(loc_t):
    l = loc_t[..., 0]
    t = loc_t[..., 1]
    r = loc_t[..., 2]
    b = loc_t[..., 3]
    c = (torch.min(l, r) / torch.max(l, r)) * \
        (torch.min(t, b) / torch.max(t, b))
    c = c.sqrt_()
    return c


def match_anchors(anns, mlvl_centers, thresholds, get_label):
    loc_targets = []
    cls_targets = []
    ctn_targets = []
    for centers in mlvl_centers:
        lx, ly = centers.size()[:2]
        loc_targets.append(centers.new_zeros((lx, ly, 4)))
        cls_targets.append(centers.new_zeros((lx, ly), dtype=torch.long))
        ctn_targets.append(centers.new_zeros((lx, ly)))

    for ann in anns:
        label = get_label(ann)
        l, t, w, h = ann['bbox']
        r = l + w
        b = t + h
        for centers, threshold, loc_t, cls_t, ctn_t in zip(mlvl_centers, thresholds, loc_targets,
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

    return loc_t, cls_t, ctn_t


class FCOSMatchAnchors:

    def __init__(self, generator, thresholds=(0, 64, 128, 256, 512, inf), get_label=lambda x: x["category_id"]):
        assert len(generator.levels) == len(thresholds) - 1
        self.generator = generator
        self.strides = self.generator.strides
        self.thresholds = list(zip(thresholds[:-1], thresholds[1:]))
        self.get_label = get_label

    def __call__(self, x, anns):
        height, width = x.shape[1:3]
        grid_sizes = calc_grid_sizes((width, height), self.strides)
        grid_sizes = [torch.Size(s) for s in grid_sizes]
        mlvl_centers = self.generator(grid_sizes, x.device, x.dtype)["centers"]
        targets = match_anchors(anns, mlvl_centers, self.thresholds, self.get_label)
        return x, targets


class FCOSMatcher:

    def __init__(self, generator, thresholds=(0, 64, 128, 256, 512, inf), get_label=lambda x: x["category_id"]):
        self.generator = generator
        self.thresholds = list(zip(thresholds[:-1], thresholds[1:]))
        self.get_label = get_label

    def __call__(self, features, targets):
        batch_size = len(targets)
        grid_sizes = [f.size()[-2:][::-1] for f in features]
        mlvl_centers = self.generator(grid_sizes, features[0].device, features[0].dtype)["centers"]
        loc_targets = []
        cls_targets = []
        ctn_targets = []
        for i in range(batch_size):
            loc_t, cls_t, ctn_t = self.match_single(targets[i], mlvl_centers)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            ctn_targets.append(ctn_t)
        loc_t = torch.stack(loc_targets, dim=0)
        cls_t = torch.stack(cls_targets, dim=0)
        ctn_t = torch.stack(ctn_targets, dim=0)
        return loc_t, cls_t, ctn_t

    def match_single(self, anns, mlvl_centers):
        return match_anchors(anns, mlvl_centers, self.thresholds, self.get_label)


class FCOSLoss(nn.Module):
    def __init__(self, loc_loss='iou', use_ctn=True, p=0.1):
        super().__init__()
        self.loc_loss = loc_loss
        self.use_ctn = use_ctn
        self.p = p

    def forward(self, loc_p, cls_p, ctn_p, loc_t, cls_t, ctn_t):
        loc_p, cls_p, ctn_p = flatten_preds(loc_p, cls_p, ctn_p)
        pos = cls_t != 0
        num_pos = pos.sum().item()
        if num_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)

        if self.loc_loss == 'iou':
            loc_p = loc_p.exp() * 4
            loc_loss = iou_loss(loc_p[pos], loc_t[pos], reduction='sum') / num_pos
        else:
            loc_t = loc_t.div_(4).log_()
            loc_loss = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction='sum') / num_pos

        cls_t = one_hot(cls_t, C=cls_p.size(-1))
        cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / num_pos
        if self.use_ctn:
            ctn_loss = F.binary_cross_entropy_with_logits(ctn_p[pos], ctn_t[pos], reduction='sum') / num_pos
            loss = loc_loss + cls_loss + ctn_loss
            if random.random() < self.p:
                print("loc: %.4f | cls: %.4f | ctn: %.4f" %
                      (loc_loss.item(), cls_loss.item(), ctn_loss.item()))
        else:
            loss = loc_loss + cls_loss
            if random.random() < self.p:
                print("loc: %.4f | cls: %.4f" %
                      (loc_loss.item(), cls_loss.item()))
        return loss


def center_based_inference(
        size, loc_p, cls_p, ctn_p, centers, conf_threshold=0.01,
        iou_threshold=0.5, topk=100, nms_method='soft_nms', use_ctn=True):
    dets = []
    bboxes = loc_p.exp_().mul_(4)
    scores, labels = cls_p[:, 1:].max(dim=-1)
    scores = scores.sigmoid_()
    if use_ctn:
        ctn_scores = ctn_p.sigmoid_()
        scores = scores.mul_(ctn_scores)

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
            bboxes, scores, iou_threshold, topk, min_score=conf_threshold)
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


class FCOSInference:

    def __init__(self, generator, conf_threshold=0.05, iou_threshold=0.5, topk=100, nms='nms', use_ctn=True):
        self.generator = generator
        self.levels = self.generator.levels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.nms = nms
        self.use_ctn = use_ctn

    def __call__(self, loc_preds, cls_preds, ctn_preds):
        grid_sizes = [p.size()[1:3] for p in loc_preds]
        centers = self.generator(grid_sizes, loc_preds[0].device, loc_preds[0].dtype)["flat"]
        stride = 2 ** self.levels[0]
        size = tuple(s * stride for s in grid_sizes[0])
        loc_p, cls_p, ctn_p = flatten_preds(loc_preds, cls_preds, ctn_preds)
        batch_size = loc_p.size(0)
        image_dets = []
        for i in range(batch_size):
            dets = center_based_inference(
                size, loc_p[i], cls_p[i], ctn_p[i], centers,
                self.conf_threshold, self.iou_threshold,
                self.topk, self.nms, self.use_ctn
            )
            image_dets.append(dets)
        return image_dets
