import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import one_hot, _tuple, _concat
from horch.detection import soft_nms_cpu, BBox, nms, calc_grid_sizes
from horch.detection.one import flatten_preds
from horch.detection.anchor.generator import AnchorGeneratorBase
from horch.nn.loss import focal_loss2


def get_fovea(cx, cy, w, h, shrunk=0.3):
    l_f = cx - 0.5 * w * shrunk
    t_f = cy - 0.5 * h * shrunk
    r_f = cx + 0.5 * w * shrunk
    b_f = cy + 0.5 * h * shrunk
    return l_f, t_f, r_f, b_f


DEFAULT_AREA_THRESHOLDS = tuple(
    (lo ** 2, hi ** 2)
    for (lo, hi) in
    ((16, 64), (32, 128), (64, 256), (128, 512), (256, 1024))
)


def get_mlvl_centers(grid_sizes, device='cpu', dtype=torch.float32):
    mlvl_centers = []
    for size in grid_sizes:
        lx, ly = size
        centers = torch.zeros(lx, ly, 2, dtype=dtype, device=device)
        centers[..., 0] = torch.arange(
            lx, dtype=dtype, device=device).view(lx, 1).expand(lx, ly) + 0.5
        centers[..., 1] = torch.arange(
            ly, dtype=dtype, device=device).view(1, ly).expand(lx, ly) + 0.5
        mlvl_centers.append(centers)
    return mlvl_centers


class FoveaAnchorGenerator(AnchorGeneratorBase):
    def __init__(self, cache=True):
        super().__init__(cache)

    def calculate(self, grid_sizes, device, dtype):
        mlvl_centers = get_mlvl_centers(grid_sizes, device, dtype)
        ret = {
            "centers": mlvl_centers
        }
        return ret


def match_anchors(anns, mlvl_centers, levels, thresholds, shrunk_pos, shrunk_neg, get_label):
    loc_targets = []
    cls_targets = []
    ignores = []
    for centers in mlvl_centers:
        lx, ly = centers.size()[:2]
        loc_targets.append(centers.new_zeros((lx, ly, 4)))
        cls_targets.append(centers.new_zeros((lx, ly), dtype=torch.long))
        ignores.append(centers.new_zeros((lx, ly), dtype=torch.uint8))

    for ann in anns:
        label = get_label(ann)
        l, t, w, h = ann['bbox']
        area = w * h
        r = l + w
        b = t + h
        for level, centers, threshold, loc_t, cls_t, ignore in zip(
                levels, mlvl_centers, thresholds, loc_targets, cls_targets, ignores):
            lo, hi = threshold
            if area <= lo or area >= hi:
                continue

            l_, t_, r_, b_, w_, h_ = [c / (2 ** level) for c in [l, t, r, b, w, h]]
            cx_ = l_ + 0.5 * w_
            cy_ = t_ + 0.5 * h_

            l_pos, t_pos, r_pos, b_pos = get_fovea(cx_, cy_, w_, h_, shrunk_pos)
            l_neg, t_neg, r_neg, b_neg = get_fovea(cx_, cy_, w_, h_, shrunk_neg)

            cx = centers[..., 0]
            cy = centers[..., 1]
            pos = (l_pos < cx) & (cx < r_pos) & (t_pos < cy) & (cy < b_pos)
            ignore |= (l_neg < cx) & (cx < r_neg) & (t_neg < cy) & (cy < b_neg)
            if not pos.any():
                continue
            ts = (torch.stack([cx - l_, cy - t_, r_ - cx, b_ - cy], dim=-1) / 4).log_()
            loc_t[pos] = ts[pos]
            cls_t[pos] = label

    loc_t = torch.cat([t.view(-1, 4) for t in loc_targets], dim=0)
    cls_t = torch.cat([t.view(-1) for t in cls_targets], dim=0)
    ignore = torch.cat([t.view(-1) for t in ignores], dim=0)
    ignore = ignore & (cls_t == 0)

    return loc_t, cls_t, ignore


class FoveaMatchAnchors:

    def __init__(self, generator, levels=(3, 4, 5, 6, 7),
                 thresholds=DEFAULT_AREA_THRESHOLDS,
                 shrunk_pos=0.3, shrunk_neg=0.4, get_label=lambda x: x["category_id"]):
        assert isinstance(generator, FoveaAnchorGenerator), \
            "Generator must be FoveaAnchorGenerator, got %s" % type(generator)
        self.generator = generator
        assert len(levels) == len(thresholds), "Thresholds don't match number of levels."
        self.levels = levels
        self.strides = [2 ** l for l in levels]
        self.thresholds = thresholds
        self.shrunk_pos = shrunk_pos
        self.shrunk_neg = shrunk_neg
        self.get_label = get_label

    def __call__(self, x, anns):
        height, width = x.shape[1:3]
        grid_sizes = calc_grid_sizes((width, height), self.strides)
        grid_sizes = [torch.Size(s) for s in grid_sizes]
        mlvl_centers = self.generator(grid_sizes, x.device, x.dtype)['centers']
        targets = match_anchors(
            anns, mlvl_centers, self.levels, self.thresholds,
            self.shrunk_pos, self.shrunk_neg, self.get_label)
        return x, targets


class FoveaMatcher:

    def __init__(self, generator, levels=(3, 4, 5, 6, 7),
                 thresholds=DEFAULT_AREA_THRESHOLDS,
                 shrunk_pos=0.3, shrunk_neg=0.4, get_label=lambda x: x["category_id"]):
        assert isinstance(generator, FoveaAnchorGenerator), \
            "Generator must be FoveaAnchorGenerator, got %s" % type(generator)
        self.generator = generator
        assert len(levels) == len(thresholds), "Thresholds don't match number of levels."
        self.levels = levels
        self.thresholds = thresholds
        self.shrunk_pos = shrunk_pos
        self.shrunk_neg = shrunk_neg
        self.get_label = get_label

    def match_single(self, mlvl_centers, anns):
        return match_anchors(
            anns, mlvl_centers, self.levels, self.thresholds,
            self.shrunk_pos, self.shrunk_neg, self.get_label)

    def __call__(self, features, targets):
        grid_sizes = [f.size()[-2:][::-1] for f in features]
        mlvl_centers = self.generator(grid_sizes, features[0].device, features[0].dtype)['centers']
        assert len(mlvl_centers) == len(self.levels)
        loc_targets = []
        cls_targets = []
        ignores = []
        for anns in targets:
            loc_t, cls_t, ignore = self.match_single(mlvl_centers, anns)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            ignores.append(ignore)
        loc_t = torch.stack(loc_targets, dim=0)
        cls_t = torch.stack(cls_targets, dim=0)
        ignore = torch.stack(ignores, dim=0)
        return loc_t, cls_t, ignore


class FoveaLoss(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, loc_p, cls_p, loc_t, cls_t, ignore):
        loc_p, cls_p = flatten_preds(loc_p, cls_p)
        pos = cls_t != 0
        num_pos = pos.sum().item()
        if num_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)
        loc_loss = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction='sum') / num_pos

        cls_t = one_hot(cls_t, C=cls_p.size(-1))
        cls_loss_pos = focal_loss2(cls_p[pos], cls_t[pos], reduction='sum')
        neg = (~pos) & (~ignore)
        cls_p_neg = cls_p[neg]
        cls_loss_neg = focal_loss2(cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
        cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos

        loss = loc_loss + cls_loss
        if random.random() < self.p:
            print("loc: %.4f | cls: %.4f" %
                  (loc_loss.item(), cls_loss.item()))
        return loss


def fovea_inference(
        loc_preds, cls_preds, mlvl_centers, conf_threshold=0.05, iou_threshold=0.5,
        topk1=1000, nms_method='soft_nms', topk2=100):
    dets = []
    mlvl_scores = []
    mlvl_labels = []
    mlvl_bboxes = []
    for loc_p, cls_p, centers in zip(loc_preds, cls_preds, mlvl_centers):
        lx, ly = centers.size()[:2]
        centers = centers.view(-1, 2)
        bboxes = loc_p.view(-1, 4)
        scores, labels = cls_p.view(-1, cls_p.size(-1))[:, 1:].max(dim=-1)
        scores = scores.sigmoid_()
        pos = scores > conf_threshold
        scores = scores[pos]
        if len(scores) == 0:
            continue
        labels = labels[pos]
        bboxes = bboxes[pos]
        centers = centers[pos]

        bboxes.exp_().mul_(4)

        bboxes[:, :2] = centers - bboxes[:, :2]
        bboxes[:, 2:] += centers

        bboxes[:, [0, 2]] /= lx
        bboxes[:, [1, 3]] /= ly

        mlvl_scores.append(scores)
        mlvl_labels.append(labels)
        mlvl_bboxes.append(bboxes)

    if len(mlvl_scores) == 0:
        return []

    scores = torch.cat(mlvl_scores, dim=0)
    labels = torch.cat(mlvl_labels, dim=0)
    bboxes = torch.cat(mlvl_bboxes, dim=0)

    if len(scores) > topk1:
        scores, indices = scores.topk(topk1)
        labels = labels[indices]
        bboxes = bboxes[indices]

    scores = scores.cpu()
    bboxes = bboxes.cpu()

    if nms_method == 'nms':
        indices = nms(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk2:
            indices = scores.topk(topk2)[1]
        else:
            indices = range(scores.size(0))
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk2, min_score=conf_threshold)
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)
    for ind in indices:
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
        }
        dets.append(det)
    return dets


class FoveaInference:

    def __init__(self, generator, conf_threshold=0.05, iou_threshold=0.5,
                 topk1=1000, nms_method='soft_nms', topk2=100):
        assert isinstance(generator, FoveaAnchorGenerator)
        self.generator = generator
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk1 = topk1
        self.nms_method = nms_method
        self.topk2 = topk2

    def __call__(self, loc_preds, cls_preds):
        grid_sizes = [p.size()[1:3] for p in loc_preds]
        mlvl_centers = self.generator(grid_sizes, loc_preds[0].device, loc_preds[0].dtype)['centers']
        image_dets = []
        batch_size = loc_preds[0].size(0)
        for i in range(batch_size):
            dets = fovea_inference(
                [p[i] for p in loc_preds],
                [p[i] for p in cls_preds],
                mlvl_centers,
                self.conf_threshold, self.iou_threshold,
                self.topk1, self.nms_method, self.topk2,
            )
            image_dets.append(dets)
        return image_dets
