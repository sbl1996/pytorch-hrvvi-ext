import random
from collections import defaultdict

from horch.common import select, _concat
from horch.detection import generate_mlvl_anchors, calc_grid_sizes
from horch.detection.anchor.generator import AnchorGeneratorBase
from horch.transforms import Transform
from horch.transforms.detection import BoxList
from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch import one_hot
from horch.nn.loss import focal_loss2

from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms, soft_nms_cpu
from toolz.curried import get
from torchvision.ops import nms as tnms


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return _concat(xs, dim=0)


class AnchorGenerator(AnchorGeneratorBase):
    def __init__(self, levels, anchors, flatten=True, with_corners=True, cache=True):
        super().__init__(levels, cache)
        assert len(levels) == len(anchors)
        self.anchor_sizes = list(anchors)
        self.flatten = flatten
        self.with_corners = with_corners

    def calculate(self, size, grid_sizes, device, dtype):
        width, height = size
        anchor_sizes = [
            a.to(device=device, dtype=dtype, copy=True)
            if torch.is_tensor(a) else torch.tensor(a, device=device, dtype=dtype)
            for a in self.anchor_sizes]
        for a in anchor_sizes:
            a.div_(a.new_tensor([width, height]))
        mlvl_anchors = generate_mlvl_anchors(
            grid_sizes, anchor_sizes, device, dtype)
        if self.flatten:
            mlvl_anchors = flatten(mlvl_anchors)
        ret = {
            "centers": mlvl_anchors
        }
        if self.with_corners:
            ret["corners"] = BBox.convert(mlvl_anchors, BBox.XYWH, BBox.LTRB)
        return ret


def coords_to_target(gt_box, anchors):
    box_txty = (gt_box[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    box_twth = (gt_box[..., 2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def target_to_coords(loc_t, anchors):
    loc_t[..., :2].mul_(anchors[..., 2:]).add_(anchors[..., :2])
    loc_t[..., 2:].exp_().mul_(anchors[..., 2:])
    return loc_t


@curry
def match_anchors(bboxes, labels, centers, corners, pos_thresh=0.7, neg_thresh=0.3, cpu=True):
    num_objects = len(bboxes)
    num_anchors = len(centers)

    if num_objects == 0:
        loc_t = centers.new_zeros(num_anchors, 4)
        cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)
        ignore = loc_t.new_zeros(num_anchors, dtype=torch.uint8) if neg_thresh else None
        return loc_t, cls_t, ignore

    ious = iou_mn(BBox.convert(bboxes, BBox.XYWH, BBox.LTRB), corners)

    max_ious, max_indices = ious.max(dim=1)
    match_ious, matches = ious.max(dim=0)
    match_ious[max_indices] = 1.
    matches[max_indices] = torch.arange(num_objects, device=centers.device)

    if cpu:
        loc_t = centers.new_zeros(num_anchors, 4)
        cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)

        pos = match_ious > pos_thresh
        matches = matches[pos]
        loc_t[pos] = coords_to_target(bboxes[matches], centers[pos])
        cls_t[pos] = labels[matches]
        non_pos = ~pos
    else:
        loc_t = coords_to_target(bboxes[matches], centers)
        cls_t = labels[matches]
        non_pos = match_ious < pos_thresh
        loc_t[non_pos] = 0
        cls_t[non_pos] = 0

    ignore = non_pos & (match_ious >= neg_thresh) if neg_thresh else None

    return loc_t, cls_t, ignore


class MatchAnchors(Transform):
    r"""
    MatchAnchors is a transform for general object detection that transforms
    annotations to localization and classification targets.

    Parameters
    ----------
    generator : AnchorGenerator
        Generator to generator anchors for difference sizes.
        Note: flatten and with_lrtb must be True.
    levels : sequence of ints
        Feature levels.
    pos_thresh : float
        IOU threshold of positive anchors.
    neg_thresh : float
        If provided, only non-positive anchors whose ious with all ground truth boxes are
        lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.

    Example:
        >>> train_transform = Compose([
        >>>     Resize((width, height)),
        >>>     ToPercentCoords(),
        >>>     ToTensor(),
        >>>     MatchAnchors(gen, levels=(3, 4, 5), pos_thresh=0.5, neg_thresh=0.4),
        >>> ])
    """

    def __init__(self, generator, pos_thresh=0.5, neg_thresh=None, get_label=lambda x: x['category_id']):
        super().__init__()
        assert generator.flatten
        assert generator.with_corners
        self.generator = generator
        self.levels = generator.levels
        self.strides = generator.strides
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label

    def __call__(self, x, anns):
        height, width = x.shape[1:3]
        centers, corners = get(["centers", "corners"], self.generator((width, height), x.device, x.dtype))

        bboxes = x.new_tensor([ann['bbox'] for ann in anns])
        bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
        labels = x.new_tensor([self.get_label(ann) for ann in anns], dtype=torch.long)

        loc_t, cls_t, ignore = match_anchors(
            bboxes, labels, centers, corners,
            self.pos_thresh, self.neg_thresh)

        if ignore is not None:
            return x, [loc_t, cls_t, ignore]
        else:
            return x, [loc_t, cls_t]


class AnchorMatcher:
    r"""

    Parameters
    ----------
    generator : AnchorGenerator
        List of anchor boxes of shape `(lx, ly, #anchors, 4)`.
    pos_thresh : float
        IOU threshold of positive anchors.
    neg_thresh : float
        If provided, only non-positive anchors whose ious with all ground truth boxes are
        lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.

    Example:
        >>> gen = AnchorGenerator(cuda(anchor_sizes))
        >>> matcher = AnchorMatcher(gen, pos_thresh=0.5)
        >>> net = OneStageDetector(backbone, fpn, head, matcher, inference)
    """

    def __init__(self, generator, pos_thresh=0.5, neg_thresh=None, get_label=lambda x: x['category_id']):
        assert generator.flatten
        assert generator.with_corners
        self.generator = generator
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label

    def __call__(self, grid_sizes, box_lists, device, dtype):
        grid_sizes = tuple(grid_sizes)
        centers, corners = get(["centers", "corners"],
                               self.generator(grid_sizes, device, dtype))

        return batched_match_anchors(
            centers, corners, box_lists, self.pos_thresh, self.neg_thresh, self.get_label
        )


def batched_match_anchors(centers, corners, box_lists, pos_thresh, neg_thresh, get_label=lambda x: x['category_id']):
    is_cpu = centers.device.type == 'cpu'
    bboxes = [
        BBox.convert(centers.new_tensor([b['bbox'] for b in boxes]), BBox.LTWH, BBox.XYWH)
        for boxes in box_lists
    ]
    labels = [
        centers.new_tensor([get_label(b) for b in boxes], dtype=torch.long)
        for boxes in box_lists
    ]

    loc_targets = []
    cls_targets = []
    ignores = []
    for i in range(len(box_lists)):
        loc_t, cls_t, ignore = match_anchors(
            bboxes[i], labels[i], centers, corners,
            pos_thresh, neg_thresh, is_cpu)
        loc_targets.append(loc_t)
        cls_targets.append(cls_t)
        ignores.append(ignore)

    loc_t = torch.stack(loc_targets, dim=0)
    cls_t = torch.stack(cls_targets, dim=0)
    if ignores[0] is not None:
        ignore = torch.stack(ignores, dim=0)
        return loc_t, cls_t, ignore
    else:
        return loc_t, cls_t


def flatten_preds(*preds):
    r"""
    Parameters
    ----------
    preds :
        [[(b, w, h, ?a, 4)]]
    """
    b = preds[0][0].size(0)
    preds_flat = []
    for ps in preds:
        p = torch.cat([p.view(b, -1, p.size(-1)) for p in ps], dim=1)
        preds_flat.append(p.squeeze(-1))
    return preds_flat


class MultiBoxLossOnline(nn.Module):

    def __init__(self, matcher, neg_pos_ratio=None, cls_loss='ce', loc_t_stds=(0.1, 0.1, 0.2, 0.2), p=0.01, prefix=""):
        super().__init__()
        self.matcher = matcher
        self.loss = MultiBoxLoss(neg_pos_ratio, cls_loss, loc_t_stds, p, prefix)

    def forward(self, loc_preds, cls_preds, box_lists):
        grid_sizes = [p.size()[1:3] for p in loc_preds]
        targets = self.matcher(grid_sizes, box_lists, loc_preds[0].device, loc_preds[0].dtype)
        loc_p, cls_p = flatten_preds(loc_preds, cls_preds)
        return self.loss(loc_p, cls_p, *targets)


class MultiBoxLoss(nn.Module):
    r"""
    Parameters
    ----------
    neg_pos_ratio : float
        Ratio between positive and negative samples used for hard negative mining.
        Default : None
    p : float
        Probability to print loss.
    cls_loss : str
        Classification loss.
        Options: ['focal', 'ce', 'bce']
        Default: 'ce'
    prefix : str
        Prefix of loss.

    Inputs
    ------
    loc_p : Tensor or List[Tensor]
        (batch_size, #anchors, 4) of list of (batch_size, w, h, ?a, 4)
    cls_p : Tensor or List[Tensor]
        (batch_size, #anchors, ?C) of list of (batch_size, w, h, ?a, ?C)
    loc_t : Tensor
        (batch_size, #anchors, 4)
    cls_t : Tensor
        (batch_size, #anchors)
    ignore : Tensor
        (batch_size, #anchors)
    """

    def __init__(self,
                 neg_pos_ratio=None,
                 cls_loss='ce',
                 loc_t_stds=(0.1, 0.1, 0.2, 0.2),
                 p=0.01,
                 prefix=""):
        super().__init__()
        assert 0 <= p <= 1
        assert cls_loss in ['focal', 'ce', 'bce'], "Classification loss must be one of ['focal', 'ce', 'bce']"
        self.neg_pos_ratio = neg_pos_ratio
        if loc_t_stds is None:
            loc_t_stds = (1, 1, 1, 1)
        self.loc_t_stds = loc_t_stds
        self.cls_loss = cls_loss
        self.prefix = prefix
        self.p = p

    def forward(self, loc_p, cls_p, loc_t, cls_t, ignore=None, *args):
        if not torch.is_tensor(loc_p):
            loc_p, cls_p = flatten_preds(loc_p, cls_p)
        pos = cls_t != 0
        neg = ~pos
        if ignore is not None:
            neg = neg & ~ignore
        num_pos = pos.sum(dim=1)
        total_pos = num_pos.sum().item()

        if total_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)

        if self.loc_t_stds:
            loc_t = loc_t / loc_t.new_tensor(self.loc_t_stds)
        loc_loss = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction='sum') / total_pos

        # Hard Negative Mining
        if self.neg_pos_ratio:
            if self.cls_loss == 'ce':
                batch_size = cls_p.size(0)
                num_classes = cls_p.size(-1)
                num_neg = self.neg_pos_ratio * num_pos

                cls_loss_all = F.cross_entropy(
                    cls_p.view(-1, num_classes), cls_t.view(-1), reduction='none').view(batch_size, -1)
                cls_loss_pos = cls_loss_all[pos].sum()
                cls_loss_neg_all = cls_loss_all.clone()  # (N, 8732)
                cls_loss_neg_all[~neg] = 0.

                cls_loss_neg = 0
                for i in range(batch_size):
                    cls_loss_neg += cls_loss_neg_all[i].topk(num_neg[i])[0].sum()

                cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos
            elif self.cls_loss == 'bce':
                assert cls_p.dim() == 2
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_p, cls_t.to(dtype=cls_p.dtype), reduction='none')
                cls_loss_pos = cls_loss[pos].sum()
                cls_loss_neg = cls_loss[neg]
                if len(cls_loss_neg) == 0:
                    cls_loss = cls_loss_pos / total_pos
                else:
                    num_neg = min(int(total_pos * self.neg_pos_ratio), len(cls_loss_neg))
                    cls_loss_neg = torch.topk(cls_loss_neg, num_neg)[0].sum()
                    cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos
            else:
                raise ValueError("Invalid cls loss `%s` for hard negative mining" % self.cls_loss)
        else:
            if self.cls_loss == 'focal':
                if cls_p.dim() - cls_t.dim() == 1:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                else:
                    cls_t = cls_t.to(dtype=cls_p.dtype)
                if ignore is not None:
                    cls_loss_pos = focal_loss2(
                        cls_p[pos], cls_t[pos], reduction='sum')
                    cls_p_neg = cls_p[neg]
                    cls_loss_neg = focal_loss2(cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
                    cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos
                else:
                    cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / total_pos
            elif self.cls_loss == 'bce':
                if cls_p.dim() - cls_t.dim() == 1:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                else:
                    cls_t = cls_t.to(dtype=cls_p.dtype)
                cls_t = cls_t.to(dtype=cls_p.dtype)
                cls_loss = F.binary_cross_entropy_with_logits(cls_p, cls_t, reduction='sum') / total_pos
            elif self.cls_loss == 'ce':
                cls_p = cls_p.view(-1, cls_p.size(-1))
                cls_t = cls_t.view(-1)
                if ignore is not None:
                    cls_loss_pos = F.cross_entropy(
                        cls_p[pos], cls_t[pos], reduction='sum')
                    cls_p_neg = cls_p[neg]
                    cls_loss_neg = F.cross_entropy(
                        cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
                    cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos
                else:
                    cls_loss = F.cross_entropy(cls_p, cls_t, reduction='sum') / total_pos
            else:
                raise ValueError("Classification loss must be one of ['focal', 'ce', 'bce']")

        loss = cls_loss + loc_loss
        if random.random() < self.p:
            if self.prefix:
                print("[%s] loc: %.4f | cls: %.4f" %
                      (self.prefix, loc_loss.item(), cls_loss.item()))
            else:
                print("loc: %.4f | cls: %.4f" %
                      (loc_loss.item(), cls_loss.item()))
        return loss


@curry
def anchor_based_inference(
        loc_p, cls_p, anchors, conf_threshold=0.01,
        iou_threshold=0.5, topk=100,
        conf_strategy='softmax',
        nms_method='soft', min_score=None,
        loc_t_stds=(0.1, 0.1, 0.2, 0.2)):
    bboxes = loc_p.view(-1, 4) * loc_p.new_tensor(loc_t_stds)
    logits = cls_p.view(-1, cls_p.size(-1))
    if conf_strategy == 'softmax':
        scores = torch.softmax(logits, dim=1)
    else:
        scores = torch.sigmoid_(logits)
    scores, labels = torch.max(scores[:, 1:], dim=1)

    if conf_threshold > 0:
        pos = scores > conf_threshold
        scores = scores[pos]
        labels = labels[pos]
        bboxes = bboxes[pos]
        anchors = anchors[pos]

    bboxes = target_to_coords(bboxes, anchors)

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms_method == 'soft':
        min_score = min_score or conf_threshold
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, min_score=min_score)
    else:
        indices = nms(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk:
            indices = scores.topk(topk)[1]
        else:
            indices = range(scores.size(0))

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
    return BoxList(dets)


def anchor_based_inference_per_class(
        loc_p, cls_p, anchors, conf_threshold=0.01,
        iou_threshold=0.5, topk=100,
        conf_strategy='softmax',
        loc_t_stds=(0.1, 0.1, 0.2, 0.2)):
    bboxes = loc_p.view(-1, 4) * loc_p.new_tensor(loc_t_stds)
    bboxes = target_to_coords(bboxes, anchors)
    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True)
    num_classes = cls_p.size(-1)
    logits = cls_p.view(-1, num_classes)
    if conf_strategy == 'softmax':
        scores = torch.softmax(logits, dim=1)
    else:
        scores = torch.sigmoid_(logits)

    dets = []
    for c in range(1, num_classes):
        c_scores = scores[:, c]
        c_bboxes = bboxes

        if conf_threshold > 0:
            pos = c_scores > conf_threshold
            c_scores = c_scores[pos]
            if len(c_scores) == 0:
                continue
            c_bboxes = c_bboxes[pos]

        indices = tnms(c_bboxes, c_scores, iou_threshold)
        c_scores = c_scores[indices]
        c_bboxes = c_bboxes[indices]
        c_bboxes = BBox.convert(
            c_bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)

        for i in range(len(c_scores)):
            det = {
                'image_id': -1,
                'category_id': c,
                'bbox': c_bboxes[i].tolist(),
                'score': c_scores[i].item(),
            }
            dets.append(det)

    dets = sorted(dets, key=lambda d: d['score'], reverse=True)
    dets = dets[:topk]
    return BoxList(dets)


class AnchorBasedInference:
    r"""
    Parameters
    ----------
    generator : AnchorGenerator
    conf_threshold : float
        Confidence threshold for filtering.
    iou_threshold : float
        IoU threshold used in nms.
    topk : int
        Top k boxes retained after nms.
    conf_strategy : str
        'softmax' for ce loss or 'sigmoid' for focal loss.
    nms : str
        'nms' for classical non-max suppression and 'soft' for soft nms.
    min_score : float
        Minimal score used in soft nms.
    """

    def __init__(self, generator, conf_threshold=0.01,
                 iou_threshold=0.5, topk=100,
                 conf_strategy='softmax', nms='soft', min_score=None,
                 per_class_nms=False, loc_t_stds=(0.1, 0.1, 0.2, 0.2)):
        assert generator.flatten
        assert generator.with_corners
        self.generator = generator
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        assert conf_strategy in [
            'softmax', 'sigmoid'], "conf_strategy must be softmax or sigmoid"
        self.conf_strategy = conf_strategy
        self.nms = nms
        self.min_score = min_score
        self.per_class_nms = per_class_nms
        if loc_t_stds is None:
            loc_t_stds = (1, 1, 1, 1)
        self.loc_t_stds = loc_t_stds

    def __call__(self, loc_preds, cls_preds):
        grid_sizes = [p.size()[1:3] for p in loc_preds]
        anchors = self.generator(grid_sizes, loc_preds[0].device, loc_preds[0].dtype)["centers"]
        loc_p, cls_p = flatten_preds(loc_preds, cls_preds)
        batch_size = loc_p.size(0)
        image_dets = [
            self.inference_single(loc_p[i], cls_p[i], anchors)
            for i in range(batch_size)
        ]
        return image_dets

    def inference_single(self, loc_p, cls_p, anchors):
        if self.per_class_nms:
            return anchor_based_inference_per_class(
                loc_p, cls_p, anchors, self.conf_threshold,
                self.iou_threshold, self.topk,
                self.conf_strategy, self.loc_t_stds
            )
        else:
            return anchor_based_inference(
                loc_p, cls_p, anchors, self.conf_threshold,
                self.iou_threshold, self.topk,
                self.conf_strategy, self.nms, self.min_score, self.loc_t_stds
            )
