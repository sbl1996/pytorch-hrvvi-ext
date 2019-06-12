import random

from horch.common import select, _concat
from horch.detection import generate_mlvl_anchors, calc_grid_sizes
from horch.detection.anchor.generator import AnchorGeneratorBase
from horch.transforms.detection import BoxList
from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch import one_hot
from horch.nn.loss import focal_loss2, loc_kl_loss

from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms, soft_nms_cpu
from toolz.curried import get


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return _concat(xs, dim=0)


class AnchorGenerator(AnchorGeneratorBase):
    def __init__(self, anchor_sizes, flatten=True, with_corners=True, cache=True):
        super().__init__(cache)
        self.anchor_sizes = anchor_sizes
        self.flatten = flatten
        self.with_corners = with_corners

    def calculate(self, grid_sizes, device, dtype):
        anchor_sizes = self.anchor_sizes.to(device=device, dtype=dtype)
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


def coords_to_target2(bboxes, anchors):
    r"""
    Parameters
    ----------
    bboxes
        (n, 4)
    anchors
        (m, 4)
    """
    bboxes = bboxes[:, None, :]
    anchors = anchors[None, :, :]
    txty = (bboxes[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    twth = (bboxes[..., 2:] / anchors[..., 2:]).log_()
    return torch.cat((txty, twth), dim=-1)


def match_anchors2(anns, centers, corners, pos_thresh=0.7, neg_thresh=0.3,
                   get_label=lambda x: x['category_id'], debug=False):
    num_anchors = len(centers)
    if len(anns) == 0:
        loc_t = centers.new_zeros(num_anchors, 4)
        cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)
        ignore = loc_t.new_zeros(num_anchors, dtype=torch.uint8) if neg_thresh else None
        return loc_t, cls_t, ignore

    bboxes = centers.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = centers.new_tensor([get_label(ann) for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, corners)

    pos = ious > pos_thresh
    cls_t, indices = (pos.long() * labels[:, None]).max(dim=0)
    loc_t_all = coords_to_target2(bboxes, centers)
    loc_t = select(loc_t_all, 0, indices)

    max_ious, max_indices = ious.max(dim=1)
    if debug:
        print(max_ious.tolist())
    loc_t[max_indices] = select(loc_t_all, 1, max_indices)
    cls_t[max_indices] = labels

    ignore = (cls_t == 0) & ((ious >= neg_thresh).sum(dim=0) != 0) if neg_thresh else None
    return loc_t, cls_t, ignore


@curry
def match_anchors(anns, centers, corners, pos_thresh=0.7, neg_thresh=0.3,
                  get_label=lambda x: x['category_id'], debug=False):
    num_anchors = len(centers)
    loc_t = centers.new_zeros(num_anchors, 4)
    cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)

    if len(anns) == 0:
        ignore = loc_t.new_zeros(num_anchors, dtype=torch.uint8) if neg_thresh else None
        return loc_t, cls_t, ignore

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([get_label(ann) for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, corners)

    pos = ious > pos_thresh
    for ipos, bbox, label in zip(pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, centers[ipos])
        cls_t[ipos] = label

    max_ious, indices = ious.max(dim=1)
    if debug:
        print(max_ious.tolist())
    loc_t[indices] = coords_to_target(bboxes, centers[indices])
    cls_t[indices] = labels

    ignore = (cls_t == 0) & ((ious >= neg_thresh).sum(dim=0) != 0) if neg_thresh else None
    return loc_t, cls_t, ignore


class MatchAnchors:
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

    def __init__(self, generator, levels, pos_thresh=0.5, neg_thresh=None, get_label=lambda x: x['category_id'],
                 debug=False):
        assert generator.flatten
        assert generator.with_corners
        self.generator = generator
        self.levels = levels
        self.strides = [ 2 ** l for l in levels ]
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label
        self.debug = debug

    def __call__(self, x, anns):
        height, width = x.shape[1:3]
        grid_sizes = calc_grid_sizes((width, height), self.strides)
        grid_sizes = [torch.Size(s) for s in grid_sizes]
        centers, corners = get(["centers", "corners"], self.generator(grid_sizes, x.device, x.dtype))
        loc_t, cls_t, ignore = match_anchors(
            anns, centers, corners,
            self.pos_thresh, self.neg_thresh, self.get_label, self.debug)

        targets = [loc_t, cls_t]
        if ignore is not None:
            targets.append(ignore)
        return x, targets


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

    def __init__(self, generator, pos_thresh=0.5, neg_thresh=None, get_label=lambda x: x['category_id'], debug=False):
        assert generator.flatten
        assert generator.with_corners
        self.generator = generator
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label
        self.debug = debug

    def __call__(self, features, box_lists):
        is_cpu = features[0].device.type == 'cpu'
        match_func = match_anchors if is_cpu else match_anchors2
        batch_size = len(box_lists)
        grid_sizes = [f.size()[-2:][::-1] for f in features]
        centers, corners = get(["centers", "corners"],
                               self.generator(grid_sizes, features[0].device, features[0].dtype))
        loc_targets = []
        cls_targets = []
        ignores = []
        for i in range(batch_size):
            loc_t, cls_t, ignore = match_func(
                box_lists[i], centers, corners,
                self.pos_thresh, self.neg_thresh, self.get_label, self.debug)
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


def target_to_coords(loc_t, anchors):
    loc_t[..., :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    loc_t[..., 2:].exp_().mul_(anchors[:, 2:])
    return loc_t


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


class MultiBoxLoss(nn.Module):
    r"""
    Parameters
    ----------
    pos_neg_ratio : float
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
    """

    def __init__(self, pos_neg_ratio=None, p=0.01, cls_loss='ce', prefix=""):
        super().__init__()
        assert 0 <= p <= 1
        assert cls_loss in ['focal', 'ce', 'bce'], "Classification loss must be one of ['focal', 'ce', 'bce']"
        self.pos_neg_ratio = pos_neg_ratio
        self.p = p
        self.cls_loss = cls_loss
        self.prefix = prefix

    def forward(self, loc_p, cls_p, loc_t, cls_t, ignore=None, *args):
        if not torch.is_tensor(loc_p):
            loc_p, cls_p = flatten_preds(loc_p, cls_p)
        # print(loc_p.shape)
        # print(cls_p.shape)
        # print(loc_t.shape)
        # print(cls_t.shape)
        pos = cls_t != 0
        neg = ~pos
        if ignore is not None:
            neg = neg & ~ignore
        num_pos = pos.sum().item()

        if num_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)
        if loc_p.size()[:-1] == pos.size():
            loc_p = loc_p[pos]
        if loc_t.size()[:-1] == pos.size():
            loc_t = loc_t[pos]
        loc_loss = F.smooth_l1_loss(
            loc_p, loc_t, reduction='sum') / num_pos

        # Hard Negative Mining
        if self.pos_neg_ratio:
            cls_loss_pos = F.cross_entropy(
                cls_p[pos], cls_t[pos], reduction='sum')

            cls_p_neg = cls_p[neg]
            cls_loss_neg = -F.log_softmax(cls_p_neg, dim=1)[..., 0]
            num_neg = min(int(num_pos / self.pos_neg_ratio), len(cls_loss_neg))
            cls_loss_neg = torch.topk(cls_loss_neg, num_neg)[0].sum()
            cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos
        else:
            if self.cls_loss == 'focal':
                if cls_p.ndimension() - cls_t.ndimension() == 1:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                else:
                    cls_t = cls_t.float()
                if ignore is not None:
                    cls_loss_pos = focal_loss2(cls_p[pos], cls_t[pos], reduction='sum')
                    cls_p_neg = cls_p[neg]
                    cls_loss_neg = focal_loss2(cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
                    cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos
                else:
                    cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / num_pos
            elif self.cls_loss == 'bce':
                assert cls_p.size() == cls_t.size()
                cls_loss = F.binary_cross_entropy_with_logits(cls_p, cls_t.float(), reduction='sum') / num_pos
            else:
                cls_p = cls_p.view(-1, cls_p.size(-1))
                cls_t = cls_t.view(-1)
                if len(cls_p) != len(cls_t):
                    cls_loss_pos = F.cross_entropy(
                        cls_p[pos], cls_t[pos], reduction='sum')
                    cls_p_neg = cls_p[neg]
                    cls_loss_neg = F.cross_entropy(
                        cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
                    cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos
                else:
                    cls_loss = F.cross_entropy(cls_p, cls_t, reduction='sum') / num_pos
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
        conf_strategy='softmax', nms_method='soft', min_score=None):
    bboxes = loc_p.view(-1, 4)
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


class KLFocalLoss(nn.Module):

    def __init__(self, p=0.01, prefix=""):
        super().__init__()
        self.p = p
        self.prefix = prefix

    def forward(self, loc_p, cls_p, log_var_p, loc_t, cls_t, ignore=None, *args):
        pos = cls_t != 0
        neg = ~pos
        if ignore is not None:
            neg = neg & ~ignore
        num_pos = pos.sum().item()
        if num_pos == 0:
            return loc_p.new_tensor(0, requires_grad=True)
        if loc_p.size()[:-1] == pos.size():
            loc_p = loc_p[pos]
            log_var_p = log_var_p[pos]
        if loc_t.size()[:-1] == pos.size():
            loc_t = loc_t[pos]

        loc_loss = loc_kl_loss(
            loc_p, log_var_p, loc_t, reduction='sum') / num_pos

        if cls_p.ndimension() - cls_t.ndimension() == 1:
            cls_t = one_hot(cls_t, C=cls_p.size(-1))
        else:
            cls_t = cls_t.float()
        if ignore is not None:
            cls_loss_pos = focal_loss2(cls_p[pos], cls_t[pos], reduction='sum')
            cls_p_neg = cls_p[neg]
            cls_loss_neg = focal_loss2(cls_p_neg, torch.zeros_like(cls_p_neg), reduction='sum')
            cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos
        else:
            cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / num_pos

        loss = cls_loss + loc_loss
        if random.random() < self.p:
            if self.prefix:
                print("[%s] loc: %.4f | cls: %.4f" %
                      (self.prefix, loc_loss.item(), cls_loss.item()))
            else:
                print("loc: %.4f | cls: %.4f" %
                      (loc_loss.item(), cls_loss.item()))
        return loss


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
                 conf_strategy='softmax', nms='soft', min_score=None):
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
        return anchor_based_inference(
            loc_p, cls_p, anchors, self.conf_threshold,
            self.iou_threshold, self.topk,
            self.conf_strategy, self.nms, self.min_score
        )
