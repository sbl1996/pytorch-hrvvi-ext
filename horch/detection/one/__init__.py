import random

from toolz import curry
from toolz.curried import get

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch import one_hot
from horch.nn.loss import focal_loss2

from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms_cpu, soft_nms_cpu


def coords_to_target(gt_box, anchors):
    box_txty = (gt_box[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    box_twth = (gt_box[..., 2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def target_to_coords(loc_t, anchors):
    loc_t[:, :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    loc_t[:, 2:].exp_().mul_(anchors[:, 2:])
    return loc_t

@curry
def match_anchors_flat(anns, anchors_xywh, anchors_ltrb, pos_thresh=0.5, neg_thresh=None,
                       get_label=get('category_id'), debug=False):
    num_anchors = len(anchors_xywh)
    loc_t = anchors_xywh.new_zeros(num_anchors, 4)
    cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)

    if len(anns) == 0:
        target = [loc_t, cls_t]
        if neg_thresh:
            target.append(loc_t.new_zeros(num_anchors, dtype=torch.uint8))
        return target

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([get_label(ann) for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, anchors_ltrb)

    max_ious, indices = ious.max(dim=1)
    if debug:
        print(max_ious.tolist())
    loc_t[indices] = coords_to_target(bboxes, anchors_xywh[indices])
    cls_t[indices] = labels

    pos = ious > pos_thresh
    for ipos, bbox, label in zip(pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, anchors_xywh[ipos])
        cls_t[ipos] = label

    target = [loc_t, cls_t]

    if neg_thresh:
        ignore = (cls_t == 0) & ((ious >= neg_thresh).sum(dim=0) != 0)
        target.append(ignore)
    return target


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return torch.cat(xs, dim=0)


class MatchAnchors:
    r"""

    Parameters
    ----------
    anchors : torch.Tensor or List[torch.Tensor]
        List of anchor boxes of shape `(lx, ly, #anchors, 4)`.
    pos_thresh : float
        IOU threshold of positive anchors.
    neg_thresh : float
        If provided, only non-positive anchors whose ious with all ground truth boxes are
        lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.
    get_label : function
        Function to extract label from annotations.
    """

    def __init__(self, anchors, pos_thresh=0.5, neg_thresh=None,
                 get_label=get('category_id'), debug=False):
        self.anchors_xywh = flatten(anchors)
        self.anchors_ltrb = BBox.convert(self.anchors_xywh, BBox.XYWH, BBox.LTRB)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label
        self.debug = debug

    def __call__(self, img, anns):
        target = match_anchors_flat(
            anns, self.anchors_xywh, self.anchors_ltrb,
            self.pos_thresh, self.neg_thresh, self.get_label, self.debug)
        return img, target


class MultiBoxLoss(nn.Module):

    def __init__(self, pos_neg_ratio=None, p=0.01, criterion='softmax', prefix=""):
        super().__init__()
        self.pos_neg_ratio = pos_neg_ratio
        self.p = p
        self.criterion = criterion
        self.prefix = prefix

    def forward(self, loc_p, cls_p, loc_t, cls_t, ignore=None, *args):
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
            if self.criterion == 'focal':
                if cls_p.ndimension() - cls_t.ndimension() == 1:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                else:
                    cls_t = cls_t.float()
                cls_loss = focal_loss2(cls_p, cls_t, reduction='sum') / num_pos
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
        conf_strategy='softmax', nms='soft_nms', soft_nms_threshold=None):
    dets = []
    bboxes = loc_p
    logits = cls_p
    if conf_strategy == 'softmax':
        scores = torch.softmax(logits, dim=1)
    else:
        scores = torch.sigmoid_(logits)
    scores = scores[:, 1:]
    scores, labels = torch.max(scores, dim=1)

    if conf_threshold > 0:
        mask = scores > conf_threshold
        scores = scores[mask]
        labels = labels[mask]
        bboxes = bboxes[mask]
        anchors = anchors[mask]

    bboxes = target_to_coords(bboxes, anchors)

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

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
        }
        dets.append(det)
    return dets


class AnchorBasedInference:

    def __init__(self, anchors, conf_threshold=0.01,
                 iou_threshold=0.5, topk=100,
                 conf_strategy='softmax', nms='soft_nms', soft_nms_threshold=None):
        self.anchors = flatten(anchors)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        assert conf_strategy in [
            'softmax', 'sigmoid'], "conf_strategy must be softmax or sigmoid"
        self.conf_strategy = conf_strategy
        self.nms = nms
        self.soft_nms_threshold = soft_nms_threshold

    def __call__(self, loc_p, cls_p):
        image_dets = []
        batch_size = loc_p.size(0)
        for i in range(batch_size):
            dets = anchor_based_inference(
                loc_p[i], cls_p[i], self.anchors,
                self.conf_threshold, self.iou_threshold,
                self.topk, self.conf_strategy, self.nms, self.soft_nms_threshold
            )
            image_dets.append(dets)
        return image_dets
