import random
import numbers
from typing import List

from toolz import curry
from toolz.curried import get, groupby

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from hutil import one_hot
from hutil.common import Args
from hutil.nn.loss import focal_loss2
from hutil import _C

from hutil.detection.bbox import BBox, transform_bbox, transform_bboxes
from hutil.detection.iou import iou_11, iou_b11, iou_1m, iou_mn

__all__ = [
    "coords_to_target", "MatchAnchors", "BBox",
    "nms_cpu", "soft_nms_cpu", "transform_bbox", "transform_bboxes", "misc_target_collate",
    "iou_1m", "iou_11", "iou_b11", "iou_mn", "draw_bboxes",
    "MultiBoxLoss", "AnchorBasedInference", "get_locations", "generate_anchors", "generate_multi_level_anchors",
    "mAP", "match_anchors", "anchor_based_inference", "batch_anchor_match", "to_pred"
]


def get_locations(size, strides):
    num_levels = int(np.log2(strides[-1]))
    lx, ly = size
    locations = [(lx, ly)]
    for _ in range(num_levels):
        # if lx == 3:
        #     lx = 1
        # else:
        #     lx = (lx - 1) // 2 + 1
        # if ly == 3:
        #     ly = 1
        # else:
        #     ly = (ly - 1) // 2 + 1
        lx = (lx - 1) // 2 + 1
        ly = (ly - 1) // 2 + 1
        locations.append((lx, ly))
    return locations[-len(strides):]


def inverse_sigmoid(x, eps=1e-3):
    x = torch.clamp(x, eps, 1 - eps)
    return (x / (1 - x)).log_()


def yolo_coords_to_target(gt_box, anchors, location):
    location = gt_box.new_tensor(location)
    box_txty = inverse_sigmoid(gt_box[:2] * location % 1)
    box_twth = (gt_box[2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def coords_to_target(gt_box, anchors, *args):
    box_txty = (gt_box[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    box_twth = (gt_box[..., 2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def target_to_coords(loc_t, anchors):
    loc_t[:, :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    loc_t[:, 2:].exp_().mul_(anchors[:, 2:])
    return loc_t


def generate_multi_level_anchors(input_size, strides=(8, 16, 32, 64, 128), aspect_ratios=(1 / 2, 1 / 1, 2 / 1),
                                 scales=(32, 64, 128, 256, 512)):
    width, height = input_size
    locations = get_locations(input_size, strides)
    if isinstance(aspect_ratios[0], numbers.Number):
        aspect_ratios_of_level = [aspect_ratios] * len(strides)
    else:
        aspect_ratios_of_level = aspect_ratios
    aspect_ratios_of_level = torch.tensor(aspect_ratios_of_level)
    anchors_of_level = []
    for (lx, ly), ars, scale in zip(locations, aspect_ratios_of_level, scales):
        if isinstance(scale, tuple):
            sw, sh = scale
        else:
            sw = sh = scale
        anchors = torch.zeros(lx, ly, len(ars), 4)
        anchors[:, :, :, 0] = (torch.arange(
            lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, len(ars)) + 0.5) / lx
        anchors[:, :, :, 1] = (torch.arange(
            ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, len(ars)) + 0.5) / ly
        anchors[:, :, :, 2] = sw * ars.sqrt() / width
        anchors[:, :, :, 3] = sh / ars.sqrt() / height
        anchors_of_level.append(anchors)
    return anchors_of_level


def generate_anchors(input_size, stride=16, aspect_ratios=(1 / 2, 1 / 1, 2 / 1), scales=(32, 64, 128, 256, 512)):
    width, height = input_size
    lx, ly = get_locations(input_size, [stride])[0]
    aspect_ratios = torch.tensor(aspect_ratios)
    scales = aspect_ratios.new_tensor(scales).view(len(scales), -1)
    num_anchors = len(aspect_ratios) * len(scales)
    anchors = torch.zeros(lx, ly, num_anchors, 4)
    anchors[:, :, :, 0] = (torch.arange(
        lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, num_anchors) + 0.5) / lx
    anchors[:, :, :, 1] = (torch.arange(
        ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, num_anchors) + 0.5) / ly
    if scales.size(1) == 2:
        sw = scales[:, [0]]
        sh = scales[:, [1]]
    else:
        sw = sh = scales
    anchors[:, :, :, 2] = (sw * aspect_ratios).view(-1) / width
    anchors[:, :, :, 3] = (sh / aspect_ratios).view(-1) / height
    return anchors


def _ensure_multi_level(xs):
    if torch.is_tensor(xs):
        return [xs]
    else:
        return xs


def to_pred(t: torch.Tensor, c: int):
    b = t.size(0)
    t = t.permute(0, 3, 2, 1).contiguous().view(b, -1, c)
    return t


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return torch.cat(xs, dim=0)

@curry
def batch_anchor_match(image_gts, anchors_xywh, anchors_ltrb, max_iou=True, pos_thresh=0.5, neg_thresh=None,
                  get_label=get('category_id'), debug=False):
    loc_t = []
    cls_t = []
    batch_size = len(image_gts)
    for i in range(batch_size):
        i_loc_t, i_cls_t = match_anchors(image_gts[i], anchors_xywh[i], anchors_ltrb[i], debug=debug)
        loc_t.append(i_loc_t)
        cls_t.append(i_cls_t)
    loc_t = torch.stack(loc_t, dim=0)
    cls_t = torch.stack(cls_t, dim=0)
    return loc_t, cls_t

@curry
def match_anchors(anns, anchors_xywh, anchors_ltrb, max_iou=True, pos_thresh=0.5, neg_thresh=None,
                  get_label=get('category_id'), debug=False):
    num_anchors = len(anchors_xywh)
    loc_t = anchors_xywh.new_zeros(num_anchors, 4)
    cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)
    if neg_thresh:
        neg = loc_t.new_ones(num_anchors, dtype=torch.uint8)

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([get_label(ann) for ann in anns])

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, anchors_ltrb)

    if max_iou:
        max_ious, indices = ious.max(dim=1)
        loc_t[indices] = coords_to_target(bboxes, anchors_xywh[indices])
        cls_t[indices] = labels
        if debug:
            print(max_ious)


    if pos_thresh:
        pos = ious > pos_thresh
        for ipos, bbox, label in zip(pos, bboxes, labels):
            loc_t[ipos] = coords_to_target(bbox, anchors_xywh[ipos])
            cls_t[ipos] = label

    target = [loc_t, cls_t]

    if pos_thresh and neg_thresh:
        ignore = (cls_t == 0) & ((ious < neg_thresh).sum(dim=0) == 0)
        target.append(ignore)

    return target


class MatchAnchors:
    r"""

    Args:
        anchors: List of anchor boxes of shape `(lx, ly, #anchors, 4)`.
        max_iou: Whether assign anchors with max ious with ground truth boxes as positive anchors.
        pos_thresh: IOU threshold of positive anchors.
        neg_thresh: If provided, only non-positive anchors whose ious with all ground truth boxes are
            lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.
        get_label: Function to extract label from annotations.
    Inputs:
        img: Input image.
        anns: Sequences of annotations containing label and bounding box.
    Outputs:
        img: Input image.
        targets:
            loc_targets:
            cls_targets:
            negs (optional): Returned when neg_thresh is provided.
    """

    def __init__(self, anchors, max_iou=True,
                 pos_thresh=0.5, neg_thresh=None,
                 get_label=get('category_id'), debug=False):
        self.anchors_xywh = flatten(anchors)
        self.anchors_ltrb = BBox.convert(self.anchors_xywh, BBox.XYWH, BBox.LTRB)
        self.max_iou = max_iou
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label
        self.debug = debug

    def __call__(self, img, anns):
        target = match_anchors(
            anns, self.anchors_xywh, self.anchors_ltrb,
            self.max_iou, self.pos_thresh, self.neg_thresh, self.get_label, self.debug)
        return img, target


class MultiBoxLoss(nn.Module):

    def __init__(self, pos_neg_ratio=None, p=0.1, criterion='softmax'):
        super().__init__()
        self.pos_neg_ratio = pos_neg_ratio
        self.p = p
        if criterion == 'softmax':
            self.criterion = F.cross_entropy
        elif criterion == 'focal':
            self.criterion = focal_loss2
        else:
            raise ValueError("criterion must be one of softmax or focal")

    def forward(self, loc_p, cls_p, loc_t, cls_t, ignore=None, *args):

        pos = cls_t != 0
        neg = ~pos
        if ignore is not None:
            neg = neg & ~ignore
        num_pos = pos.sum().item()
        if loc_p.size()[:-1] == pos.size():
            loc_p = loc_p[pos]
        loc_t = loc_t[pos]
        loc_loss = F.smooth_l1_loss(
            loc_p, loc_t, reduction='sum') / num_pos

        # Hard Negative Mining
        if self.pos_neg_ratio:
            cls_loss_pos = self.criterion(
                cls_p[pos], cls_t[pos], reduction='sum')

            cls_p_neg = cls_p[neg]
            cls_loss_neg = -F.log_softmax(cls_p_neg, dim=1)[..., 0]
            num_neg = min(num_pos / self.pos_neg_ratio, len(cls_loss_neg))
            cls_loss_neg = torch.topk(cls_loss_neg, num_neg)[0].sum()
            cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos
        else:
            if self.criterion == focal_loss2:
                cls_t = one_hot(cls_t, C=cls_p.size(-1))
            cls_loss_pos = self.criterion(
                cls_p[pos], cls_t[pos], reduction='sum')

            cls_loss_neg = self.criterion(
                cls_p[neg], cls_t[neg], reduction='sum')
            cls_loss = (cls_loss_pos + cls_loss_neg) / num_pos

        loss = cls_loss + loc_loss
        if random.random() < self.p:
            print("loc: %.4f | cls: %.4f" %
                  (loc_loss.item(), cls_loss.item()))
        return loss


@curry
def anchor_based_inference(
        loc_p, cls_p, anchors, conf_threshold=0.01,
        iou_threshold=0.5, topk=100,
        conf_strategy='softmax', nms='soft_nms'):
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
        if len(scores) > topk:
            indices = scores.topk(topk)[1]
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, conf_threshold=conf_threshold / 100)
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
                 conf_strategy='softmax', nms='soft_nms'):
        self.anchors = flatten(anchors)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        assert conf_strategy in [
            'softmax', 'sigmoid'], "conf_strategy must be softmax or sigmoid"
        self.conf_strategy = conf_strategy
        self.nms = nms

    def __call__(self, loc_pred, cls_pred, *args):
        image_dets = []
        batch_size = loc_pred.size(0)
        for i in range(batch_size):
            dets = anchor_based_inference(
                loc_pred[i], cls_pred[i], self.anchors,
                self.conf_threshold, self.iou_threshold,
                self.topk, self.conf_strategy, self.nms
            )
            image_dets.append(dets)
        return image_dets


def nms_cpu(boxes, confidences, iou_threshold=0.5):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: Same length as boxes
        iou_threshold (float): Default value is 0.5
    Returns:
        indices: (N,)
    """
    return _C.nms_cpu(boxes, confidences, iou_threshold)


def soft_nms_cpu(boxes, confidences, iou_threshold=0.5, topk=100, conf_threshold=0.01):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: Same length as boxes
        iou_threshold (float): Default value is 0.5
        topk (int): Topk to remain
        conf_threshold (float): Filter bboxes whose score is less than it to speed up
    Returns:
        indices:
    """
    topk = min(len(boxes), topk)
    return _C.soft_nms_cpu(boxes, confidences, iou_threshold, topk, conf_threshold)


def mAP(detections: List[BBox], ground_truths: List[BBox], iou_threshold=.5):
    r"""
    Args:
        detections: sequences of BBox with `confidence`
        ground_truths: same size sequences of BBox
        iou_threshold:
    """
    image_dts = groupby(lambda b: b.image_id, detections)
    image_gts = groupby(lambda b: b.image_id, ground_truths)
    image_ids = image_gts.keys()
    maps = []
    for i in image_ids:
        i_dts = image_dts.get(i, [])
        i_gts = image_gts[i]
        class_dts = groupby(lambda b: b.category_id, i_dts)
        class_gts = groupby(lambda b: b.category_id, i_gts)
        classes = class_gts.keys()
        aps = []
        for c in classes:
            if c not in class_dts:
                aps.append(0)
                continue
            aps.append(AP(class_dts[c], class_gts[c], iou_threshold))
        maps.append(np.mean(aps))
    return np.mean(maps)

def AP(dts: List[BBox], gts: List[BBox], iou_threshold):
    TP = np.zeros(len(dts), dtype=np.uint8)
    n_positive = len(gts)
    seen = np.zeros(n_positive)
    for i, dt in enumerate(dts):
        ious = [iou_11(dt.bbox, gt.bbox) for gt in gts]
        j_max, iou_max = max(enumerate(ious), key=lambda x: x[1])
        if iou_max > iou_threshold:
            if not seen[j_max]:
                TP[i] = 1
                seen[j_max] = 1
    FP = 1 - TP
    acc_fp = np.cumsum(FP)
    acc_tp = np.cumsum(TP)
    recall = acc_tp / n_positive
    precision = acc_tp / (acc_fp + acc_tp)
    ap = average_precision(recall, precision)[0]
    return ap


def average_precision(recall, precision):
    mrec = [0, *recall, 1]
    mpre = [0, *precision, 0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap += np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mpre[:-1], mrec[:-1], ii


@curry
def misc_target_collate(batch):
    input, target = zip(*batch)
    return default_collate(input), Args(target)


def draw_bboxes(img, anns, categories=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for ann in anns:
        if isinstance(ann, BBox):
            ann = ann.to_ann()
        bbox = ann["bbox"]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if categories:
            ax.text(bbox[0], bbox[1],
                    categories[ann["category_id"]], fontsize=12)
    return fig, ax
