import random
import sys
from typing import List, Any

from toolz import curry
from toolz.curried import get, countby, identity, valmap, groupby

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from hutil import one_hot
from hutil.common import Args
from hutil.nn.loss import focal_loss2
import hutil._C.detection as CD

__all__ = [
    "coords_to_target", "MultiLevelAnchorMatching", "BBox",
    "nms_cpu", "soft_nms_cpu", "non_max_suppression",
    "transform_bbox", "transform_bboxes", "box_collate_fn",
    "iou_1m", "iou_11", "iou_b11", "iou_mn_cpu", "draw_bboxes",
    "MultiBoxLoss", "get_anchors", "MultiLevelAnchorInference",
    "get_locations"]

def get_locations(size, strides):
    num_levels = int(np.log2(strides[-1]))
    lx, ly = size
    locations = [(lx, ly)]
    for _ in range(num_levels):
        if lx == 3:
            lx = 1
        else:
            lx = (lx - 1) // 2 + 1
        if ly == 3:
            ly = 1
        else:
            ly = (ly - 1) // 2 + 1
        locations.append((lx, ly))
    return locations[-len(strides):]


def inverse_sigmoid(x, eps=1e-3):
    x = torch.clamp(x, eps, 1-eps)
    return (x / (1 - x)).log_()
    

def yolo_coords_to_target(gt_box, anchors, location):
    location = gt_box.new_tensor(location)
    box_txty = inverse_sigmoid(gt_box[:2] * location % 1)
    box_twth = (gt_box[2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def coords_to_target(gt_box, anchors, *args):
    box_txty = (gt_box[:2] - anchors[..., :2]) \
        / anchors[..., 2:]
    box_twth = (gt_box[2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def get_anchors(lx, ly, sizes):
    anchors = torch.zeros(lx, ly, len(sizes), 4)
    anchors[:, :, :, 0] = (torch.arange(
        lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, len(sizes)) + 0.5) / lx
    anchors[:, :, :, 1] = (torch.arange(
        ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, len(sizes)) + 0.5) / ly
    anchors[:, :, :, 2:] = sizes
    return anchors


class MultiLevelAnchorMatching:
    r"""

    Args:
        multi_level_anchors: List of anchor boxes of shape `(lx, ly, #anchors, 4)`.
        max_iou: Whether assign anchors with max ious with ground truth boxes as positive anchors.
        pos_thresh: IOU threshold of positive anchors.
        neg_thresh: If provided, only non-positive anchors whose ious with all ground truth boxes are
            lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.
        get_label: Function to extract label from annotations.
        get_bbox: Function to extract bounding box from annotations. The bounding box must be a sequence
            containing [xmin, ymin, width, height].
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

    def __init__(self, multi_level_anchors, max_iou=True, 
        pos_thresh=0.5, neg_thresh=None, 
        get_label=lambda x: x['category_id'] + 1, 
        get_bbox=get("bbox"), 
        coords_to_target=coords_to_target, 
        debug=False):
        self.multi_level_anchors = multi_level_anchors
        self.max_iou = max_iou
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = get_label
        self.get_bbox = get_bbox
        self.coords_to_target = coords_to_target
        self.debug = debug

    def __call__(self, img, anns):
        locations = []
        flat_anchors = []
        loc_targets = []
        cls_targets = []
        negs = []
        for anchors in self.multi_level_anchors:
            lx, ly = anchors.size()[:2]
            locations.append((lx, ly))
            anchors = anchors.view(-1, 4)
            flat_anchors.append(anchors)
            num_anchors = anchors.size(0)
            loc_targets.append(torch.zeros(num_anchors, 4))
            cls_targets.append(torch.zeros(num_anchors, dtype=torch.long))
            if self.neg_thresh:
                negs.append(torch.ones(num_anchors, dtype=torch.uint8))
            else:
                negs.append(None)

        for ann in anns:
            label = self.get_label(ann)
            bbox = torch.tensor(transform_bbox(
                self.get_bbox(ann), BBox.LTWH, BBox.XYWH))

            max_ious = []
            for anchors, loc_t, cls_t, neg, location in zip(flat_anchors, loc_targets, cls_targets, negs, locations):
                ious = iou_1m(bbox, anchors, format=BBox.XYWH)
                max_ious.append(ious.max(dim=0))

                if self.pos_thresh:
                    pos = ious > self.pos_thresh
                    if pos.sum() != 0:
                        loc_t[pos] = self.coords_to_target(bbox, anchors[pos], location)
                        cls_t[pos] = label

                if self.neg_thresh:
                    neg &= ious < self.neg_thresh


            f_i, (max_iou, ind) = max(
                enumerate(max_ious), key=lambda x: x[1][0])
            if self.debug:
                print("Feature map %d: %f" % (f_i, max_iou))
                print("BBox   %s" % bbox.tolist())
                print("Anchor %s" % flat_anchors[f_i][ind].tolist())
            loc_targets[f_i][ind] = self.coords_to_target(
                bbox, flat_anchors[f_i][ind], locations[f_i])
            cls_targets[f_i][ind] = label

        if self.neg_thresh:
            ignores = [ ~neg for neg in negs ]
        # if len(flat_anchors) == 1:
        #     loc_targets = loc_targets[0]
        #     cls_targets = cls_targets[0]
        #     ignores = ignores[0]
        
        targets = [loc_targets, cls_targets]
        if self.neg_thresh:
            targets.append(ignores)
        return img, targets


class MultiBoxLoss(nn.Module):

    def __init__(self, neg_pos_ratio=None, p=0.1, criterion='softmax'):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.p = p
        if criterion == 'softmax':
            self.criterion = F.softmax
        elif criterion == 'focal':
            self.criterion = focal_loss2
        else:
            raise ValueError("criterion must be one of softmax or focal")

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets, ignores=None, *args):
        cls_loss = 0
        loc_loss = 0
        if ignores is None:
            ignores = [None] * len(loc_preds)
        for loc_p, cls_p, loc_t, cls_t, ignore in zip(loc_preds, cls_preds, loc_targets, cls_targets, ignores):
            pos = cls_t != 0
            num_pos = pos.sum().item()
            if num_pos == 0:
                continue

            loc_loss += F.smooth_l1_loss(
                loc_p[pos], loc_t[pos], reduction='sum') / num_pos

            # Hard Negative Mining
            if self.neg_pos_ratio:
                cls_loss_pos = self.criterion(
                    cls_p[pos], cls_t[pos], reduction='sum')

                mask = ~pos
                if ignore is not None:
                    mask = mask & (~ignore)
                cls_p_neg = cls_p[mask]
                cls_loss_neg = -F.log_softmax(cls_p_neg, dim=1)[..., 0]
                num_neg = min(self.neg_pos_ratio * num_pos, len(cls_loss_neg))
                cls_loss_neg = torch.topk(cls_loss_neg, num_neg)[0].sum()
                cls_loss += (cls_loss_pos + cls_loss_neg) / num_pos
            elif ignore is not None:
                if self.criterion == focal_loss2:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                cls_loss_pos = self.criterion(
                    cls_p[pos], cls_t[pos], reduction='sum')

                mask = ~pos & (~ignore)
                cls_loss_neg = self.criterion(
                    cls_p[mask], cls_t[mask], reduction='sum')
                cls_loss += (cls_loss_pos + cls_loss_neg) / num_pos
            else:
                if self.criterion == F.softmax:
                    cls_p = cls_p.transpose(1, -1)
                elif self.criterion == focal_loss2:
                    cls_t = one_hot(cls_t, C=cls_p.size(-1))
                cls_loss += self.criterion(cls_p, cls_t,
                                        reduction='sum') / num_pos
        loss = cls_loss + loc_loss
        if random.random() < self.p:
            print("loc: %.4f | cls: %.4f" %
                  (loc_loss.item(), cls_loss.item()))
        return loss


class MultiLevelAnchorInference:

    def __init__(self, size, multi_level_anchors, conf_threshold=0.01, topk_per_level=300, topk=100, iou_threshold=0.45, conf_strategy='softmax'):
        self.width, self.height = size
        self.multi_level_anchors = multi_level_anchors
        self.conf_threshold = conf_threshold
        self.topk_per_level = topk_per_level
        self.topk = topk
        self.iou_threshold = iou_threshold
        assert conf_strategy in [
            'softmax', 'sigmoid'], "conf_strategy must be softmax or sigmoid"
        self.conf_strategy = conf_strategy

    def __call__(self, loc_preds, cls_preds, *args):
        detections = []
        batch_size = loc_preds[0].size(0)
        for i in range(batch_size):
            boxes = []
            confs = []
            labels = []
            for loc_p, cls_p, anchors in zip(loc_preds, cls_preds, self.multi_level_anchors):
                loc_p = loc_p[i]
                cls_p = cls_p[i]
                anchors = anchors.view(-1, 4)

                if self.conf_strategy == 'softmax':
                    conf = torch.softmax(cls_p, dim=1)
                else:
                    conf = torch.sigmoid_(cls_p)
                conf = conf[..., 1:]
                conf, label = torch.max(conf, dim=1)

                mask = conf > self.conf_threshold
                conf = conf[mask]
                label = label[mask]
                box = loc_p[mask]
                anchors = anchors[mask]

                box[:, :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
                box[:, 2:].exp_().mul_(anchors[:, 2:])
                box[:, [0, 2]] *= self.width
                box[:, [1, 3]] *= self.height

                if len(conf) > self.topk_per_level:
                    conf, indices = conf.topk(self.topk_per_level)
                    box = box[indices]
                    label = label[indices]

                boxes.append(box)
                confs.append(conf)
                labels.append(label)

            boxes = torch.cat(boxes, dim=0)
            confs = torch.cat(confs, dim=0)
            labels = torch.cat(labels, dim=0)

            boxes = transform_bboxes(
                boxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
            confs = confs.cpu()
            indices = soft_nms_cpu(
                boxes, confs, self.iou_threshold, self.topk)
            for ind in indices:
                detections.append(
                    BBox(
                        image_name=i,
                        class_id=labels[ind].item(),
                        box=boxes[ind].tolist(),
                        confidence=confs[ind].item(),
                        box_format=BBox.LTRB,
                    )
                )

        return detections


def nms_cpu(boxes, confidences, iou_threshold=0.5):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: Same length as boxes
        iou_threshold (float): Default value is 0.5
    Returns:
        indices: (N,)
    """
    return CD.nms_cpu(boxes, confidences, iou_threshold)


def soft_nms_cpu(boxes, confidences, iou_threshold=0.5, topk=100, conf_threshold=0.01):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: Same length as boxes
        iou_threshold (float): Default value is 0.5
    Returns:
        indices:
    """
    topk = min(len(boxes), topk)
    return CD.soft_nms_cpu(boxes, confidences, iou_threshold, topk, conf_threshold)


def soft_nms(boxes, confidences, iou_threshold, topk=10):
    r"""
    Args:
        boxes(tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: (N,)
    Returns:
        indices: (N,)
    """
    confidences = confidences.clone()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    zero = boxes.new_tensor(0)
    indices = []

    while True:
        i = confidences.argmax()
        # print(i)
        # print(confidences[i])
        indices.append(i)
        if len(indices) >= topk:
            break
        xx1 = torch.max(x1[i], x1)
        yy1 = torch.max(y1[i], y1)
        xx2 = torch.min(x2[i], x2)
        yy2 = torch.min(y2[i], y2)

        w = torch.max(zero, xx2 - xx1 + 1)
        h = torch.max(zero, yy2 - yy1 + 1)

        inter = w * h
        ious = inter / (areas[i] + areas - inter)
        mask = ious >= iou_threshold
        confidences[mask] *= (1 - ious)[mask]
    return boxes.new_tensor(indices, dtype=torch.long)


def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    r"""
    Args:
        boxes(tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: (N,)
        max_boxes(int):
        iou_threshold(float):
    Returns:
        indices: (N,)
    """
    N = len(boxes)
    confs, orders = confidences.sort(descending=True)
    boxes = boxes[orders]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    suppressed = confidences.new_zeros(N, dtype=torch.uint8)

    zero = boxes.new_tensor(0)

    for i in range(N):
        if suppressed[i] == 1:
            continue
        xx1 = torch.max(x1[i], x1[i+1:])
        yy1 = torch.max(y1[i], y1[i+1:])
        xx2 = torch.min(x2[i], x2[i+1:])
        yy2 = torch.min(y2[i], y2[i+1:])

        w = torch.max(zero, xx2 - xx1 + 1)
        h = torch.max(zero, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[i+1:] - inter)
        suppressed[i+1:][iou > iou_threshold] = 1
    return orders[torch.nonzero(suppressed == 0).squeeze(1)]


class BBox:
    LTWH = 0  # [xmin, ymin, width, height]
    LTRB = 1  # [xmin, ymin, xmax,  ymax]
    XYWH = 2  # [cx,   cy,   width, height]

    def __init__(self, image_name, class_id, box, confidence=None, box_format=1):
        self.image_name = image_name
        self.class_id = class_id
        self.confidence = confidence
        self.box = transform_bbox(
            box, format=box_format, to=1)

    def __repr__(self):
        return "BBox(image_name=%s, class_id=%s, box=%s, confidence=%s)" % (
            self.image_name, self.class_id, self.box, self.confidence
        )


def draw_bboxes(img, anns, with_label=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for ann in anns:
        bbox = ann["bbox"]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if with_label:
            ax.text(bbox[0], bbox[1], ann["label"], fontsize=12)
    return fig, ax


def boxes_ltwh_to_ltrb(boxes, inplace=False):
    if inplace:
        boxes[..., 2:] += boxes[..., :2]
        return boxes
    boxes_lt = boxes[..., :2]
    boxes_wh = boxes[..., 2:]
    boxes_rb = boxes_lt + boxes_wh
    return torch.cat((boxes_lt, boxes_rb), dim=-1)


def boxes_ltwh_to_xywh(boxes, inplace=False):
    if inplace:
        boxes[..., :2] += boxes[..., 2:] / 2
        return boxes
    boxes_lt = boxes[..., :2]
    boxes_wh = boxes[..., 2:]
    boxes_xy = boxes_lt - boxes_wh / 2
    return torch.cat((boxes_lt, boxes_xy), dim=-1)


def boxes_ltrb_to_ltwh(boxes, inplace=False):
    if inplace:
        boxes[..., 2:] -= boxes[..., :2]
        return boxes
    boxes_lt = boxes[..., :2]
    boxes_rb = boxes[..., 2:]
    boxes_wh = boxes_rb - boxes_lt
    return torch.cat((boxes_lt, boxes_wh), dim=-1)


def boxes_ltrb_to_xywh(boxes, inplace=False):
    if inplace:
        boxes[..., 2:] -= boxes[..., :2]
        boxes[..., :2] += boxes[..., 2:] / 2
        return boxes
    boxes_lt = boxes[..., :2]
    boxes_rb = boxes[..., 2:]
    boxes_wh = boxes_rb - boxes_lt
    boxes_xy = (boxes_lt + boxes_rb) / 2
    return torch.cat((boxes_xy, boxes_wh), dim=-1)


def boxes_xywh_to_ltrb(boxes, inplace=False):
    if inplace:
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]
        return boxes
    boxes_xy = boxes[..., :2]
    boxes_wh = boxes[..., 2:]
    halve = boxes_wh / 2
    xymin = boxes_xy - halve
    xymax = boxes_xy + halve
    return torch.cat((xymin, xymax), dim=-1)


def boxes_xywh_to_ltwh(boxes, inplace=False):
    if inplace:
        boxes[..., :2] -= boxes[..., 2:] / 2
        return boxes
    boxes_lt = boxes[..., :2] - boxes[..., 2:] / 2
    boxes_wh = boxes[..., 2:]
    return torch.cat((boxes_lt, boxes_wh), dim=-1)


def box_ltwh_to_xywh(box):
    l, t, w, h = box
    x = l + w / 2
    y = t + h / 2
    return [x, y, w, h]


def box_ltwh_to_ltrb(box):
    l, t, w, h = box
    r = l + w
    b = t + h
    return [l, t, r, b]


def box_ltrb_to_ltwh(box):
    l, t, r, b = box
    w = r - l
    h = b - t
    return [l, t, w, h]


def box_ltrb_to_xywh(box):
    l, t, r, b = box
    x = (l + r) / 2
    y = (t + b) / 2
    w = r - l
    h = b - t
    return [x, y, w, h]


def box_xywh_to_ltwh(box):
    x, y, w, h = box
    l = x - w / 2
    t = y - h / 2
    return [l, t, w, h]


def box_xywh_to_ltrb(box):
    x, y, w, h = box
    hw = w / 2
    hh = h / 2
    l = x - hw
    t = y - hh
    r = x + hw
    b = y + hh
    return [l, t, r, b]


def transform_bbox(box, format=BBox.LTWH, to=BBox.XYWH):
    r"""Transform the bounding box between different formats.

    Args:
        box(sequences of int): For detail of the box and formats, see `BBox`.
        format: Format of given bounding box.
        to: Target format.

    """
    if format == BBox.LTWH:
        if to == BBox.LTWH:
            return list(box)
        elif to == BBox.LTRB:
            return box_ltwh_to_ltrb(box)
        else:
            return box_ltwh_to_xywh(box)
    elif format == BBox.LTRB:
        if to == BBox.LTWH:
            return box_ltrb_to_ltwh(box)
        elif to == BBox.LTRB:
            return list(box)
        else:
            return box_ltrb_to_xywh(box)
    else:
        if to == BBox.LTWH:
            return box_xywh_to_ltwh(box)
        elif to == BBox.LTRB:
            return box_xywh_to_ltrb(box)
        else:
            return list(box)


def transform_bboxes(boxes, format=BBox.LTWH, to=BBox.XYWH, inplace=False):
    r"""
    Transform the bounding box between different formats.

    Args:
        boxes(*, 4): For detail of the box and formats, see `BBox`.
        format: Format of given bounding box.
        to: Target format.

    """
    if format == BBox.LTWH:
        if to == BBox.LTWH:
            return boxes
        elif to == BBox.LTRB:
            return boxes_ltwh_to_ltrb(boxes, inplace=inplace)
        else:
            return boxes_ltwh_to_xywh(boxes, inplace=inplace)
    elif format == BBox.LTRB:
        if to == BBox.LTWH:
            return boxes_ltrb_to_ltwh(boxes, inplace=inplace)
        elif to == BBox.LTRB:
            return boxes
        else:
            return boxes_ltrb_to_xywh(boxes, inplace=inplace)
    else:
        if to == BBox.LTWH:
            return boxes_xywh_to_ltwh(boxes, inplace=inplace)
        elif to == BBox.LTRB:
            return boxes_xywh_to_ltrb(boxes, inplace=inplace)
        else:
            return boxes


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def iou_1m(box, boxes, format=BBox.LTRB):
    r"""
    Calculates one-to-many ious by corners([xmin, ymin, xmax, ymax]).

    Args:
        box: (4,)
        boxes: (*, 4)

    Returns:
        ious: (*,)
    """
    box = transform_bboxes(box, format=format, to=BBox.LTRB)
    boxes = transform_bboxes(boxes, format=format, to=BBox.LTRB)
    xi1 = torch.max(boxes[..., 0], box[0])
    yi1 = torch.max(boxes[..., 1], box[1])
    xi2 = torch.min(boxes[..., 2], box[2])
    yi2 = torch.min(boxes[..., 3], box[3])
    inter_w = torch.relu(xi2 - xi1)
    inter_h = torch.relu(yi2 - yi1)
    inter_area = inter_w * inter_h
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[..., 2] - boxes[..., 0]) * \
        (boxes[..., 3] - boxes[..., 1])
    union_area = boxes_area + box_area - inter_area
    iou = inter_area / union_area
    return iou

class IoUMN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        ious = CD.iou_mn_forward_cpu(boxes1, boxes2)
        ctx.save_for_backward(boxes1, boxes2, ious)

        return ious

    @staticmethod
    def backward(ctx, dious):
        dboxes1, dboxes2 = CD.iou_mn_backward_cpu(dious.contiguous(), *ctx.saved_variables)
        return dboxes1, dboxes2

def iou_mn_cpu(boxes1, boxes2):
    r"""
    Calculates IoU between boxes1 of size m and boxes2 of size n;

    Args:
        boxes1: (m, 4)
        boxes2: (n, 4)
    Returns:
        ious: (m, n)
    """
    return IoUMN.apply(boxes1, boxes2)

def iou_b11(boxes1, boxes2):
    r"""
    Calculates batch one-to-one ious by corners([xmin, ymin, xmax, ymax]).

    Args:
        boxes1: (*, 4)
        boxes2: (*, 4)

    Returns:
        ious: (*,)
    """

    xi1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    yi1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    xi2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    yi2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    xdiff = xi2 - xi1
    ydiff = yi2 - yi1
    inter_area = xdiff * ydiff
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    iou = inter_area / (boxes1_area + boxes2_area - inter_area)
    return iou.masked_fill_(xdiff < 0, 0).masked_fill_(ydiff < 0, 0)


def iou_11(box1, box2):
    r"""
    Args:
        box1: [left, top, right, bottom]
        box2: [left, top, right, bottom]
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    box1_area = (box1[2] - box1[0]) * \
        (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * \
        (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def mAP(detections: List[BBox], ground_truths: List[BBox], iou_threshold=.5):
    r"""
    Args:
        detections: sequences of BBox with `confidence`
        ground_truths: same size sequences of BBox
    """
    ret = []
    class_detections = groupby(lambda b: b.class_id, detections)
    class_ground_truths = groupby(lambda b: b.class_id, ground_truths)
    classes = class_ground_truths.keys()
    for c in classes:
        if c not in class_detections:
            ret.append(0)
            continue

        dects = class_detections[c]
        gts = class_ground_truths[c]
        n_positive = len(gts)

        dects = sorted(dects, key=lambda b: b.confidence, reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        seen = {k: np.zeros(n)
                for k, n in countby(lambda b: b.image_name, gts).items()}

        image_gts = groupby(lambda b: b.image_name, gts)
        for i, d in enumerate(dects):
            gt = image_gts.get(d.image_name) or []
            iou_max = sys.float_info.min
            for j, g in enumerate(gt):
                iou = iou_11(d.box, g.box)
                if iou > iou_max:
                    iou_max = iou
                    j_max = j
            if iou_max > iou_threshold:
                if not seen[d.image_name][j_max]:
                    TP[i] = 1
                    seen[d.image_name][j_max] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        recall = acc_TP / n_positive
        precision = np.divide(acc_TP, (acc_FP + acc_TP))
        t = average_precision(recall, precision)
        ret.append(t[0])
    return sum(ret) / len(ret)


def average_precision(recall, precision):
    mrec = [0, *recall, 1]
    mpre = [0, *precision, 0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap += np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    return ap, mpre[:-1], mrec[:-1], ii


def scale_box(bbox, src_size, dst_size, format=BBox.LTWH):
    if format == BBox.LTWH:
        iw, ih = src_size
        ow, oh = dst_size
        sw = ow / iw
        sh = oh / ih
        bbox[0] *= sw
        bbox[1] *= sh
        bbox[2] *= sw
        bbox[3] *= sh
        return bbox
    else:
        raise NotImplementedError


@curry
def box_collate_fn(batch, get_label=get("category_id"), get_bbox=get('bbox')):
    x, y = zip(*batch)
    ground_truths = []
    for i in range(len(y)):
        for ann in y[i]:
            ground_truths.append(
                BBox(
                    image_name=i,
                    class_id=get_label(ann),
                    box=get_bbox(ann),
                    box_format=BBox.LTWH,
                )
            )
    return default_collate(x), Args(ground_truths)
