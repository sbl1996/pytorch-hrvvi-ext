import sys
from typing import List, Any
from enum import Enum

from toolz.curried import get, countby, identity, valmap, groupby

import numpy as np

import torch
from torch.utils.data.dataloader import default_collate

from hutil.common import Args


def non_max_suppression(boxes, confidences, max_boxes, iou_threshold, inplace=False):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        confidences: (N,)
        max_boxes (int): 
        iou_threshold (float):
    Returns:
        indices: (N,)
    """
    N = len(boxes)
    if N <= max_boxes:
        return list(range(N))
    if not inplace:
        boxes = boxes.clone()
        confidences = confidences.clone()
    boxes = boxes.view(-1, 4)
    confidences = confidences.view(-1)
    indices = []
    while True:
        ind = confidences.argmax()
        indices.append(ind.item())
        boxes_iou = iou_1m(boxes[ind], boxes)
        mask = boxes_iou > iou_threshold
        boxes.masked_fill_(mask.unsqueeze(-1), 0)
        confidences.masked_fill_(mask, 0)
        if len(indices) >= max_boxes or confidences.sum() == 0:
            return indices


class BoundingBox:
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
        return "BoundingBox(image_name=%s, class_id=%s, box=%s, confidence=%s)" % (
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


def transform_bbox(box, format=BoundingBox.LTWH, to=BoundingBox.XYWH):
    r"""Transform the bounding box between different formats.

    Args:
        box (sequences of int): For detail of the box and formats, see `BoundingBox`.
        format: Format of given bounding box.
        to: Target format.

    """
    if format == BoundingBox.LTWH:
        if to == BoundingBox.LTWH:
            return list(box)
        elif to == BoundingBox.LTRB:
            return box_ltwh_to_ltrb(box)
        else:
            return box_ltwh_to_xywh(box)
    elif format == BoundingBox.LTRB:
        if to == BoundingBox.LTWH:
            return box_ltrb_to_ltwh(box)
        elif to == BoundingBox.LTRB:
            return list(box)
        else:
            return box_ltrb_to_xywh(box)
    else:
        if to == BoundingBox.LTWH:
            return box_xywh_to_ltwh(box)
        elif to == BoundingBox.LTRB:
            return box_xywh_to_ltrb(box)
        else:
            return list(box)


def transform_bboxes(boxes, format=BoundingBox.LTWH, to=BoundingBox.XYWH, inplace=False):
    r"""
    Transform the bounding box between different formats.

    Args:
        boxes (*, 4): For detail of the box and formats, see `BoundingBox`.
        format: Format of given bounding box.
        to: Target format.

    """
    if format == BoundingBox.LTWH:
        if to == BoundingBox.LTWH:
            return boxes
        elif to == BoundingBox.LTRB:
            return boxes_ltwh_to_ltrb(boxes, inplace=inplace)
        else:
            return boxes_ltwh_to_xywh(boxes, inplace=inplace)
    elif format == BoundingBox.LTRB:
        if to == BoundingBox.LTWH:
            return boxes_ltrb_to_ltwh(boxes, inplace=inplace)
        elif to == BoundingBox.LTRB:
            return boxes
        else:
            return boxes_ltrb_to_xywh(boxes, inplace=inplace)
    else:
        if to == BoundingBox.LTWH:
            return boxes_xywh_to_ltwh(boxes, inplace=inplace)
        elif to == BoundingBox.LTRB:
            return boxes_xywh_to_ltrb(boxes, inplace=inplace)
        else:
            return boxes


def iou_1m(box, boxes):
    r"""
    Calculates one-to-many ious by corners ([xmin, ymin, xmax, ymax]).

    Args:
        box: (4,)
        boxes: (*, 4)

    Returns:
        ious: (*,)
    """
    xi1 = torch.max(boxes[..., 0], box[0])
    yi1 = torch.max(boxes[..., 1], box[1])
    xi2 = torch.min(boxes[..., 2], box[2])
    yi2 = torch.min(boxes[..., 3], box[3])
    xdiff = xi2 - xi1
    ydiff = yi2 - yi1
    inter_area = xdiff * ydiff
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[..., 2] - boxes[..., 0]) * \
        (boxes[..., 3] - boxes[..., 1])
    union_area = boxes_area + box_area - inter_area

    iou = inter_area / union_area
    return iou.masked_fill_(xdiff < 0, 0).masked_fill_(ydiff < 0, 0)


def iou_b11(boxes1, boxes2):
    r"""
    Calculates batch one-to-one ious by corners ([xmin, ymin, xmax, ymax]).

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
    xdiff = xi2 - xi1
    ydiff = yi2 - yi1
    inter_area = xdiff * ydiff
    box1_area = (box1[2] - box1[0]) * \
        (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * \
        (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    if xdiff < 0 or ydiff < 0:
        iou = 0
    return iou


def mAP(detections: List[BoundingBox], ground_truths: List[BoundingBox], iou_threshold=.5):
    r"""
    Args:
        detections: sequences of BoundingBox with `confidence`
        ground_truths: same size sequences of BoundingBox
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


def scale_box(bbox, src_size, dst_size, format=BoundingBox.LTWH):
    if format == BoundingBox.LTWH:
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


def box_collate_fn(batch, label_field="category_id", box_field='bbox'):
    x, y = zip(*batch)
    ground_truths = []
    for i in range(len(y)):
        for ann in y[i]:
            ground_truths.append(
                BoundingBox(
                    image_name=i,
                    class_id=ann[label_field],
                    box=ann[box_field],
                    box_format=BoundingBox.LTWH,
                )
            )
    return default_collate(x), Args(ground_truths)
