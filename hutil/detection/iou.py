import torch
from hutil.detection.bbox import BBox, transform_bboxes
from hutil import _C


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
    inter_w = torch.relu(xi2 - xi1)
    inter_h = torch.relu(yi2 - yi1)
    inter_area = inter_w * inter_h
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                  (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                  (boxes2[..., 3] - boxes2[..., 1])
    iou = inter_area / (boxes1_area + boxes2_area - inter_area)
    return iou


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


class IoUMN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        ious = _C.iou_mn_forward(boxes1, boxes2)
        ctx.save_for_backward(boxes1, boxes2, ious)

        return ious

    @staticmethod
    def backward(ctx, dious):
        dboxes1, dboxes2 = _C.iou_mn_backward(
            dious.contiguous(), *ctx.saved_variables)
        return dboxes1, dboxes2


def iou_mn(boxes1, boxes2):
    r"""
    Calculates IoU between boxes1 of size m and boxes2 of size n;

    Args:
        boxes1: (m, 4)
        boxes2: (n, 4)
    Returns:
        ious: (m, n)
    """
    return IoUMN.apply(boxes1, boxes2)
