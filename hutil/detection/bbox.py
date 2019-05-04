import numpy as np

import torch

class BBox:
    LTWH = 0  # [xmin, ymin, width, height]
    LTRB = 1  # [xmin, ymin, xmax,  ymax]
    XYWH = 2  # [cx,   cy,   width, height]

    @staticmethod
    def to_absolute(bbox, size, inplace=True):
        if torch.is_tensor(bbox):
            return bboxes_to_absolute(bbox, size, inplace=inplace)
        else:
            return bbox_to_absolute(bbox, size, inplace=inplace)

    @staticmethod
    def to_percent(bbox, size, inplace=True):
        if torch.is_tensor(bbox):
            return bboxes_to_percent(bbox, size, inplace=inplace)
        else:
            return bbox_to_percent(bbox, size, inplace=inplace)

    @staticmethod
    def convert(bbox, format=0, to=1, inplace=False):
        if torch.is_tensor(bbox) or (isinstance(bbox, np.ndarray) and inplace):
            return transform_bboxes(bbox, format=format, to=to, inplace=inplace)
        else:
            return transform_bbox(bbox, format, to)

    def __init__(self, image_id, category_id, bbox, score=None, format=1, area=None, segmentation=None, **kwargs):
        self.image_id = image_id
        self.category_id = category_id
        self.score = score
        self.bbox = transform_bbox(
            bbox, format=format, to=1)
        self.area = area or get_bbox_area(bbox, format=format)
        self.segmentation = segmentation

    def __repr__(self):
        return "BBox(image_id=%s, category_id=%s, bbox=%s, score=%s, area=%s)" % (
            self.image_id, self.category_id, self.bbox, self.score, self.area
        )

    def to_ann(self):
        bbox = transform_bbox(
            self.bbox, format=1, to=0)
        ann = {
            'image_id': self.image_id,
            'category_id': self.category_id,
            'score': self.score,
            'bbox': bbox,
            'area': self.area,
        }
        if self.segmentation is not None:
            ann['segmentation'] = self.segmentation
        return ann


def get_bbox_area(bbox, format=BBox.LTRB):
    if format == BBox.LTRB:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    else:
        return bbox[2] * bbox[3]


def bbox_to_percent(bbox, size, inplace=False):
    if not inplace:
        bbox = list(bbox)
    w, h = size
    bbox[0] /= w
    bbox[1] /= h
    bbox[2] /= w
    bbox[3] /= h
    return bbox


def bbox_to_absolute(bbox, size, inplace=False):
    if not inplace:
        bbox = list(bbox)
    w, h = size
    bbox[0] *= w
    bbox[1] *= h
    bbox[2] *= w
    bbox[3] *= h
    return bbox


def bboxes_to_percent(bboxes, size, inplace=False):
    if not inplace:
        bboxes = bboxes.clone()
    w, h = size
    bboxes[..., 0] /= w
    bboxes[..., 1] /= h
    bboxes[..., 2] /= w
    bboxes[..., 3] /= h
    return bboxes


def bboxes_to_absolute(bboxes, size, inplace=False):
    if not inplace:
        bboxes = bboxes.clone()
    w, h = size
    bboxes[..., 0] *= w
    bboxes[..., 1] *= h
    bboxes[..., 2] *= w
    bboxes[..., 3] *= h
    return bboxes


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
    return torch.cat((boxes_xy, boxes_wh), dim=-1)


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

    Parameters
    ----------
    boxes : array_like
    format
        Format of given bounding box. For detail of the box and formats, see `BBox`.
    to
        Target format.
    inplace : bool
        Whether to transform bboxes inplace.
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
