from horch import _C


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
