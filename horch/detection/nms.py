from horch import _C


def nms(boxes, scores, iou_threshold=0.5):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        scores: Same length as boxes
        iou_threshold (float): Default value is 0.5
    Returns:
        indices: (N,)
    """
    return _C.nms(boxes, scores, iou_threshold)


def soft_nms_cpu(boxes, scores, iou_threshold=0.5, topk=100, min_score=0.01):
    r"""
    Args:
        boxes (tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
        scores: Same length as boxes
        iou_threshold (float): Default value is 0.5
        topk (int): Topk to remain
        min_score (float): Filter bboxes whose score is less than it to speed up
    Returns:
        indices:
    """
    topk = min(len(boxes), topk)
    return _C.soft_nms_cpu(boxes, scores, iou_threshold, topk, min_score)
