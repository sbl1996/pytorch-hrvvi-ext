from typing import List

import numpy as np
from toolz.curried import groupby

from hutil.detection.bbox import BBox
from hutil.detection.iou import iou_11


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
