from typing import List

import numpy as np
from toolz.curried import groupby

from horch.detection.bbox import BBox
from horch.detection.iou import iou_11


def mean_average_precision(dts: List[BBox], gts: List[BBox], iou_threshold=.5):
    r"""
    Args:
        dts: sequences of BBox with `confidence`
        gts: same size sequences of BBox
        iou_threshold:
    """
    class_dts = groupby(lambda b: b.category_id, dts)
    class_gts = groupby(lambda b: b.category_id, gts)
    # classes = set(class_gts.keys()).union(set(class_dts.keys()))
    classes = class_gts.keys()
    aps = []
    for c in classes:
        if c not in class_dts:
            aps.append(0)
            continue
        i_gts = groupby(lambda b: b.image_id, class_gts[c])
        n_positive = len(class_gts[c])
        dts = sorted(class_dts[c], key=lambda b: b.score, reverse=True)
        TP = np.zeros(len(dts), dtype=np.uint8)
        seen = {
            i: np.zeros(len(gts))
            for i, gts in i_gts.items()
        }
        for i, dt in enumerate(dts):
            image_id = dt.image_id
            if image_id not in i_gts:
                continue
            ious = [iou_11(dt.bbox, gt.bbox) for gt in i_gts[dt.image_id]]
            j_max, iou_max = max(enumerate(ious), key=lambda x: x[1])
            if iou_max > iou_threshold:
                if not seen[image_id][j_max]:
                    TP[i] = 1
                    seen[image_id][j_max] = 1
        FP = 1 - TP
        acc_fp = np.cumsum(FP)
        acc_tp = np.cumsum(TP)
        recall = acc_tp / n_positive
        precision = acc_tp / (acc_fp + acc_tp)
        ap = average_precision_pr(precision, recall)[0]
        aps.append(ap)
    return np.mean(aps)


def average_precision_pr(precision, recall):
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
