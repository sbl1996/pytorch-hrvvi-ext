import time
from toolz import curry
from toolz.curried import get, groupby

import cv2
import torch
import numpy as np

from ignite.metrics import Accuracy as IgniteAccuracy
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from hutil.functools import lmap
from hutil.detection import mAP, BBox


class Average(Metric):

    def __init__(self, output_transform):
        super().__init__(output_transform)

    def reset(self):
        self._num_examples = 0
        self._sum = 0

    def update(self, output):
        val, N = output
        self._sum += val * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Metric must have at least one example before it can be computed')
        return self._sum / self._num_examples


def topk_accuracy(input, target, k):
    r"""
    Args:
        input:  (batch, C, *)
        target: (batch, *)
    """
    num_examples = np.prod(target.size())
    topk_pred = torch.topk(input, k=k, dim=1)[1]
    num_corrects = torch.sum(topk_pred == target.unsqueeze(1)).item()
    accuracy = num_corrects / num_examples
    return accuracy, num_examples


@curry
def take_until_eos(eos_index, tokens):
    for i, token in enumerate(tokens):
        if token == eos_index:
            return tokens[:i]
    return tokens


def bleu(y_pred, y, eos_index):
    y_pred = y_pred.argmax(dim=1)
    output = lmap(take_until_eos(eos_index), y_pred.tolist())
    target = lmap(take_until_eos(eos_index), y.tolist())
    target = lmap(lambda x: [x], target)
    score = corpus_bleu(
        target, output, smoothing_function=SmoothingFunction().method1)
    return score


class TopKAccuracy(Average):
    r"""
    Args:
        k: default to 5
    Inputs:
        y_pred: (batch_size, C, ...) or (batch_size, ...)  
        y:      (batch_size, ...)  
    """

    def __init__(self, k=5):
        self.k = k
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y = get(["y_pred", "y"], output)
        y_pred = y_pred[0]
        y = y[0]
        return topk_accuracy(y_pred, y, k=self.k)


class TrainLoss(Average):
    r"""
    Reuse training loss to avoid extra computations.
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        loss, batch_size = get(["loss", "batch_size"], output)
        return loss, batch_size


class Loss(Average):
    r"""
    Reuse training loss to avoid extra computations.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y, batch_size = get(["y_pred", "y", "batch_size"], output)
        loss = self.criterion(*y_pred, *y).item()
        return loss, batch_size


class Accuracy(IgniteAccuracy):
    r"""
    Inputs:
        y_pred: (batch_size, C, ...) or (batch_size, ...)  
        y:      (batch_size, ...)  
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y = get(["y_pred", "y"], output)
        return y_pred[0], y[0]


class Bleu(Average):

    def __init__(self, eos_index):
        self.eos_index = eos_index
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        y_pred, y, batch_size = get(["y_pred", "y", "batch_size"], output)
        y_pred = y_pred[0]
        y = y[0]
        return bleu(y_pred, y, self.eos_index), batch_size


class LossD(Average):

    def __init__(self):
        super().__init__(self.output_transform)

    def output_transform(self, output):
        lossD, batch_size = get(["lossD", "batch_size"], output)
        return lossD, batch_size


class LossG(Average):

    def __init__(self):
        super().__init__(self.output_transform)

    def output_transform(self, output):
        lossG, batch_size = get(["lossG", "batch_size"], output)
        return lossG, batch_size


class COCOEval(Metric):

    def __init__(self, inference, annotations, iou_type='bbox'):
        self.inference = inference
        self.annotations = annotations
        self.iou_type = iou_type
        super().__init__()

    def reset(self):
        from pycocotools.coco import COCO
        self.coco_gt = COCO(self.annotations, verbose=False)
        self.res = []

    def update(self, output):
        y, y_pred, batch_size = get(
            ["y", "y_pred", "batch_size"], output)
        image_gts = y[0]
        image_dets = self.inference(*y_pred)
        for dets, gts in zip(image_dets, image_gts):
            image_id = gts[0]['image_id']
            for d in dets:
                d['image_id'] = image_id
                self.res.append(d)

    def compute(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools.mask import encode

        if self.iou_type == 'segm':
            for dt in self.res:
                img = self.coco_gt.imgs[dt['image_id']]
                width = img['width']
                height = img['height']
                l, t, w, h = [int(v) for v in dt['bbox']]
                r = l + w
                b = t + h
                l = max(0, l)
                t = max(0, t)
                r = min(r, width)
                b = min(b, height)
                w = r - l
                h = b - t
                m = np.zeros((height, width), dtype=np.uint8)
                segm = cv2.resize(dt['segmentation'], (w, h))
                m[t:b, l:r] = segm
                dt['segmentation'] = encode(np.asfortranarray(m))
        elif self.iou_type == 'bbox':
            for dt in self.res:
                img = self.coco_gt.imgs[dt['image_id']]
                width = img['width']
                height = img['height']
                sw = width / dt['scale_w']
                sh = height / dt['scale_h']
                l, t, w, h = [int(v) for v in dt['bbox']]
                l *= sw
                t *= sh
                w *= sw
                h *= sh
                dt['bbox'] = [l, t, w, h]

        coco_dt = self.coco_gt.loadRes(self.res)
        ev = COCOeval(self.coco_gt, coco_dt,
                      iouType=self.iou_type, verbose=False)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats[0]


def get_ap(values):
    values = np.array([np.mean(values), values[0], values[5]])
    return values


class CocoAveragePrecision(Average):
    r"""
    Args:

    Inputs:
        y (list of list of hutil.detection.BBox): ground truth bounding boxes
        y_pred: (batch_size, h, w, c)
        predict: y_pred -> detected bounding boxes like `y` with additional `confidence`
    """

    def __init__(self, inference, iou_threshold=np.arange(0.5, 1, 0.05), get_value=get_ap):
        self.inference = inference
        self.iou_threshold = iou_threshold
        self.get_value = get_value
        super().__init__(self.output_transform)

    def output_transform(self, output):
        y, y_pred, batch_size = get(
            ["y", "y_pred", "batch_size"], output)
        image_gts = y[0]
        image_dets = self.inference(*y_pred)
        for dets, gts in zip(image_dets, image_gts):
            image_id = gts[0]['image_id']
            for d in dets:
                d['image_id'] = image_id

        image_dets = [
            [BBox(**ann, format=BBox.LTWH) for ann in idets]
            for idets in image_dets
        ]
        image_gts = [
            [BBox(**ann, format=BBox.LTWH) for ann in igts]
            for igts in image_gts
        ]

        values = np.array([np.mean([mAP(image_dets[i], image_gts[i],
                                        threshold) for i in range(batch_size)])
                           for threshold in self.iou_threshold])
        values = self.get_value(values)
        return values, batch_size


class MeanAveragePrecision(Average):
    r"""
    Args:

    Inputs:
        y (list of list of hutil.detection.BBox): ground truth bounding boxes
        y_pred: (batch_size, h, w, c)
        predict: y_pred -> detected bounding boxes like `y` with additional `confidence`
    """

    def __init__(self, inference, iou_threshold=0.5):
        self.inference = inference
        self.iou_threshold = iou_threshold
        super().__init__(self.output_transform)

    def output_transform(self, output):
        y, y_pred, batch_size = get(
            ["y", "y_pred", "batch_size"], output)
        image_gts = y[0]
        image_dets = self.inference(*y_pred)

        for dets, gts in zip(image_dets, image_gts):
            image_id = gts[0]['image_id']
            for d in dets:
                d['image_id'] = image_id

        image_dets = [
            [BBox(**ann, format=BBox.LTWH) for ann in idets]
            for idets in image_dets
        ]
        image_gts = [
            [BBox(**ann, format=BBox.LTWH) for ann in igts]
            for igts in image_gts
        ]
        values = np.mean([mAP(image_dets[i], image_gts[i],
                              self.iou_threshold) for i in range(batch_size)])
        return values, batch_size
