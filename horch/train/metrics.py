import time

from PIL import Image
from toolz import curry
from toolz.curried import get

import cv2
import torch
import numpy as np

from ignite.metrics import Accuracy as IgniteAccuracy
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from horch.functools import lmap
from horch.detection import BBox
from horch.detection.eval import mean_average_precision


class Average(Metric):

    def __init__(self, output_transform):
        super().__init__(output_transform)
        self._num_examples = 0
        self._sum = 0

    def reset(self):
        self._num_examples = 0
        self._sum = 0

    def update(self, output):
        val, n = output
        self._sum += val * n
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Metric must have at least one example before it can be computed')
        return self._sum / self._num_examples


def topk_accuracy(input, target, k=5):
    r"""
    Parameters
    ----------
    input : torch.Tensor
        Tensor of shape (batch_size, num_classes, ...)
    target : torch.Tensor
        Tensor of shape (batch_size, ...)
    k : int
        Default: 5
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


def bleu(preds, y, eos_index):
    preds = preds.argmax(dim=1)
    output = lmap(take_until_eos(eos_index), preds.tolist())
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
        preds: (batch_size, C, ...) or (batch_size, ...)
        y:      (batch_size, ...)  
    """

    def __init__(self, k=5):
        self.k = k
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        preds, target = get(["preds", "target"], output)
        preds = preds[0]
        target = target[0]
        return topk_accuracy(preds, target, k=self.k)


class TrainLoss(Average):
    r"""
    Reuse training loss to avoid extra computations.
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    @staticmethod
    def output_transform(output):
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
        preds, target, batch_size = get(["preds", "target", "batch_size"], output)
        loss = self.criterion(*preds, *target).item()
        return loss, batch_size


class Accuracy(IgniteAccuracy):
    r"""
    Inputs:
        preds: (batch_size, C, ...) or (batch_size, ...)
        y:      (batch_size, ...)  
    """

    def __init__(self):
        super().__init__(output_transform=self.output_transform)

    @staticmethod
    def output_transform(output):
        preds, target = get(["preds", "target"], output)
        return preds[0], target[0]


class Bleu(Average):

    def __init__(self, eos_index):
        self.eos_index = eos_index
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        preds, target, batch_size = get(["preds", "target", "batch_size"], output)
        preds = preds[0]
        target = target[0]
        return bleu(preds, target, self.eos_index), batch_size


class LossD(Average):

    def __init__(self):
        super().__init__(self.output_transform)

    @staticmethod
    def output_transform(output):
        lossD, batch_size = get(["lossD", "batch_size"], output)
        return lossD, batch_size


class LossG(Average):

    def __init__(self):
        super().__init__(self.output_transform)

    @staticmethod
    def output_transform(output):
        lossG, batch_size = get(["lossG", "batch_size"], output)
        return lossG, batch_size


class COCOEval(Metric):

    def __init__(self, annotations, iou_type='bbox'):
        self.annotations = annotations
        self.iou_type = iou_type
        super().__init__()

    def reset(self):
        from hpycocotools.coco import COCO
        self.coco_gt = COCO(self.annotations, verbose=False)
        self.res = []

    def update(self, output):
        target, image_dets, batch_size = get(
            ["target", "preds", "batch_size"], output)
        image_gts = target[0]
        for dets, gts in zip(image_dets, image_gts):
            image_id = gts[0]['image_id']
            for d in dets:
                d = {**d, 'image_id': image_id}
                self.res.append(d)

    def compute(self):
        from hpycocotools.coco import COCO
        img_ids = list(set([ d['image_id'] for d in self.res ]))
        imgs = self.coco_gt.loadImgs(img_ids)
        ann_ids = self.coco_gt.getAnnIds(imgIds=img_ids)
        anns = self.coco_gt.loadAnns(ann_ids)
        annotations = {
            **self.annotations,
            'images': imgs,
            'annotations': anns,
        }
        coco_gt = COCO(annotations, verbose=False)

        from hpycocotools.cocoeval import COCOeval
        from hpycocotools.mask import encode
        for dt in self.res:
            img = self.coco_gt.imgs[dt['image_id']]
            width = img['width']
            height = img['height']
            l, t, w, h = dt['bbox']
            l *= width
            t *= height
            w *= width
            h *= height
            dt['bbox'] = [l, t, w, h]
            if 'segmentation' in dt and self.iou_type == 'segm':
                r = int(l + w)
                b = int(t + h)
                l = max(0, int(l))
                t = max(0, int(t))
                r = min(r, width)
                b = min(b, height)
                w = r - l
                h = b - t
                m = np.zeros((height, width), dtype=np.uint8)
                segm = cv2.resize(dt['segmentation'], (w, h), interpolation=cv2.INTER_NEAREST)
                m[t:b, l:r] = segm
                dt['segmentation'] = encode(np.asfortranarray(m))

        coco_dt = coco_gt.loadRes(self.res)
        ev = COCOeval(coco_gt, coco_dt,
                      iouType=self.iou_type, verbose=False)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats[0]


def get_ap(values):
    values = np.array([np.mean(values), values[0], values[5]])
    return values


def mean_iou(pred, gt, num_classes):
    n = 0
    miou = 0
    for c in range(num_classes):
        class_iou = class_seg_iou(pred, gt, c)
        if class_iou is not None:
            n += 1
            miou += class_iou
    return miou / n


def class_seg_iou(pred, gt, class_):
    p = pred == class_
    g = gt == class_
    tp = (p & g).sum()
    fn = (~p & g).sum()
    fp = (p & ~g).sum()
    denominator = tp + fn + fp
    if denominator == 0:
        return None
    else:
        return tp / denominator


class SegmentationMeanIoU(Average):
    r"""
    Args:

    Inputs:
        target (list of list of horch.Detection.BBox): ground truth bounding boxes
        preds: (batch_size, h, w, c)
        predict: preds -> detected bounding boxes like `target` with additional `confidence`
    """

    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        super().__init__(self.output_transform)

    def output_transform(self, output):
        targets, preds, batch_size = get(
            ["target", "preds", "batch_size"], output)
        gts = targets[0]
        if isinstance(gts[0], Image.Image):
            gts = [np.array(img) for img in gts]
        elif torch.is_tensor(gts):
            gts = gts.cpu().byte().numpy()

        v = np.mean([
            mean_iou(preds[i], gts[i], self.num_classes)
            for i in range(batch_size)
        ])

        return v, batch_size


class PixelAccuracy(Average):
    r"""
    Args:

    Inputs:
        target (list of list of horch.Detection.BBox): ground truth bounding boxes
        preds: (batch_size, h, w, c)
        predict: preds -> detected bounding boxes like `target` with additional `confidence`
    """

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        super().__init__(self.output_transform)

    def output_transform(self, output):
        targets, preds, batch_size = get(
            ["target", "preds", "batch_size"], output)

        gts = targets[0]
        if isinstance(gts[0], Image.Image):
            gts = [np.array(img) for img in gts]
        elif torch.is_tensor(gts):
            gts = gts.cpu().byte().numpy()

        accs = []
        for i in range(batch_size):
            g = gts[i]
            p = preds[i]
            tp = (g == p).sum()
            if self.ignore_index is not None:
                tp += (g == self.ignore_index).sum()
            accs.append(tp / np.prod(g.shape))
        acc = np.mean(accs)
        return acc, batch_size


class CocoAveragePrecision(Average):
    r"""
    Args:

    Inputs:
        target (list of list of horch.Detection.BBox): ground truth bounding boxes
        preds: (batch_size, h, w, c)
        predict: preds -> detected bounding boxes like `target` with additional `confidence`
    """

    def __init__(self, iou_threshold=np.arange(0.5, 1, 0.05), get_value=get_ap):
        self.iou_threshold = iou_threshold
        self.get_value = get_value
        super().__init__(self.output_transform)

    def output_transform(self, output):
        target, image_dets, batch_size = get(
            ["target", "preds", "batch_size"], output)
        image_gts = target[0]
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

        values = np.array([np.mean([mean_average_precision(image_dets[i], image_gts[i],
                                        threshold) for i in range(batch_size)])
                           for threshold in self.iou_threshold])
        values = self.get_value(values)
        return values, batch_size


class MeanAveragePrecision(Metric):
    r"""
    Args:

    Inputs:
        target (list of list of horch.Detection.BBox): ground truth bounding boxes
        preds: (batch_size, h, w, c)
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        super().__init__()

    def reset(self):
        self.dts = []
        self.gts = []

    def update(self, output):
        target, image_dets, batch_size = get(
            ["target", "preds", "batch_size"], output)
        image_gts = target[0]

        for dts, gts in zip(image_dets, image_gts):
            image_id = gts[0]['image_id']
            for d in dts:
                d = {**d, 'image_id': image_id}
                self.dts.append(d)
            self.gts.extend(gts)

    def compute(self):
        dts = [BBox(**ann, format=BBox.LTWH) for ann in self.dts]
        gts = [BBox(**ann, format=BBox.LTWH) for ann in self.gts]

        mAP = mean_average_precision(dts, gts, iou_threshold=self.iou_threshold)

        return mAP
