import time

from PIL import Image
from ignite.engine import Events
from toolz import curry
from toolz.curried import get, groupby

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from ignite.metrics import Accuracy as IgniteAccuracy
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from sklearn.metrics import roc_auc_score, f1_score

from horch.ops import inverse_sigmoid
from horch.functools import lmap


class IAverage(Metric):
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
        return self._sum / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine, name):
        result = self.compute()
        if torch.is_tensor(result) and len(result.shape) == 0:
            result = result.item()
        engine.state.metrics[name] = result

    def attach(self, engine, name):
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)


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


class EpochSummary(Metric):

    def __init__(self, metric_func):
        super().__init__()
        self.metric_func = metric_func

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, output):
        preds, target = get(["preds", "target"], output)
        self.preds.append(preds[0])
        self.targets.append(target[0])

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.metric_func(preds, targets)


class ROCAUC(Metric):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, output):
        preds, target = get(["preds", "target"], output)
        self.preds.append(preds[0])
        self.targets.append(target[0])

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        y_score = F.softmax(preds, dim=1)[:, 1].cpu().numpy()
        y_true = targets.cpu().numpy()
        return roc_auc_score(y_true, y_score)


class Bleu(Average):

    def __init__(self, eos_index):
        self.eos_index = eos_index
        super().__init__(output_transform=self.output_transform)

    def output_transform(self, output):
        preds, target, batch_size = get(["preds", "target", "batch_size"], output)
        preds = preds[0]
        target = target[0]
        return bleu(preds, target, self.eos_index), batch_size


class LossD(IAverage):

    def __init__(self):
        super().__init__(self.output_transform)

    @staticmethod
    def output_transform(output):
        lossD, batch_size = get(["lossD", "batch_size"], output)
        return lossD, batch_size


class LossG(IAverage):

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
        img_ids = list(set([d['image_id'] for d in self.res]))
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

    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        super().__init__(self.output_transform)

    def output_transform(self, output):
        targets, preds, batch_size = get(
            ["target", "preds", "batch_size"], output)
        targets = targets[0]
        preds = preds[0].argmax(dim=1)

        if isinstance(targets[0], Image.Image):
            targets = [np.array(img) for img in targets]
        elif torch.is_tensor(targets):
            targets = targets.cpu().byte().numpy()

        preds = preds.cpu().byte().numpy()

        v = np.mean([
            mean_iou(preds[i], targets[i], self.num_classes)
            for i in range(batch_size)
        ])

        return v, batch_size


class PixelAccuracy(Average):

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        super().__init__(self.output_transform)

    def output_transform(self, output):
        targets, preds, batch_size = get(
            ["target", "preds", "batch_size"], output)

        ys = targets[0]
        ps = preds[0]
        ps = ps.argmax(dim=1)

        accs = []
        for i in range(batch_size):
            y = ys[i]
            p = ps[i]
            tp = (y == p).sum()
            if self.ignore_index is not None:
                tp += (y == self.ignore_index).sum()
            accs.append(tp.cpu().item() / np.prod(y.shape))
        acc = np.mean(accs)
        return acc, batch_size


class F1Score(Metric):
    r"""
    """

    def __init__(self, threshold=0.5, ignore_index=None, eps=1e-8, from_logits=True):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.eps = eps
        self.from_logits = from_logits
        super().__init__(self.output_transform)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, output):
        tp, fp, fn = output
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        p = self.tp / (self.tp + self.fp + self.eps)
        r = self.tp / (self.tp + self.fn + self.eps)

        f1 = 2 * p * r / (p + r + self.eps)
        return f1

    def output_transform(self, output):
        targets, preds, batch_size = get(
            ["target", "preds", "batch_size"], output)

        y = targets[0]
        p = preds[0]
        if p.ndim == 4:
            if p.size(1) == 1:
                p = p.squeeze(1)
                if self.from_logits:
                    p = torch.sigmoid(p)
            elif p.size(1) == 2:
                if self.from_logits:
                    p = torch.softmax(p, dim=1)[:, 1, :, :]
        elif p.ndim == 3:
            if self.from_logits:
                p = torch.sigmoid(p)
        p = p > self.threshold
        p = p.long()
        y = y.long()

        if self.ignore_index is None:
            w = torch.ones_like(y)
        else:
            w = (y != self.ignore_index).long()
        tp = torch.sum(p * y * w).item()
        fp = torch.sum((1 - p) * y * w).item()
        fn = torch.sum(p * (1 - y) * w).item()
        return tp, fp, fn