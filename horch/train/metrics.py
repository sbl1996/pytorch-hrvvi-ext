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
from horch.detection import BBox
from horch.detection.eval import mean_average_precision, average_precision


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


class F1Score(Average):
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

        ys = targets[0]
        ps = preds[0]
        ps = ps.argmax(dim=1)

        p = ps.cpu().byte().numpy().ravel()
        y = ys.cpu().byte().numpy().ravel()
        sample_weight = y != self.ignore_index
        f1 = f1_score(y, p, sample_weight=sample_weight)
        return f1, batch_size



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

    def __init__(self, iou_threshold=0.5, interpolation='11point', ignore_difficult=True, class_names=None):
        assert interpolation in ['all', '11point']
        self.iou_threshold = iou_threshold
        self.interpolation = interpolation
        self.ignore_difficult = ignore_difficult
        self.class_names = class_names
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

        aps = average_precision(dts, gts, self.iou_threshold, self.interpolation == '11point', self.ignore_difficult)
        mAP = np.mean(list(aps.values()))
        if self.class_names:
            num_classes = len(self.class_names)
            d = {}
            for i in range(num_classes):
                d[self.class_names[i]] = aps.get(i + 1, 0) * 100
            d['ALL'] = mAP * 100
            d = pd.DataFrame({'mAP': d}).transpose()
            pd.set_option('precision', 1)
            print(d)
        return mAP


class MeanAveragePrecision2(Metric):
    r"""
    Args:

    Inputs:
        target (list of list of horch.Detection.BBox): ground truth bounding boxes
        preds: (batch_size, h, w, c)
    """

    def __init__(self, iou_threshold=0.5, interpolation='11point', ignore_difficult=True, class_names=None):
        assert interpolation in ['all', '11point']
        self.iou_threshold = iou_threshold
        self.interpolation = interpolation
        self.ignore_difficult = ignore_difficult
        self.class_names = class_names
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

        i2gts = groupby(lambda b: b.image_id, gts)
        i2dts = groupby(lambda b: b.image_id, dts)
        img_ids = i2gts.keys()

        det_boxes = []
        det_labels = []
        det_scores = []
        true_boxes = []
        true_labels = []
        true_difficulties = []

        for img_id in img_ids:
            igts = i2gts[img_id]
            idts = i2dts[img_id]
            det_boxes.append(torch.tensor([b.bbox for b in idts]))
            det_labels.append(torch.tensor([b.category_id for b in idts], dtype=torch.long))
            det_scores.append(torch.tensor([b.score for b in idts]))

            true_boxes.append(torch.tensor([b.bbox for b in igts]))
            true_labels.append(torch.tensor([b.category_id for b in igts], dtype=torch.long))
            true_difficulties.append(torch.tensor([b.is_difficult for b in igts], dtype=torch.uint8))

        ap, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        mAP = np.mean([ v for v in ap.values() if v != 0 ])
        return mAP


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)



def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label map
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)

        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision