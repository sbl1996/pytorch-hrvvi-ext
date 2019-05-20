from typing import Sequence
from math import sqrt

from toolz import curry

import numpy as np

import torch
from torch.utils.data.dataloader import default_collate

from horch.common import Args

from horch.detection.bbox import BBox
from horch.detection.iou import iou_11, iou_b11, iou_1m, iou_mn
from horch.detection.anchor import find_priors_kmeans, find_priors_coco
from horch.detection.nms import nms, soft_nms_cpu
from horch.detection.eval import mAP

__all__ = [
    "BBox", "nms", "soft_nms_cpu", "misc_target_collate",
    "iou_1m", "iou_11", "iou_b11", "iou_mn", "draw_bboxes",
    "get_locations", "calc_anchor_sizes", "generate_anchors",
    "generate_mlvl_anchors", "generate_anchors_with_priors",
    "find_priors_kmeans", "mAP", "find_priors_coco"
]


def _pair(x):
    if not isinstance(x, Sequence):
        return x, x
    else:
        return x


def get_locations(size, strides, pad_threshold=3):
    num_levels = int(np.log2(strides[-1]))
    lx, ly = size
    locations = [(lx, ly)]
    for _ in range(num_levels):
        if lx <= pad_threshold:
            lx -= 2
        else:
            lx = (lx - 1) // 2 + 1
        if ly <= pad_threshold:
            ly -= 2
        else:
            ly = (ly - 1) // 2 + 1
        locations.append((lx, ly))
    return locations[-len(strides):]


def calc_anchor_sizes(size, aspect_ratios, scales=(1,), dtype=torch.float32):
    w, h = _pair(size)
    sw = [w * sqrt(ars) * s for ars in aspect_ratios for s in scales]
    sh = [h / sqrt(ars) * s for ars in aspect_ratios for s in scales]
    return torch.tensor([sw, sh], dtype=dtype).transpose(1, 0)


def generate_mlvl_anchors(input_size, strides, anchor_sizes):
    width, height = input_size
    locations = get_locations(input_size, strides)
    anchors_of_level = []
    for (lx, ly), sizes in zip(locations, anchor_sizes):
        anchors = torch.zeros(lx, ly, len(sizes), 4)
        anchors[:, :, :, 0] = (torch.arange(
            lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, len(sizes)) + 0.5) / lx
        anchors[:, :, :, 1] = (torch.arange(
            ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, len(sizes)) + 0.5) / ly
        anchors[:, :, :, 2] = sizes[:, 0] / width
        anchors[:, :, :, 3] = sizes[:, 1] / height
        anchors_of_level.append(anchors)
    return anchors_of_level


def generate_anchors(input_size, stride=16, aspect_ratios=(1 / 2, 1 / 1, 2 / 1), scales=(32, 64, 128, 256, 512)):
    width, height = input_size
    lx, ly = get_locations(input_size, [stride])[0]
    aspect_ratios = torch.tensor(aspect_ratios)
    scales = aspect_ratios.new_tensor(scales).view(len(scales), -1)
    num_anchors = len(aspect_ratios) * len(scales)
    anchors = torch.zeros(lx, ly, num_anchors, 4)
    anchors[:, :, :, 0] = (torch.arange(
        lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, num_anchors) + 0.5) / lx
    anchors[:, :, :, 1] = (torch.arange(
        ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, num_anchors) + 0.5) / ly
    if scales.size(1) == 2:
        sw = scales[:, [0]]
        sh = scales[:, [1]]
    else:
        sw = sh = scales
    anchors[:, :, :, 2] = (sw * aspect_ratios).view(-1) / width
    anchors[:, :, :, 3] = (sh / aspect_ratios).view(-1) / height
    return anchors


def generate_anchors_with_priors(input_size, stride, priors):
    lx, ly = get_locations(input_size, [stride])[0]
    num_anchors = len(priors)
    anchors = torch.zeros(lx, ly, num_anchors, 4)
    anchors[:, :, :, 0] = (torch.arange(
        lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, num_anchors) + 0.5) / lx
    anchors[:, :, :, 1] = (torch.arange(
        ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, num_anchors) + 0.5) / ly
    anchors[:, :, :, 2:] = priors
    return anchors


def misc_target_collate(batch):
    input, target = zip(*batch)
    target = [default_collate(t) if torch.is_tensor(t[0]) else t for t in target]
    return default_collate(input), Args(target)


def misc_collate(batch):
    input, target = zip(*batch)
    if torch.is_tensor(input[0]):
        input = default_collate(input)
    else:
        if any([torch.is_tensor(t) for t in input[0]]):
            input = [default_collate(t) if torch.is_tensor(t[0]) else t for t in zip(*input)]
        input = Args(input)
    if any([torch.is_tensor(t) for t in target[0]]):
        target = [default_collate(t) if torch.is_tensor(t[0]) else t for t in zip(*target)]
    if len(target[0]) == 0:
        target = []
    else:
        target = Args(target)
    return input, target


def draw_bboxes(img, anns, categories=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for ann in anns:
        if isinstance(ann, BBox):
            ann = ann.to_ann()
        bbox = ann["bbox"]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if categories:
            ax.text(bbox[0], bbox[1],
                    categories[ann["category_id"]], fontsize=12,
                    bbox=dict(boxstyle="round",
                              ec=(1., 0.5, 0.5),
                              fc=(1., 0.8, 0.8),
                              facecolor='red', alpha=0.5, edgecolor='red'
                              )
                    )
    return fig, ax
