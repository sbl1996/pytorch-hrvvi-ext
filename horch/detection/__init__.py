import colorsys
import random
from typing import Sequence
from math import sqrt

import numpy as np

import torch
from torch.utils.data.dataloader import default_collate

from horch.common import ProtectedSeq, tuplify
from horch.detection.bbox import BBox
from horch.detection.iou import iou_11, iou_b11, iou_1m, iou_mn
from horch.detection.anchor.finder import find_priors_kmeans, find_priors_coco
from horch.detection.nms import nms, soft_nms_cpu, softer_nms_cpu
from horch.detection.eval import mean_average_precision

__all__ = [
    "BBox", "nms", "soft_nms_cpu",
    "iou_1m", "iou_11", "iou_b11", "iou_mn", "draw_bboxes",
    "calc_grid_sizes", "calc_anchor_sizes",
    "generate_mlvl_anchors",
    "find_priors_kmeans", "mean_average_precision", "find_priors_coco", "softer_nms_cpu"
]


def _pair(x):
    if not isinstance(x, Sequence):
        return x, x
    else:
        return x


def calc_grid_sizes(size, strides, pad_threshold=3):
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
        lx = max(lx, 1)
        ly = max(ly, 1)
        locations.append((lx, ly))
    return tuple(locations[-len(strides):])


def calc_anchor_sizes(
        length,
        aspect_ratios=(1 / 2, 1 / 1, 2 / 1),
        scales=tuple(2 ** p for p in [0, 1 / 3, 2 / 3]),
        device='cpu', dtype=torch.float32):
    w, h = tuplify(length, 2)
    sw = [w * sqrt(ars) * s for ars in aspect_ratios for s in scales]
    sh = [h / sqrt(ars) * s for ars in aspect_ratios for s in scales]
    return torch.tensor([sw, sh], device=device, dtype=dtype).transpose(1, 0)


def calc_mlvl_anchor_sizes(
        lengths=(32, 64, 128, 256, 512),
        aspect_ratios=(1 / 2, 1 / 1, 2 / 1),
        scales=tuple(2 ** p for p in [0, 1 / 3, 2 / 3]),
        device='cpu', dtype=torch.float32):
    mlvl_anchors_sizes = [
        calc_anchor_sizes(length, aspect_ratios, scales, device, dtype)
        for length in lengths
    ]
    return torch.stack(mlvl_anchors_sizes)


def calc_ssd_priors(
        size,
        scales=(0.1, 0.2, 0.375, 0.55, 0.725, 0.9),
        aspect_ratios=(
                (1 / 2, 1, 2 / 1),
                (1 / 3, 1 / 2, 1, 2 / 1, 3 / 1),
                (1 / 3, 1 / 2, 1, 2 / 1, 3 / 1),
                (1 / 3, 1 / 2, 1, 2 / 1, 3 / 1),
                (1 / 2, 1, 2 / 1),
                (1 / 2, 1, 2 / 1),
        ),
        extra_for_one=True):
    width, height = size
    num_levels = len(scales)
    aspect_ratios = tuplify(aspect_ratios, num_levels)
    mlvl_priors = []
    for i in range(num_levels):
        s = scales[i]
        ps = []
        for a in aspect_ratios[i]:
            sw = s * sqrt(a) * width
            sh = s / sqrt(a) * height
            ps.append((sw, sh))
            if extra_for_one and a == 1:
                if i == num_levels - 1:
                    s_extra = 1.0
                else:
                    s_extra = sqrt(s * scales[i + 1])
                ps.append((s_extra * width, s_extra * height))
        mlvl_priors.append(ps)
    return mlvl_priors

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_bboxes(img, anns, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16, figsize=(10, 10)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not colors:
        if categories:
            colors = random_colors(len(categories))
        else:
            colors = ['w' for _ in range(100)]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for ann in anns:
        c = ann["category_id"]
        if isinstance(ann, BBox):
            ann = ann.to_ann()
        bbox = ann["bbox"]
        color = colors[c]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=linewidth,
                         alpha=0.7, edgecolor=color, facecolor='none')

        ax.add_patch(rect)
        if categories:
            text = "%s %.2f" % (categories[c], ann['score'])
            ax.text(bbox[0], bbox[1] + label_offset, text,
                    color=color, size=fontsize, backgroundcolor="none")
    return fig, ax


def generate_mlvl_anchors(grid_sizes, anchor_sizes, device='cpu', dtype=torch.float32):
    mlvl_anchors = []
    for (lx, ly), sizes in zip(grid_sizes, anchor_sizes):
        anchors = sizes.new_zeros((lx, ly, len(sizes), 4))
        anchors[:, :, :, 0] = (torch.arange(
            lx, dtype=dtype, device=device).view(lx, 1, 1).expand(lx, ly, len(sizes)) + 0.5) / lx
        anchors[:, :, :, 1] = (torch.arange(
            ly, dtype=dtype, device=device).view(1, ly, 1).expand(lx, ly, len(sizes)) + 0.5) / ly
        anchors[:, :, :, 2] = sizes[:, 0]
        anchors[:, :, :, 3] = sizes[:, 1]
        anchors.clamp_(min=0.0, max=1.0)
        mlvl_anchors.append(anchors)
    return mlvl_anchors