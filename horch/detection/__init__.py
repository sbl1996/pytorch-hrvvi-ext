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
    "calc_grid_sizes", "calc_anchor_sizes", "generate_anchors",
    "generate_mlvl_anchors", "generate_anchors_with_priors",
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


#
#
# def generate_mlvl_anchors(levels=(), lengths=(32, 64, 128, 256, 512), aspect_ratios=(1/2, 1/1, 2/1), scales=):
#     width, height = input_size
#     strides = [ 2 ** l for l in levels ]
#     aspect_ratios = torch.tensor(aspect_ratios)
#     locations = calc_grid_sizes(input_size, strides)
#     mlvl_anchors = []
#     for (lx, ly), scale, stride in zip(locations, lengths, strides):
#         anchors = torch.zeros(lx, ly, len(aspect_ratios), 4)
#         anchors[:, :, :, 0] = (torch.arange(
#             lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, len(aspect_ratios)) + 0.5) / lx
#         anchors[:, :, :, 1] = (torch.arange(
#             ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, len(aspect_ratios)) + 0.5) / ly
#         anchors[:, :, :, 2] = (stride * aspect_ratios).view(-1) / width
#         anchors[:, :, :, 3] = (stride / aspect_ratios).view(-1) / height
#         mlvl_anchors.append(anchors)
#     return mlvl_anchors


def generate_anchors(input_size, stride=16, aspect_ratios=(1 / 2, 1 / 1, 2 / 1), scales=(32, 64, 128, 256, 512)):
    # width, height = input_size
    # lx, ly = get_locations(input_size, [stride])[0]
    # aspect_ratios = torch.tensor(aspect_ratios)
    # scales = aspect_ratios.new_tensor(scales).view(len(scales), -1)
    # num_anchors = len(aspect_ratios) * len(scales)
    # anchors = torch.zeros(lx, ly, num_anchors, 4)
    # anchors[:, :, :, 0] = (torch.arange(
    #     lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, num_anchors) + 0.5) / lx
    # anchors[:, :, :, 1] = (torch.arange(
    #     ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, num_anchors) + 0.5) / ly
    # if scales.size(1) == 2:
    #     sw = scales[:, [0]]
    #     sh = scales[:, [1]]
    # else:
    #     sw = sh = scales
    # anchors[:, :, :, 2] = (sw * aspect_ratios).view(-1) / width
    # anchors[:, :, :, 3] = (sh / aspect_ratios).view(-1) / height
    # return anchors
    pass


def generate_anchors_with_priors(input_size, stride, priors):
    # # lx, ly = get_locations(input_size, [stride])[0]
    # # num_anchors = len(priors)
    # # anchors = torch.zeros(lx, ly, num_anchors, 4)
    # # anchors[:, :, :, 0] = (torch.arange(
    # #     lx, dtype=torch.float).view(lx, 1, 1).expand(lx, ly, num_anchors) + 0.5) / lx
    # # anchors[:, :, :, 1] = (torch.arange(
    # #     ly, dtype=torch.float).view(1, ly, 1).expand(lx, ly, num_anchors) + 0.5) / ly
    # # anchors[:, :, :, 2:] = priors
    # return anchors
    pass


def draw_bboxes(img, anns, categories=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for ann in anns:
        if isinstance(ann, BBox):
            ann = ann.to_ann()
        bbox = ann["bbox"]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=2,
                         edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        if categories:
            ax.text(bbox[0], bbox[1],
                    categories[ann["category_id"]], fontsize=12,
                    bbox=dict(boxstyle="round",
                              # ec=(1., 0.5, 0.5),
                              # fc=(1., 0.8, 0.8),
                              facecolor='green', alpha=0.5, edgecolor='green'
                              ),
                    color='white',
                    )
    return fig, ax


def generate_mlvl_anchors(grid_sizes, anchor_sizes, device='cpu', dtype=torch.float32):
    mlvl_anchors = []
    for (lx, ly), sizes in zip(grid_sizes, anchor_sizes):
        anchors = anchor_sizes.new_zeros((lx, ly, len(sizes), 4))
        anchors[:, :, :, 0] = (torch.arange(
            lx, dtype=dtype, device=device).view(lx, 1, 1).expand(lx, ly, len(sizes)) + 0.5) / lx
        anchors[:, :, :, 1] = (torch.arange(
            ly, dtype=dtype, device=device).view(1, ly, 1).expand(lx, ly, len(sizes)) + 0.5) / ly
        anchors[:, :, :, 2] = sizes[:, 0]
        anchors[:, :, :, 3] = sizes[:, 1]
        mlvl_anchors.append(anchors)
    return mlvl_anchors
