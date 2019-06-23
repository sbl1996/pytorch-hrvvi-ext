from typing import Sequence

import cv2
import random
import numpy as np

import albumentations as A
from albumentations import DualTransform, denormalize_bbox, denormalize_bboxes, normalize_bbox
from albumentations import BasicTransform
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensor as AToTensor
from horch.common import tuplify
from toolz import first


def iou_1m(box, boxes, eps=0):
    r"""
    Calculates one-to-many ious.

    Parameters
    ----------
    box : ``Sequences[Number]``
        A bounding box.
    boxes : ``array_like``
        Many bounding boxes.

    Returns
    -------
    ious : ``array_like``
        IoUs between the box and boxes.
    """
    xi1 = np.maximum(boxes[..., 0], box[0])
    yi1 = np.maximum(boxes[..., 1], box[1])
    xi2 = np.minimum(boxes[..., 2], box[2])
    yi2 = np.minimum(boxes[..., 3], box[3])
    xdiff = xi2 - xi1 + eps
    ydiff = yi2 - yi1 + eps
    inter_area = xdiff * ydiff
    box_area = (box[2] - box[0] + eps) * (box[3] - box[1] + eps)
    boxes_area = (boxes[..., 2] - boxes[..., 0] + eps) * \
        (boxes[..., 3] - boxes[..., 1] + eps)
    union_area = boxes_area + box_area - inter_area

    iou = inter_area / union_area
    iou[xdiff < 0] = 0
    iou[ydiff < 0] = 0
    return iou


class RandomExpand(DualTransform):
    """
    Expand the given PIL Image to random size.

    This is popularly used to train the SSD-like detectors.

    Parameters
    ----------
    ratios : ``tuple``
        Range of expand ratio.
    """

    def __init__(self, ratios=(1, 4), mean=(0.485, 0.456, 0.406), max_pixel_value=255.0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.ratios = ratios
        self.mean = mean
        self.max_pixel_value = max_pixel_value

    def get_params_dependent_on_targets(self, params):
        height, width = params['image'].shape[:2]
        ratio = random.uniform(*self.ratios)
        h_start = random.uniform(0, height * ratio - height)
        w_start = random.uniform(0, width * ratio - width)
        return {
            'ratio': ratio,
            'h_start': h_start,
            'w_start': w_start
        }

    @property
    def targets_as_params(self):
        return ['image']

    def apply(self, img, ratio=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        expand_image = np.zeros((int(rows * ratio), int(cols * ratio), img.shape[2]), dtype=img.dtype)

        mean = np.array(self.mean, dtype=np.float32)
        mean *= self.max_pixel_value

        expand_image[:, :, :] = mean
        expand_image[int(h_start):int(h_start + rows), int(w_start):int(w_start + cols)] = img
        return expand_image

    def apply_to_bbox(self, bbox, ratio=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = bbox
        x_min = (x_min * cols + w_start) / cols / ratio
        y_min = (y_min * rows + h_start) / rows / ratio
        x_max = (x_max * cols + w_start) / cols / ratio
        y_max = (y_max * rows + h_start) / rows / ratio
        return [x_min, y_min, x_max, y_max]


class RandomSampleCrop(DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, min_scale=0.3, max_scale=1, max_aspect_ratio=2, constraints=None, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_aspect_ratio = max_aspect_ratio
        self.constraints = constraints or (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    def apply(self, img, h_start=0, w_start=0, h_end=0, w_end=0, **params):
        if h_start == h_end:
            return img
        return F.crop(img, w_start, h_start, w_end, h_end)

    def get_params_dependent_on_targets(self, params):
        height, width = params['image'].shape[:2]
        bboxes = denormalize_bboxes(params['bboxes'], height, width)
        bboxes = np.array(bboxes)[:, :4]

        candidates = [(0, 0, width, height)]
        for min_iou, max_iou in self.constraints:
            min_iou = -np.inf if min_iou is None else min_iou
            max_iou = np.inf if max_iou is None else max_iou

            for _ in range(50):
                scale = random.uniform(self.min_scale, self.max_scale)
                aspect_ratio = random.uniform(
                    max(1 / self.max_aspect_ratio, scale * scale),
                    min(self.max_aspect_ratio, 1 / (scale * scale)))
                crop_h = int(height * scale / np.sqrt(aspect_ratio))
                crop_w = int(width * scale * np.sqrt(aspect_ratio))

                crop_t = random.randrange(height - crop_h)
                crop_l = random.randrange(width - crop_w)
                crop_b = crop_t + crop_h
                crop_r = crop_l + crop_w
                patch = np.array((crop_l, crop_t, crop_r, crop_b))
                ious = iou_1m(patch, bboxes, eps=1)
                if min_iou <= ious.min() and ious.max() <= max_iou:
                    candidates.append((crop_l, crop_t, crop_r, crop_b))
                    break

        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
        while candidates:
            l, t, r, b = candidates.pop(np.random.randint(0, len(candidates)))
            mask = (l < centers[:, 0]) & (centers[:, 0] < r) & (
                    t < centers[:, 1]) & (centers[:, 1] < b)
            if not mask.any():
                continue
            indices = np.nonzero(mask)[0].tolist()
            return {
                'indices': indices,
                'h_start': t,
                'w_start': l,
                'h_end': b,
                'w_end': r,
            }
        return {
            'indices': [],
            'h_start': 0,
            'w_start': 0,
            'h_end': 0,
            'w_end': 0,
        }

    def apply_to_bboxes(self, bboxes, indices=(), h_start=0, w_start=0, h_end=0, w_end=0, **params):
        if h_start == h_end:
            return bboxes
        bboxes = [
            F.bbox_crop(bboxes[i][:4], w_start, h_start, w_end, h_end, **params) + bboxes[i][4:]
            for i in indices
        ]
        return bboxes

    @property
    def targets_as_params(self):
        return ['image', 'bboxes']


class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Default 0.3
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, max_part_shift=0.3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_part_shift = max_part_shift

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params):
        h, w = params['image'].shape[:2]
        bboxes = params['bboxes']
        bbox = random.choice(bboxes)
        bbox = denormalize_bbox(bbox, h, w)
        h_max_shift = int((bbox[3] - bbox[1]) * self.max_part_shift)
        w_max_shift = int((bbox[2] - bbox[0]) * self.max_part_shift)

        x_min = bbox[0] - random.randint(-w_max_shift, w_max_shift)
        x_min = max(0, x_min)
        x_max = bbox[2] + random.randint(-w_max_shift, w_max_shift)
        x_max = min(x_max, w)

        y_min = bbox[1] - random.randint(-h_max_shift, h_max_shift)
        y_min = max(0, y_min)
        y_max = bbox[3] + random.randint(-h_max_shift, h_max_shift)
        y_max = min(y_max, h)
        return {'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
                }

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, **params)
        # return F.bbox_crop(bbox, y_max - y_min, x_max - x_min, h_start, w_start, **params)

    @property
    def targets_as_params(self):
        return ['image', 'bboxes']


def SSDTransform(size, color_jitter=True, expand=(1, 4), scale=(0.1, 1), mean=(0.485, 0.456, 0.406)):
    transforms = []
    if color_jitter:
        transforms += [
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ]
    transforms += [
        RandomExpand(ratios=expand, mean=mean, p=0.5),
        A.OneOf([
            A.NoOp(p=1),
            RandomSampleCrop(min_scale=scale[0], max_scale=scale[1]),
        ], p=1),
        A.HorizontalFlip(),
        A.Resize(size[1], size[0]),
    ]
    return transforms


class PostTransform(BasicTransform):

    def __init__(self):
        super().__init__(False, p=1.0)

    @property
    def targets(self):
        return {'image': self.apply,
                'mask': self.apply_to_mask,
                'masks': self.apply_to_masks,
                'bboxes': self.apply_to_bboxes,
                'keypoints': self.apply_to_keypoints}

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError('Method apply_to_bbox is not implemented in class ' + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError('Method apply_to_keypoint is not implemented in class ' + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        bboxes = [list(bbox) for bbox in bboxes]
        return [self.apply_to_bbox(bbox[:4], **params) + bbox[4:] for bbox in bboxes]

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = [list(keypoint) for keypoint in keypoints]
        return [self.apply_to_keypoint(keypoint[:4], **params) + keypoint[4:] for keypoint in keypoints]

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]


class ToPercentCoords(PostTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Default 0.3
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self):
        super().__init__()

    def apply_to_bbox(self, bbox, rows=0, cols=0, **params):
        return normalize_bbox(bbox, rows, cols)


def identity(*args):
    return args


class Compose:

    def __init__(self, transforms):
        i = 0
        ts = []
        for t in transforms:
            if isinstance(t, Sequence):
                ts.extend(t)
                i += 1
            elif isinstance(t, BasicTransform):
                ts.append(t)
                i += 1
            else:
                break
        self.transforms = A.Compose(
            ts,
            bbox_params={'format': 'coco', 'min_area': 0, 'min_visibility': 0.25,
                         'label_fields': ['category_id', 'id', 'image_id', 'area', 'isdifficult', 'iscrowd']})
        self.post_transforms = transforms[i:]

    def __call__(self, img, anns):
        keys = set(k for d in anns for k in d.keys())
        annotations = unzip_dict(anns, keys)
        annotations['bboxes'] = annotations['bbox']
        del annotations['bbox']
        annotations['image'] = img
        annotations = self.transforms(**annotations)
        annotations['bbox'] = annotations['bboxes']
        del annotations['bboxes']
        img = annotations['image']
        del annotations['image']
        anns = zip_dict(annotations, keys)
        for t in self.post_transforms:
            img, anns = t(img, anns)
        return img, anns


def ToTensor():
    return AToTensor(normalize={'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)})


def unzip_dict(dicts, keys):
    res = {}
    for key in keys:
        res[key] = []
    for d in dicts:
        for k in keys:
            res[k].append(d[k])
    return res


def zip_dict(d, keys):
    length = len(d[first(keys)])
    res = [
        {k: d[k][i] for k in keys}
        for i in range(length)
    ]
    return res


def Resize(size):
    w, h = tuplify(size, 2)
    return A.Resize(h, w)