import random
import numpy as np

from albumentations import DualTransform


class RandomExpand(DualTransform):
    """
    Expand the given PIL Image to random size.

    This is popularly used to train the SSD-like detectors.

    Parameters
    ----------
    ratios : ``tuple``
        Range of expand ratio.
    """

    def __init__(self, ratios=(1, 4), mean=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.ratios = ratios
        self.mean = mean

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

    def apply(self, img, ratio=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        expand_image = np.zeros((int(rows * ratio), int(cols * ratio), img.shape[2]), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(h_start):int(h_start + rows),
        int(w_start):int(w_start + cols)] = img
        return expand_image

    def apply_to_bbox(self, bbox, ratio=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = bbox
        x_min = (x_min * cols + w_start) / cols / ratio
        y_min = (y_min * rows + h_start) / rows / ratio
        x_max = (x_max * cols + w_start) / cols / ratio
        y_max = (y_max * rows + h_start) / rows / ratio
        return [x_min, y_min, x_max, y_max]