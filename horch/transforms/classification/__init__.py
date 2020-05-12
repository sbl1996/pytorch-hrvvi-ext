import numbers

import numpy as np

import torch
import torchvision.transforms.functional as F

from horch.transforms import Transform
from horch.transforms.classification.autoaugment import CIFAR10Policy, ImageNetPolicy


class Cutout(Transform):
    """Randomly mask out one or more patches from an image.

    Optimal length: 16 for CIFAR10, 8 for CIFAR100, 20 for SVHN, 24 or 32 for STL10

    Note:
        It should be put after ToTensor().

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class PadToSquare(Transform):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image

            - reflect: pads with reflection of image without repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, fill=0, padding_mode='constant'):
        super().__init__()
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'auto']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        w, h = img.size
        if w == h:
            return img
        elif w > h:
            d = w - h
            pt = d // 2
            pb = d - pt
            padding = (0, pt, 0, pb)
        else:
            d = h - w
            pl = d // 2
            pr = d - pl
            padding = (pl, 0, pr, 0)

        if self.padding_mode == 'auto':
            fill = img.getpixel((0, 0))
            padding_mode = 'constant'
        else:
            fill = self.fill
            padding_mode = self.padding_mode
        return F.pad(img, padding, fill, padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.\
            format(self.fill, self.padding_mode)



# class RandomRotation(object):
#     """Rotate the image by angle.
#
#     Args:
#         degrees (sequence or float or int): Range of degrees to select from.
#             If degrees is a number instead of sequence like (min, max), the range of degrees
#             will be (-degrees, +degrees).
#         resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
#             An optional resampling filter. See `filters`_ for more information.
#             If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
#         expand (bool, optional): Optional expansion flag.
#             If true, expands the output to make it large enough to hold the entire rotated image.
#             If false or omitted, make the output image the same size as the input image.
#             Note that the expand flag assumes rotation around the center and no translation.
#         center (2-tuple, optional): Optional center of rotation.
#             Origin is the upper left corner.
#             Default is the center of the image.
#
#     .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
#
#     """
#
#     def __init__(self, degrees, resample=False, expand=False, center=None, fillcolor='auto'):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             if len(degrees) != 2:
#                 raise ValueError("If degrees is a sequence, it must be of len 2.")
#             self.degrees = degrees
#
#         self.resample = resample
#         self.expand = expand
#         self.center = center
#         self.fillcolor = fillcolor
#
#     @staticmethod
#     def get_params(degrees):
#         """Get parameters for ``rotate`` for a random rotation.
#
#         Returns:
#             sequence: params to be passed to ``rotate`` for random rotation.
#         """
#         angle = random.uniform(degrees[0], degrees[1])
#
#         return angle
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be rotated.
#
#         Returns:
#             PIL Image: Rotated image.
#         """
#
#         angle = self.get_params(self.degrees)
#         fillcolor = self.fillcolor
#         if self.fillcolor == 'auto':
#             fillcolor = img.getpixel((0, 0))
#         return img.rotate(angle, self.resample, self.expand, self.center, fillcolor=fillcolor)
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
#         format_string += ', resample={0}'.format(self.resample)
#         format_string += ', expand={0}'.format(self.expand)
#         if self.center is not None:
#             format_string += ', center={0}'.format(self.center)
#         format_string += ', fillcolor={0}'.format(self.fillcolor)
#         format_string += ')'
#         return format_string
