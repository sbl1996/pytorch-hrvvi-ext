import random

import torchvision.transforms.functional as TF

from hutil.detection import BBox
from hutil.transforms import JointTransform, Compose
from hutil.transforms.detection.functional import resize, center_crop, hflip, hflip2, vflip, vflip2, to_absolute_coords, to_percent_coords


class Resize(JointTransform):
    """Resize the image and bounding boxes.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)        
    Inputs:
        img (PIL Image): Image to be resized.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)
    """

    def __init__(self, size):
        super().__init__(resize(size=size))
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + "(size=%s)".format(self.size)


class CenterCrop(JointTransform):
    """Crops the given PIL Image at the center and transform the bounding boxes.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    Inputs:
        img (PIL.Image): Image to be cropped.
        anns (sequences of dict): Sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, w, h) or (cx, cy, w, h).
    """

    def __init__(self, size):
        super().__init__(center_crop(output_size=size))
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + "(size=%s)".format(self.size)


class ToTensor(JointTransform):

    def __init__(self):
        super().__init__(lambda x, y: (TF.to_tensor(x), y))

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToPercentCoords(JointTransform):

    def __init__(self):
        super().__init__(to_percent_coords)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToAbsoluteCoords(JointTransform):

    def __init__(self):
        super().__init__(to_absolute_coords)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomHorizontalFlip(JointTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, format=BBox.LTWH):
        super().__init__()
        self.p = p
        if format == BBox.LTWH or format == BBox.XYWH:
            self.f = hflip
        elif format == BBox.LTRB:
            self.f = hflip2
        else:
            raise ValueError("invalid bounding box format")

    def __call__(self, img, anns):
        """
        Args:
            img (PIL Image): Image to be flipped.
            anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
                (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)
        """
        if random.random() < self.p:
            return self.f(img, anns)
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(JointTransform):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, format=BBox.LTWH):
        self.p = p
        if format == BBox.LTWH or format == BBox.XYWH:
            self.f = vflip
        elif format == BBox.LTRB:
            self.f = vflip2
        else:
            raise ValueError("invalid bounding box format")

    def __call__(self, img, anns):
        """
        Args:
            img (PIL Image): Image to be flipped.
            anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
                (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)
        """
        if random.random() < self.p:
            return self.f(img, anns)
        return img, anns

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
