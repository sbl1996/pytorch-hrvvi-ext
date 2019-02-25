import random

import torchvision.transforms.functional as TF

import hutil.transforms.detection as DF
from hutil.detection import BoundingBox


class Transform:

    def __init__(self):
        pass

    def __call__(self, input, target):
        pass


class JointTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input, target)


class InputTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return self.transform(input), target


class TargetTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, input, target):
        return input, self.transform(target)


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


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
        super().__init__(DF.resize(size=size))
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
        super().__init__(DF.center_crop(output_size=size))
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
        super().__init__(DF.to_percent_coords)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToAbsoluteCoords(JointTransform):

    def __init__(self):
        super().__init__(DF.to_absolute_coords)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, format=BoundingBox.LTWH):
        self.p = p
        if format == BoundingBox.LTWH or format == BoundingBox.XYWH:
            self.f = DF.hflip
        elif format == BoundingBox.LTRB:
            self.f = DF.hflip2
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


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, format=BoundingBox.LTWH):
        self.p = p
        if format == BoundingBox.LTWH or format == BoundingBox.XYWH:
            self.f = DF.vflip
        elif format == BoundingBox.LTRB:
            self.f = DF.vflip2
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
