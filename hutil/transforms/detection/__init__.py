import random
import math

from toolz.curried import get
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF

from hutil.detection import BBox
from hutil.transforms import JointTransform, Compose
from hutil.transforms.detection.functional import resize, center_crop, hflip, hflip2, vflip, vflip2, to_absolute_coords, to_percent_coords, resized_crop, crop


def iou_1m(box, boxes):
    r"""
    Calculates one-to-many ious by corners([xmin, ymin, xmax, ymax]).

    Args:
        box: (4,)
        boxes: (*, 4)

    Returns:
        ious: (*,)
    """
    xi1 = np.maximum(boxes[..., 0], box[0])
    yi1 = np.maximum(boxes[..., 1], box[1])
    xi2 = np.minimum(boxes[..., 2], box[2])
    yi2 = np.minimum(boxes[..., 3], box[3])
    xdiff = xi2 - xi1
    ydiff = yi2 - yi1
    inter_area = xdiff * ydiff
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[..., 2] - boxes[..., 0]) * \
        (boxes[..., 3] - boxes[..., 1])
    union_area = boxes_area + box_area - inter_area

    iou = inter_area / union_area
    iou[xdiff < 0] = 0
    iou[ydiff < 0] = 0
    return iou


class RandomExpand(JointTransform):

    def __init__(self, ratios=(1, 4)):
        self.ratios = ratios

    def __call__(self, img, anns):
        width, height = img.size
        ratio = random.uniform(*self.ratios)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = Image.new(
            img.mode, (int(width*ratio), int(height*ratio)))
        expand_image.paste(img, (int(left), int(top)))

        new_anns = []
        for ann in anns:
            bbox = list(ann['bbox'])
            bbox[0] += left
            bbox[1] += top
            new_anns.append({**ann, "bbox": bbox})
        return expand_image, new_anns


class RandomSampleCrop(JointTransform):

    def __init__(self, min_ious=[0.1, 0.3, 0.5, 0.9], aspect_ratio_constraint=(0.5, 2)):
        self.min_ious = min_ious
        self.get_bbox = get("bbox")
        min_ars, max_ars = aspect_ratio_constraint
        self.min_ars = min_ars
        self.max_ars = max_ars

    def __call__(self, img, anns):
        width, height = img.size
        min_iou = random.choice(self.min_ious)
        boxes = np.stack([self.get_bbox(ann) for ann in anns])
        boxes[:, 2:] += boxes[:, :2]
        for _ in range(50):
            w = random.uniform(0.3 * width, width)
            h = random.uniform(0.3 * height, height)

            if h / w < self.min_ars or h / w > self.max_ars:
                continue

            l = random.uniform(0, width - w)
            t = random.uniform(0, height - h)
            r = l + w
            b = t + h

            patch = np.array([l, t, r, b])
            ious = iou_1m(patch, boxes)
            if ious.min() < min_iou:
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            mask = (l < centers[:, 0]) & (centers[:, 0] < r) & (
                t < centers[:, 1]) & (centers[:, 1] < b)

            if not mask.any():
                continue
            indices = np.nonzero(mask)[0].tolist()
            return crop(img, get(indices, anns), t, l, h, w)
        return img, anns


class RandomResizedCrop(JointTransform):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
        drop: whether to drop a object if the center of it is not in the crop
    Inputs:
        img (PIL Image): Image to be resized.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, drop=True):
        super().__init__()
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.drop = drop

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, anns):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(
            img, anns, i, j, h, w, self.size, self.interpolation, self.drop)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4)
                                                    for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4)
                                                    for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        format_string += ', drop={0})'.format(
            self.drop)
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
        super().__init__(resize(size=size))
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + "(size=%s)" % (self.size,)


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
