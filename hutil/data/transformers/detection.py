from toolz import curry
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor


class Compose:
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


@curry
def Resize(img, anns, size):
    """Resize the image and bounding boxes.

    Args:
        img (PIL.Image): input image
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)
        size (tuple): tuple of (height, width)

    Example:
        >>> transform = Resize(size=(224, 224))
        >>> 
    """
    h, w = size
    sw = w / img.width
    sh = h / img.height
    img = img.resize((w, h))
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] *= sw
        bbox[1] *= sh
        bbox[2] *= sw
        bbox[3] *= sh
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


@curry
def ImageTransform(img, anns, f):
    return f(img), anns


@curry
def TargetTransform(img, anns, f):
    return img, f(anns)
