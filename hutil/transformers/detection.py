import numbers
from toolz import curry
from PIL import Image


def draw_box(im, anns):
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for ann in anns:
        bbox = ann["bbox"]
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)


@curry
def center_crop(img, anns, output_size):
    r"""Crops the given PIL Image at the center and transform the bounding boxes.

    Args:
        img (PIL.Image): Image to be cropped.
        anns (sequences of dict): Sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, w, h) or (cx, cy, w, h).
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, anns, i, j, th, tw)


@curry
def crop(img, anns, i, j, h, w):
    """Crop the given PIL Image and transform the bounding boxes.

    Args:
        img (PIL.Image): Image to be cropped.
        anns (sequences of dict): Sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, w, h) or (cx, cy, w, h).
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    """
    img = img.crop((j, i, j + w, i + h))
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] -= j
        bbox[1] -= i
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


@curry
def resize(img, anns, size):
    """Resize the image and bounding boxes.

    Args:
        img (PIL Image): Image to be resized.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax) or (xmin, ymin, w, h) or (cx, cy, w, h)
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)        
    """
    w, h = img.size
    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img, anns
        if w < h:
            ow = size
            sw = sh = ow / w
            oh = int(sh * h)
        else:
            oh = size
            sw = sh = oh / h
            ow = int(sw * w)
    else:
        oh, ow = size
        sw = ow / w
        sh = oh / h

    img = img.resize((ow, oh), resample=2)
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] *= sw
        bbox[1] *= sh
        bbox[2] *= sw
        bbox[3] *= sh
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def to_percent_coords(img, anns):
    w, h = img.size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] /= w
        bbox[1] /= h
        bbox[2] /= w
        bbox[3] /= h
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def to_absolute_coords(img, anns):
    w, h = img.size
    new_anns = []
    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] *= w
        bbox[1] *= h
        bbox[2] *= w
        bbox[3] *= h
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def hflip(img, anns):
    """Horizontally flip the given PIL Image and transform the bounding boxes.

    Args:
        img (PIL Image): Image to be flipped.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, w, h) or (cx, cy, w, h)
    """
    w, h = img.size
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_anns = []

    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[0] = w - (bbox[0] + bbox[2])
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def hflip2(img, anns):
    """Horizontally flip the given PIL Image and transform the bounding boxes.

    Args:
        img (PIL Image): Image to be flipped.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax)
    """
    w, h = img.size
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_anns = []

    for ann in anns:
        bbox = list(ann['bbox'])
        l = bbox[0]
        bbox[0] = w - bbox[2]
        bbox[2] = w - l
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def vflip(img, anns):
    """Vertically flip the given PIL Image and transform the bounding boxes.

    Args:
        img (PIL Image): Image to be flipped.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, w, h) or (cx, cy, w, h)
    """
    w, h = img.size
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_anns = []

    for ann in anns:
        bbox = list(ann['bbox'])
        bbox[1] = h - (bbox[1] + bbox[3])
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns


def vflip2(img, anns):
    """Vertically flip the given PIL Image and transform the bounding boxes.

    Args:
        img (PIL Image): Image to be flipped.
        anns (sequences of dict): sequences of annotation of objects, containing `bbox` of 
            (xmin, ymin, xmax, ymax)
    """
    w, h = img.size
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_anns = []

    for ann in anns:
        bbox = list(ann['bbox'])
        t = bbox[1]
        bbox[1] = h - bbox[3]
        bbox[3] = h - t
        new_anns.append({**ann, "bbox": bbox})
    return img, new_anns
