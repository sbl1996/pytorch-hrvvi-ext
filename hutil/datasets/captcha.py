import os
import pickle
import random
import string
from copy import deepcopy

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ALPHABET_DIGITS = string.digits + string.ascii_letters


class Captcha(Dataset):

    files = {
        "train": "train.pt",
        "val": "val.pt",
        "test": "test.pt",
    }

    def __init__(self, root, split="train", letters=ALPHABET_DIGITS, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.letters = letters
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set

        path = os.path.join(self.root, self.files[self.split])
        with open(path, 'rb') as f:
            d = pickle.load(f)

        self.data = d["data"]
        self.labels = d["labels"].astype(np.int64)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CaptchaOnline(Dataset):

    def __init__(self, image, size=50000, nchars=4, letters=ALPHABET_DIGITS, transform=None, online=True, **kwargs):
        self.image = image
        self.size = size
        self.nchars = nchars
        self.letters = letters
        self.transform = transform
        self.letters = letters
        self.num_classes = len(self.letters)
        self.online = online
        self.kwargs = kwargs

        if not self.online:
            self.data = [self.gen_captcha(nchars) for _ in range(size)]

    def gen_captcha(self, nchars):
        labels = [random.randrange(self.num_classes) for _ in range(nchars)]
        labels = np.array(labels, dtype=np.int64)
        chars = [self.letters[i] for i in labels]
        img = self.image.generate_image(
            chars, noise_dots=.3, noise_curve=.3, **self.kwargs)
        return img, labels

    def __getitem__(self, index):
        if self.online:
            img, target = self.gen_captcha(self.nchars)
        else:
            img, target = self.data[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return self.size


class CaptchaDetectionOnline(Dataset):

    def __init__(self, image, size=50000, nchars=4, letters=ALPHABET_DIGITS, transform=None, online=True, **kwargs):
        self.image = image
        self.size = size
        self.nchars = nchars
        self.letters = letters
        self.num_classes = len(self.letters)
        self.transform = transform
        self.online = online
        self.kwargs = kwargs
        self.table = [0] * 256
        for i, c in enumerate(letters):
            self.table[ord(c)] = i+1

        from pycocotools.mask import area, encode
        self.area_fn = area
        self.encode_fn = encode

        if not self.online:
            self.data = [self.gen_captcha(self.nchars, i) for i in range(size)]

    def to_coco(self):
        categories = []
        for i, c in enumerate(self.letters):
            categories.append({
                'supercategory': 'letter',
                'id': i + 1,
                'name': c
            })
        images = []
        annotations = []
        assert not self.online, "Only non-online dataset could be transformed to coco style"
        ann_id = 0
        for i in range(self.size):
            anns = self.data[i][1]
            img = {
                "file_name": "%d.jpg" % i,
                "height": self.image._height,
                "width": self.image._width,
                "id": i,
            }
            images.append(img)
            for ann in anns:
                ann = deepcopy(ann)
                ann['id'] = ann_id
                ann_id += 1
                annotations.append(ann)
        dataset = {
            'categories': categories,
            'images': images,
            'annotations': annotations,
        }
        return dataset

    def gen_captcha(self, nchars, image_id=None):
        labels = [random.randrange(self.num_classes) for _ in range(nchars)]
        chars = [self.letters[i] for i in labels]
        img, anns = self.image.generate_image(
            chars, noise_dots=.3, noise_curve=.3, **self.kwargs)
        for ann, label in zip(anns, labels):
            ann['image_id'] = image_id
            ann['category_id'] = label + 1
            segm = np.asfortranarray(ann['segmentation'], dtype=np.uint8)
            segm = self.encode_fn(segm)
            ann['segmentation'] = segm
            ann['area'] = self.area_fn(segm)
            ann['iscrowd'] = 0

        return img, anns

    def __getitem__(self, index):
        if self.online:
            img, target = self.gen_captcha(self.nchars)
        else:
            img, target = self.data[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return self.size


class CaptchaSegmentationOnline(Dataset):

    def __init__(self, image, size=50000, nchars=4, letters=ALPHABET_DIGITS, transform=None, online=True, **kwargs):
        self.image = image
        self.size = size
        self.nchars = nchars
        self.letters = letters
        self.num_classes = len(self.letters)
        self.transform = transform
        self.online = online
        self.kwargs = kwargs
        self.table = [0] * 256
        for i, c in enumerate(letters):
            self.table[ord(c)] = i+1

        if not self.online:
            self.data = [self.gen_captcha(self.nchars, i) for i in range(size)]

    def gen_captcha(self, nchars, image_id=None):
        labels = [random.randrange(self.num_classes) for _ in range(nchars)]
        chars = [self.letters[i] for i in labels]
        img, bboxes, mask = self.image.generate_image(
            chars, noise_dots=.3, noise_curve=.3, return_bbox=True, return_mask=True, **self.kwargs)
        mask = mask.point(self.table)

        return img, mask

    def __getitem__(self, index):
        if self.online:
            img, target = self.gen_captcha(self.nchars)
        else:
            img, target = self.data[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return self.size
