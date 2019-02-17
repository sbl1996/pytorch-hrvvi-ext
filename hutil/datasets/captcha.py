import os
import pickle
import random
import string

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

        if not self.online:
            self.data = [self.gen_captcha(self.nchars) for _ in range(size)]

    def gen_captcha(self, nchars):
        labels = [random.randrange(self.num_classes) for _ in range(nchars)]
        chars = [self.letters[i] for i in labels]
        img, bboxes = self.image.generate_image(
            chars, noise_dots=.3, noise_curve=.3, return_bbox=True, **self.kwargs)
        annotations = [
            {
                "category_id": labels[i],
                "bbox": bboxes[i]
            } for i in range(nchars)]
        return img, annotations

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
