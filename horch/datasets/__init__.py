import bisect
import math

import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from torch.utils.data import Dataset, ConcatDataset as TorchConcatDataset
from torchvision.transforms import Compose

from horch.transforms import InputTransform
from horch.datasets.captcha import Captcha, CaptchaDetectionOnline, CaptchaOnline, CaptchaSegmentationOnline
from horch.datasets.coco import CocoDetection
from horch.datasets.voc import VOCDetection, VOCSegmentation, VOCDetectionConcat
from horch.datasets.svhn import SVHNDetection
from horch.datasets.animefaces import AnimeFaces


class Fullset(Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        return self.transform(input, target)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "Fullset(%s)" % self.dataset


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices

        self._transform = None

        self.transform = transform

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, Compose):
            transform = InputTransform(transform)
        self._transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            return self.transform(img, target)

        return img, target

    def get_image(self, idx):
        return self.dataset.get_image(self.indices[idx])

    def get_target(self, idx):
        return self.dataset.get_target(self.indices[idx])

    def get_class(self, idx):
        return self.dataset.get_class(self.indices[idx])

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        fmt_str = 'Subset of ' + self.dataset.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str


def train_test_split(dataset, test_ratio, shuffle=False, transform=None, test_transform=None, random_state=None):
    if isinstance(transform, Compose):
        transform = InputTransform(transform)
    if isinstance(test_transform, Compose):
        test_transform = InputTransform(test_transform)

    n = len(dataset)
    train_indices, test_indices = sklearn_train_test_split(
        list(range(n)), test_size=test_ratio, shuffle=shuffle, random_state=random_state)
    ds_train = Subset(dataset, train_indices, transform)
    ds_test = Subset(dataset, test_indices, test_transform)
    return ds_train, ds_test


class CachedDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None] * len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]

    def get_class(self, idx):
        return self.dataset.get_class(idx)


def batchify(ds, batch_size):
    n = len(ds)
    n_batches = math.ceil(n / batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch = [ds[j] for j in range(start, end)]
        yield batch


class CombineDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets
        assert len(set(len(ds) for ds in datasets)) == 1, "All datasets must be of the same length"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(ds[idx] for ds in self.datasets)


class ConcatDataset(TorchConcatDataset):

    def get_class(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_class(sample_idx)
