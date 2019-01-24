import numpy as np
from torch.utils.data import random_split, Dataset, Subset

__all__ = ["train_test_split"]


def train_test_split(dataset, test_ratio, random=False):
    num_examples = len(dataset)
    num_test_examples = int(num_examples * test_ratio)
    num_train_examples = num_examples - num_test_examples
    if random:
        return random_split(dataset, [num_train_examples, num_test_examples])
    else:
        train_set = Subset(dataset, np.arange(num_train_examples))
        test_set = Subset(dataset, np.arange(num_train_examples, num_examples))
        return train_set, test_set


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
