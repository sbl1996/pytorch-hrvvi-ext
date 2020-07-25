import numpy as np
from sklearn.model_selection import KFold as R_KFold
from horch.datasets import Subset


class KFold(R_KFold):
    def __init__(self, n_splits='warn', shuffle=True, transform=None, test_transform=None, random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.transform = transform
        self.test_transform = test_transform

    def split(self, ds, y=None, groups=None):
        n = len(ds)
        for train_indices, test_indices in super().split(list(range(n))):
            ds_train = Subset(ds, train_indices, self.transform)
            ds_test = Subset(ds, test_indices, self.test_transform)
            yield ds_train, ds_test


def k_fold(ds, n_splits=5, shuffle=True, transform=None, test_transform=None, random_state=None):

    n = len(ds)
    kf = KFold(n_splits, shuffle, random_state)
    for train_indices, test_indices in kf.split(list(range(n))):
        ds_train = Subset(ds, train_indices, transform)
        ds_test = Subset(ds, test_indices, test_transform)
        yield ds_train, ds_test


def cross_val_score(fit_fn, ds, cv: KFold, verbose=0):
    scores = []
    for i, (ds_train, ds_val) in enumerate(cv.split(ds)):
        if verbose == 1:
            print(f"Round {i+1}")
        score = fit_fn(ds_train, ds_val, verbose)
        print(score)
        scores.append(score)
    return scores
