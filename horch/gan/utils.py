from math import ceil

import numpy as np


def l1_norm(x, xs):
    ds = np.linalg.norm(xs - x, ord=1, axis=1)
    return ds


def l2_norm(x, xs):
    ds = np.linalg.norm(xs - x, axis=1)
    return ds


def knn(x, xs, k=5, batch_size=10000, dist='l2'):
    r"""
    Calculate k-nearest-neighbors.

    Parameters
    ----------
    x : ndarray
        Pixel array of the image.
    xs : ndarray
        Pixel arrays of images.
    k : int
        Number of neighbors.
    batch_size : int
        Mini-batch size.
    dist : function
        Function used to calculate the distance of two images.
    """
    assert dist in ['l1', 'l2']
    if dist == 'l1':
        dist = l1_norm
    elif dist == 'l2':
        dist = l2_norm

    n = len(xs)
    x = x.reshape(-1).astype(np.int8)
    xs = xs.reshape(n, -1).astype(np.int8)
    n_batches = ceil(n / batch_size)

    dists = []
    for i in range(n_batches):
        start = i * batch_size
        end = min(n, (i+1) * batch_size)
        dists.append(dist(x, xs[start:end]))
    dists = np.concatenate(dists)
    indices = np.argpartition(dists, k)[:k]
    return indices, dists[indices]