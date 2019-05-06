import numpy as np
import torch

from hutil._numpy import iou_mn, iou_11
from hutil.detection import BBox


def kmeans(X, k, max_iter=300, tol=1e-6, verbose=True):
    n, d = X.shape
    centers = X[np.random.choice(n, size=k, replace=False)]
    for i in range(max_iter):
        dist = 1 - iou_mn(X, centers)
        y = np.argmin(dist, axis=1)
        loss = 0
        for ki in range(k):
            kx = X[y == ki]
            if kx.size != 0:
                center = np.mean(kx, axis=0)
                iou = iou_11(center, centers[ki])
                loss += 1 - iou
                centers[ki] = center
        loss /= k
        if verbose:
            print("Iter %d: %.6f" % (i, loss))
        if loss < tol:
            break
    return y, centers


def find_centers_kmeans(bboxes, k, max_iter=100, verbose=True):
    r"""
    Find bounding box centers by kmeans.

    Parameters
    ----------
    bboxes : ``numpy.ndarray``
        Bounding boxes of normalized [xmin, ymin, xmax, ymax].
    k : ``int``
        Number of clusters (priors).
    max_iter : ``int``
        Maximum numer of iterations. Default: 100
    verbose: ``bool``
        Whether to print info.
    """
    centers = kmeans(bboxes, k, max_iter, verbose=verbose)[1]
    if verbose:
        mean_iou = iou_mn(bboxes, centers).max(axis=1).mean()
        print("Mean IoU: %.4f" % mean_iou)
    return centers


def find_priors_kmeans(sizes, k, max_iter=100, verbose=True):
    r"""
    Find bounding box centers by kmeans.

    Parameters
    ----------
    sizes : ``numpy.ndarray``
        Bounding boxes of normalized [width, height].
    k : ``int``
        Number of clusters (priors).
    max_iter : ``int``
        Maximum numer of iterations. Default: 100
    verbose: ``bool``
        Whether to print info.
    """
    bboxes = np.concatenate([np.full_like(sizes, 0.5), sizes], axis=-1)
    bboxes = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB, inplace=True)
    centers = find_centers_kmeans(bboxes, k, max_iter, verbose)
    centers[:, 2] -= centers[:, 0]
    centers[:, 3] -= centers[:, 1]
    priors = centers[:, 2:]
    return torch.from_numpy(priors).float()
