from collections import Sequence
from math import ceil

import numpy as np
from scipy import linalg

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor

from horch.common import CUDA
from horch.train._utils import to_device


class _ImageDataset(Dataset):

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, item):
        img = self.imgs[item]
        if self.transform is not None:
            img = self.transform(img)
        return img,

    def __len__(self):
        return len(self.imgs)


def batchify(tensors, batch_size=32):
    assert len(set([len(t) for t in tensors])) == 1
    n = len(tensors[0])
    n_batches = ceil(n / batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end = min(n, (i + 1) * batch_size)
        batch = tuple(t[start:end] for t in tensors)
        yield batch


def batch_apply(inputs, model, func=lambda x: x, batch_size=32, device=None):
    device = device or ('cuda' if CUDA else 'cpu')

    model.eval()
    model.to(device)

    if torch.is_tensor(inputs):
        inputs = (inputs,)

    if isinstance(inputs, Sequence) and all(torch.is_tensor(t) for t in inputs):
        it = batchify(inputs, batch_size=batch_size)
    else:
        transforms = Compose([
            ToTensor()
        ])
        ds = _ImageDataset(inputs, transforms)
        it = DataLoader(ds, batch_size=batch_size)

    preds = []
    for batch in it:
        x = to_device(batch, device)
        if torch.is_tensor(x):
            x = (x,)
        with torch.no_grad():
            p = func(model(*x))
        preds.append(p)
    preds = torch.cat(preds, dim=0)
    return preds


def inception_score(imgs, model, batch_size=32, device=None):
    r"""
    Parameters
    ----------
    imgs : List[Image] or ndarray or tensor
        `imgs` could be a list of PIL Images or uint8 ndarray of shape (N, H, W, C)
        or a float tensor of shape (N, C, H, W)
    """
    pyxs = batch_apply(imgs, model, lambda p: F.softmax(p, dim=1), batch_size, device)
    py = torch.mean(pyxs, dim=0)
    score = (pyxs * (pyxs / py).log_()).sum(dim=1).mean().exp().item()
    return score


def calculate_activation_statistics(imgs, model, batch_size=32, device=None):
    preds = batch_apply(imgs, model, lambda x: x, batch_size, device).cpu().numpy()
    mu = np.mean(preds, axis=0)
    sigma = np.cov(preds, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

