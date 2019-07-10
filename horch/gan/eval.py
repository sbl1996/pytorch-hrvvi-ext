import numpy as np
from scipy import linalg

import torch
import torch.nn.functional as F
from scipy.stats import entropy
from toolz import curry
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
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
        return img

    def __len__(self):
        return len(self.imgs)


def inception_score(imgs, model, batch_size=32, device=None):
    device = device or ('cuda' if CUDA else 'cpu')

    transforms = Compose([
        ToTensor()
    ])
    ds = _ImageDataset(imgs, transforms)
    data_loader = DataLoader(ds, batch_size=batch_size)

    model.eval()
    model.to(device)

    pyxs = []
    for batch in data_loader:
        x = to_device(batch, device)
        with torch.no_grad():
            p = F.softmax(model(x), dim=1)
        pyxs.append(p)

    pyxs = torch.cat(pyxs, dim=0)
    py = torch.mean(pyxs, dim=0)
    score = (pyxs * (pyxs / py).log_()).sum(dim=1).mean().exp().item()
    return score


def calculate_activation_statistics(imgs, model, batch_size=32, device=None):
    device = device or ('cuda' if CUDA else 'cpu')

    transforms = Compose([
        ToTensor()
    ])
    ds = _ImageDataset(imgs, transforms)
    data_loader = DataLoader(ds, batch_size=batch_size)

    model.eval()
    model.to(device)

    preds = []
    for batch in data_loader:
        x = to_device(batch, device)
        with torch.no_grad():
            p = model(x).view(x.size(0), -1)
        preds.append(p.cpu().numpy())
    preds = np.concatenate(preds)
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

