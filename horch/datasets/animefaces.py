import os
import re

import numpy as np
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url

from horch.datasets.utils import download_google_drive


class AnimeFaces(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_info = {
        'unlabeled48': {
            'url': 'https://drive.google.com/open?id=16tAUOr4QCs-0jObqFAUPRX9Fv_tO48k1',
            'tgz_filename': 'animefaces.tar.gz',
            'tgz_md5': '5b94b54ddcc4abba492357509ce7b1d0',
            'filename': 'unlabeled48_X.npy',
            'md5': 'a66112bec4a2ab4467eda5c68eccd69a'
        },
        'unlabeled96': {
            # 'url': 'https://drive.google.com/open?id=16tAUOr4QCs-0jObqFAUPRX9Fv_tO48k1',
            'tgz_filename': 'animefaces96.tar.gz',
            'tgz_md5': '47701650112cd743e35f8a31a6854642',
            'filename': 'unlabeled96_X.npy',
            'md5': '4b75a17e80d28b140ce4bafc0bb51884'
        },
    }

    def __init__(self, root, split='unlabeled48',
                 transform=None, target_transform=None,
                 download=False):

        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.data = []
        self.targets = []

        fp = os.path.join(root, self.split_info[self.split]['filename'])
        self.data = np.load(fp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = -1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        info = self.split_info[self.split]
        filename, md5 = info['tgz_filename'], info['tgz_md5']
        fpath = os.path.join(root, filename)
        if not check_integrity(fpath, md5):
            return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        info = self.split_info[self.split]
        download_google_drive(info['url'], self.root, info['tgz_filename'], info['tgz_md5'])

        # extract file
        with tarfile.open(os.path.join(self.root, info['tgz_filename']), "r:gz") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Split: {}".format(self.split)