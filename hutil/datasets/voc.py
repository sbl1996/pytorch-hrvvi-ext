import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity

from hutil.datasets.utils import download_google_drive

# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}

DETECTION_CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "total",
]

DETECTION_CATEGORY_TO_IDX = {name: i for i,
                             name in enumerate(DETECTION_CATEGORIES)}


class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.image_set = image_set

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        self.voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(self.voc_root, 'JPEGImages')
        annotation_dir = os.path.join(self.voc_root, 'Annotations')

        if download:
            self.download()

        splits_dir = os.path.join(self.voc_root, 'ImageSets', "Main")

        split_f = os.path.join(splits_dir, image_set + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg")
                       for x in file_names]
        self.annotations = [os.path.join(
            annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(self.annotations[index])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def download(self):
        import tarfile
        if os.path.isdir(self.voc_root):
            print("Dataset found. Skip download or extract")
            return

        download_url(self.url, self.root, self.filename, self.md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r") as tar:
            tar.extractall(path=self.root)

    def parse_object(self, obj):
        x1 = int(obj['bndbox']['xmin'])
        y1 = int(obj['bndbox']['ymin'])
        x2 = int(obj['bndbox']['xmax'])
        y2 = int(obj['bndbox']['ymax'])
        w = x2 - x1
        h = y2 - y1
        bbox = [x1, y1, w, h]
        return {
            'category_id': DETECTION_CATEGORY_TO_IDX[obj['name']],
            'bbox': bbox
        }

    def parse_voc_xml(self, path):
        with open(path, "rb") as f:
            d = xmltodict.parse(f)
        ann = d['annotation']
        objects = ann['object']
        if isinstance(objects, OrderedDict):
            objects = [objects]
        return [self.parse_object(obj) for obj in objects]


TRAINAUG_FILE = {
    "name": "trainaug.tar",
    "md5": "7677cd72fdefc1f4d23beb556c0e87dc",
    "url": "https://drive.google.com/open?id=1inOFikLz9oOW85s4nuAlCZ_1XesU_zn2",
}


class VOCSegmentation(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val`` or ``trainaug``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):
        self.root = Path(root).expanduser().absolute()
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.image_set = image_set
        self.augmented = image_set == 'trainaug'

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        self.voc_root = self.root / base_dir
        image_dir = self.voc_root / 'JPEGImages'
        mask_dir = self.voc_root / 'SegmentationClass'
        if self.augmented:
            mask_dir = self.voc_root / 'SegmentationClassAug'

        if download:
            self.download()

        splits_dir = self.voc_root / 'ImageSets' / 'Segmentation'
        split_f = splits_dir / (image_set.rstrip('\n') + '.txt')

        if not split_f.exists():
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or image_set="trainaug"')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / (x + ".jpg") for x in file_names]
        self.masks = [mask_dir / (x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def download(self):
        import tarfile

        if self.voc_root.is_dir():
            print("VOC found. Skip download or extract")
        else:
            download_url(self.url, self.root, self.filename, self.md5)

            with tarfile.open(self.root / self.filename, "r") as tar:
                tar.extractall(path=self.root)

        if self.augmented:
            mask_dir = self.voc_root / 'SegmentationClassAug'
            if mask_dir.is_dir():
                print("SBT found. Skip download or extract")
            else:
                file_id = re.match(
                    r"https://drive.google.com/open\?id=(.*)", TRAINAUG_FILE['url']).group(1)
                filename = TRAINAUG_FILE['name']
                download_google_drive(
                    file_id, self.voc_root, filename, TRAINAUG_FILE['md5'])

                file_path = self.voc_root / filename
                with tarfile.open(file_path, "r") as tar:
                    tar.extractall(path=self.voc_root)
                split_f = self.voc_root / 'trainaug.txt'
                splits_dir = self.voc_root / 'ImageSets' / 'Segmentation'
                split_f.rename(splits_dir / split_f.name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
