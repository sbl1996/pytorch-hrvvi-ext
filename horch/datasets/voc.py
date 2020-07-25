import bisect
import copy
import json
import os
import re
from pathlib import Path

import xmltodict
from PIL import Image
from toolz.curried import groupby
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity

from horch.datasets.utils import download_google_drive

# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012',
        # "ann_file_url": "https://drive.google.com/open?id=1v98GB2D7oc6OoP8NdIHayZbt-8V6Fc5Q",
        "ann_file_url": "https://drive.google.com/open?id=1f_MTZr4ypkY83yahZ61zCkLb6z3Bo_7Y",
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
        'base_dir': 'VOCdevkit/VOC2007',
        "ann_file_url": "https://drive.google.com/open?id=189LC78-tuvJXKawqwirFlFzflqvAI9oA",
        # "ann_file_url": "https://drive.google.com/open?id=1dwgWLM4qxe5aT3o46y0Jz10DKmh17w6W",
    }
}

TEST_DATASET_YEAR_DICT = {
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': 'VOCdevkit/VOC2007',
        'ann_file_url': "https://drive.google.com/open?id=1sAT2wgrMNFqDsUWom4foQ-WtxwA_IS7e",
        # "ann_file_url": "https://drive.google.com/open?id=1BGSle9xH6B_voeUE4Mp0YYYFmKZfxu0B",
    }
}


VOC_CATEGORIES = [
    "__background__",
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
]

VOC_CATEGORY_TO_IDX = {name: i for i,
                       name in enumerate(VOC_CATEGORIES)}


class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``trainval`` or ``test``
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
                 image_set='trainval',
                 download=False,
                 transform=None):
        self.root = Path(root).expanduser().absolute()
        self.year = year
        self.image_set = image_set
        if image_set == 'test':
            dataset_dict = TEST_DATASET_YEAR_DICT
        else:
            dataset_dict = DATASET_YEAR_DICT
        self.url = dataset_dict[year]['url']
        self.filename = dataset_dict[year]['filename']
        self.md5 = dataset_dict[year]['md5']

        base_dir = dataset_dict[year]['base_dir']
        self.voc_root = self.root / base_dir
        self.image_dir = self.voc_root / 'JPEGImages'
        self.ann_file_url = dataset_dict[year]['ann_file_url']
        self.ann_file = self.voc_root / ("%s%s.json" % (image_set, year))

        if download:
            self.download()

        from hpycocotools.coco import COCO
        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)

        self.coco = COCO(self.data, verbose=False)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def to_coco(self, indices=None):
        if indices is None:
            return self.data
        ids = [self.ids[i] for i in indices]
        images = self.coco.loadImgs(ids)
        ann_ids = self.coco.getAnnIds(ids)
        annotations = self.coco.loadAnns(ann_ids)
        return {
            **self.data,
            "images": images,
            "annotations": annotations,
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs([img_id])[0]['file_name']

        img = Image.open(self.image_dir / path).convert('RGB')

        if self.transform is not None:
            img, anns = self.transform(img, anns)
        return img, anns

    def __len__(self):
        return len(self.ids)

    def download(self):
        import tarfile
        if self.voc_root.is_dir() and self.ann_file.exists():
            print("Dataset found. Skip download or extract.")
            return

        if not self.voc_root.is_dir():
            download_url(self.url, self.root, self.filename, self.md5)
            with tarfile.open(self.root / self.filename, "r") as tar:
                tar.extractall(path=self.root)

        if not self.ann_file.exists():
            google_drive_match = re.match(
                r"https://drive.google.com/open\?id=(.*)", self.ann_file_url)
            file_id = google_drive_match.group(1)
            download_google_drive(file_id, self.voc_root, self.ann_file.name)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Year: {}\n'.format(self.year)
        fmt_str += '    ImageSet: {}\n'.format(self.image_set)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


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


class VOCDetectionConcat(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._data = merge_coco(datasets)
        self._img_anns = groupby(lambda x: x['image_id'], self._data['annotations'])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        img = self.datasets[dataset_idx][sample_idx][0]
        anns = self._img_anns[idx]
        return img, anns

    def to_coco(self):
        return copy.deepcopy(self._data)


def merge_coco(datasets):
    all_annotations = [ds.to_coco() for ds in datasets]
    for i in range(len(all_annotations) - 1):
        assert all_annotations[i]['categories'] == all_annotations[i + 1]['categories']
    images = all_annotations[0]['images']
    annotations = all_annotations[0]['annotations']

    image_id = images[-1]['id'] + 1
    ann_id = annotations[-1]['id'] + 1
    for d in all_annotations[1:]:
        d_images = d['images']
        n = len(d_images)
        assert [img['id'] for img in d_images] == list(range(n))
        img_anns = groupby(lambda x: x['image_id'], d['annotations'])
        for i in range(n):
            img = d_images[i]
            anns = img_anns[img['id']]
            img = {
                **img,
                'id': image_id,
            }
            for ann in anns:
                annotations.append({
                    **ann,
                    'id': ann_id,
                    'image_id': image_id
                })
                ann_id += 1
            image_id += 1
            images.append(img)
    return {
        **all_annotations[0],
        'images': images,
        'annotations': annotations
    }
