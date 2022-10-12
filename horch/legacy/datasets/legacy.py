from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import xmltodict
from PIL import Image
from horch.datasets.voc import TEST_DATASET_YEAR_DICT, DATASET_YEAR_DICT, VOC_CATEGORIES, VOC_CATEGORY_TO_IDX
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val`` or ``test``
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
        self.transform = transform

        base_dir = dataset_dict[year]['base_dir']
        self.voc_root = self.root / base_dir
        image_dir = self.voc_root / 'JPEGImages'
        annotation_dir = self.voc_root / 'Annotations'

        if download:
            self.download()

        splits_dir = self.voc_root / 'ImageSets' / "Main"

        split_f = splits_dir / (image_set + '.txt')

        if not split_f.exists():
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid '
                'image_set from the VOC ImageSets/Main folder.')

        with open(split_f, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / (x + ".jpg") for x in self.file_names]
        self.annotations = [annotation_dir /
                            (x + ".xml") for x in self.file_names]
        assert (len(self.images) == len(self.annotations))

    def to_coco(self, indices=None):
        if indices is None:
            indices = range(len(self))
        categories = []
        for i, c in enumerate(VOC_CATEGORIES[1:]):
            categories.append({
                'supercategory': 'object',
                'id': i + 1,
                'name': c
            })
        images = []
        annotations = []
        ann_id = 0
        for i in indices:
            info = self.parse_voc_xml(self.annotations[i])
            anns = info['annotations']
            img = {
                "file_name": (self.file_names[i] + ".jpg"),
                "height": info['height'],
                "width": info['width'],
                "id": i,
            }
            images.append(img)
            for ann in anns:
                ann = deepcopy(ann)
                w, h = ann['bbox'][2:]
                ann['area'] = w * h
                ann['iscrowd'] = 0
                ann['image_id'] = i
                ann['id'] = ann_id
                ann_id += 1
                annotations.append(ann)
        dataset = {
            'categories': categories,
            'images': images,
            'annotations': annotations,
        }
        return dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index])
        anns = self.parse_voc_xml(self.annotations[index])['annotations']
        for ann in anns:
            ann['image_id'] = index

        if self.transform is not None:
            img, anns = self.transform(img, anns)

        return img, anns

    def __len__(self):
        return len(self.images)

    def download(self):
        import tarfile
        if self.voc_root.is_dir():
            print("Dataset found. Skip download or extract")
            return

        download_url(self.url, self.root, self.filename, self.md5)

        with tarfile.open(self.root / self.filename, "r") as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def parse_object(self, obj):
        x1 = int(obj['bndbox']['xmin'])
        y1 = int(obj['bndbox']['ymin'])
        x2 = int(obj['bndbox']['xmax'])
        y2 = int(obj['bndbox']['ymax'])
        w = x2 - x1
        h = y2 - y1
        bbox = [x1, y1, w, h]
        return {
            'category_id': VOC_CATEGORY_TO_IDX[obj['name']],
            'bbox': bbox
        }

    def parse_voc_xml(self, path):
        with open(path, "rb") as f:
            d = xmltodict.parse(f)
        info = d['annotation']
        objects = info['object']
        size = info['size']
        if isinstance(objects, OrderedDict):
            objects = [objects]
        return {
            'width': int(size['width']),
            'height': int(size['height']),
            'annotations': [self.parse_object(obj) for obj in objects],
        }

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

