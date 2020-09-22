import random
import os
import json
from PIL import Image

from hhutil.io import save_json, fmt_path, read_json

from torch.utils.data import Dataset

# https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py


class CocoDetection(Dataset):

    def __init__(self, root, ann_file, transform=None):
        from hpycocotools.coco import COCO
        self.root = root
        self.ann_file = ann_file

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

    def get_image(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs([img_id])[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        return img

    def get_target(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        target = coco.loadAnns(ann_ids)
        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs([img_id])[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def extract(ann_file, d, indices, suffix, img_dir=None):
    sub_d = {
        k: v
        for k, v in d.items()
        if k not in ['images', 'annotations']
    }
    images = d['images']
    sub_images = [images[i] for i in indices]
    sub_images_ids = set(img['id'] for img in sub_images)
    sub_annotations = [ann for ann in d['annotations'] if ann['image_id'] in sub_images_ids]
    sub_d['images'] = sub_images
    sub_d['annotations'] = sub_annotations
    save_json(sub_d, ann_file.parent / (ann_file.stem + "_" + suffix + ".json"))

def sample(ann_file, k):
    ann_file = fmt_path(ann_file)

    d = read_json(ann_file)
    images = d['images']

    n = len(images)
    indices = list(range(n))
    random.shuffle(indices)
    sub_indices = indices[:k]

    extract(ann_file, d, sub_indices, suffix="sub")


def train_test_split(ann_file, test_ratio, seed=0):
    ann_file = fmt_path(ann_file)

    d = read_json(ann_file)

    n = len(d['images'])
    n_test = int(n * test_ratio)
    n_train = n - n_test
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    extract(ann_file, d, train_indices, "train")
    extract(ann_file, d, test_indices, "test")
