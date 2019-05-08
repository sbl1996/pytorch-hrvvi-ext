import re
import json
import tarfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from horch.datasets.utils import download_google_drive

SPLIT_FILES = {
    "train": {
        "name": "train.tar",
        "md5": "5e2cf0c808741ec1d53039c1986d328b",
        "url": "https://drive.google.com/open?id=1eexDhJC6UpCghrxqwSyqKnqDRmHInP9B",
    },
    "test": {
        "name": "test.tar",
        "md5": "0af70bdc326b095e5d8d0ab9840f9324",
        "url": "https://drive.google.com/open?id=1tO8N7b3iyU0ucRj2XffD0dZz8CGgFjL7",
    }
}


class SVHNDetection(Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and targets simultaneously and returns a transformed version of them.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 download=False):
        self.root = Path(root).expanduser().absolute()
        self.split = split
        self.transform = transform
        self.filename = SPLIT_FILES[split]['name']
        self.md5 = SPLIT_FILES[split]['md5']
        self.url = SPLIT_FILES[split]["url"]
        self.img_dir = self.root / split
        self.ann_filename = split + ".json"
        self.ann_dir = self.root / "annotations"
        self.ann_dir.mkdir(exist_ok=True, parents=True)
        self.ann_file = self.ann_dir / self.ann_filename

        if download:
            self.download()

        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)

        from hpycocotools.coco import COCO
        self.coco = COCO(self.data, verbose=False)
        self.ids = list(self.coco.imgs.keys())

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
            tuple: (image, anns) where target is a dictionary of the XML tree.
        """

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(self.img_dir / path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def download(self):
        if self.img_dir.is_dir() and self.ann_file.exists():
            print("Dataset found. Skip download or extract")
            return

        google_drive_match = re.match(
            r"https://drive.google.com/open\?id=(.*)", self.url)
        if google_drive_match:
            file_id = google_drive_match.group(1)
            download_google_drive(file_id, self.root, self.filename, self.md5)
        else:
            download_url(self.url, self.root, self.filename, self.md5)

        file_path = self.root / self.filename
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=self.root)
        ann_file = self.root / self.ann_filename
        ann_file.rename(self.ann_file)
