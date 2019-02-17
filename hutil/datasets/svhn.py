import re
from pathlib import Path
import pickle
import tarfile

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity

SPLIT_FILES = {
    "train": {
        "name": "train.tar",
        "md5": "318f6caf5b82a73193bc436a80c55f3e",
        "url": "https://drive.google.com/open?id=1Hhl1fXxo9VOorwIKKOnDNZqFdwZy3Haj",
    },
    "test": {
        "name": "test.tar",
        "md5": "ddf20d0d71dcb0dd8796b6fe7021ba13",
        "url": "https://drive.google.com/open?id=1EW8BvfTX-rmoJF0hpeB5FnGFcvagK_op",
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
        self.detection_dir = self.root / "Detection"
        self.base_dir = self.detection_dir / split

        if download:
            self.download()

        self.ann_file = self.base_dir / "annotations.pt"

        with open(self.ann_file, "rb") as f:
            data = pickle.load(f)

        self.images = [self.base_dir / x['name']
                       for x in data]
        self.annotations = [x['annotations'] for x in data]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.annotations[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def download(self):
        if self.base_dir.is_dir():
            print("Dataset found. Skip download or extract")
            return

        google_drive_match = re.match(
            r"https://drive.google.com/open\?id=(.*)", self.url)
        if google_drive_match:
            file_id = google_drive_match.group(1)
            download_google_drive(file_id, self.root, self.filename, self.md5)
        else:
            download_url(self.url, self.root, self.filename, self.md5)

        self.detection_dir.mkdir(exist_ok=True)

        file_path = self.root / self.filename
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=self.detection_dir)


def download_google_drive(file_id, root, filename, md5):
    fpath = root / filename

    root.mkdir(exist_ok=True)

    # downloads file
    if fpath.is_file() and check_integrity(fpath, md5):
        print('Using downloaded and verified file: %s' % fpath)
    else:
        from google_drive_downloader import GoogleDriveDownloader as gdd
        gdd.download_file_from_google_drive(
            file_id=file_id,
            dest_path=fpath,
        )
