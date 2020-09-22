import random
import numpy as np

from torch.utils.data import Dataset


def rand_bbox(shape, lam):
    H = shape[1]
    W = shape[2]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def one_hot(size, target):
    x = np.zeros(size, dtype=np.float32)
    x[target] = 1.
    return x


class CutMix(Dataset):

    def __init__(self, dataset, num_classes=10, num_mix=1, beta=1., prob=0.5):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        image, label = self.dataset[index]
        lb_onehot = one_hot(self.num_classes, label)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            image2, label2 = self.dataset[rand_index]
            lb2_onehot = one_hot(self.num_classes, label2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)
            image[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.shape[1] * image.shape[2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return image, lb_onehot

    def __len__(self):
        return len(self.dataset)