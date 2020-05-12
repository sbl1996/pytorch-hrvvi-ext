import argparse

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from horch.core import load_yaml_config
from horch.core.catalog.helper import get_optimizer, get_lr_scheduler
from horch.datasets import train_test_split
from horch.models.modules import Conv2d, Flatten
from horch.train import manual_seed
from horch.train.trainer import Trainer
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy
from horch.transforms import Compose

class LeNet5(nn.Sequential):

    def __init__(self, num_classes=10):
        super().__init__(
            Conv2d(1, 6, kernel_size=5, norm_layer='default', activation='default'),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv2d(6, 16, kernel_size=5, norm_layer='default', activation='default'),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(8 * 8 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes),
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MNIST.')
    parser.add_argument('-c', '--config', help='config file')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    manual_seed(cfg.seed)

    train_transform = Compose(cfg.Dataset.Train.transforms)
    val_transform = Compose(cfg.Dataset.Val.transforms)
    test_transform = Compose(cfg.Dataset.Test.transforms)

    data_home = cfg.Dataset.data_home
    ds = MNIST(data_home, train=True, download=True)
    ds = train_test_split(ds, test_ratio=0.2, random=True)[1]
    ds_train, ds_val = train_test_split(
        ds, test_ratio=cfg.Dataset.Split.test_ratio, random=cfg.Dataset.Split.random,
        transform=train_transform,
        test_transform=val_transform,
    )
    ds_test = MNIST(data_home, train=False, download=True)
    ds_test = train_test_split(ds_test, test_ratio=0.2, random=True, test_transform=test_transform)[1]

    net = eval(cfg.Model)(**cfg.get(cfg.Model))
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(cfg.Optimizer, net)
    lr_scheduler = get_lr_scheduler(cfg.LRScheduler, optimizer)

    if cfg.get("Mixup"):
        mixup = True
        mixup_alpha = cfg.Mixup.alpha
    else:
        mixup = False
        mixup_alpha = None

    metrics = {
        'loss': TrainLoss(),
        'acc': Accuracy(mixup=mixup),
    }

    test_metrics = {
        'loss': Loss(criterion),
        'acc': Accuracy(),
    }

    trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                      metrics, test_metrics, save_path=cfg.save_path, mixup_alpha=mixup_alpha)

    train_loader = DataLoader(ds_train,
                              batch_size=cfg.Dataset.Train.batch_size,
                              num_workers=cfg.Dataset.Train.get("num_workers", 2),
                              shuffle=cfg.Dataset.Train.get("shuffle", True),
                              pin_memory=cfg.Dataset.Train.get("pin_memory", True))
    val_loader = DataLoader(ds_val,
                            batch_size=cfg.Dataset.Val.batch_size,
                            num_workers=cfg.Dataset.Val.get("num_workers", 2))
    test_loader = DataLoader(ds_test,
                             batch_size=cfg.Dataset.Test.batch_size,
                             num_workers=cfg.Dataset.Test.get("num_workers", 2))

    trainer.fit(train_loader, cfg.epochs, val_loader=val_loader, eval_freq=cfg.get("eval_freq", 1), progress_bar=True)

    trainer.evaluate(test_loader)