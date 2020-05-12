import argparse

import torch.nn as nn
from horch.core.catalog.helper import get_optimizer, get_lr_scheduler

from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10

from horch.core import load_yaml_config
from horch.datasets import train_test_split
from horch.nn.loss import CrossEntropyLoss
from horch.train import manual_seed
from horch.train.trainer import Trainer
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy
from horch.models.cifar import *

from torchvision.transforms import Compose

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CIFAR10.')
    parser.add_argument('-c', '--config', help='config file')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    manual_seed(cfg.seed)

    train_transform = Compose(cfg.Dataset.Train.transforms)
    test_transform = Compose(cfg.Dataset.Test.transforms)

    data_home = cfg.Dataset.data_home
    ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
    ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

    ds_train = train_test_split(ds_train, test_ratio=0.01, random=True)[1]
    ds_test = train_test_split(ds_test, test_ratio=0.01, random=True)[1]

    net = eval(cfg.Model)(**cfg.get(cfg.Model))
    criterion = CrossEntropyLoss(label_smoothing=cfg.get("label_smooth"))

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
                              num_workers=cfg.Dataset.Train.get("num_workers", 1),
                              shuffle=cfg.Dataset.Train.get("shuffle", True),
                              pin_memory=cfg.Dataset.Train.get("pin_memory", True))
    test_loader = DataLoader(ds_test,
                             batch_size=cfg.Dataset.Test.batch_size,
                             num_workers=cfg.Dataset.Test.get("num_workers", 1))

    trainer.fit(train_loader, cfg.epochs, val_loader=test_loader, eval_freq=cfg.get("eval_freq", 1), progress_bar=True)