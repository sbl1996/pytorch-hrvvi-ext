import argparse

from horch.core.catalog.helper import get_optimizer, get_lr_scheduler, get_dataloader

import torch
from torchvision.datasets import CIFAR10

from horch.core import load_yaml_config
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

    if cfg.get("benchmark"):
        torch.backends.cudnn.benchmark = True

    train_transform = Compose(cfg.Dataset.Train.transforms)
    test_transform = Compose(cfg.Dataset.Test.transforms)

    data_home = cfg.Dataset.data_home
    ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
    ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

    train_loader = get_dataloader(cfg.Dataset.Train, ds_train)
    test_loader = get_dataloader(cfg.Dataset.Test, ds_test)

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
                      metrics, test_metrics, save_path=cfg.save_path, mixup_alpha=mixup_alpha,
                      fp16=cfg.get("fp16", False))

    trainer.fit(train_loader, cfg.epochs, val_loader=test_loader,
                eval_freq=cfg.get("eval_freq", 1), save_freq=cfg.get("save_freq"),
                n_saved=cfg.get("n_saved", 1), progress_bar=cfg.get("prograss_bar", False))
