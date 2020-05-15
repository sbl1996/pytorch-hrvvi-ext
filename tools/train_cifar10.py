import argparse

from horch.core.catalog.helper import get_optimizer, get_lr_scheduler, get_dataloader, get_model

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10

from horch.core import load_yaml_config
from horch.config import cfg as global_cfg, load_from_dict
from horch.datasets import train_test_split
from horch.nn.loss import CrossEntropyLoss
from horch.train import manual_seed
from horch.train.classification.mix import get_mix
from horch.train.classification.trainer import Trainer
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy

import horch.models.cifar

from torchvision.transforms import Compose

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CIFAR10.')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='resume from checkpoints')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    if cfg.get("Global"):
        global_cfg.merge_from_other_cfg(load_from_dict(cfg.get("Global")))

    manual_seed(cfg.seed)

    if cfg.get("benchmark"):
        torch.backends.cudnn.benchmark = True

    train_transform = Compose(cfg.Dataset.Train.transforms)
    test_transform = Compose(cfg.Dataset.Test.transforms)

    data_home = cfg.Dataset.data_home
    ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
    ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

    if cfg.get("Debug") and cfg.Debug.get("subset"):
        ratio = cfg.Debug.subset
        ds_train = train_test_split(ds_train, test_ratio=ratio, random=True)[1]
        ds_test = train_test_split(ds_test, test_ratio=ratio, random=True)[1]

    train_loader = get_dataloader(cfg.Dataset.Train, ds_train)
    test_loader = get_dataloader(cfg.Dataset.Test, ds_test)

    net = get_model(cfg.Model, horch.models.cifar)

    criterion = CrossEntropyLoss(label_smoothing=cfg.get("label_smooth"))

    optimizer = get_optimizer(cfg.Optimizer, net)
    lr_scheduler = get_lr_scheduler(cfg.LRScheduler, optimizer)

    mix = get_mix(cfg.get("Mix"))

    metrics = {
        'loss': TrainLoss(),
        'acc': Accuracy(mix),
    }

    test_metrics = {
        'loss': Loss(nn.CrossEntropyLoss()),
        'acc': Accuracy(),
    }

    trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                      metrics, test_metrics, save_path=cfg.save_path, mix=mix,
                      fp16=cfg.get("fp16", False))

    if args.resume:
        trainer.resume()

    trainer.fit(train_loader, cfg.epochs, val_loader=test_loader,
                eval_freq=cfg.get("eval_freq", 1), save_freq=cfg.get("save_freq"),
                n_saved=cfg.get("n_saved", 1), progress_bar=cfg.get("prograss_bar", False))
