import argparse

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from horch.core import load_yaml_config, register_op
from horch.config import cfg as global_cfg, load_from_dict
from horch.core.catalog.helper import get_optimizer, get_lr_scheduler, get_dataloader
from horch.datasets import train_test_split, Fullset
from horch.models.modules import Conv2d, Flatten
from horch.nn.loss import CrossEntropyLoss
from horch.train import manual_seed
from horch.train.classification.mix import get_mix
from horch.train.classification.trainer import Trainer
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy
from horch.transforms import Compose

from torch.optim.lr_scheduler import MultiStepLR

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

    register_op(MultiStepLR, serialize=False)

    parser = argparse.ArgumentParser(description='Train MNIST.')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='resume from checkpoints')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    if cfg.get("Global"):
        global_cfg.merge_from_other_cfg(load_from_dict(cfg.get("Global")))

    manual_seed(cfg.seed)

    train_transform = Compose(cfg.Dataset.Train.transforms)
    val_transform = Compose(cfg.Dataset.Val.transforms)
    test_transform = Compose(cfg.Dataset.Test.transforms)

    data_home = cfg.Dataset.data_home
    ds = MNIST(data_home, train=True, download=True)
    ds_test = MNIST(data_home, train=False, download=True)
    if cfg.get("Debug") and cfg.Debug.get("subset"):
        ratio = cfg.Debug.subset
        ds = train_test_split(ds, test_ratio=ratio, random=True)[1]
        ds_test = train_test_split(ds_test, test_ratio=ratio, random=True)[1]

    ds_train, ds_val = train_test_split(
        ds, test_ratio=cfg.Dataset.Split.test_ratio, random=cfg.Dataset.Split.random,
        transform=train_transform,
        test_transform=val_transform,
    )
    ds_test = Fullset(ds_test, test_transform)

    train_loader = get_dataloader(cfg.Dataset.Train, ds_train)
    val_loader = get_dataloader(cfg.Dataset.Val, ds_val)
    test_loader = get_dataloader(cfg.Dataset.Test, ds_test)

    net = eval(cfg.Model)(**cfg.get(cfg.Model))

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
                      metrics, test_metrics, save_path=cfg.save_path, mix=mix)

    if args.resume:
        trainer.resume()

    trainer.fit(train_loader, cfg.epochs, val_loader=val_loader,
                eval_freq=cfg.get("eval_freq", 1), save_freq=cfg.get("save_freq"),
                n_saved=cfg.get("n_saved", 1), progress_bar=cfg.get("prograss_bar", False))

    trainer.evaluate(test_loader)