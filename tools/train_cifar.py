import argparse

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100

from hhutil.io import read_text

from horch.config.helper import get_transform, get_data_loader, get_model, get_lr_scheduler, get_optimizer, get_mix
from horch.datasets import train_test_split
from horch.defaults import update_defaults

from horch.nn.loss import CrossEntropyLoss

import horch.models.cifar
import horch.optim.lr_scheduler

from horch.config.config import get_config, override
from horch.train import manual_seed
from horch.train.cls import Trainer
from horch.train.cls.metrics import Accuracy
from horch.train.metrics import TrainLoss, Loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CIFAR10/CIFAR100.')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-r', '--resume', help='resume from checkpoints')
    args = parser.parse_args()
    print(read_text(args.config))
    cfg = get_config(args.config)
    dataset = cfg.Dataset.type
    assert dataset in ["CIFAR10", 'CIFAR100']

    fp16 = cfg.get("fp16", False)
    if fp16:
        assert cfg.device == 'gpu'

    update_defaults(cfg.get("Global", {}))

    manual_seed(cfg.seed)

    data_home = cfg.Dataset.data_home

    train_transform = get_transform(cfg.Dataset.Train.transforms)
    test_transform = get_transform(cfg.Dataset.Test.transforms)

    if dataset == "CIFAR10":
        num_classes, CIFAR = 10, CIFAR10
    else:
        num_classes, CIFAR = 100, CIFAR100

    if cfg.get("hpo"):
        import nni
        RCV_CONFIG = nni.get_next_parameter()
        for k, v in RCV_CONFIG.items():
            ks = k.split(".")
            override(cfg, ks, v)

    ds_train = CIFAR(data_home, train=True, transform=train_transform)
    ds_test = CIFAR(data_home, train=False, transform=test_transform)

    if cfg.get("Debug") and cfg.Debug.get("subset"):
        ratio = cfg.Debug.subset
        ds_train = train_test_split(ds_train, ratio)[1]
        ds_test = train_test_split(ds_test, ratio)[1]

    use_mix = cfg.get("Mix") is not None
    if use_mix:
        cfg.Mix.num_classes = num_classes
        ds_train = get_mix(cfg.Mix, ds_train)

    train_loader = get_data_loader(cfg.Dataset.Train, ds_train)
    test_loader = get_data_loader(cfg.Dataset.Test, ds_test)

    cfg.Model.num_classes = num_classes
    model = get_model(cfg.Model, horch.models.cifar)

    criterion = CrossEntropyLoss(non_sparse=use_mix, label_smoothing=cfg.get("label_smooth"))

    epochs = cfg.epochs
    optimizer = get_optimizer(cfg.Optimizer, model.parameters())
    lr_scheduler = get_lr_scheduler(cfg.LRScheduler, optimizer, epochs)

    train_metrics = {
        'loss': TrainLoss()
    }
    if not use_mix:
        train_metrics['acc'] = Accuracy()

    test_metrics = {
        'loss': Loss(CrossEntropyLoss()),
        'acc': Accuracy(),
    }

    trainer = Trainer(model, criterion, optimizer, lr_scheduler, train_metrics, test_metrics,
                      work_dir=cfg.work_dir, fp16=fp16, device=cfg.get("device", 'auto'))

    eval_freq = cfg.get("eval_freq", 1)
    save_freq = cfg.get("save_freq")
    work_dir = cfg.get("work_dir")

    if args.resume:
        trainer.load()

    if save_freq:
        assert work_dir is not None

    trainer.fit(train_loader, epochs, test_loader, eval_freq, save_freq)