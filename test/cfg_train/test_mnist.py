import torch.nn as nn

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from horch.train.lr_scheduler import CosineAnnealingLR
from horch.core import load_yaml_config
from horch.datasets import train_test_split
from horch.models.modules import Conv2d, Flatten
from horch.train import manual_seed
from horch.train.trainer2 import Trainer
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy
from horch.transforms import Compose

cfg = load_yaml_config('/test/cfg_train/mnist.yaml')
manual_seed(cfg.seed)


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


train_transform = Compose(cfg.Dataset.Train.transforms)
val_transform = Compose(cfg.Dataset.Val.transforms)
test_transform = Compose(cfg.Dataset.Test.transforms)

data_home = cfg.Dataset.data_home
ds = MNIST(data_home, train=True, download=True)
ds = train_test_split(ds, test_ratio=0.1, random=True)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=cfg.Dataset.Split.test_ratio, random=cfg.Dataset.Split.random,
    transform=train_transform,
    test_transform=val_transform,
)
ds_test = MNIST(data_home, train=False, download=True)
ds_test = train_test_split(ds_test, test_ratio=0.1, random=True, test_transform=test_transform)[1]

net = eval(cfg.Model)(**cfg.get(cfg.Model))
criterion = nn.CrossEntropyLoss()


def get_optimizer(cfg):
    optimizer_type = cfg.type
    cfg.pop("type")
    optimizer = eval(optimizer_type)(net.parameters(), **cfg)
    return optimizer


def get_lr_scheduler(cfg, optimizer):
    lr_scheduler_type = cfg.type
    cfg.pop("type")
    lr_scheduler = eval(lr_scheduler_type)(optimizer, **cfg)
    return lr_scheduler


optimizer = get_optimizer(cfg.Optimizer)
lr_scheduler = get_lr_scheduler(cfg.LRScheduler, optimizer)

metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

test_metrics = {
    'loss': Loss(criterion),
    'acc': Accuracy(),
}

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics, test_metrics, save_path=cfg.save_path)

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

trainer.fit(train_loader, cfg.epochs, val_loader=val_loader, eval_freq=cfg.get("eval_freq", 1))

trainer.evaluate(test_loader)