import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad

from horch.datasets import train_test_split
from horch.models.modules import Conv2d, Flatten
from horch.train import manual_seed
from horch.train.lr_scheduler import CosineAnnealingLR
from horch.train.model_selection import KFold, cross_val_score

from horch.train.v2.cls import Trainer
from horch.train.v2.metrics import TrainLoss, Loss
from horch.train.v2.cls.metrics import Accuracy

manual_seed(0)


class LeNet5(nn.Sequential):

    def __init__(self):
        super().__init__(
            Conv2d(1, 6, kernel_size=5, norm='default', act='default'),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv2d(6, 16, kernel_size=5, norm='default', act='default'),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(8 * 8 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )


train_transform = Compose([
    Pad(2),
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
])

test_transform = Compose([
    Pad(2),
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
])

data_home = "/Users/hrvvi/Code/study/pytorch/datasets"
ds = MNIST(data_home, train=True, download=False)
ds = train_test_split(ds, test_ratio=0.02, shuffle=True)[1]

cv = KFold(
    5, True, train_transform, test_transform, random_state=0)


def fit_fn(ds_train, ds_val, verbose):

    net = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # lr_scheduler = MultiStepLR(optimizer, [10, 20], gamma=0.1)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001, warmup=5, warmup_eta_min=0.01)

    metrics = {
        'loss': TrainLoss(),
        'acc': Accuracy(),
    }

    test_metrics = {
        'loss': Loss(criterion),
        'acc': Accuracy(),
    }

    trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                      metrics=metrics, test_metrics=test_metrics,
                      work_dir="./checkpoints/MNIST-LeNet5")
    trainer._verbose = False
    # summary(net, (1, 32, 32))

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=128)

    accs = trainer.fit(train_loader, 5, val_loader=val_loader)['acc']
    return accs[-1], max(accs)

cross_val_score(fit_fn, ds, cv, verbose=1)