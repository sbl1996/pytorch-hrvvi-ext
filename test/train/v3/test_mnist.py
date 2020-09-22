import math

import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad

from horch.datasets import train_test_split
from horch.models.layers import Conv2d
from horch.nn import Flatten
from horch.train import manual_seed
from horch.optim.lr_scheduler import CosineAnnealingLR

from horch.train.v3.callbacks import Callback
from horch.train.v3.cls import CNNLearner
from horch.train.v3.metrics import TrainLoss, Loss
from horch.train.v3.cls.metrics import Accuracy

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

data_home = "/Users/hrvvi/Code/study/pytorch/datasets/MNIST"
ds = MNIST(data_home, train=True, download=False)
ds = train_test_split(ds, test_ratio=0.1, shuffle=True)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.05, shuffle=True,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = MNIST(data_home, train=False, download=False, transform=test_transform)
ds_test = train_test_split(ds_test, test_ratio=0.1, shuffle=True)[1]

mul = 1
batch_size = 128
steps_per_epoch = math.ceil(len(ds_train) / batch_size)
min_lr = 0.01 * mul

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=min_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
lr_scheduler = CosineAnnealingLR(optimizer, epochs=30, min_lr=0.001, warmup_epoch=5, warmup_min_lr=0.001)

train_metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

eval_metrics = {
    'loss': Loss(criterion),
    'acc': Accuracy(),
}

trainer = CNNLearner(net, criterion, optimizer, lr_scheduler,
                     train_metrics=train_metrics, eval_metrics=eval_metrics,
                     work_dir="./checkpoints/MNIST-LeNet5")
trainer.load()

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ds_val, batch_size=128)
test_loader = DataLoader(ds_test, batch_size=128)

# class TrySave(Callback):
#
#     def after_epoch(self, state):
#         if state['epoch'] == 3:
#             self.learner.save()
trainer.fit(train_loader, 5, val_loader=val_loader)
# trainer.save()
# print(trainer.fit(train_loader, 2, val_loader=val_loader))

trainer.evaluate(test_loader)