import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split
from horch.nn import CrossEntropyLoss, DropPath
from horch.optim.lr_scheduler import CosineAnnealingLR
from horch.models.cifar.preactresnet import ResNet
from horch.nas.nasnet.genotypes import Genotype
from horch.train import manual_seed
from horch.train.v3.callbacks import Callback

from horch.train.v3.cls import CNNLearner
from horch.train.v3.cls.metrics import Accuracy
from horch.train.v3.metrics import TrainLoss, Loss
from horch.transforms.classification import CIFAR10Policy, Cutout

manual_seed(0)
# torch.backends.cudnn.benchmark = True

train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    # CIFAR10Policy(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    Cutout(1, 16),
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

data_home = "/Users/hrvvi/Code/study/pytorch/datasets/CIFAR10"
ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)
ds_train = train_test_split(ds_train, test_ratio=0.01)[1]
ds_test = train_test_split(ds_test, test_ratio=0.01)[1]

epochs = 200
# net = PreActResNet(28, 10)
net = ResNet(16, 1)
criterion = CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
lr_scheduler = CosineAnnealingLR(optimizer, epochs, min_lr=0)

train_metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

eval_metrics = {
    'loss': Loss(CrossEntropyLoss()),
    'acc': Accuracy(),
}

trainer = CNNLearner(net, criterion, optimizer, lr_scheduler,
                     train_metrics=train_metrics, eval_metrics=eval_metrics, work_dir="models/WRN")

# summary(net, (3, 32, 32))

train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(ds_test, batch_size=32)

trainer.fit(train_loader, epochs, val_loader=test_loader, val_freq=5)

trainer.evaluate(test_loader)