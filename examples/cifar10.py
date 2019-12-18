import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts
from horch.models.utils import summary
from horch.models.cifar.efficientnet import efficientnet_b0
from horch.train import Trainer, Save
from horch.train.metrics import Accuracy, TrainLoss
from horch.transforms.ext import Cutout, CIFAR10Policy

# Data Augmentation

train_transform = Compose([
    RandomCrop(32, padding=4, fill=128),
    RandomHorizontalFlip(),
    CIFAR10Policy(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    Cutout(1, 16),
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Dataset

data_home = "datasets/CIFAR10"
ds = CIFAR10(data_home, train=True, download=True)
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

# Define network, loss and optimizer
net = efficientnet_b0(num_classes=10, dropout=0.3, drop_connect=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)

# Define metrics

metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

# Put it together to get a Trainer

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, save_path="./checkpoints", name="CIFAR10-EfficientNet")

# Show number of parameters

summary(net, (3, 32, 32))

# Define batch size

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

# Train and save the best model by val loss (lower is better) after 600 epochs
# Actually, there is no need to do early stopping using CosineAnnealingWarmRestarts.

trainer.fit(train_loader, 630, val_loader=val_loader, save=Save.ByMetric("-val_loss", patience=600))

# Evaluate on test set (You should get accuracy above 97%)

trainer.evaluate(test_loader)