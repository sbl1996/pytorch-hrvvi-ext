from multiprocessing.context import SpawnContext

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split
from horch.train.lr_scheduler import CosineAnnealingLR
from horch.models.utils import summary
from horch.models.cifar.efficientnet import efficientnet_b0
from horch.train import Trainer, Save
from horch.train.metrics import TrainLoss
from horch.train.metrics.classification import Accuracy
from horch.train.trainer import print_lr
from horch.transforms.ext import Cutout, CIFAR10Policy


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
ds = train_test_split(ds, test_ratio=0.5)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

net = LeNet5()
# net = efficientnet_b0(num_classes=10, dropout=0.3, drop_connect=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
lr_scheduler = CosineAnnealingLR(optimizer, 100, eta_min=1e-3, warmup=5, warmup_eta_min=1e-3)


metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, save_path="./checkpoints", name="CIFAR10-EfficientNet")

# summary(net, (3, 32, 32))

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2,
                          pin_memory=True, multiprocessing_context=SpawnContext())
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

trainer.fit(train_loader, 630, val_loader=val_loader,
            save=Save.ByMetric("-val_loss", patience=600),
            callbacks=[print_lr])

trainer.evaluate(test_loader)