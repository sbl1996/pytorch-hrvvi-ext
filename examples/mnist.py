import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad

from horch.datasets import train_test_split
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts
from horch.models.utils import summary
from horch.train import Trainer, Save
from horch.train.metrics import TrainLoss
from horch.train.metrics.classification import Accuracy


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
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


# Data Augmentation

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

# Dataset

data_home = "datasets"
ds = MNIST(data_home, train=True, download=True)
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = MNIST(data_home, train=False, download=True, transform=test_transform)

# Define network, loss and optimizer
net = LeNet5()
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
                  metrics=metrics, save_path="./checkpoints", name="MNIST-LeNet5")

# Show number of parameters

summary(net, (1, 32, 32))

# Define batch size

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

# Train

trainer.fit(train_loader, 70, val_loader=val_loader, save=Save.ByMetric('-val_loss', patience=50))

# Evaluate

trainer.evaluate(test_loader)