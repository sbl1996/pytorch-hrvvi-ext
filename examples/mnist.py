import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad
from horch.datasets import train_test_split
from horch.train.lr_scheduler import CosineAnnealingLR
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy
from horch.train.classification.trainer import Trainer

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

data_home = "datasets"
ds = MNIST(data_home, train=True, download=True)
ds = train_test_split(ds, test_ratio=0.1, random=True)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = MNIST(data_home, train=False, download=True, transform=test_transform)

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4, nesterov=True)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001, warmup=5, warmup_eta_min=0.1)

metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(mixup=True),
}

test_metrics = {
    'loss': Loss(criterion),
    'acc': Accuracy(),
}

# summary(net, (1, 32, 32))

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics, test_metrics, save_path="./checkpoints/MNIST-LeNet5", mixup_alpha=1.0)

# trainer.resume()

trainer.fit(train_loader, 35, val_loader=val_loader, eval_freq=5, save_freq=10, n_saved=2)