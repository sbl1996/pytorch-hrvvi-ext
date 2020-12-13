import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip

from horch.transforms.classification import CIFAR10Policy, Cutout

from horch.nn import CrossEntropyLoss
from horch.optim.lr_scheduler import CosineAnnealingLR
from horch.models.cifar.preactresnet import ResNet
from horch.train import manual_seed

from horch.train.cls import CNNLearner
from horch.train.cls.metrics import Accuracy
from horch.train.metrics import TrainLoss, Loss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
manual_seed(0)

train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    # CIFAR10Policy(),
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
    Cutout(1, 16),
])

test_transform = Compose([
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
])

data_home = gpath('datasets/CIFAR10')
ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)

batch_size = 128
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
test_loader = DataLoader(ds_test, batch_size=batch_size * 8, shuffle=False, pin_memory=True, num_workers=2)

net = ResNet(28, 10)
criterion = CrossEntropyLoss()

epochs = 200
base_lr = 0.1
optimizer = SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = CosineAnnealingLR(optimizer, epochs, min_lr=0)


train_metrics = {
    "loss": TrainLoss(),
    "acc": Accuracy(),
}

eval_metrics = {
    "loss": Loss(CrossEntropyLoss()),
    "acc": Accuracy(),
}

learner = CNNLearner(
    net, criterion, optimizer, lr_scheduler,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=gpath('models/WRN'), fp16=True)

learner.fit(train_loader, epochs, test_loader, val_freq=1)