import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split, CombineDataset
from horch.defaults import set_defaults
from horch.nas.nasnet.search.gdas import Network, TauSchedule

from horch.optim.lr_scheduler import CosineAnnealingLR
from horch.train import manual_seed
from horch.train.cls.metrics import Accuracy
from horch.train.v3.darts import DARTSLearner
from horch.train.v3.metrics import TrainLoss, Loss

manual_seed(0)

train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
])

valid_transform = Compose([
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
])

data_home = "/Users/hrvvi/Code/study/pytorch/datasets/CIFAR10"
ds = CIFAR10(root=data_home, train=True, download=True)

ds = train_test_split(ds, test_ratio=0.02, shuffle=True)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.5, shuffle=True,
    transform=train_transform, test_transform=valid_transform)
ds = CombineDataset(ds_train, ds_val)

train_loader = DataLoader(ds, batch_size=256, pin_memory=True, num_workers=2)
val_loader = DataLoader(ds_val, batch_size=256, pin_memory=True, num_workers=2)

set_defaults({
    'relu': {
        'inplace': False,
    },
    'bn': {
        'affine': False,
    }
})
model = Network(4, 8)
criterion = nn.CrossEntropyLoss()

epochs = 250
optimizer_model = SGD(model.model_parameters(), 0.025, momentum=0.9, weight_decay=3e-4, nesterov=True)
optimizer_arch = Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
lr_scheduler = CosineAnnealingLR(optimizer_model, epochs=epochs, min_lr=1e-5)

train_metrics = {
    "loss": TrainLoss(),
    "acc": Accuracy(),
}

eval_metrics = {
    "loss": Loss(criterion),
    "acc": Accuracy(),
}


learner = DARTSLearner(
    model, criterion, optimizer_arch, optimizer_model, lr_scheduler,
    train_metrics=train_metrics, eval_metrics=eval_metrics, work_dir='models/DARTS', fp16=False)

learner.fit(train_loader, epochs, val_loader, callbacks=[TauSchedule(10, 0.1)])
