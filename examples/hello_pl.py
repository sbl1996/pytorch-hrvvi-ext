import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

import pytorch_lightning as pl

from horch.datasets import train_test_split
from horch.models.cifar.shufflenetv2 import ShuffleNetV2
from horch.train.lr_scheduler import CosineAnnealingLR
from horch.transforms.classification.autoaugment import Cutout

train_transform = Compose([
    RandomCrop(32, padding=4, fill=128),
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

data_home = "datasets/CIFAR10"
ds = CIFAR10(data_home, train=True, download=True)
ds = train_test_split(ds, test_ratio=0.5)[1]
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transform,
    test_transform=test_transform,
)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)


class CIFARModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = ShuffleNetV2(32, [128, 256, 512], [4, 8, 4], 512, num_classes=10, use_se=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, 100, eta_min=1e-3, warmup=5, warmup_eta_min=1e-3)
        return [optimizer], [lr_scheduler]

model = CIFARModel()

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

trainer = pl.Trainer(gpus=1)

trainer.fit(model)
