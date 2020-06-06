import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose, Pad, Lambda

from horch.config import cfg
from horch.datasets import train_test_split, CombineDataset
from horch.nas.darts.model_search_gdas import Network
from horch.nas.darts.trainer import DARTSTrainer
from horch.train import manual_seed
from horch.train.metrics import TrainLoss, Loss
from horch.train.metrics.classification import Accuracy

manual_seed(0)

train_transform = Compose([
    Pad(2),
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
    Lambda(lambda x: x.expand(3, -1, -1))
])


root = '/Users/hrvvi/Code/study/pytorch/datasets'
ds_all = MNIST(root=root, train=True, download=True, transform=train_transform)

ds = train_test_split(ds_all, test_ratio=0.001, random=True)[1]
ds_train, ds_val = train_test_split(ds, test_ratio=0.5, random=True)
ds = CombineDataset(ds_train, ds_val)

train_loader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=2)
val_loader = DataLoader(ds_val, batch_size=2, pin_memory=True, num_workers=2)

cfg.relu.inplace = False
cfg.bn.affine = False

criterion = nn.CrossEntropyLoss()
tau_max, tau_min = 10, 0.1
model = Network(8, 8, steps=4, multiplier=4, stem_multiplier=1, tau=tau_max)

optimizer_arch = Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
optimizer_model = SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3e-4)
lr_scheduler = CosineAnnealingLR(optimizer_model, T_max=50, eta_min=0.001)

metrics = {
    "loss": TrainLoss(),
    "acc": Accuracy(),
}

test_metrics = {
    "loss": Loss(criterion),
    "acc": Accuracy(),
}

trainer = DARTSTrainer(model, criterion, [optimizer_arch, optimizer_model], lr_scheduler,
                       metrics, test_metrics, save_path='checkpoints/DARTS')

def tau_schedule(engine, trainer):
    iteration = engine.state.iteration
    iters_per_epoch = engine.state.epoch_length
    steps = iteration / iters_per_epoch
    model.tau = (tau_max - tau_min) * (1 - steps / engine.state.max_epochs) + tau_min


trainer.fit(train_loader, 20, val_loader, eval_freq=3, save_freq=2, callbacks=[tau_schedule])

# trainer.resume()
#
# trainer.fit(train_loader, None, val_loader, eval_freq=2)