from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split
from horch.nn import CrossEntropyLoss, DropPath
from horch.optim.lr_scheduler import CosineAnnealingLR
from horch.models.cifar.nasnet import NASNet
from horch.nas.darts.genotypes import Genotype
from horch.train import manual_seed
from horch.train.callbacks import Callback

from horch.train.cls import CNNLearner
from horch.train.cls import Accuracy
from horch.train.metrics import TrainLoss, Loss
from horch.transforms.classification import Cutout

manual_seed(0)
# torch.backends.cudnn.benchmark = True

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

data_home = "/Users/hrvvi/Code/study/pytorch/datasets/CIFAR10"
ds_train = CIFAR10(data_home, train=True, download=True, transform=train_transform)
ds_test = CIFAR10(data_home, train=False, download=True, transform=test_transform)
ds_train = train_test_split(ds_train, test_ratio=0.01)[1]
ds_test = train_test_split(ds_test, test_ratio=0.01)[1]

PC_DARTS_cifar = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_5x5', 1), ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)
    ],
    reduce_concat=[2, 3, 4, 5]
)

drop_path = 0.3
epochs = 600
# net = NASNet(36, 20, True, drop_path, 10, PC_DARTS_cifar)
net = NASNet(4, 5, True, drop_path, 10, PC_DARTS_cifar)
criterion = CrossEntropyLoss(auxiliary_weight=0.4)
optimizer = SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
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
                     train_metrics=train_metrics, eval_metrics=eval_metrics, work_dir="../train/v3/models")

# summary(net, (3, 32, 32))

train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(ds_test, batch_size=32)


class DropPathSchedule(Callback):

    def __init__(self, drop_path):
        super().__init__()
        self.drop_path = drop_path

    def begin_epoch(self, state):
        drop_path = self.drop_path * (state['epoch'] / state['epochs'])
        for m in self.learner.model.modules():
            if isinstance(m, DropPath):
                m.p = drop_path

trainer.fit(train_loader, epochs, val_loader=test_loader, val_freq=5,
            callbacks=[DropPathSchedule(drop_path)])

trainer.evaluate(test_loader)