# Overview
[pytorch-hrvvi-ext](https://github.com/sbl1996/pytorch-hrvvi-ext) is my extension to PyTorch, which contains many "out of the box" tools to facilitate my everyday study. It is very easy to use them and integrate them to your projects.
I will call it `horch` below because of `import horch`.

# Install

```bash
pip install pybind11

# Install with no extras
pip install -U git+https://github.com/sbl1996/pytorch-hrvvi-ext.git

# Install with extras if you want to use these provided datasets
pip install -U git+https://github.com/sbl1996/hpycocotools.git
pip install -U git+https://github.com/sbl1996/pytorch-hrvvi-ext.git#egg=pytorch-hrvvi-ext[coco]

# Install unstable version with latest features
pip install -U git+https://github.com/sbl1996/pytorch-hrvvi-ext.git@dev#egg=pytorch-hrvvi-ext[coco]
```


# Hightlights

## Trainer
`Trainer` is written on [ignite](https://github.com/pytorch/ignite), providing the following features:

- Train your network in few lines without writing loops explicitly.
- Automatic gpu support like Keras
- Metric for both CV and NLP (Loss, Accuracy, Top-K Accuracy, mAP, BLEU)
- Checkpoints of the whole trainer by epochs or metrics

## Datasets
`horch` contains many datasets wrapped by me providing `torchvison.datasets` style API. Some of them is much easier to train than VOC or COCO and more suitable for *BEGINNERS* in object detection.

- CaptchaDetectionOnline: generate captcha image and bounding boxes of chars online
- SVHNDetection: [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset for object detection
- VOCDetection: enhanced `torchvision.datasets.VOCDetection` with test set
- VOCSegmentation: enhanced `torchvision.datasets.VOCVOCSegmentation` with [trainaug](http://home.bharathh.info/pubs/codes/SBD/download.html) set
- Costom COCO format dataset with COCOEval

## Transforms
Transoforms in `horch` transform inputs and targets of datasets simultaneously, which is more flexible than `torchvison.transforms` and makes it easier to do data augmentation for object detection with `torchvision.transforms` style API.

- Resize
- CenterCrop
- RandomResizedCrop
- RandomSampleCrop
- ToPercentCoords
- RandomHorizontalFlip
- SSDTransform

### Detection
`horch.detection` provides many useful functions for object detection includeing:

- BBox: bounding box class supporting three formats (LTRB, XYWH, LTWH)
- transform_bbox: transform bounding box between three formats
- iou_1m: calculate IOU between 1 and many
- non_max_suppression

### Models

#### Backbone
- MobileNetV2 (pretrained)
- ShuffleNetV2 (pretrained)
- SqueezeNet (pretrained)
- SNet
- Darknet53
- ResNet

#### FPN
- FPN (standard)
- FPN2 (bottom up)
- stacked_fpn (alternant top down and bottom up)
- ContextEnhance (N->1)

#### Head
- SSDHead
- RetinaHead
- SharedDWConvHead

#### Model
- SSD (SSDHead + extra layers)
- RefineDet (AnchorRefineInference, RefineLoss)
- FCOS (FCOSTransform, FCOSInference, FCOSLoss)

#### Combination 

- Standard: MatchAnchors + Backbone + (FPN) + Head + AnchorBasedInference + MultiBoxLoss
- SSD: Backbone + SSD
- RetinaNet: Backbone + FPN + RetinaHead + MultiBoxLoss(criterion='focal')
- RefineDet: Backbone + AnchorRefineInference + RefineDet + RefineLoss
- FCOS: FCOSTransform + Backbone + FPN + RetinaHead(num_anchors=1, num_classes=num_classes+1) + FCOSInference + FCOS + FCOSLoss

### Others
- train_test_split: Split a dataset to a train set and a test set with different (or same) transforms
- init_weights: Initialize weights of your model in the right and easy way
- Fullset: Transform your dataset to `horch` style dataset

# Examples


## CIFAR10

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split, Fullset
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts
from horch.models.utils import summary
from horch.models.cifar.efficientnet import efficientnet_b0
from horch.train import Trainer, Save
from horch.train.metrics import Accuracy, TrainLoss
from horch.transforms import InputTransform
from horch.transforms.ext import Cutout, CIFAR10Policy

# Data Augmentation

train_transforms = InputTransform(
    Compose([
        RandomCrop(32, padding=4, fill=128),
        RandomHorizontalFlip(),
        CIFAR10Policy(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        Cutout(1, 16),
    ])
)

test_transform = InputTransform(
    Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
)

# Dataset

data_home = "datasets/CIFAR10"
ds = CIFAR10(data_home, train=True, download=True)
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.02,
    transform=train_transforms,
    test_transform=test_transform,
)
ds_test = Fullset(CIFAR10(data_home, train=False,
                          download=True), test_transform)

# Define network, loss and optimizer
net = efficientnet_b0(num_classes=10, dropout=0.2, drop_connect=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.05, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)

# Define metrics

metrics = {
    'loss': TrainLoss(),
    'acc': Accuracy(),
}

# Put it together with Trainer

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, save_path="./checkpoints", name="CIFAR10-EfficientNet")

# Show number of parameters

summary(net, (3, 32, 32))

# Define batch size

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=128)
val_loader = DataLoader(ds_val, batch_size=128)

# Train and save good models by val loss (lower is better) after first 40 epochs
# Hint: In fact, there is no need to do early stopping when using CosineAnnealingWarmRestarts.

trainer.fit(train_loader, 630, val_loader=val_loader, save=Save.ByMetric("-val_loss", patience=600))

# Evaluate on test set (You should get accuracy above 97%)

trainer.evaluate(test_loader)
```

## SVHNDetection + RefineDet

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader

import horch
from horch import cuda
from horch.datasets import train_test_split, Fullset, CocoDetection

from horch.train import Trainer, Save, misc_collate
from horch.train.optimizer import AdamW
from horch.train.metrics import TrainLoss, COCOEval
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts

from horch.transforms import Compose, ToTensor
from horch.transforms.detection import Resize, ToPercentCoords, SSDTransform

from horch.detection import generate_mlvl_anchors,  find_priors_coco
from horch.detection.one import MatchAnchors

from horch.models.utils import summary
from horch.models.backbone import SNet
from horch.models.detection.refinedet import RefineLoss, AnchorRefineInference, RefineDet

# Describe the dataset

width = 192
height = 96
levels = (3, 4, 5)
priors_per_level = 3
num_classes = 11  # num_classes + *1* (background)

# Define the path to the dataset

root = "/Users/hrvvi/Code/study/pytorch/datasets/SVHN/train"
ds = CocoDetection(root, "/Users/hrvvi/Code/study/pytorch/datasets/SVHN/annotations/train.json")

# Find priors and generate anchors automaticly

strides = [2 ** l for l in levels]
num_levels = len(levels)
priors = find_priors_coco(ds, k=num_levels * priors_per_level)
anchor_sizes = priors.view(num_levels, priors_per_level, 2) * \ 
    torch.tensor([width, height], dtype=torch.float32)
anchors = generate_mlvl_anchors((width, height), strides, anchor_sizes)

# Data augmentation
# Use SSDTransform may improve AP

train_transform = Compose([
    # SSDTransform((width, HEIGHT), color_jitter=False, expand=(1,3), min_area_frac=0.4),
    Resize((width, height)),
    ToPercentCoords(),
    ToTensor(),
    MatchAnchors(anchors, pos_thresh=0.5),
])

test_transform = Compose([
    Resize((width, height)),
    ToPercentCoords(),
    ToTensor(),
])

# Use only 0.005% data to try

rest, ds = train_test_split(ds, test_ratio=0.0005)

ds_train = Fullset(ds, train_transform)
ds_val = Fullset(ds, test_transform)

# Define model
# Choose `bn` as normalization layer if batch size is enough (e.g > 32)
horch.models.set_default_norm_layer('gn')

f_channels = 64
backbone = SNet(version=49, feature_levels=(3,4,5))
inference = AnchorRefineInference(cuda(anchors), neg_threshold=0.01, r_topk=200, d_topk=100)
net = RefineDet(backbone, anchor_sizes.size(1), num_classes, f_channels, inference, extra_levels=None)
summary(net, (3, height, width))

criterion = RefineLoss(neg_threshold=0.01, p=1)  # p means the propability to print loss


# Choose optimizer and lr_scheduler

optimizer = AdamW(filter(lambda x: x.requires_grad,
                        net.parameters()), lr=0.001, weight_decay=1e-4)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)  # T_0 should equal to number of epochs

metrics = {
    'loss': TrainLoss(),
}

test_metrics = {
    "AP": COCOEval(ds.dataset.to_coco(ds.indices))
}

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, test_metrics=test_metrics,
                  save_path="./checkpoints", name="RefineDet-SVHN")


# Choose batch size

train_loader = DataLoader(
    ds_train, batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(
    ds_val, batch_size=64, collate_fn=misc_collate)

# Train 100 epochs, evaluate every 10 epochs and save model with highest AP

trainer.fit(train_loader, 100, val_loader=(val_loader, 10), save=Save.ByMetric("val_AP", patience=80))

```


## SVHNDetection + FoveaBox

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader

import horch
from horch import cuda
from horch.datasets import SVHNDetection, train_test_split, Fullset, Subset

from horch.train import Trainer, Save, ValSet, misc_collate
from horch.train.optimizer import AdamW
from horch.train.metrics import TrainLoss, COCOEval
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts

from horch.transforms.detection import Compose, Resize, ToTensor, ToPercentCoords

from horch.detection import get_locations

from horch.models.utils import summary
from horch.models.detection import OneStageDetector
from horch.models.backbone import ShuffleNetV2
from horch.models.detection.enhance import stacked_fpn
from horch.models.detection.head import RetinaHead
from horch.models.detection.fovea import FoveaLoss, FoveaInference, FoveaTransform, get_mlvl_centers

num_classes = 1 + 10
width = 192
height = 96
levels = [3, 4, 5]
strides = [2 ** l for l in levels]
locations = get_locations((width, height), strides)
mlvl_centers = get_mlvl_centers(locations)

area_thresholds = [
    (lo ** 2, hi ** 2)
    for (lo, hi) in
    [(16, 64), (32, 128), (64, 256)]
]


train_transform = Compose([
    Resize((width, height)),
    ToTensor(),
    FoveaTransform(mlvl_centers, levels, thresholds=area_thresholds),
])

test_transform = Compose([
    Resize((width, height)),
    ToPercentCoords(),
    ToTensor(),
])


data_home = '/Users/hrvvi/Code/study/pytorch/datasets/SVHN/'
ds = SVHNDetection(data_home, split='train', download=True)

ds_rest, ds = train_test_split(ds, test_ratio=0.0005)

ds_train, ds_val = train_test_split(
    ds, test_ratio=0.1,
    transform=train_transform,
    test_transform=test_transform)
ds_tv = Subset(ds, ds_train.indices[:int(0.5 * len(ds_train))], test_transform)

horch.models.set_default_norm_layer('gn')

backbone = ShuffleNetV2(mult=0.5, feature_levels=levels)
fpn = stacked_fpn(2, backbone.out_channels, 64)
head = RetinaHead(1, num_classes, 64, num_layers=2, concat=False)
inference = FoveaInference(cuda(mlvl_centers), conf_threshold=0.05, topk1=100, topk2=20)

net = OneStageDetector(backbone, fpn, head, inference)

criterion = FoveaLoss(p=1)
optimizer = AdamW(filter(lambda x: x.requires_grad,
                        net.parameters()), lr=0.001)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

metrics = {
    'loss': TrainLoss(),
}

test_metrics = {
    'mAP': COCOEval(ds.to_coco()),
}

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, test_metrics=test_metrics,
                  save_path="./checkpoints", name="Fovea-SVHN")

train_loader = DataLoader(
    ds_train, batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(
    ds_val, batch_size=32, collate_fn=misc_collate)
tv_loader = DataLoader(
    ds_tv, batch_size=32, collate_fn=misc_collate)

trainer.fit1(train_loader, epochs=50,
             validate=[
                 ValSet(val_loader, per_epochs=10, name="val"),
                 ValSet(tv_loader, per_epochs=5, name="train"),
             ], save=Save.ByMetric("train_mAP", patience=10))
```