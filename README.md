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

## SVHNDetection

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader

from horch import cuda
from horch.datasets import train_test_split, Fullset, CocoDetection

from horch.train import Trainer, Save
from horch.train.metrics import TrainLoss, COCOEval
from horch.train.lr_scheduler import CosineAnnealingWarmRestarts

from horch.transforms import Compose, ToTensor
from horch.transforms.detection import Resize, ToPercentCoords, SSDTransform

from horch.detection import generate_mlvl_anchors, misc_target_collate, find_priors_coco
from horch.detection.one import MatchAnchors

from horch.models.utils import summary
from horch.models.detection import get_basic_levels, get_extra_levels
from horch.models.detection.backbone import SNet
from horch.models.detection.refinedet import RefineLoss, AnchorRefineInference, RefineDet

# Describe your dataset

WIDTH = 256
HEIGHT = 128
LEVELS = (4, 5, 6)
PRIORS_PER_LEVEL = 3
NUM_CLASSES = 11  # num_classes + *1* (background)

# Define the path to your dataset

root = "/Users/hrvvi/Code/study/pytorch/datasets/SVHN/train"
ds = CocoDetection(root, "/Users/hrvvi/Code/study/pytorch/datasets/SVHN/annotations/train.json")

# Find priors and generate anchors automaticly

STRIDES = [2 ** l for l in LEVELS]
NUM_LEVELS = len(LEVELS)
priors = find_priors_coco(ds, k=NUM_LEVELS * PRIORS_PER_LEVEL)
ANCHOR_SIZES = priors.view(NUM_LEVELS, PRIORS_PER_LEVEL, 2) * torch.tensor([WIDTH, HEIGHT], dtype=torch.float32)
ANCHORS = generate_mlvl_anchors((WIDTH, HEIGHT), STRIDES, ANCHOR_SIZES)

# Data augmentation
# Use SSDTransform may improve AP

train_transform = Compose([
    # SSDTransform((WIDTH, HEIGHT), color_jitter=False, expand=(1,3), min_area_frac=0.4),
    Resize((WIDTH, HEIGHT)),
    ToPercentCoords(),
    ToTensor(),
    MatchAnchors(ANCHORS, pos_thresh=0.5),
])

test_transform = Compose([
    Resize((WIDTH, HEIGHT)),
    ToPercentCoords(),
    ToTensor(),
])

# Use only 0.005% data to try

rest, ds = train_test_split(ds, test_ratio=0.0005)

ds_train = Fullset(ds, train_transform)
ds_val = Fullset(ds, test_transform)

# Define model
# Choose `bn` as normalization layer if batch size is enough (e.g > 32)

f_channels = 64
norm_layer = 'gn'
backbone = SNet(version=49, feature_levels=get_basic_levels(LEVELS), norm_layer=norm_layer)
inference = AnchorRefineInference(cuda(ANCHORS), neg_threshold=0.01, r_topk=200, d_topk=100)
net = RefineDet(backbone, ANCHOR_SIZES.size(1), NUM_CLASSES, f_channels, inference,
                norm_layer=norm_layer, extra_levels=get_extra_levels(LEVELS))
summary(net, (3, HEIGHT, WIDTH))

criterion = RefineLoss(neg_threshold=0.01, p=1)  # p means the propability to print loss


# Choose optimizer and lr_scheduler

optimizer = Adam(filter(lambda x: x.requires_grad,
                        net.parameters()), lr=0.001, weight_decay=1e-4)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)  # T_0 should equal to number of epochs

metrics = {
    'loss': TrainLoss(),
}

test_metrics = {
    "AP": COCOEval(ds.dataset.to_coco(ds.indices))
}

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, evaluate_metrics=test_metrics,
                  save_path="./checkpoints", name="RefineDet-SVHN")


# Choose batch size

train_loader = DataLoader(
    ds_train, batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(
    ds_val, batch_size=64, collate_fn=misc_target_collate)

# Train 100 epochs, evaluate every 10 epochs and save model with highest AP

trainer.fit(train_loader, 100, val_loader=(val_loader, 10), save=Save.ByMetric("val_AP", patience=80))

```

## CIFAR10

```python

# Data Preparation

train_transforms = InputTransform(
    Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
)

test_transform = InputTransform(
    Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
)

data_home = gpath("datasets/CIFAR10")
ds = CIFAR10(data_home, train=True, download=True)
ds_train, ds_val = train_test_split(
    ds, test_ratio=0.04,
    transform=train_transforms,
    test_transform=test_transform,
)


# Define network, loss and optimizer

net = ResNet(WideSEBasicBlock, [4,4,4], k=2)
net.apply(init_weights(nonlinearity='relu'))
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=1e-1, momentum=0.9, dampening=0, weight_decay=5e-4, nesterov=True)
lr_scheduler = MultiStepLR(optimizer, [40, 80, 110], gamma=0.2)


# Define metrics

metrics = {
    'loss': Loss(),
    'acc': Accuracy(),
}

# Put it together with Trainer

trainer = Trainer(net, criterion, optimizer, lr_scheduler, metrics=metrics, save_path=gpath("models"), name="CIFAR10-SE-WRN28-2")

# Show number of parameters

summary(net, (3,32,32))

# Define batch size

train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(ds_val, batch_size=128)

# Train and save good models by val loss (lower is better) after first 40 epochs

trainer.fit(train_loader, 100, val_loader=val_loader, save_by_metric='-val_loss', patience=40)
```

## CaptchaDetectionOnline
```python

letters = "0123456789abcdefghijkmnopqrstuvwxyzABDEFGHJKMNRT"
NUM_CLASSES = len(letters) + 1
WIDTH = 128
HEIGHT = 48
LOCATIONS = [
    (8, 3),
    (4, 2),
]
ASPECT_RATIOS = [
    (1, 2, 1/2),
    (1, 2, 1/2),
]
ASPECT_RATIOS = [torch.tensor(ars) for ars in ASPECT_RATIOS]
NUM_FEATURE_MAPS = len(ASPECT_RATIOS)
SCALES = compute_scales(NUM_FEATURE_MAPS, 0.2, 0.9)
DEFAULT_BOXES = [
    compute_default_boxes(lx, ly, scale, ars)
    for (lx, ly), scale, ars in zip(LOCATIONS, SCALES, ASPECT_RATIOS)
]


# Define captcha dataset

fonts = [
    gpath("fonts/msyh.ttf"),
    gpath("fonts/sfsl0800.pfb.ttf"),
    gpath("fonts/SimHei.ttf"),
    gpath("fonts/Times New Roman.ttf"),
]

font_sizes = (28, 32, 36, 40, 44, 48)
image = ImageCaptcha(
    WIDTH, HEIGHT, fonts=fonts, font_sizes=font_sizes)

transform = Compose([
    ToPercentCoords(),
    ToTensor(),
    SSDTransform(SCALES, DEFAULT_BOXES, NUM_CLASSES),
])

test_transform = Compose([
    ToTensor(),
])

ds_train = CaptchaDetectionOnline(
    image, size=50000, letters=letters, rotate=20, transform=transform)
ds_val = CaptchaDetectionOnline(
    image, size=1000, letters=letters, rotate=20, transform=test_transform, online=False)
 
    
# Define network, loss and optimizer

out_channels = [
    (NUM_CLASSES + 4) * len(ars)
    for ars in ASPECT_RATIOS
]
net = DSOD([3, 4, 4, 4], 36, out_channels=out_channels, reduction=1)
net.apply(init_weights(nonlinearity='relu'))
criterion = SSDLoss(NUM_CLASSES)
optimizer = Adam(net.parameters(), lr=3e-4)
lr_scheduler = MultiStepLR(optimizer, [40, 70, 100], gamma=0.1)


# Define metrics for training and testing

metrics = {
    'loss': TrainLoss(),
}
test_metrics = {
    'mAP': MeanAveragePrecision(
        SSDInference(
            width=WIDTH, height=HEIGHT,
            f_default_boxes=[ cuda(d) for d in DEFAULT_BOXES ],
            num_classes=NUM_CLASSES,
        )
    )
}

# Put it together with Trainer

trainer = Trainer(net, criterion, optimizer, lr_scheduler,
                  metrics=metrics, evaluate_metrics=test_metrics,
                  save_path=gpath("models"), name="DSOD-CAPTCHA-48")

# Show numbers of parameters

summary(net, (3,HEIGHT, WIDTH))


# Define batch size

train_loader = DataLoader(
    ds_train, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(
    ds_val, batch_size=32, collate_fn=box_collate_fn)

# Train and save by val mAP (higher is better) after first 10 epochs

trainer.fit(train_loader, 15, val_loader=val_loader, save_by_metric='val_mAP', patience=10)
```