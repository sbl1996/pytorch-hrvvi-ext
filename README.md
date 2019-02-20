# Overview
[pytorch-hrvvi-ext](https://github.com/sbl1996/pytorch-hrvvi-ext) is my extension to PyTorch, which contains many "out of the box" tools to facilitate my everyday study. It is very easy to use them and integrate them to your projects.
I will call it `hutil` below because of `import hutil`.

# Install

```bash
pip3 install -U --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pytorch-hrvvi-ext
```

# Hightlights

## Trainer
`Trainer` is written on [ignite](https://github.com/pytorch/ignite), providing the following features:

- Train your network in few lines without writing loops explicitly.
- Automatic gpu support like Keras
- Metric for both CV and NLP (Loss, Accuracy, Top-K Accuracy, mAP, BLEU)
- Checkpoints of the whole trainer by epochs or metrics
- Send metric history to WeChat

## Datasets
`hutil` contains many datasets wrapped by me providing `torchvison.datasets` style API. Some of them is much easier to train than VOC or COCO and more suitable for *BEGINNERS* in object detection. Now it contains the following datasets:

- CaptchaDetectionOnline: generate captcha image and bounding boxes of chars online
- SVHNDetection: [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset for object detection
- CocoDetection: unreleased dataset of torchvison with `hutil`'s transforms
- VOCDetection: unreleased dataset of torchvison with `hutil`'s transforms

## Transforms
Transoforms in `hutil` transform inputs and targets of datasets simultaneously, which is more flexible than `torchvison.transforms` and makes it easier to do data augmentation for object detection with `torchvision.transforms` style API. The following transoforms is provided now:

- Resize
- CenterCrop
- ToPercentCoords
- Compose
- InputTransform
- TargetTransform

## Others
- train_test_split: Split a dataset to a train set and a test set with different (or same) transforms
- Fullset: Transform your dataset to `hutil`' style dataset

# Examples

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
ds_test = CIFAR10(data_home, train=False, download=True)


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
test_loader = DataLoader(ds_test, batch_size=128)
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