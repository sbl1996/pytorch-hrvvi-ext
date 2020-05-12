from horch.core.catalog.catalog import register_op
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ToTensor, ColorJitter, Resize, RandomVerticalFlip, \
    Normalize, Pad, RandomResizedCrop
from horch.transforms.classification import Cutout, CIFAR10Policy

ops = [
    RandomHorizontalFlip,
    RandomCrop,
    ToTensor,
    ColorJitter,
    Resize,
    RandomVerticalFlip,
    Normalize,
    Pad,
    RandomResizedCrop,
    Cutout,
    CIFAR10Policy,
]


for cls in ops:
    register_op(cls)