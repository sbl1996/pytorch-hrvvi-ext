from horch.core.catalog.catalog import register_op
from horch.train.lr_scheduler import CosineAnnealingLR

ops = [
    CosineAnnealingLR,
]


for cls in ops:
    register_op(cls, serialize=False)