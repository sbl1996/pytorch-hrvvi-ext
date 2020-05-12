from horch.core.catalog.catalog import register_op
from torch.optim import SGD, Adam

ops = [
    SGD,
    Adam
]


for cls in ops:
    register_op(cls, serialize=False)