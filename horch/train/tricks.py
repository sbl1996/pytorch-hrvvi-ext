import torch
import torch.nn as nn


# LR warmup

# Zero gamma (bn of residual)

# No bias decay
def get_params(model):
    for m in model.modules():
        if 'Conv' in m.__class__.__name__ or isinstance(m, nn.Linear):
            for name, p in m.named_parameters():
                if name == 'weight':
                    yield p

# FP16

# ResNet-D (Efficient)
# 1. stride=2 in conv3x3
# 2. replace conv1x1 of stride=2 with avgpool 2x2 followed by conv1x1

# Cosine decay

# Label smoothing

# Mixup