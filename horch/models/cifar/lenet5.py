import torch.nn as nn

from horch.nn import Flatten
from horch.models.layers import Conv2d, Linear


class LeNet5(nn.Module):

    def __init__(self, in_channels=1, num_classes=10, dropout=None):
        super().__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 6, 5, stride=1, padding=0, act='def'),
            nn.MaxPool2d(2, 2),
            Conv2d(6, 16, 5, stride=1, padding=0, act='def'),
            nn.MaxPool2d(2, 2),
        )

        classifier = [
            Flatten(),
            Linear(400, 120, act='def'),
            Linear(120, 84, act='def'),
            Linear(84, num_classes)
        ]
        if dropout:
            classifier.insert(0, nn.Dropout(dropout))
            classifier.insert(2, nn.Dropout(dropout))
            classifier.insert(4, nn.Dropout(dropout))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x