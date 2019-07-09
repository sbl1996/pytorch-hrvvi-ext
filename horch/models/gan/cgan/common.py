import torch.nn as nn
from horch.models.modules import ConditionalBatchNorm2d
from torch.nn.utils import spectral_norm


class Generator(nn.Module):

    def __init__(self, in_channels, channels, out_channels, num_classes=10, size=(32, 32), use_sn=True):
        super().__init__()
        self.h, self.w = size
        self.in_channels = in_channels
        self.dense = nn.Linear(in_channels, (self.h // 8) * (self.w // 8) * channels * 8)

        self.bn1 = ConditionalBatchNorm2d(channels * 8, num_classes)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn2 = ConditionalBatchNorm2d(channels * 4, num_classes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn3 = ConditionalBatchNorm2d(channels * 2, num_classes)
        self.relu3 = nn.ReLU(True)
        self.conv3 = nn.ConvTranspose2d(channels * 2, channels * 1, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn4 = ConditionalBatchNorm2d(channels * 1, num_classes)
        self.relu4 = nn.ReLU(True)
        self.conv4 = nn.ConvTranspose2d(channels * 1, out_channels, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        if use_sn:
            spectral_norm(self.dense)
            spectral_norm(self.conv1)
            spectral_norm(self.conv2)
            spectral_norm(self.conv3)
            spectral_norm(self.conv4)

    def forward(self, x, y):
        x = self.dense(x).view(x.size(0), -1, self.h // 8, self.w // 8)

        x = self.bn1(x, y)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, y)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.bn3(x, y)
        x = self.relu3(x)
        x = self.conv3(x)

        x = self.bn4(x, y)
        x = self.relu4(x)
        x = self.conv4(x)

        x = self.tanh(x)
        return x
