import torch.nn as nn
from torch.nn.utils import spectral_norm

from horch.models.modules import ConditionalBatchNorm2d


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


class Discriminator(nn.Module):
    def __init__(self, in_channels, channels=64, out_channels=1, num_classes=10, size=(32, 32), leaky_slope=0.1, use_sn=True, project_y=True):
        super().__init__()
        self.h, self.w = size
        self.out_channels = out_channels
        self.project_y = project_y
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, True),
            nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, True),
        )
        self.dense = nn.Linear((self.h // 8) * (self.w // 8) * channels * 8, out_channels)

        if project_y:
            self.embedding = nn.Embedding(num_classes, (self.h // 8) * (self.w // 8) * channels * 8)

        if use_sn:
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
            spectral_norm(self.dense)
            if project_y:
                spectral_norm(self.embedding)

    def forward(self, x, y):
        x = self.conv(x).view(x.size(0), -1)
        out = self.dense(x)
        if self.project_y:
            embedded_y = self.embedding(y)
            out += (embedded_y * x).sum(dim=1, keepdim=True)
        return out
