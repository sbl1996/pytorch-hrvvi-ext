import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.gan.common import ResBlock
from torch.nn.utils import spectral_norm

from horch.models.modules import seq, SelfAttention2, ConditionalBatchNorm2d


class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, use_sn=False):
        super().__init__()
        shortcut = nn.Sequential()
        # shortcut.scale = MaxUnpool2d(True)
        shortcut.scale = nn.UpsamplingNearest2d(scale_factor=2)
        shortcut.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.shortcut = shortcut

        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu1 = nn.ReLU(True)

        self.scale = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_sn:
            spectral_norm(self.shortcut.conv)
            spectral_norm(self.conv1)
            spectral_norm(self.conv2)

    def forward(self, x, y):
        identity = x

        x = self.bn1(x, y)
        x = self.relu1(x)
        x = self.scale(x)
        x = self.conv1(x)
        x = self.bn2(x, y)
        x = self.relu2(x)
        x = self.conv2(x)
        return x + self.shortcut(identity)


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, channels=64, out_channels=3, num_classes=10, use_sn=True, non_local=True):
        super().__init__()
        self.in_channels = in_channels
        self.non_local = non_local
        self.dense = nn.Linear(in_channels, 6 * 6 * channels * 8)
        self.block1 = GenResBlock(channels * 8, channels * 4, num_classes, use_sn=use_sn)
        self.block2 = GenResBlock(channels * 4, channels * 2, num_classes, use_sn=use_sn)
        if non_local:
            self.attn = SelfAttention2(channels * 2)
        self.block3 = GenResBlock(channels * 2, channels * 1, num_classes, use_sn=use_sn)
        self.bn = ConditionalBatchNorm2d(channels * 1, num_classes)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(channels * 1, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        if use_sn:
            spectral_norm(self.dense)

            if non_local:
                spectral_norm(self.attn.conv_theta)
                spectral_norm(self.attn.conv_phi)
                spectral_norm(self.attn.conv_g)
                spectral_norm(self.attn.conv_attn)

            spectral_norm(self.conv)

    def forward(self, x, y):
        x = self.dense(x).view(x.size(0), -1, 6, 6)
        x = self.block1(x, y)
        x = self.block2(x, y)
        if self.non_local:
            x = self.attn(x)
        x = self.block3(x, y)
        x = self.bn(x, y)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x


class ResNetDiscriminator(nn.Module):

    def __init__(self, in_channels=3, channels=64, out_channels=1, num_classes=10, use_sn=True, use_bn=False,
                 project_y=True):
        super().__init__()
        self.out_channels = out_channels
        self.project_y = project_y
        self.conv = nn.Sequential(
            ResBlock(in_channels, channels * 1, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 1, channels * 2, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 2, channels * 4, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 4, channels * 8, 'down', use_sn=use_sn, use_bn=use_bn),
            ResBlock(channels * 8, channels * 16, None, use_sn=use_sn, use_bn=use_bn),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dense = nn.Linear(channels * 16, out_channels)

        if project_y:
            self.embedding = nn.Embedding(num_classes, channels * 16)

        if use_sn:
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
