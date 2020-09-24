import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.layers import Conv2d, Pool2d


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_theta = Conv2d(in_channels, in_channels // 8, kernel_size=1)

        self.conv_phi = Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.pool_phi = nn.MaxPool2d(kernel_size=2, stride=(2, 2))

        self.conv_g = Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.pool_g = Pool2d(kernel_size=2, stride=2, type='max')

        self.conv_attn = Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.size()
        theta = self.conv_theta(x)
        theta = theta.view(b, -1, h * w)

        phi = self.conv_phi(x)
        phi = self.pool_phi(phi)
        phi = phi.view(b, -1, h * w // 4)

        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = F.softmax(attn, dim=-1)

        g = self.conv_g(x)
        g = self.pool_g(g)
        g = g.view(b, -1, h * w // 4)

        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(b, -1, h, w)
        attn_g = self.conv_attn(attn_g)

        x = x + self.sigma * attn_g
        return x


class SelfAttention2(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        channels = in_channels // reduction
        self.conv_theta = Conv2d(in_channels, channels, kernel_size=1)
        self.conv_phi = Conv2d(in_channels, channels, kernel_size=1)
        self.conv_g = Conv2d(in_channels, channels, kernel_size=1)
        self.conv_attn = Conv2d(channels, in_channels, kernel_size=1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.size()
        theta = self.conv_theta(x)
        theta = theta.view(b, -1, h * w)

        phi = self.conv_phi(x)
        phi = phi.view(b, -1, h * w)

        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = F.softmax(attn, dim=-1)

        g = self.conv_g(x)
        g = g.view(b, -1, h * w)

        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(b, -1, h, w)
        attn_g = self.conv_attn(attn_g)

        x = x + self.sigma * attn_g
        return x


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, momentum=0.001):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class SharedConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, embedding, momentum=0.001):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.embedding = embedding

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
