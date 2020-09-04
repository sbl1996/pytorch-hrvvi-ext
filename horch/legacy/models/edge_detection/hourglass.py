import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Norm, Act, Conv2d


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = Norm('default', in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)
        self.bn2 = Norm('default', channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = Norm('default', channels)
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.relu = Act('default')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, channels, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, channels, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    def __init__(self, num_stacks=2, num_blocks=1, num_classes=1, depth=3, block=Bottleneck):
        super(HourglassNet, self).__init__()

        self.in_channels = 64
        self.channels = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.in_channels, 1)
        self.layer2 = self._make_residual(block, self.in_channels, 1)
        self.layer3 = self._make_residual(block, self.channels, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.channels * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.channels, depth))
            res.append(self._make_residual(block, self.channels, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        self.side1 = Conv2d(256, 1, 1)
        self.fuse = Conv2d(1 + num_stacks, 1, 1)
        nn.init.constant_(self.fuse.weight, 1 / (1 + num_stacks))
        nn.init.constant_(self.fuse.bias, 0)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, in_channels, out_channels):
        return Conv2d(in_channels, out_channels, kernel_size=1,
                      norm='default', act='default')

    def forward(self, x):
        size = x.size()[2:4]
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out.append(self.side1(x))
        x = self.maxpool(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        ps = []
        for x in out:
            if x.size()[2:4] != size:
                x = F.interpolate(x, size, mode='bilinear', align_corners=False)
            ps.append(x)
        p = torch.cat(ps, dim=1)
        p = self.fuse(p)
        if self.training:
            return ps, p
        else:
            return p


# class DeepSupervisionLoss(nn.Module):
#
#     def __init__(self, p=0):
#         self.p = p
#
#     def __call__(self, preds, fuse, target):
#         target = target.type_as(target)
#         loss = f1_loss(torch.sigmoid(fuse.squeeze(1)), target)
#         for p in preds:
#             loss += f1_loss(torch.sigmoid(p.squeeze(1)), target)
#         loss /= len(preds) + 1
#         return loss