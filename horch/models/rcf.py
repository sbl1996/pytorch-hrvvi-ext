import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

from horch.models.modules import Conv2d
from horch.nn.loss import f1_loss
import random


def conv(in_channels, out_channels):
    side = Conv2d(in_channels, out_channels, 1, norm_layer='default')
    nn.init.normal_(side[0].weight, 0, 0.01)
    # nn.init.constant_(side[0].bias, 0)
    return side

class RCF(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vgg16_bn(pretrained=True)
        del backbone.avgpool
        del backbone.classifier
        f = backbone.features

        self.conv11 = f[:3]
        self.conv12 = f[3:6]
        self.pool1 = f[6]
        self.conv21 = f[7:10]
        self.conv22 = f[10:13]
        self.pool2 = f[13]
        self.conv31 = f[14:17]
        self.conv32 = f[17:20]
        self.conv33 = f[20:23]
        self.pool3 = f[23]
        self.conv41 = f[24:27]
        self.conv42 = f[27:30]
        self.conv43 = f[30:33]
        self.pool4 = f[33]
        self.conv51 = f[34:37]
        self.conv52 = f[37:40]
        self.conv53 = f[40:43]

        self.side11 = conv(64, 21)
        self.side12 = conv(64, 21)
        self.side21 = conv(128, 21)
        self.side22 = conv(128, 21)
        self.side31 = conv(256, 21)
        self.side32 = conv(256, 21)
        self.side33 = conv(256, 21)
        self.side41 = conv(512, 21)
        self.side42 = conv(512, 21)
        self.side43 = conv(512, 21)
        self.side51 = conv(512, 21)
        self.side52 = conv(512, 21)
        self.side53 = conv(512, 21)

        self.fuse1 = conv(21, 1)
        self.fuse2 = conv(21, 1)
        self.fuse3 = conv(21, 1)
        self.fuse4 = conv(21, 1)
        self.fuse5 = conv(21, 1)

        self.fuse = Conv2d(5, 1, 1)
        nn.init.constant_(self.fuse.weight, 1/5)
        nn.init.constant_(self.fuse.bias, 0)

    def forward(self, x):
        size = x.shape[2:4]
        ps = []

        x = self.conv11(x)
        c = self.side11(x)
        x = self.conv12(x)
        c += self.side12(x)
        ps.append(self.fuse1(c))

        x = self.pool1(x)

        x = self.conv21(x)
        c = self.side21(x)
        x = self.conv22(x)
        c += self.side22(x)
        p = self.fuse2(c)
        p = F.interpolate(p, size, mode='bilinear', align_corners=False)
        ps.append(p)

        x = self.pool2(x)

        x = self.conv31(x)
        c = self.side31(x)
        x = self.conv32(x)
        c += self.side32(x)
        x = self.conv33(x)
        c += self.side33(x)
        p = self.fuse3(c)
        p = F.interpolate(p, size, mode='bilinear', align_corners=False)
        ps.append(p)

        x = self.pool3(x)

        x = self.conv41(x)
        c = self.side41(x)
        x = self.conv42(x)
        c += self.side42(x)
        x = self.conv43(x)
        c += self.side43(x)
        p = self.fuse4(c)
        p = F.interpolate(p, size, mode='bilinear', align_corners=False)
        ps.append(p)

        x = self.pool4(x)

        x = self.conv51(x)
        c = self.side51(x)
        x = self.conv52(x)
        c += self.side52(x)
        x = self.conv53(x)
        c += self.side53(x)
        p = self.fuse5(c)
        p = F.interpolate(p, size, mode='bilinear', align_corners=False)
        ps.append(p)

        p = torch.cat(ps, dim=1)
        p = self.fuse(p)

        return ps, p


class RCFLoss(nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, ps, p, target):
        target = target.type_as(p)
        loss = f1_loss(torch.sigmoid(torch.squeeze(p, 1)), target)
        for p in ps:
            loss += f1_loss(torch.sigmoid(torch.squeeze(p, 1)), target)
        loss /= len(ps) + 1
        if random.random() < self.p:
            print("loss: %.4f" % loss.item())
        return loss
