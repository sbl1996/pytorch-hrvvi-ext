import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d

class CASENet(nn.Module):
    def __init__(self, backbone, side_in_channels, num_classes):
        super().__init__()
        self.backbone = backbone
        self.side1 = Conv2d(side_in_channels[0], 1, 1,
                            norm_layer='default', activation='default')
        self.side2 = Conv2d(side_in_channels[1], 1, 1,
                            norm_layer='default', activation='default')
        self.side3 = Conv2d(side_in_channels[2], 1, 1,
                            norm_layer='default', activation='default')
        self.side5 = Conv2d(side_in_channels[3], num_classes, 1)

        self.conv = nn.Conv2d(4 * num_classes, num_classes, 1, groups=num_classes)

    def forward(self, x):
        size = x.shape[2:4]
        c1, c2, c3, _, c5 = self.backbone(x)
        s1 = self.side1(c1)
        s1 = F.interpolate(s1, size, mode='bilinear', align_corners=False)
        s2 = self.side2(c2)
        s2 = F.interpolate(s2, size, mode='bilinear', align_corners=False)
        s3 = self.side3(c3)
        s3 = F.interpolate(s3, size, mode='bilinear', align_corners=False)
        s5 = self.side5(c5)
        s5 = F.interpolate(s5, size, mode='bilinear', align_corners=False)

        xs = []
        for i in range(s5.size(1)):
            xs.append(s5[:, [i], :, :])
            xs.extend([s1, s2, s3])
        x = torch.cat(xs, dim=1)
        x = self.conv(x)
        return x
