import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d


class CASENet(nn.Module):
    def __init__(self, backbone, side_in_channels, num_classes, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.side1 = Conv2d(side_in_channels[0], 1, 1,
                            norm_layer='default', activation='default')
        self.side2 = Conv2d(side_in_channels[1], 1, 1,
                            norm_layer='default', activation='default')
        self.side3 = Conv2d(side_in_channels[2], 1, 1,
                            norm_layer='default', activation='default')
        self.side5 = Conv2d(side_in_channels[3], num_classes, 1)

        self.dropout = nn.Dropout2d(dropout)
        self.pred = nn.Conv2d(4 * num_classes, num_classes, 1, groups=num_classes)

    def get_param_groups(self):
        group1 = self.backbone.parameters()
        layers = [
            self.side1, self.side2, self.side3, self.side5, self.pred
        ]
        group2 = [
            p
            for l in layers
            for p in l.parameters()
        ]
        return [group1, group2]

    def forward(self, x):
        size = x.shape[2:4]
        c1, c2, c3, _, c5 = self.backbone(x)

        c1 = self.dropout(c1)
        c2 = self.dropout(c2)
        c3 = self.dropout(c3)
        c5 = self.dropout(c5)

        c1 = self.side1(c1)
        c1 = F.interpolate(c1, size, mode='bilinear', align_corners=False)
        c2 = self.side2(c2)
        c2 = F.interpolate(c2, size, mode='bilinear', align_corners=False)
        c3 = self.side3(c3)
        c3 = F.interpolate(c3, size, mode='bilinear', align_corners=False)
        c5 = self.side5(c5)
        c5 = F.interpolate(c5, size, mode='bilinear', align_corners=False)

        xs = []
        for i in range(c5.size(1)):
            xs.append(c5[:, [i], :, :])
            xs.extend([c1, c2, c3])
        x = torch.cat(xs, dim=1)

        x = self.dropout(x)
        x = self.pred(x)
        return x
