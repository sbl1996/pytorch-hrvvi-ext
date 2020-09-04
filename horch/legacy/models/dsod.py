import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d, seq, Norm, Act


class StemBlock(nn.Sequential):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(3, channels, kernel_size=3, stride=2,
                   norm='default', act='default'),
            Conv2d(channels, channels, kernel_size=3, stride=1,
                   norm='default', act='default'),
            Conv2d(channels, channels * 2, kernel_size=3, stride=1,
                   norm='default', act='default'),
        )


class DenseUnit(nn.Module):

    def __init__(self, in_channels, mid_channels=192, growth_rate=48):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=1,
                   norm='default', act='default'),
            Conv2d(mid_channels, growth_rate, kernel_size=3,
                   norm='default', act='default'),
        )

    def forward(self, x):
        x = torch.cat([x, self.conv(x)], dim=1)
        return x


class Transition(nn.Sequential):

    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1,
                           norm='default', act='default')
        if pool:
            self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)


class DenseSupervision1(nn.Module):

    def __init__(self, inC, outC=256):
        super(DenseSupervision1, self).__init__()
        self.model_name = 'DenseSupervision'

        self.right = nn.Sequential(
            # nn.BatchNorm2d(inC),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(inC,outC,1),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False)
        )

    def forward(self, x1, x2):
        # x1 should be f1
        right = self.right(x1)
        return torch.cat([x2, right], 1)


class DenseSupervision(nn.Module):

    def __init__(self, inC, outC=128):
        super(DenseSupervision, self).__init__()
        self.model_name = 'DenseSupervision'
        self.left = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False)
        )
        self.right = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3, 2, 1, bias=False)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], 1)


class BasicBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels // 2, kernel_size=1,
                            norm='default', act='default')
        self.conv2 = Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2,
                            norm='default', act='default')


class DenseNet(nn.Module):

    def __init__(self,
                 stem_channels=64,
                 mid_channels=192,
                 growth_rate=48,
                 num_units=(6, 8, 8, 8)):
        super().__init__()

        self.init_block = StemBlock(stem_channels)
        in_channels = stem_channels * 2
        out_channels = [in_channels]
        for i, n in enumerate(num_units):
            stage = nn.Sequential()
            stage.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            for j in range(n):
                stage.add_module("unit%d" % (j + 1),
                                 DenseUnit(in_channels, mid_channels, growth_rate))
                in_channels += growth_rate
            if i != len(num_units) - 1:
                stage.add_module(
                    "trans", Transition(in_channels, in_channels))
            out_channels.append(in_channels)
            self.add_module("stage%d" % (i + 1), stage)
        self.post_activ = seq(
            ("bn", Norm("default", in_channels)),
            ("relu", Act("default")),
        )

        del self.stage4.pool
        print(out_channels)
        self.trans = Transition(out_channels[-1], 256)

        self.ds1 = DenseSupervision1(out_channels[-3], 256)
        self.ds2 = DenseSupervision(512, 256)
        self.ds3 = DenseSupervision(512, 128)
        self.ds4 = DenseSupervision(256, 128)
        self.ds5 = DenseSupervision(256, 128)
        self.out_channels = [out_channels[-3], 512, 512, 256, 256, 256]

    def forward(self, x):
        x = self.init_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        f1 = x
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.trans(x)

        f2 = self.ds1(f1, x)
        f3 = self.ds2(f2)
        f4 = self.ds3(f3)
        f5 = self.ds4(f4)
        f6 = self.ds5(f5)

        return f1, f2, f3, f4, f5, f6


class DenseNet2(nn.Module):

    def __init__(self,
                 stem_channels=64,
                 mid_channels=192,
                 growth_rate=48,
                 num_units=(6, 8, 8, 8)):
        super().__init__()

        self.init_block = StemBlock(stem_channels)
        in_channels = stem_channels * 2
        out_channels = [in_channels]
        for i, n in enumerate(num_units):
            stage = nn.Sequential()
            stage.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            for j in range(n):
                stage.add_module("unit%d" % (j + 1),
                                 DenseUnit(in_channels, mid_channels, growth_rate))
                in_channels += growth_rate
            if i != len(num_units) - 1:
                stage.add_module(
                    "trans", Transition(in_channels, in_channels))
            out_channels.append(in_channels)
            self.add_module("stage%d" % (i + 1), stage)
        self.post_activ = seq(
            ("bn", Norm("default", in_channels)),
            ("relu", Act("default")),
        )

        del self.stage4.pool

        self.trans = Transition(out_channels[-1], out_channels[-1])
        self.proj = Transition(out_channels[-1], 512)

        self.extra1 = BasicBlock(512, 512)
        self.extra2 = BasicBlock(512, 256)
        self.extra3 = BasicBlock(256, 256)
        self.extra4 = BasicBlock(256, 256)
        self.out_channels = [out_channels[-3], 512, 512, 256, 256, 256]

    def forward(self, x):
        xs = []
        x = self.init_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        f1 = x
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.trans(x)
        x = self.proj(x)
        f2 = x
        f3 = self.extra1(f2)
        f4 = self.extra2(f3)
        f5 = self.extra3(f4)
        f6 = self.extra4(f5)

        return f1, f2, f3, f4, f5, f6


class DenseNet3(nn.Module):

    def __init__(self,
                 stem_channels=64,
                 mid_channels=192,
                 growth_rate=48,
                 num_units=(6, 8, 8, 8)):
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", StemBlock(stem_channels))
        in_channels = stem_channels * 2
        for i, n in enumerate(num_units):
            stage = nn.Sequential()
            if i != len(num_units) - 1:
                stage.add_module("trans", Transition(in_channels, in_channels))
            for j in range(n):
                stage.add_module("unit%d" % (j + 1),
                                 DenseUnit(in_channels, mid_channels, growth_rate))
                in_channels += growth_rate
            self.features.add_module("stage%d" % (i + 1), stage)
        self.features.add_module("post_activ", seq(
            ("bn", Norm("default", in_channels)),
            ("relu", Act("default")),
        ))

        # self.extra1 = BasicBlock(512, 512)
        # self.extra2 = BasicBlock(512, 256)
        # self.extra3 = BasicBlock(256, 256)
        # self.extra4 = BasicBlock(256, 256)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':
    m = DenseNet()
    input = torch.randn(1, 3, 300, 300)
    o = m(input)
    for ii in o:
        print(ii.shape)
