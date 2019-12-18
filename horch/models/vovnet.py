import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import get_norm_layer, Conv2d, get_activation


class OSA(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, n=5):
        super().__init__()
        self.convs = nn.ModuleList([])
        channels = in_channels
        for i in range(n):
            self.convs.append(Conv2d(channels, mid_channels, kernel_size=3,
                                     norm_layer='default', activation='default'))
            channels = mid_channels
        self.project = Conv2d(in_channels + mid_channels * n, out_channels, kernel_size=1,
                              norm_layer='default', activation='default')

    def forward(self, x):
        xs = [x]
        for conv in self.convs:
            x = conv(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.project(x)
        return x


class VoVNet(nn.Module):
    def __init__(
            self,
            stem_channels=64,
            mid_channels=(64, 80, 96, 112),
            out_channels=(128, 256, 384, 512),
            num_modules=(1, 1, 1, 1),
            num_classes=1000):
        super().__init__()
        num_stages = 5
        assert len(mid_channels) == len(out_channels) == len(num_modules) == num_stages - 1

        self.features = nn.Sequential()
        self.features.add_module("init_block", nn.Sequential(
            Conv2d(3, stem_channels, kernel_size=3, stride=2,
                   norm_layer='default', activation='default'),
            Conv2d(stem_channels, stem_channels, kernel_size=3,
                   norm_layer='default', activation='default'),
            Conv2d(stem_channels, stem_channels * 2, kernel_size=3,
                   norm_layer='default', activation='default'),
        ))
        in_channels = stem_channels * 2
        for i, m, o, n in zip(range(num_stages - 1), mid_channels, out_channels, num_modules):
            stage = nn.Sequential()
            stage.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            for j in range(n):
                stage.add_module("unit%d" % (j + 1), OSA(in_channels, m, o))
                in_channels = o
            self.features.add_module("stage%d" % (i + 1), stage)

        self.features.add_module("post_activ", nn.Sequential(
            get_norm_layer("default", in_channels),
            get_activation("default"),
        ))
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))

        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)


def get_vovnet(convs,
               model_name=None,
               **kwargs):
    stem_channels = 64
    if convs == 27:
        mid_channels = (64, 80, 96, 112)
        out_channels = (128, 256, 384, 512)
        num_modules = (1, 1, 1, 1)
    elif convs == 39:
        mid_channels = (128, 160, 192, 224)
        out_channels = (256, 512, 768, 1024)
        num_modules = (1, 1, 2, 2)
    elif convs == 57:
        mid_channels = (128, 160, 192, 224)
        out_channels = (256, 512, 768, 1024)
        num_modules = (1, 1, 4, 3)
    else:
        raise ValueError("Unsupported VoVNet version with number of layers {}".format(convs))

    return VoVNet(stem_channels, mid_channels, out_channels, num_modules, **kwargs)


def vovnet27_slim(**kwargs):
    return get_vovnet(blocks=27, model_name="vovnet27_slim", **kwargs)


def vovnet39(**kwargs):
    return get_vovnet(blocks=39, model_name="vovnet39", **kwargs)


def vovnet57(**kwargs):
    return get_vovnet(blocks=57, model_name="vovnet57", **kwargs)