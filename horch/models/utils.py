import torch
import torch.nn as nn
from horch.common import _concat
from toolz import curry
from horch.ext.summary import summary


def get_last_conv(m):
    r"""
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, nn.Conv2d), m.modules())
    return list(convs)[-1]


def get_in_channels(mod):
    r"""
    Get the output channels of the last conv layer of a block.
    """
    for m in list(mod.modules()):
        if isinstance(m, nn.BatchNorm2d):
            return m.num_features
        elif isinstance(m, nn.Conv2d):
            return m.in_channels
        else:
            continue
    raise ValueError("Cannot get output channels.")


def calc_out_channels(mod):
    r"""
    Get the output channels of the last conv layer of a block.
    """
    in_channels = get_in_channels(mod)
    x = torch.randn(1, in_channels, 32, 32)
    with torch.no_grad():
        x = mod(x)
    return x.size(1)


def get_out_channels(mod):
    r"""
    Get the output channels of the last conv layer of a block.
    """
    for m in reversed(list(mod.modules())):
        if isinstance(m, nn.BatchNorm2d):
            return m.num_features
        elif isinstance(m, nn.Conv2d):
            return m.out_channels
        else:
            continue
    raise ValueError("Cannot get output channels.")


@curry
def conv_to_atrous(mod, rate):
    r"""
    Convert a 3x3 Conv2d to Atrous Convolution.
    """
    def f(m):
        if 'Conv2d' in type(m).__name__ and m.kernel_size != (1, 1):
            kh, kw = m.kernel_size
            ph = ((kh - 1) * (rate - 1) + kh - 1) // 2
            pw = ((kw - 1) * (rate - 1) + kw - 1) // 2
            m.padding = (ph, pw)
            m.stride = (1, 1)
            m.dilation = (rate, rate)
    mod.apply(f)
    return mod


def remove_stride_padding(mod):
    r"""
    Convert a 3x3 Conv2d to Atrous Convolution.
    """
    def f(m):
        if 'Conv2d' in type(m).__name__ and m.kernel_size != (1, 1):
            m.padding = (0, 0)
            m.stride = (1, 1)
    mod.apply(f)
    return mod


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model


def clip(model, tol=1e-6):
    for p in model.parameters():
        p = p.data
        p[p.abs() < tol] = 0
    return model


def weight_init_normal(module, mean, std):
    def f(m):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            nn.init.normal_(m.weight, mean, std)
    module.apply(f)


def bias_init_constant(module, val):
    def f(m):
        name = type(m).__name__
        if "Linear" in name or "Conv" in name:
            if m.bias is not None:
                nn.init.constant_(m.bias, val)
    module.apply(f)


def set_bn_momentum(module, val):
    def f(m):
        name = type(m).__name__
        if "BatchNorm" in name:
            m.momentum = val
    module.apply(f)


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def freeze_bn(module):
    def f(m):
        name = type(m).__name__
        if "BatchNorm" in name:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    module.apply(f)