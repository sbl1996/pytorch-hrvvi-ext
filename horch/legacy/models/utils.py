import torch
import torch.nn as nn
from toolz import curry

from thop import profile as thop_profile, clever_format
from thop.vision.basic_hooks import count_relu

from horch.ext.summary import summary
from horch.models.modules import Swish, HardSwish, HardSigmoid


def count_sigmoid(m, x, y):
    """
    Using this approximation for exponetial operation:  exp(x) = 1 + x + x^2/2! + .. + x^9/9!
    For sigmoid f(x) = 1/(1+exp(x)): there are totally 10 add ops, 9 division ops(2! are considered as constant).
    Since it is element-wise operation. The final ops is about(10+9)*num_elements.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 19 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_swish(m, x, y):
    """
    swish = x*sigmoid(x). So the total ops is 20*num_elements. See definition of count_sigmoid.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 20 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_hsigmoid(m, x, y):
    """
    hsigmoid = relu6(x + 3) / 6. So the total ops is 20*num_elements. See definition of count_sigmoid.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 3 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_hswish(m, x, y):
    """
    hswish = x*hsigmoid(x). So the total ops is 20*num_elements. See definition of count_sigmoid.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 4 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])


THOP_CUSTOM_OPS = {
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    HardSwish: count_hswish,
    HardSigmoid: count_hsigmoid,
    nn.ReLU: count_relu,
    nn.PReLU: count_relu,
    nn.ELU: count_relu,
    nn.LeakyReLU: count_relu,
    nn.ReLU6: count_relu,
}


def profile(model: nn.Module, inputs, verbose=True):
    return thop_profile(model, inputs, custom_ops=THOP_CUSTOM_OPS, verbose=verbose)


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
    mod.eval()
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


def freeze_bn(module, eval=True, requires_grad=True):
    def f(m):
        name = type(m).__name__
        if "BatchNorm" in name:
            if eval:
                m.frozen = True
                m.eval()
            m.weight.requires_grad = requires_grad
            m.bias.requires_grad = requires_grad

    module.apply(f)
