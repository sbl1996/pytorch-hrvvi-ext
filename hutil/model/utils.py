import torch
from toolz import curry


def get_last_conv(m):
    r"""
    Get the last conv layer in an Module.
    """
    convs = filter(lambda kv: 'conv' in kv[0], m.named_modules())
    return max(convs, key=lambda kv: kv[0])[1]


def get_out_channels(m):
    r"""
    Get the output channels of the last conv layer of a block.
    """
    return get_last_conv(m).out_channels


@curry
def conv_to_atrous(m, rate):
    r"""
    Convert a 3x3 Conv2d to Atrous Convolution.
    """
    if 'Conv2d' in type(m).__name__ and m.kernel_size != (1, 1):
        kh, kw = m.kernel_size
        ph = int(((kh - 1) * (rate - 1) + kh - 1) / 2)
        pw = int(((kw - 1) * (rate - 1) + kw - 1) / 2)
        m.padding = (ph, pw)
        m.stride = (1, 1)
        m.dilation = (rate, rate)
    return m