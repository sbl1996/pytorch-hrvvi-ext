from pathlib import Path

import torch
import torch.nn as nn
from horch.common import _concat
from horch.ext.summary import summary
from toolz import curry
from horch.datasets.utils import download_google_drive


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
        p[p.abs() < tol] = 0
    return model


def get_loc_cls_preds(ps, num_classes, concat=True):
    loc_preds = []
    cls_preds = []
    b = ps[0].size(0)
    for p in ps:
        p = p.permute(0, 3, 2, 1).contiguous().view(
            b, -1, 4 + num_classes)
        loc_p = p[..., :4]
        loc_preds.append(loc_p)
        cls_p = p[..., 4:]
        if cls_p.size(-1) == 1:
            cls_p = cls_p[..., 0]
        cls_preds.append(cls_p)
    if concat:
        loc_p = _concat(loc_preds, dim=1)
        cls_p = _concat(cls_preds, dim=1)
        return loc_p, cls_p
    return loc_preds, cls_preds


def load_state_dict_from_google_drive(file_id, filename, md5, model_dir=None, map_location=None):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    """

    if model_dir is None:
        from torch.hub import _get_torch_home
        torch_home = _get_torch_home()
        torch_home = Path(torch_home)
        model_dir = torch_home / 'checkpoints'
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    download_google_drive(file_id, model_dir, filename, md5)
    cached_file = model_dir / filename
    return torch.load(cached_file, map_location=map_location)


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
