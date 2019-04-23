import torch
import torch.nn as nn
import torch.nn.functional as F


def get_groups(channels, max_groups):
    g = max_groups
    while channels % g != 0:
        g = g // 2
    return g


def get_normalization(name, channels):
    if isinstance(name, nn.Module):
        return name
    if name == 'bn':
        return nn.BatchNorm2d(channels)
    elif name == 'gn':
        num_groups = get_groups(channels, 32)
        return nn.GroupNorm(channels // num_groups, channels)
    else:
        raise NotImplementedError


def get_activation(name):
    if isinstance(name, nn.Module):
        return name
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    else:
        return NotImplementedError


def Conv2d(in_channels, out_channels,
           kernel_size=3, stride=1,
           padding='same', dilation=1, groups=1,
           normalization=None, activation=None):

    if padding == 'same':
        padding = (kernel_size - 1) // 2
    layers = []
    bias = normalization is None
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, dilation, groups, bias))
    if normalization is not None:
        layers.append(get_normalization(normalization, out_channels))
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)
