from typing import Dict

from yacs.config import CfgNode as CN

DEFAULTS = {
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
    },
    'norm_layer': 'bn',
    'activation': 'relu',
    'relu': {
        'inplace': True,
    },
    'relu6': {
        'inplace': True,
    },
    'leaky_relu': {
        'negative_slope': 0.1,
        'inplace': True,
    },
    'hswish': {
        'inplace': True,
    }
}


def load_from_dict(d: Dict):
    cfg = CN()
    for k, v in d.items():
        if isinstance(v, dict):
            v = load_from_dict(v)
        setattr(cfg, k, v)
    return cfg


cfg = load_from_dict(DEFAULTS)


def get_cfg_defaults():
    return cfg.clone()
