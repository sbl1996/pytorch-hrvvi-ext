from difflib import get_close_matches
from typing import Mapping, Sequence, Union

from cerberus import Validator

DEFAULTS = {
    'bn': {
        'momentum': 0.1,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
    },
    'norm': 'bn',
    'act': 'relu',
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
    },
    'swish': {
        'inplace': True,
    },
    'seed': 42,
}

_defaults_schema = {
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
    },
    'relu': {
        'inplace': {'type': 'boolean'},
    },
    'act': {'type': 'string', 'allowed': ['relu', 'swish', 'mish', 'leaky_relu', 'sigmoid']},
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn']},
    'seed': {'type': 'integer'},
    'no_bias_decay': {'type': 'boolean'},
}


def set_defaults(kvs: Mapping):
    def _set_defaults(kvs, prefix):
        for k, v in kvs.items():
            if isinstance(v, dict):
                _set_defaults(v, prefix + (k,))
            else:
                set_default(prefix + (k,), v)
    return _set_defaults(kvs, ())


def set_default(keys: Union[str, Sequence[str]], value):
    def loop(d, keys, schema):
        k = keys[0]
        if k not in d:
            match = get_close_matches(k, d.keys())
            if match:
                raise KeyError("No such key `%s`, maybe you mean `%s`" % (k, match[0]))
            else:
                raise KeyError("No key `%s` in %s" % (k, d))
        if len(keys) == 1:
            v = Validator({k: schema[k]})
            if not v.validate({k: value}):
                raise ValueError(v.errors)
            d[k] = value
        else:
            loop(d[k], keys[1:], schema[k])

    if isinstance(keys, str):
        keys = [keys]
    loop(DEFAULTS, keys, _defaults_schema)


def update_defaults(new_d, d=DEFAULTS):
    for k, v in new_d.items():
        if k not in d:
            raise KeyError(str(k))
        if isinstance(v, dict):
            update_defaults(v, d[k])
        else:
            d[k] = new_d[k]