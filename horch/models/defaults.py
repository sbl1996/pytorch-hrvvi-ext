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


def set_default(keys, value):
    if isinstance(keys, str):
        keys = [keys]
    global DEFAULTS
    d = DEFAULTS
    for k in keys[:-1]:
        assert k in d
        d = d[k]
    k = keys[-1]
    assert k in d
    d[k] = value


def get_default(keys):
    if isinstance(keys, str):
        keys = [keys]
    global DEFAULTS
    d = DEFAULTS
    for k in keys:
        d = d.get(k)
        if d is None:
            return d
    return d


def get_default_activation():
    return get_default('activation')


def set_default_activation(name):
    set_default('activation', name)


def get_default_norm_layer():
    return get_default('norm_layer')


def set_default_norm_layer(name):
    set_default('norm_layer', name)