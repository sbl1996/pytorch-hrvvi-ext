from horch.nas.operations import OPS

PRIMITIVES_nas_bench_201 = [
    'none',
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
]
PRIMITIVES_tiny = [
    'none',
    'skip_connect',
    'sep_conv_3x3',
]
PRIMITIVES_small = [
    'none',
    'skip_connect',
    'sep_conv_3x3',
    'max_pool_3x3',
]
PRIMITIVES_medium = [
    'none',
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'conv_3x1_1x3',
]
PRIMITIVES_large = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_3x1_1x3',
]
PRIMITIVES_huge = [
    'skip_connect',
    'nor_conv_1x1',
    'max_pool_3x3',
    'avg_pool_3x3',
    'nor_conv_3x3',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'conv_3x1_1x3',
    'sep_conv_5x5',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
    'att_squeeze',
]
PRIMITIVES_darts = [
    'none',
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
CONFIG = {
    'primitives': PRIMITIVES_darts,
}


def set_primitives(search_space):
    if isinstance(search_space, list):
        for k in search_space:
            if not k in OPS:
                raise ValueError("Not supported operation: %s" % k)
        CONFIG['primitives'] = search_space
    elif search_space == 'tiny':
        CONFIG['primitives'] = PRIMITIVES_tiny
    elif search_space == 'small':
        CONFIG['primitives'] = PRIMITIVES_small
    elif search_space == 'medium':
        CONFIG['primitives'] = PRIMITIVES_medium
    elif search_space == 'large':
        CONFIG['primitives'] = PRIMITIVES_large
    elif search_space == 'huge':
        CONFIG['primitives'] = PRIMITIVES_huge
    elif search_space == 'darts':
        CONFIG['primitives'] = PRIMITIVES_darts
    else:
        raise ValueError("No search space %s" % search_space)


def get_primitives():
    return CONFIG['primitives']