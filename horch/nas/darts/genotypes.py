from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES_small = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
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

PRIMITIVES = [
    'none',
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'max_pool_3x3',
    'avg_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    # 'sep_conv_7x7',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5',
    # 'conv_3x1_1x3',
    # 'conv_7x1_1x7',
    # 'att_squeeze',
]

NASNet = Genotype(
    normal=[
        (('sep_conv_5x5', 1), ('sep_conv_3x3', 0)),
        (('sep_conv_5x5', 0), ('sep_conv_3x3', 0)),
        (('avg_pool_3x3', 1), ('skip_connect', 0)),
        (('avg_pool_3x3', 0), ('avg_pool_3x3', 0)),
        (('sep_conv_3x3', 1), ('skip_connect', 1)),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        (('sep_conv_5x5', 1), ('sep_conv_7x7', 0)),
        (('max_pool_3x3', 1), ('sep_conv_7x7', 0)),
        (('avg_pool_3x3', 1), ('sep_conv_5x5', 0)),
        (('skip_connect', 3), ('avg_pool_3x3', 2)),
        (('sep_conv_3x3', 2), ('max_pool_3x3', 1)),
    ],
    reduce_concat=[4, 5, 6],
)

PNASNet = Genotype(
    normal=[
        (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
        (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
        (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
        (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
        (('sep_conv_3x3', 0), ('skip_connect', 1)),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
        (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
        (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
        (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
        (('sep_conv_3x3', 0), ('skip_connect', 1)),
    ],
    reduce_concat=[2, 3, 4, 5, 6],
)

DARTS_V1 = Genotype(
    normal=[
        (('sep_conv_3x3', 1), ('sep_conv_3x3', 0)),  # step 1
        (('skip_connect', 0), ('sep_conv_3x3', 1)),  # step 2
        (('skip_connect', 0), ('sep_conv_3x3', 1)),  # step 3
        (('sep_conv_3x3', 0), ('skip_connect', 2))  # step 4
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        (('max_pool_3x3', 0), ('max_pool_3x3', 1)),  # step 1
        (('skip_connect', 2), ('max_pool_3x3', 0)),  # step 2
        (('max_pool_3x3', 0), ('skip_connect', 2)),  # step 3
        (('skip_connect', 2), ('avg_pool_3x3', 0))  # step 4
    ],
    reduce_concat=[2, 3, 4, 5],
)

# DARTS: Differentiable Architecture Search, ICLR 2019
DARTS_V2 = Genotype(
    normal=[
        (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)),  # step 1
        (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)),  # step 2
        (('sep_conv_3x3', 1), ('skip_connect', 0)),  # step 3
        (('skip_connect', 0), ('dil_conv_3x3', 2))  # step 4
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        (('max_pool_3x3', 0), ('max_pool_3x3', 1)),  # step 1
        (('skip_connect', 2), ('max_pool_3x3', 1)),  # step 2
        (('max_pool_3x3', 0), ('skip_connect', 2)),  # step 3
        (('skip_connect', 2), ('max_pool_3x3', 1))  # step 4
    ],
    reduce_concat=[2, 3, 4, 5],
)

# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
SETN = Genotype(
    normal=[
        (('skip_connect', 0), ('sep_conv_5x5', 1)),
        (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)),
        (('sep_conv_5x5', 1), ('sep_conv_5x5', 3)),
        (('max_pool_3x3', 1), ('conv_3x1_1x3', 4))],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        (('sep_conv_3x3', 0), ('sep_conv_5x5', 1)),
        (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
        (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
        (('avg_pool_3x3', 0), ('skip_connect', 1))],
    reduce_concat=[2, 3, 4, 5],
)

# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019
GDAS_V1 = Genotype(
    normal=[
        (('skip_connect', 0), ('skip_connect', 1)),
        (('skip_connect', 0), ('sep_conv_5x5', 2)),
        (('sep_conv_3x3', 3), ('skip_connect', 0)),
        (('sep_conv_5x5', 4), ('sep_conv_3x3', 3))],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)),
        (('sep_conv_5x5', 2), ('sep_conv_5x5', 1)),
        (('dil_conv_5x5', 2), ('sep_conv_3x3', 1)),
        (('sep_conv_5x5', 0), ('sep_conv_5x5', 1))],
    reduce_concat=[2, 3, 4, 5],
)

G1 = [
    ('avg_pool_3x3', 1), ('nor_conv_1x1', 0),
    ('sep_conv_5x5', 1), ('skip_connect', 0),
    ('skip_connect', 1), ('sep_conv_5x5', 3),
    ('skip_connect', 0), ('sep_conv_5x5', 4)
]


Networks = {'DARTS_V1': DARTS_V1,
            'DARTS_V2': DARTS_V2,
            'DARTS': DARTS_V2,
            'NASNet': NASNet,
            'GDAS_V1': GDAS_V1,
            'PNASNet': PNASNet,
            'SETN': SETN,
            }


# This function will return a Genotype from a dict.
def build_genotype_from_dict(xdict):
    def remove_value(nodes):
        return [tuple([(x[0], x[1]) for x in node]) for node in nodes]

    genotype = Genotype(
        normal=remove_value(xdict['normal']),
        normal_concat=xdict['normal_concat'],
        reduce=remove_value(xdict['reduce']),
        reduce_concat=xdict['reduce_concat'],
    )
    return genotype
