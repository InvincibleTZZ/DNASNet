from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'normal normal_concat')

"""
Operation sets
"""

FORWARD_PRIMITIVES = [
    'avg_pool_3x3_p',
    'avg_pool_3x3_n',
    #'max_pool_3x3_p',
    #'max_pool_3x3_n',
    #'sep_conv_3x3_p',
    #'sep_conv_3x3_n',
    #'sep_conv_5x5_p',
    #'sep_conv_5x5_n',
    #'dil_conv_3x3_p',
    #'dil_conv_3x3_n',
    #'dil_conv_5x5_p',
    #'dil_conv_5x5_n',
    'conv_3x3_p',
    'conv_3x3_n',
    'conv_5x5_p',
    'conv_5x5_n',
]

PRIMITIVES = list(FORWARD_PRIMITIVES)
BACK_PRIMITIVES = [f'{primitive}_back' for primitive in FORWARD_PRIMITIVES]
ALL_PRIMITIVES = PRIMITIVES + BACK_PRIMITIVES

SHD_PRIMITIVES = [
    'identity',
    'linear',
]
SHD_BACK_PRIMITIVES = [f'{primitive}_back' for primitive in SHD_PRIMITIVES]

SHD_TSKIP_DELAYS = [2, 4, 8, 12]
SHD_TSKIP_PRIMITIVES = (
    ['zero'] +
    [f'identity_d{delay}' for delay in SHD_TSKIP_DELAYS] +
    [f'linear_d{delay}' for delay in SHD_TSKIP_DELAYS]
)

SHD_TSKIP2_DELAYS = [1, 2, 4, 8, 16, 32]
SHD_TSKIP2_PRIMITIVES = (
    ['zero'] +
    [f'identity_d{delay}' for delay in SHD_TSKIP2_DELAYS] +
    [f'linear_d{delay}' for delay in SHD_TSKIP2_DELAYS]
)

SHD_TSKIP2_STABLE_DELAYS = [1, 2, 4, 8, 16, 32]
SHD_TSKIP2_STABLE_PRIMITIVES_NOZERO = (
    [f'identity_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS] +
    [f'linear_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS]
)
SHD_TSKIP2_STABLE_PRIMITIVES = (
    ['zero'] + list(SHD_TSKIP2_STABLE_PRIMITIVES_NOZERO)
)

SHD_TSKIP2_STABLE_EI_PRIMITIVES_NOZERO = (
    [f'identity_p_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS] +
    [f'identity_n_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS] +
    [f'linear_p_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS] +
    [f'linear_n_d{delay}' for delay in SHD_TSKIP2_STABLE_DELAYS]
)
SHD_TSKIP2_STABLE_EI_PRIMITIVES = (
    ['zero'] + list(SHD_TSKIP2_STABLE_EI_PRIMITIVES_NOZERO)
)


"""====== SNN Archirtecture By Other Methods"""


dvsc10_skip2 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_skip1 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_n', 2), ('conv_3x3_p', 1),
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_base0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 3),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_base1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_base2 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 3), ('conv_5x5_n', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_EE_base2 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 3), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p_back', 2),
        ('avg_pool_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_base3 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_n', 0),
        ('conv_3x3_p_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

 
dvsc10_stdp = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1), 
        ('conv_5x5_p', 0), ('avg_pool_3x3_p', 2), 
        ('conv_5x5_p', 1), ('avg_pool_3x3_n', 2)], 
    normal_concat=range(2, 5))

dvsc10_2 = Genotype(normal=[
    ('conv_3x3_p', 0), ('conv_3x3_n', 1),
    ('conv_3x3_n', 1), ('avg_pool_3x3_p', 0),
    ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
    ('avg_pool_3x3_n_back', 2),
    ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

dvsc10_1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        #('conv_3x3_p_back', 2),
        #('conv_3x3_n_back', 2)
        ],
    normal_concat=range(2, 5)
) 

dvsc10_0 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_n_back', 3)
        ],
    normal_concat=range(2, 5)
)


cifar_stdp = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_n', 0), 
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('conv_3x3_p', 2), ('conv_3x3_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)], 
    normal_concat=range(2, 5),
)
cifar_100 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0), 
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('conv_3x3_p', 2), ('conv_3x3_p', 1),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)
        ], 
    normal_concat=range(2, 5),
)

cifar_final = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

cifar_final_new = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_p', 0),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

cifar_final_new_noback = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        #('conv_3x3_n_back', 2),
        #('conv_3x3_p_back', 2)
        ],
    normal_concat=range(2, 5),
)


cifar_hebb_seed_42 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0), 
        ('avg_pool_3x3_p', 1), ('conv_3x3_p', 0), 
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 2), 
        ('avg_pool_3x3_p_back', 2), 
        ('avg_pool_3x3_p_back', 2)], 
        normal_concat=range(2, 5))

cifar_hebb_seed_41 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('avg_pool_3x3_p', 0), 
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 2), 
        ('conv_5x5_p', 1), ('avg_pool_3x3_n', 0), 
        ('avg_pool_3x3_n_back', 1), 
        ('avg_pool_3x3_n_back', 0)], 
        normal_concat=range(2, 5))

cifar_hebb_seed_0 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), 
        ('conv_3x3_p', 2), ('conv_5x5_p', 0), 
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 3), 
        ('avg_pool_3x3_p_back', 2), 
        ('conv_5x5_p_back', 2)], 
        normal_concat=range(2, 5))

cifar_rate_only_seed_0 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_p', 0), 
        ('avg_pool_3x3_n', 2), ('conv_5x5_p', 1), 
        ('avg_pool_3x3_n', 0), ('avg_pool_3x3_n', 1), 
        ('avg_pool_3x3_n_back', 1), 
        ('avg_pool_3x3_n_back', 0)], 
        normal_concat=range(2, 5))

cifar_rate_only_seed_41 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_p', 0), 
        ('avg_pool_3x3_n', 2), ('conv_3x3_p', 0), 
        ('avg_pool_3x3_n', 0), ('avg_pool_3x3_n', 1), 
        ('avg_pool_3x3_n_back', 1), 
        ('avg_pool_3x3_n_back', 0)], 
        normal_concat=range(2, 5))

cifar_rate_only_seed_42 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_p', 0), 
        ('avg_pool_3x3_n', 2), ('conv_3x3_p', 1), 
        ('avg_pool_3x3_n', 0), ('avg_pool_3x3_n', 1), 
        ('avg_pool_3x3_n_back', 0), 
        ('avg_pool_3x3_n_back', 1)], 
        normal_concat=range(2, 5))

cifar_stdp_seed_0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0), 
        ('conv_3x3_p', 2), ('conv_3x3_p', 1), 
        ('avg_pool_3x3_n', 3), ('conv_3x3_p', 1), 
        ('conv_5x5_p_back', 2), ('conv_3x3_p_back', 0)], 
    normal_concat=range(2, 5))

cifar_stdp_seed_41 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_3x3_p', 0), 
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 2), 
        ('conv_5x5_p', 2), ('conv_3x3_p', 0), 
        ('conv_5x5_p_back', 1), ('conv_3x3_p_back', 3)], 
    normal_concat=range(2, 5))    

cifar_stdp_seed_42 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0), 
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('avg_pool_3x3_p', 2), ('avg_pool_3x3_n', 0), 
        ('avg_pool_3x3_p_back', 1), 
        ('avg_pool_3x3_n_back', 1)], 
        normal_concat=range(2, 5))    


shd_stdp = Genotype(
    normal=[
        ('linear', 0), ('identity', 1),
        ('linear', 0), ('identity', 2),
        ('linear', 2), ('identity', 3),
    ],
    normal_concat=range(2, 5),
)

shd_tskip = Genotype(
    normal=[
        ('linear_d2', 0),
        ('linear_d4', 1),
        ('linear_d8', 2),
    ],
    normal_concat=range(3),
)

shd_tskip2 = Genotype(
    normal=[
        ('linear_d2', 'layer0_forward0'),
        ('linear_d2', 'layer0_forward1'),
        ('linear_d2', 'layer1_forward0'),
        ('linear_d2', 'layer1_forward1'),
        ('linear_d2', 'layer2_forward0'),
        ('linear_d2', 'layer2_forward1'),
        ('linear_d2', 'layer3_forward0'),
        ('linear_d2', 'layer3_forward1'),
        ('linear_d4', 'layer0_back_from1'),
        ('linear_d4', 'layer1_back_from2'),
        ('linear_d4', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_stable = Genotype(
    normal=[
        ('linear_d8', 'layer0_forward0'),
        ('linear_d8', 'layer0_forward1'),
        ('linear_d2', 'layer1_forward0'),
        ('linear_d8', 'layer1_forward1'),
        ('linear_d8', 'layer2_forward0'),
        ('linear_d1', 'layer2_forward1'),
        ('linear_d4', 'layer3_forward0'),
        ('linear_d2', 'layer3_forward1'),
        ('linear_d1', 'layer0_back_from1'),
        ('linear_d8', 'layer1_back_from2'),
        ('linear_d2', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_stable_seed42 = shd_tskip2_stable

shd_tskip2_random_seed20260507 = Genotype(
    normal=[
        ('identity_d1', 'layer0_forward0'),
        ('linear_d16', 'layer0_forward1'),
        ('linear_d4', 'layer1_forward0'),
        ('identity_d32', 'layer1_forward1'),
        ('linear_d2', 'layer2_forward0'),
        ('identity_d1', 'layer2_forward1'),
        ('identity_d2', 'layer3_forward0'),
        ('identity_d2', 'layer3_forward1'),
        ('identity_d1', 'layer0_back_from1'),
        ('linear_d16', 'layer1_back_from2'),
        ('identity_d32', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_random_seed20260508 = Genotype(
    normal=[
        ('identity_d1', 'layer0_forward0'),
        ('identity_d4', 'layer0_forward1'),
        ('identity_d1', 'layer1_forward0'),
        ('identity_d4', 'layer1_forward1'),
        ('identity_d16', 'layer2_forward0'),
        ('linear_d8', 'layer2_forward1'),
        ('identity_d2', 'layer3_forward0'),
        ('identity_d8', 'layer3_forward1'),
        ('linear_d4', 'layer0_back_from1'),
        ('linear_d2', 'layer1_back_from2'),
        ('linear_d8', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_random_seed20260509 = Genotype(
    normal=[
        ('linear_d32', 'layer0_forward0'),
        ('identity_d1', 'layer0_forward1'),
        ('linear_d4', 'layer1_forward0'),
        ('linear_d2', 'layer1_forward1'),
        ('identity_d8', 'layer2_forward0'),
        ('linear_d1', 'layer2_forward1'),
        ('linear_d16', 'layer3_forward0'),
        ('linear_d2', 'layer3_forward1'),
        ('identity_d1', 'layer0_back_from1'),
        ('linear_d32', 'layer1_back_from2'),
        ('linear_d2', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_random_seed20260510 = Genotype(
    normal=[
        ('identity_d16', 'layer0_forward0'),
        ('linear_d8', 'layer0_forward1'),
        ('linear_d1', 'layer1_forward0'),
        ('linear_d8', 'layer1_forward1'),
        ('identity_d8', 'layer2_forward0'),
        ('identity_d32', 'layer2_forward1'),
        ('identity_d1', 'layer3_forward0'),
        ('identity_d4', 'layer3_forward1'),
        ('identity_d1', 'layer0_back_from1'),
        ('identity_d1', 'layer1_back_from2'),
        ('identity_d2', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_random_linear_seed20260511 = Genotype(
    normal=[
        ('linear_d16', 'layer0_forward0'),
        ('linear_d4', 'layer0_forward1'),
        ('linear_d2', 'layer1_forward0'),
        ('linear_d8', 'layer1_forward1'),
        ('linear_d16', 'layer2_forward0'),
        ('linear_d32', 'layer2_forward1'),
        ('linear_d2', 'layer3_forward0'),
        ('linear_d8', 'layer3_forward1'),
        ('linear_d4', 'layer0_back_from1'),
        ('linear_d16', 'layer1_back_from2'),
        ('linear_d4', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_ei_balanced_selected = Genotype(
    normal=[
        ('linear_n_d8', 'layer0_forward0'),
        ('linear_n_d8', 'layer0_forward1'),
        ('linear_n_d1', 'layer1_forward0'),
        ('linear_p_d4', 'layer1_forward1'),
        ('linear_p_d2', 'layer2_forward0'),
        ('linear_p_d2', 'layer2_forward1'),
        ('linear_p_d8', 'layer3_forward0'),
        ('linear_n_d4', 'layer3_forward1'),
        ('linear_p_d4', 'layer0_back_from1'),
        ('linear_p_d8', 'layer1_back_from2'),
        ('linear_p_d1', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)

shd_tskip2_random_ei_linear_seed20260512 = Genotype(
    normal=[
        ('linear_n_d2', 'layer0_forward0'),
        ('linear_n_d4', 'layer0_forward1'),
        ('linear_n_d1', 'layer1_forward0'),
        ('linear_p_d16', 'layer1_forward1'),
        ('linear_p_d16', 'layer2_forward0'),
        ('linear_n_d2', 'layer2_forward1'),
        ('linear_p_d4', 'layer3_forward0'),
        ('linear_n_d2', 'layer3_forward1'),
        ('linear_p_d16', 'layer0_back_from1'),
        ('linear_p_d2', 'layer1_back_from2'),
        ('linear_p_d2', 'layer2_back_from3'),
    ],
    normal_concat=range(4),
)


dvsg_seed_0 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_3x3_p', 0), 
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), 
        ('conv_3x3_p', 2), ('avg_pool_3x3_p', 0), 
        ('avg_pool_3x3_p_back', 1), 
        ('conv_3x3_p_back', 3)], 
        normal_concat=range(2, 5))

dvsg_seed_42 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('avg_pool_3x3_p', 0), 
        ('avg_pool_3x3_p', 2), ('conv_5x5_p', 0), 
        ('avg_pool_3x3_p', 2), ('conv_5x5_p', 1), 
        ('avg_pool_3x3_n_back', 1), 
        ('conv_3x3_p_back', 2)], 
        normal_concat=range(2, 5))

dvsg_seed_41 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0), 
        ('conv_5x5_p', 2), ('conv_3x3_p', 0), 
        ('conv_3x3_p', 3), ('conv_5x5_p', 1), 
        ('conv_5x5_p_back', 1), 
        ('conv_5x5_p_back', 1)], 
        normal_concat=range(2, 5))    


dvsg_seed_1 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('avg_pool_3x3_p', 2), ('conv_3x3_p', 0), 
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('avg_pool_3x3_n_back', 2), 
        ('conv_3x3_p_back', 3)], 
        normal_concat=range(2, 5))

dvsg_seed_2026 = Genotype(
    normal=[('avg_pool_3x3_p', 1), ('conv_5x5_p', 0), 
    ('avg_pool_3x3_n', 2), ('avg_pool_3x3_p', 1), 
    ('conv_3x3_p', 2), ('avg_pool_3x3_n', 3), 
    ('conv_5x5_p_back', 0), 
    ('avg_pool_3x3_n_back', 0)], 
    normal_concat=range(2, 5))

dvsg_l4_seed_39 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('avg_pool_3x3_p', 0), 
        ('conv_3x3_p', 2), ('conv_3x3_p', 0), 
        ('conv_5x5_p', 0), ('conv_3x3_p', 2), 
        ('conv_5x5_p_back', 0), 
        ('conv_5x5_p_back', 2)], 
        normal_concat=range(2, 5))

dvsg_l4_seed_41 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0), 
        ('conv_5x5_p', 0), ('conv_5x5_p', 1), 
        ('avg_pool_3x3_n', 3), ('conv_5x5_p', 2), 
        ('avg_pool_3x3_p_back', 1), 
        ('avg_pool_3x3_p_back', 2)], 
        normal_concat=range(2, 5))

dvsg_l4_seed_42 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_p', 0), 
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 2), 
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1), 
        ('avg_pool_3x3_n_back', 0), 
        ('avg_pool_3x3_n_back', 3)], 
        normal_concat=range(2, 5))

dvsg_l4_seed_0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1), 
        ('conv_3x3_p', 2), ('conv_5x5_p', 0), 
        ('conv_5x5_p', 2), ('conv_3x3_p', 3), 
        ('conv_5x5_p_back', 0), 
        ('avg_pool_3x3_p_back', 2)], 
        normal_concat=range(2, 5))


dvsg_random1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 3),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsg_random2 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_p', 0),
        ('conv_5x5_n', 2), ('conv_3x3_n', 0),
        ('conv_5x5_p', 3), ('conv_3x3_n', 0),
        ('avg_pool_3x3_p_back', 0),
        ('conv_3x3_p_back', 0)
    ],
    normal_concat=range(2, 5)
)

dvsg_random3 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random4 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_5x5_p', 2), ('conv_3x3_p', 0),
        ('conv_3x3_p', 2), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random5 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_3x3_p', 2),
        ('conv_5x5_p', 0), ('conv_3x3_p', 2),
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random6 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 0), ('conv_5x5_p', 2),
        ('conv_3x3_p', 1), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p_back', 2),
        ('conv_5x5_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random7 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 0), ('conv_5x5_p', 2),
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),
        ('avg_pool_3x3_n_back', 2),
        ('conv_5x5_n_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random8 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 0), ('conv_5x5_p', 2),
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p_back', 2),
        ('conv_5x5_n_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random9 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_n', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_n', 1),
        ('avg_pool_3x3_p_back', 1),
        ('conv_5x5_n_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsg_random10 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 2),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 3),
        ('avg_pool_3x3_p_back', 1),
        ('conv_5x5_n_back', 0)
    ],
    normal_concat=range(2, 5)
)