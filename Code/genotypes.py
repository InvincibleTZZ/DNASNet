from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'normal normal_concat')

"""
Operation sets
"""

PRIMITIVES = [ 
         
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
    'conv_5x5_n'
]


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
