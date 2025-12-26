import numpy as np


state = {
    'fields': {
        '_type': 'lattice_environment',
        'bin_volume': 1.0,
        'A': {
            'concentration':  np.random.uniform(low=0,high=1,size=(2,2)).astype('float32'),
            'exchange':  np.zeros(shape=(2,2)).astype('float32'),
        },
        'B': {
            'concentration': np.random.uniform(low=0, high=1, size=(2, 2)).astype('float32'),
            'exchange': np.zeros(shape=(2, 2)).astype('float32'),
        },
        'count_conc_adapter': {
            '_type': 'step',
            '_inputs': {
                'exchange': 'map[array[count]]',
                'volume': 'float',
            },
            '_outputs': {
                'concentrations': 'map[array[concentration]]',
            },
            'inputs': {
                'exchange': ['*', 'count'],
                'volume': ['bin_volume'],
            },
            'outputs': {
                'concentrations': ['*', 'concentration'],
            }
        },
    },
    'diffusion': {
        '_type': 'process',
        '_inputs': {
            'fields': 'map[array[concentration]]',  # this will use array apply
        },
        '_outputs': {
            'fields': 'map[array[concentration]]',
        },
        'inputs': {
            'fields': ['fields', '*', 'concentration'],
        },
        'outputs': {
            'fields': ['fields', '*', 'concentration'],
        },
    },
    'particle': {
        '_type': 'process',
        '_inputs': {
            'substrates': 'map[concentration]',   # this will use the concentrations apply
        },
        '_outputs': {
            'substrates': 'map[count]',
        },
        'inputs': {
            'substrates': ['fields', '*', 'concentration', 0, 0],
        },
        'outputs': {
            'substrates': ['fields', '*', 'exchange', 0, 0],
        },
    },
}