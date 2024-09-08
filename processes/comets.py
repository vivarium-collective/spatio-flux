import numpy as np
from process_bigraph import Composite
from processes.spatial_dfba import dfba_config
from processes import core
from plot.plot import plot_time_series, plot_species_distributions_to_gif


# TODO -- need to do this to register???
from processes.spatial_dfba import DynamicFBA
from processes.diffusion_advection import DiffusionAdvection


def run_comets():
    n_bins = (10, 10)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f'[{i},{j}]'] = {
                '_type': 'process',
                'address': 'local:DynamicFBA',
                'config': dfba_config(
                    model_file='TESTING'  # load the same model for all processes
                ),
                'inputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j]
                    }
                }
            }

    composite_state = {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'spatial_dfba': dfba_processes_dict,
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': (10, 10),
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': {
                    'glucose': 1e-1,
                    'acetate': 1e-1,
                    'biomass': 1e-1,
                },
                'advection_coeffs': {
                    'glucose': (0, 0),
                    'acetate': (0, 0),
                    'biomass': (0, 0),
                },
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }
    }

    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='comets.json', outdir='out')

    sim.update({}, 60.0)
    comets_results = sim.gather_results()
    # print(comets_results)

    # plot timeseries
    plot_time_series(
        comets_results,
        coordinates=[(0, 0), (5, 5)],
    )

    # plot 2d video
    plot_species_distributions_to_gif(
        comets_results,
        out_dir='out',
        filename='comets_results.gif',
        title='',
        skip_frames=1)


if __name__ == '__main__':
    run_comets()
