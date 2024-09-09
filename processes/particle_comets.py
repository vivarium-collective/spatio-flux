import numpy as np
from process_bigraph import Composite
from processes.dfba import dfba_config
from processes import core
from viz.plot import (
    plot_time_series, plot_species_distributions_with_particles_to_gif)


# TODO -- need to do this to register???
from processes.dfba import DynamicFBA
from processes.diffusion_advection import DiffusionAdvection
from processes.particles_motion import ParticlesMotion


def run_particle_comets(
        total_time=10.0,
        bounds=(10, 10),
        n_bins=(8, 8),
):

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    # initialize particles
    n_particles_per_species = [3, 3, 5]  # Number of particles per species
    diffusion_rates = [0.4, 0.2, 0.05]  # Diffusion rates per species
    advection_rates = [(0, 0), (0, 0), (0, -0.1)]  # Advection vectors per species

    particles = ParticlesMotion.initialize_particles(
        n_particles_per_species=n_particles_per_species,
        bounds=bounds,
        diffusion_rates=diffusion_rates,
        advection_rates=advection_rates,
        size_range=(100, 200),
    )

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
                'bounds': bounds,
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
        },
        'particles': particles,
        'particles_process': {
            '_type': 'process',
            'address': 'local:ParticlesMotion',
            'config': {
                'n_bins': n_bins,
                'bounds': bounds,
                'default_diffusion_rate': 1e-1,
                'default_advection_rate': (0, 0),
                'default_add_probability': 0.1,
                'boundary_to_add': ['top'],
                # 'boundary_to_remove': ['bottom'],
            },
            'inputs': {
                'particles': ['particles'],
                'fields': ['fields']
            },
            'outputs': {
                'particles': ['particles'],
                'fields': ['fields']
            }
        }
    }

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='particle_comets.json', outdir='out')

    # run simulation
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = sim.gather_results()
    # print(comets_results)

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        particle_comets_results,
        coordinates=[(0, 0), (5, 5)],
        out_dir='out',
        filename='particle_comets_timeseries.png'
    )

    # # plot 2d video
    # plot_species_distributions_to_gif(
    #     particle_comets_results,
    #     out_dir='out',
    #     filename='particle_comets_fields.gif',
    #     title='',
    #     skip_frames=1)

    plot_species_distributions_with_particles_to_gif(
        particle_comets_results,
        out_dir='out',
        filename='particle_comets_with_fields.gif',
        title='',
        skip_frames=1,
        bounds=bounds,
    )


if __name__ == '__main__':
    run_particle_comets()
