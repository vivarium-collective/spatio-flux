import numpy as np
from process_bigraph import Composite
from processes.spatial_dfba import dfba_config
from processes import core
from plot.fields import plot_time_series, plot_species_distributions_to_gif


# TODO -- need to do this to register???
from processes.spatial_dfba import DynamicFBA
from processes.diffusion_advection import DiffusionAdvection
from processes.particles import Particles


def run_particle_comets():
    n_bins = (8, 8)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    # initialize particles
    n_particles_per_species = [10, 10, 10]  # Number of particles per species
    env_size = [10, 10]  #((0, 10), (0, 10))  # Environment size (xmin, xmax), (ymin, ymax)
    diffusion_rates = [0.5, 0.2, 0.05]  # Diffusion rates per species
    advection_rates = [(0, 0), (0, 0), (0, -0.1)]  # Advection vectors per species

    particles = Particles.initialize_particles(
        n_particles_per_species=n_particles_per_species,
        env_size=env_size,
        diffusion_rates=diffusion_rates,
        advection_rates=advection_rates
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
        },
        'particles': particles,
        'particles_process': {
            '_type': 'process',
            'address': 'local:Particles',
            'config': {
                'n_bins': n_bins,
                'bounds': (10, 10),
                'default_diffusion_rate': 1e-1,
                'default_advection_rate': (0, 0),
                'add_probability': 0.1,
                'boundary_to_add': ['top'],
                'boundary_to_remove': ['bottom'],
                # 'particle_initial_position': (0, 0)
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

    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='particle_comets.json', outdir='out')

    sim.update({}, 60.0)
    particle_comets_results = sim.gather_results()
    # print(comets_results)

    # plot timeseries
    plot_time_series(
        particle_comets_results,
        coordinates=[(0, 0), (5, 5)],
        out_dir='out',
        filename='particle_comets_results.png'
    )

    # plot 2d video
    plot_species_distributions_to_gif(
        particle_comets_results,
        out_dir='out',
        filename='particle_comets_results.gif',
        title='',
        skip_frames=1)


if __name__ == '__main__':
    run_particle_comets()
