"""
Particles process
"""
import uuid
import numpy as np
from process_bigraph import Process, Composite
from processes import core  # import the core from the processes package
from plot.particles import plot_particles


class Particles(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'default_advection_rate': {'_type': 'tuple[float,float]', '_default': (0, 0)},
        # 'diffusion_coeffs': 'map[float]',
        # 'advection_coeffs': 'map[tuple[float,float]]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.env_size = (
            (0, self.config['bounds'][0]),
            (0, self.config['bounds'][1])
        )

    def inputs(self):
        return {
            'particles': 'any',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def outputs(self):
        return {
            'particles': {
                '_type': 'any',
                '_apply': 'set'
            },
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def update(self, state, interval):
        particles = state['particles']

        new_particles = []
        for particle in particles:
            updated_particle = particle.copy()
            # Apply diffusion and advection
            dx, dy = np.random.normal(0, particle['diffusion_rate'], 2) + particle['advection']

            new_x_position = particle['position'][0] + dx
            new_y_position = particle['position'][1] + dy

            # enforce boundary
            if new_x_position < self.env_size[0][0]:
                new_x_position = self.env_size[0][0]
            elif new_x_position > self.env_size[0][1]:
                new_x_position = self.env_size[0][1]

            if new_y_position < self.env_size[1][0]:
                new_y_position = self.env_size[1][0]
            elif new_y_position > self.env_size[1][1]:
                new_y_position = self.env_size[1][1]

            new_position = (new_x_position, new_y_position)
            updated_particle['position'] = new_position

            new_particles.append(updated_particle)

        return {
            'particles': new_particles,
            # 'fields': state['fields']
        }


core.register_process('Particles', Particles)


def initialize_particles(n_particles_per_species, env_size, diffusion_rates, advection_rates=None):
    """
    Initialize particle positions for multiple species.

    Parameters:
    - n_particles_per_species: List of numbers of particles for each species.
    - env_size: Tuple indicating the xlim and ylim of the environment, as ((xmin, xmax), (ymin, ymax)).
    - diffusion_rates: List of diffusion rates for each species.
    - advection_rates: List of advection vectors for each species.

    Returns:
    - particles: List of dictionaries representing particles.
    """
    advection_rates = advection_rates or [(0.0, 0.0) for s in range(len(n_particles_per_species))]
    particles = []
    species_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend as needed

    for species_idx, n_particles in enumerate(n_particles_per_species):
        color = species_colors[species_idx % len(species_colors)]
        diffusion_rate = diffusion_rates[species_idx]
        advection_rate = advection_rates[species_idx]

        for _ in range(n_particles):
            particle = {
                'id': str(uuid.uuid4()),
                'position': tuple(np.random.uniform(low=[env_size[0][0], env_size[1][0]],
                                              high=[env_size[0][1], env_size[1][1]],
                                              size=2)),
                'size': np.random.uniform(10, 100),
                'color': color,
                'diffusion_rate': diffusion_rate,
                'advection': advection_rate
            }
            particles.append(particle)

    return particles


def run_particles(
    total_time=100,  # Total frames
):

    # initialize particles
    n_particles_per_species = [10, 10, 10]  # Number of particles per species
    env_size = ((0, 10), (0, 10))  # Environment size (xmin, xmax), (ymin, ymax)
    diffusion_rates = [0.5, 0.2, 0.05]  # Diffusion rates per species
    advection_rates = [(0, 0), (0, 0), (0, -0.1)]  # Advection vectors per species

    # initialize
    particles = initialize_particles(n_particles_per_species, env_size, diffusion_rates,
                                     advection_rates=advection_rates)

    # initialize fields
    n_bins = (10, 10)
    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)

    composite_state = {
        'fields': {
            'glucose': initial_glucose,
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
    sim.save(filename='particles.json', outdir='out')

    # simulate
    sim.update({}, total_time)

    # gather results
    particles_results = sim.gather_results()
    particles_results = particles_results[('emitter',)]

    particles_history = [p['particles'] for p in particles_results]
    # print(particles_history)

    # plot particles
    plot_particles(
        total_time=total_time,
        history=particles_history,
        env_size=((0, n_bins[0]), (0, n_bins[1])),
        out_dir='out',
        filename='particles.gif',
    )


if __name__ == '__main__':
    run_particles()
