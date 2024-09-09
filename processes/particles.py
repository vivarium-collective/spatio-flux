"""
Particles process
"""
import uuid
import numpy as np
from process_bigraph import Process, Composite
from processes import core  # import the core from the processes package
from plot.particles import plot_particles


import numpy as np
import uuid

import numpy as np
import uuid


class Particles(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'default_advection_rate': {'_type': 'tuple[float,float]', '_default': (0, 0)},
        'add_probability': {'_type': 'float', '_default': 0.1},  # probability of adding particles
        'boundary_to_add': {'_type': 'list', '_default': ['left', 'right']},  # which boundaries to add particles
        'boundary_to_remove': {'_type': 'list', '_default': ['left', 'right', 'top', 'bottom']},
        # which boundaries to remove particles
        'n_particles_per_species': {'_type': 'list', '_default': [10, 10]},  # number of particles for each species
        'species_colors': {'_type': 'list', '_default': ['b', 'g', 'r', 'c', 'm', 'y', 'k']},  # colors for each species
        'diffusion_rates': {'_type': 'list', '_default': [1e-1, 1e-2]},  # diffusion rates for species
        'advection_rates': {'_type': 'list', '_default': [(0.0, 0.0), (0.0, 0.0)]},  # advection rates for species
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

    @staticmethod
    def initialize_particles(
            n_particles_per_species,
            env_size,
            diffusion_rates,
            advection_rates=None,
            species_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend as needed
    ):
        """
        Initialize particle positions for multiple species.
        """
        advection_rates = advection_rates or [(0.0, 0.0) for _ in range(len(n_particles_per_species))]
        particles = []

        for species_idx, n_particles in enumerate(n_particles_per_species):
            color = species_colors[species_idx % len(species_colors)]
            diffusion_rate = diffusion_rates[species_idx]
            advection_rate = advection_rates[species_idx]

            for _ in range(n_particles):
                particle = {
                    'id': str(uuid.uuid4()),
                    'position': tuple(np.random.uniform(
                        low=[0, 0],
                        high=[env_size[0], env_size[1]],
                        size=2)),
                    'size': np.random.uniform(10, 100),  # Random size between 10 and 100
                    'color': color,
                    'diffusion_rate': diffusion_rate,
                    'advection': advection_rate
                }
                particles.append(particle)

        return particles

    def update(self, state, interval):
        particles = state['particles']
        fields = state['fields']  # Retrieve the fields

        new_particles = []
        new_fields = {
            mol_id: np.zeros_like(field)
            for mol_id, field in fields.items()}
        for particle in particles:
            updated_particle = particle.copy()
            # Apply diffusion and advection
            dx, dy = np.random.normal(0, particle['diffusion_rate'], 2) + particle['advection']

            new_x_position = particle['position'][0] + dx
            new_y_position = particle['position'][1] + dy

            # Check and remove particles if they hit specified boundaries
            if self.check_boundary_hit(new_x_position, new_y_position):
                continue  # Remove particle if it hits a boundary

            new_position = (new_x_position, new_y_position)
            updated_particle['position'] = new_position

            # Retrieve local field concentration for each particle
            x, y = self.get_bin_position(new_position)
            local_field_concentrations = self.get_local_field_values(fields, x, y)

            # MARKER: Insert what to do for the given particle based on local_field_concentrations
            for mol_id, conc in local_field_concentrations.items():
                new_conc = max(conc - 0.01, 0)
                # pass
                new_fields[mol_id][x, y] = new_conc - conc  # Update the field concentration at the particle's location

            new_particles.append(updated_particle)

        # Probabilistically add new particles at user-defined boundaries
        for boundary in self.config['boundary_to_add']:
            if np.random.rand() < self.config['add_probability']:
                new_particle = {
                    'id': str(uuid.uuid4()),
                    'position': self.get_boundary_position(boundary),
                    'size': np.random.uniform(10, 100),  # Random size for new particles
                    'color': self.config['species_colors'][0],  # Use the first color for new particles
                    'diffusion_rate': self.config['default_diffusion_rate'],
                    'advection': self.config['default_advection_rate']
                }
                new_particles.append(new_particle)

        return {
            'particles': new_particles,
            'fields': new_fields
        }

    def get_bin_position(self, position):
        x, y = position
        x_bins, y_bins = self.config['n_bins']
        x_min, x_max = self.env_size[0]
        y_min, y_max = self.env_size[1]

        # Convert the particle's (x, y) position to the corresponding bin in the 2D grid
        x_bin = int((x - x_min) / (x_max - x_min) * x_bins)
        y_bin = int((y - y_min) / (y_max - y_min) * y_bins)

        # Correct any potential out-of-bound indices
        x_bin = min(max(x_bin, 0), x_bins - 1)
        y_bin = min(max(y_bin, 0), y_bins - 1)

        return x_bin, y_bin

    def get_local_field_values(self, fields, x_bin, y_bin):
        """
        Retrieve local field values for a particle based on its position.

        Parameters:
        - fields: dict of 2D numpy arrays representing fields, keyed by molecule ID.
        - position: Tuple (x, y) representing the particle's position.

        Returns:
        - local_values: dict of field concentrations at the particle's location, keyed by molecule ID.
        """
        local_values = {}
        for mol_id, field in fields.items():
            local_values[mol_id] = field[x_bin, y_bin]

        return local_values

    def check_boundary_hit(self, new_x_position, new_y_position):
        # Check if the particle hits any of the boundaries to be removed
        if 'left' in self.config['boundary_to_remove'] and new_x_position < self.env_size[0][0]:
            return True
        if 'right' in self.config['boundary_to_remove'] and new_x_position > self.env_size[0][1]:
            return True
        if 'top' in self.config['boundary_to_remove'] and new_y_position > self.env_size[1][1]:
            return True
        if 'bottom' in self.config['boundary_to_remove'] and new_y_position < self.env_size[1][0]:
            return True
        return False

    def get_boundary_position(self, boundary):
        if boundary == 'left':
            return (self.env_size[0][0], np.random.uniform(*self.env_size[1]))
        elif boundary == 'right':
            return (self.env_size[0][1], np.random.uniform(*self.env_size[1]))
        elif boundary == 'top':
            return (np.random.uniform(*self.env_size[0]), self.env_size[1][1])
        elif boundary == 'bottom':
            return (np.random.uniform(*self.env_size[0]), self.env_size[1][0])


core.register_process('Particles', Particles)


def run_particles(
    total_time=100,  # Total frames
):

    # initialize particles
    n_particles_per_species = [10, 10, 10]  # Number of particles per species
    env_size = [10, 10]  #((0, 10), (0, 10))  # Environment size (xmin, xmax), (ymin, ymax)
    diffusion_rates = [0.5, 0.2, 0.05]  # Diffusion rates per species
    advection_rates = [(0, 0), (0, 0), (0, -0.1)]  # Advection vectors per species

    # initialize
    particles = Particles.initialize_particles(
        n_particles_per_species=n_particles_per_species,
        env_size=env_size,
        diffusion_rates=diffusion_rates,
        advection_rates=advection_rates
    )

    # initialize fields
    n_bins = (10, 10)
    initial_glucose = np.random.uniform(low=0, high=20, size=(n_bins[0], n_bins[1]))

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
        # total_time=total_time,
        history=particles_history,
        env_size=((0, n_bins[0]), (0, n_bins[1])),
        out_dir='out',
        filename='particles.gif',
    )


if __name__ == '__main__':
    run_particles()
