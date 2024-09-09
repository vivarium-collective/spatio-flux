"""
Particles process
=================

A process for simulating the motion of particles in a 2D environment.
"""

import uuid
import numpy as np
from process_bigraph import Process, Composite
from processes import core
from viz.plot import plot_species_distributions_with_particles_to_gif, plot_particles


class Particles(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'default_advection_rate': {'_type': 'tuple[float,float]', '_default': (0, 0)},
        'default_add_probability': {'_type': 'float', '_default': 0.1},  # probability of adding particles
        'boundary_to_add': {'_type': 'list', '_default': ['left', 'right']},  # which boundaries to add particles
        'boundary_to_remove': {'_type': 'list', '_default': ['left', 'right', 'top', 'bottom']},
        # which boundaries to remove particles
        'n_particles_per_species': {'_type': 'list', '_default': [10, 10]},  # number of particles for each species
        'species_colors': {'_type': 'list', '_default': ['b', 'g', 'r', 'c', 'm', 'y', 'k']},  # colors for each species
        'diffusion_rates': {'_type': 'map', '_default': {}},  # diffusion rates for species by color with {'color': rate}
        'advection_rates': {'_type': 'map', '_default': {}},  # advection rates for species by color with {'color': rate}
        'add_probability': {'_type': 'map', '_default': {}},  # probability of adding particles
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.env_size = (
            (0, self.config['bounds'][0]),
            (0, self.config['bounds'][1])
        )

        self.diffusion_rates = {
            color: self.config['diffusion_rates'].get(color, self.config['default_diffusion_rate'])
            for color in self.config['species_colors']}
        self.advection_rates = {
            color: self.config['advection_rates'].get(color, self.config['default_advection_rate'])
            for color in self.config['species_colors']}
        self.add_probability = {
            color: self.config['add_probability'].get(color, self.config['default_add_probability'])
            for color in self.config['species_colors']}

    def inputs(self):
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
            bounds,
            species_colors=None,  # Extend as needed
            size_range=(10, 100)
    ):
        """
        Initialize particle positions for multiple species.
        """
        if species_colors is None:
            species_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # advection_rates = advection_rates or [(0.0, 0.0) for _ in range(len(n_particles_per_species))]
        particles = []

        for species_idx, n_particles in enumerate(n_particles_per_species):
            color = species_colors[species_idx % len(species_colors)]
            for _ in range(n_particles):
                particle = {
                    'id': str(uuid.uuid4()),
                    'position': tuple(np.random.uniform(
                        low=[0, 0],
                        high=[bounds[0], bounds[1]],
                        size=2)),
                    'size': np.random.uniform(size_range[0], size_range[1]),
                    'color': color,
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
            color = particle['color']

            # Apply diffusion and advection
            dx, dy = np.random.normal(0, self.diffusion_rates[color], 2) + self.advection_rates[color]

            new_x_position = particle['position'][0] + dx
            new_y_position = particle['position'][1] + dy

            # Check and remove particles if they hit specified boundaries
            if self.check_boundary_hit(new_x_position, new_y_position):
                continue  # Remove particle if it hits a boundary

            new_position = (new_x_position, new_y_position)
            updated_particle['position'] = new_position

            # Retrieve local field concentration for each particle
            x, y = self.get_bin_position(new_position)
            local_field_concentrations = self.get_local_field_values(fields, column=x, row=y)

            # MARKER: Insert what to do for the given particle based on local_field_concentrations
            local_biomass = local_field_concentrations.get('biomass')
            if local_biomass:
                # Michaelis-Menten-like rate law for uptake
                max_uptake_rate = 0.1  # maximum uptake rate (tunable)
                half_saturation = 1  # half-saturation constant (tunable, determines how quickly saturation occurs)
                uptake_rate = (max_uptake_rate * local_biomass) / (half_saturation + local_biomass)

                # Particle uptake rate is proportional to its size
                absorbed_biomass = float(uptake_rate * particle['size'])

                size = updated_particle['size']
                updated_particle['size'] = max(size + 0.01*absorbed_biomass, 0.0)
                if local_biomass - absorbed_biomass < 0.0:
                    absorbed_biomass = local_biomass
                new_fields['biomass'][y, x] = -absorbed_biomass

            new_particles.append(updated_particle)

        # Probabilistically add new particles at user-defined boundaries
        for boundary in self.config['boundary_to_add']:
            for color, prob in self.add_probability.items():
                if np.random.rand() < prob:
                    new_particle = {
                        'id': str(uuid.uuid4()),
                        'position': self.get_boundary_position(boundary),
                        'size': np.random.uniform(10, 100),  # Random size for new particles
                        'color': color,
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

    def get_local_field_values(self, fields, column, row):
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
            local_values[mol_id] = field[row, column]

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
        bounds=(10, 10),  # Bounds of the environment
        n_bins=(20, 20),  # Number of bins in the x and y directions
):

    # initialize particles
    colors = ['b', 'g', 'r']
    n_particles_per_species = [5, 5, 5]  # Number of particles per species
    custom_diffusion_rates = {}
    custom_advection_rates = {
        'r': (0, -0.1),
    }
    custom_add_probability = {
        'r': 0.4,
    }
    default_add_probability = 0.0

    # initialize
    particles = Particles.initialize_particles(
        n_particles_per_species=n_particles_per_species,
        bounds=bounds,
    )

    # initialize fields
    initial_biomass = np.random.uniform(low=0, high=20, size=(n_bins[0], n_bins[1]))

    composite_state = {
        'fields': {
            'biomass': initial_biomass,
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
                'default_add_probability': default_add_probability,
                'boundary_to_add': ['top'],
                # 'boundary_to_remove': ['bottom'],
                'species_colors': colors,
                'diffusion_rates': custom_diffusion_rates,
                'advection_rates': custom_advection_rates,
                'add_probability': custom_add_probability,
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
    sim.save(filename='particles.json', outdir='out')

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    particles_results = sim.gather_results()
    emitter_results = particles_results[('emitter',)]

    particles_history = [p['particles'] for p in emitter_results]
    # print(particles_history)

    print('Plotting...')
    # plot particles
    plot_particles(
        # total_time=total_time,
        history=particles_history,
        env_size=((0, bounds[0]), (0, bounds[1])),
        out_dir='out',
        filename='particles.gif',
    )

    plot_species_distributions_with_particles_to_gif(
        particles_results,
        out_dir='out',
        filename='particle_with_fields.gif',
        title='',
        skip_frames=1,
        bounds=bounds,
    )


if __name__ == '__main__':
    run_particles()
