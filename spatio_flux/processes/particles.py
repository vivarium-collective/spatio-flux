"""
Particles process
=================

A process for simulating the motion of particles in a 2D environment.
"""

import uuid
import numpy as np
from process_bigraph import Process, Composite
from spatio_flux import core
from spatio_flux.viz.plot import plot_species_distributions_with_particles_to_gif, plot_particles


# TODO -- make particle type
particle_type = {
    'id': 'string',
    'position': 'tuple[float,float]',
    'size': 'float',
}
core.register('particle', particle_type)


class Particles(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'advection_rate': {'_type': 'tuple[float,float]', '_default': (0, 0)},
        # adding/removing particles at boundaries
        'add_probability': {'_type': 'float', '_default': 0.0},  # TODO -- make probability type
        'boundary_to_add': {'_type': 'list', '_default': ['left', 'right']},  # which boundaries to add particles
        'boundary_to_remove': {'_type': 'list', '_default': ['left', 'right', 'top', 'bottom']},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.env_size = (
            (0, self.config['bounds'][0]),
            (0, self.config['bounds'][1])
        )

    def inputs(self):
        return {
            'particles': {
                '_type': 'list[particle]',
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
            n_particles,
            bounds,
            size_range=(10, 100)
    ):
        """
        Initialize particle positions for multiple species.
        """
        # advection_rates = advection_rates or [(0.0, 0.0) for _ in range(len(n_particles_per_species))]
        particles = []
        for _ in range(n_particles):
            particle = {
                'id': str(uuid.uuid4()),
                'position': tuple(np.random.uniform(
                    low=[0, 0],
                    high=[bounds[0], bounds[1]],
                    size=2)),
                'size': np.random.uniform(size_range[0], size_range[1]),
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
            dx, dy = np.random.normal(0, self.config['diffusion_rate'], 2) + self.config['advection_rate']

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
            if np.random.rand() < self.config['add_probability']:
                new_particle = {
                    'id': str(uuid.uuid4()),
                    'position': self.get_boundary_position(boundary),
                    'size': np.random.uniform(10, 100),  # Random size for new particles
                    # 'local': {}  # TODO local field values
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


# Helper functions to get specs and states
def get_particles_spec(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        diffusion_rate=1e-1,
        advection_rate=(0, 0),
        add_probability=0.0,
        boundary_to_add=['top'],
):
    return {
            '_type': 'process',
            'address': 'local:Particles',
            'config': {
                'n_bins': n_bins,
                'bounds': bounds,
                'diffusion_rate': diffusion_rate,
                'advection_rate': advection_rate,
                'add_probability': add_probability,
                'boundary_to_add': boundary_to_add,
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


def get_particles_state(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        n_particles=15,
        diffusion_rate=0.1,
        advection_rate=(0, -0.1),
        boundary_to_add=None,
        add_probability=0.4,
):

    # initialize particles
    if boundary_to_add is None:
        boundary_to_add = ['top']

    particles = Particles.initialize_particles(
        n_particles=n_particles,
        bounds=bounds,
    )
    # initialize fields
    initial_biomass = np.random.uniform(low=0, high=20, size=(n_bins[1], n_bins[0]))

    return {
        'fields': {
            'biomass': initial_biomass,
        },
        'particles': particles,
        'particles_process': get_particles_spec(
            n_bins=n_bins,
            bounds=bounds,
            diffusion_rate=diffusion_rate,
            advection_rate=advection_rate,
            add_probability=add_probability,
            boundary_to_add=boundary_to_add,
        )
    }


def run_particles(
        total_time=100,  # Total frames
        bounds=(10.0, 20.0),  # Bounds of the environment
        n_bins=(20, 40),  # Number of bins in the x and y directions
):
    # initialize particles state
    composite_state = get_particles_state(
        n_bins=n_bins,
        bounds=bounds,
        n_particles=10,
        diffusion_rate=0.1,
        advection_rate=(0, -0.1),
        add_probability=0.4,
        boundary_to_add=['top'],
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='particles.json', outdir='out')

    # TODO -- save a viz figure of the initial state

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
