"""
Particles process
=================

A process for simulating the motion of particles in a 2D environment.
"""
import base64
import uuid

import numpy as np
from process_bigraph import Process, default


INITIAL_MASS_RANGE = (1E-3, 1.0)  # Default mass range for particles


def get_bin_position(position, n_bins, env_size):
    x, y = position
    x_bins, y_bins = n_bins #self.config['n_bins']
    x_min, x_max = env_size[0]
    y_min, y_max = env_size[1]

    # Convert the particle's (x, y) position to the corresponding bin in the 2D grid
    x_bin = int((x - x_min) / (x_max - x_min) * x_bins)
    y_bin = int((y - y_min) / (y_max - y_min) * y_bins)

    # Correct any potential out-of-bound indices
    x_bin = min(max(x_bin, 0), x_bins - 1)
    y_bin = min(max(y_bin, 0), y_bins - 1)

    return x_bin, y_bin


def get_local_field_values(fields, column, row):
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
        local_values[mol_id] = field[column, row]

    return local_values


def short_id():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')



def generate_single_particle_state(config=None):
    """
    Initialize a single particle with random properties.
    """
    config = config or {}
    bounds = config['bounds']
    n_bins = config['n_bins']
    fields = config.get('fields', {})
    mol_ids = fields.keys()
    mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

    # get particle properties
    position = config.get('position', tuple(np.random.uniform(low=[0, 0], high=[bounds[0], bounds[1]], size=2)))
    mass = config.get('mass', np.random.uniform(mass_range[0], mass_range[1]))
    x, y = get_bin_position(position, n_bins, ((0.0, bounds[0]), (0.0, bounds[1])))
    # TODO update local and exchange values
    local = get_local_field_values(fields, column=x, row=y)
    exchanges = {f: 0.0 for f in mol_ids}  # TODO exchange rates

    return {
        'position': position,
        'local': local,
        'mass': mass,
        'exchange': exchanges
    }


class Particles(Process):
    config_schema = {
        # environment size and resolution
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',

        # particle movement
        'diffusion_rate': default('float', 1e-1),
        'advection_rate': default('tuple[float,float]', (0, 0)),

        # adding/removing particles at boundaries
        'add_probability': 'float', # TODO -- make probability type

        # which boundaries to add particles
        'boundary_to_add': default('list[boundary_side]', ['top']),
        'boundary_to_remove': default('list[boundary_side]', ['left', 'right', 'top', 'bottom'])
    }

    def initialize(self, config):
        self.env_size = (
            (0, config['bounds'][0]),
            (0, config['bounds'][1])
        )

    def inputs(self):
        return {
            'particles': 'map[particle]',
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
            'particles': 'map[particle]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def initial_state(self, config=None):
        return {}

    @staticmethod
    def generate_state(config=None):
        """
        Initialize particle positions for multiple species.
        """
        config = config or {}
        fields = config.get('fields', {})
        n_bins = config.get('n_bins', (1,1))
        bounds = config.get('bounds',(1.0,1.0))
        n_particles = config.get('n_particles', 15)
        mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

        # assert n_bins from the shape of the first field array
        if len(fields) > 0:
            fields_bins = fields[list(fields.keys())[0]].shape
            if fields_bins != n_bins:
                raise ValueError(
                    f"Shape of fields {fields_bins} does not match n_bins {n_bins}"
                )

        # advection_rates = advection_rates or [(0.0, 0.0) for _ in range(len(n_particles_per_species))]
        particles = {}
        for _ in range(n_particles):
            id = short_id()
            particles[id] = generate_single_particle_state(config={
                'bounds': bounds,
                'mass_range': mass_range,
                'n_bins': n_bins,
                'fields': fields,
            })
            particles[id]['id'] = id

        return {
            'particles': particles}


    def update(self, state, interval):
        particles = state['particles']
        fields = state['fields']  # Retrieve the fields

        new_particles = {'_remove': [], '_add': {}}
        new_fields = {
            mol_id: np.zeros_like(field)
            for mol_id, field in fields.items()}

        for particle_id, particle in particles.items():
            updated_particle = {'exchange': {}}

            # Apply diffusion and advection
            dx, dy = np.random.normal(0, self.config['diffusion_rate'], 2) + self.config['advection_rate']

            new_x_position = particle['position'][0] + dx
            new_y_position = particle['position'][1] + dy

            # Check and remove particles if they hit specified boundaries
            if self.check_boundary_hit(new_x_position, new_y_position):
                new_particles['_remove'].append(particle_id)
                continue  # Remove particle if it hits a boundary
            # clip if hit a boundary
            x_min, x_max = self.env_size[0]
            y_min, y_max = self.env_size[1]
            if not (x_min <= new_x_position <= x_max and y_min <= new_y_position <= y_max):
                buffer = 0.0001  # TODO -- make parameter?
                buffer_x_min = x_min + (x_max - x_min) * buffer
                buffer_x_max = x_max - (x_max - x_min) * buffer
                buffer_y_min = y_min + (y_max - y_min) * buffer
                buffer_y_max = y_max - (y_max - y_min) * buffer

                new_x_position = np.clip(new_x_position, buffer_x_min, buffer_x_max)
                new_y_position = np.clip(new_y_position, buffer_y_min, buffer_y_max)

            new_position = (new_x_position, new_y_position)
            updated_particle['position'] = (dx, dy) # new_position

            # Retrieve local field concentration for each particle
            x, y = get_bin_position(new_position, self.config['n_bins'], self.env_size)

            # Update local environment values for each particle
            updated_particle['local'] = get_local_field_values(fields, column=x, row=y)

            # Apply exchanges to fields and reset
            exchange = particle['exchange']

            # print(f'particle {particle_id} exchange: {exchange}')

            for mol_id, exchange_rate in exchange.items():
                new_fields[mol_id][x, y] += exchange_rate
                updated_particle['exchange'][mol_id] = 0.0

            new_particles[particle_id] = updated_particle

        # Probabilistically add new particles at user-defined boundaries
        for boundary in self.config['boundary_to_add']:
            if np.random.rand() < self.config['add_probability']:
                position = self.get_boundary_position(boundary)
                # use the function to get a new particle
                new_particle = generate_single_particle_state({
                    'bounds': self.config['bounds'],
                    'n_bins': self.config['n_bins'],
                    'fields': fields,
                    'position': position,
                })
                particle_id = short_id()
                new_particle['id'] = particle_id # Generate a unique ID for the new particle
                new_particles['_add'][particle_id] = new_particle

        return {
            'particles': new_particles,
            'fields': new_fields
        }

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
            return self.env_size[0][0], np.random.uniform(*self.env_size[1])
        elif boundary == 'right':
            return self.env_size[0][1], np.random.uniform(*self.env_size[1])
        elif boundary == 'top':
            return np.random.uniform(*self.env_size[0]), self.env_size[1][1]
        elif boundary == 'bottom':
            return np.random.uniform(*self.env_size[0]), self.env_size[1][0]


class MinimalParticle(Process):
    config_schema = {
        'reactions': {
            '_type': 'map[reaction]',
            '_default': {
                'grow': {
                    'vmax': 0.01,
                    'reactant': 'glucose',
                    'product': 'mass',
                },
                'release': {
                    'vmax': 0.001,
                    'reactant': 'mass',
                    'product': 'detritus',
                }
            }
        }
    }

    def initialize(self, config):
        self.reactions = config['reactions']

    def inputs(self):
        return {
            'mass': 'float',
            'substrates': 'map[positive_float]'
        }

    def outputs(self):
        return {
            'mass': 'float',
            'substrates': 'map[float]'
        }

    def update(self, state, interval):
        substrates = state['substrates']
        exchanges = {mol_id: 0.0 for mol_id in substrates}
        mass = state['mass']
        mass_change = 0.0

        for reaction in self.reactions.values():
            reactant = reaction['reactant']
            if reactant not in substrates and reactant != 'mass':
                exchanges[reactant] = 0.0
                # raise ValueError(f"Reactant '{reactant}' not found in substrates or mass.")
            product = reaction['product']
            vmax = reaction['vmax']

            conc = mass if reactant == 'mass' else substrates.get(reactant, 0.0)
            rate = vmax * conc

            if reactant == 'mass':
                mass_change -= rate
            else:
                exchanges[reactant] -= rate

            if product == 'mass':
                mass_change += rate
            else:
                exchanges[product] = exchanges.get(product, 0.0) + rate

        return {
            'mass': mass_change,
            'substrates': exchanges
        }
