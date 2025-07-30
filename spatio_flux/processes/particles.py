"""
Particles Process
=================

Simulates particle motion in a 2D environment with diffusion, advection,
and probabilistic birth/death at the boundaries. Each particle moves,
reads local field values, and can contribute to the environment.
"""
import base64
import uuid
import numpy as np
from process_bigraph import Process, default

# Constants
INITIAL_MASS_RANGE = (1E-3, 1.0)


def short_id(length=6):
    """
    Generate a short, URL-safe unique ID string.

    Parameters:
        length (int): Number of bytes to use from the UUID. Determines uniqueness.
                      Each byte adds ~1.33 characters to the final ID.

    Returns:
        str: A base64-encoded ID string of approximately 1.33 * length characters.

    Example:
        length=6 → ~8-character ID with ~2.8e14 possible values
        length=4 → ~6-character ID with ~4.3e9 possible values
    """
    raw = uuid.uuid4().bytes[:length]
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def get_bin_position(position, n_bins, env_size):
    x, y = position
    x_bins, y_bins = n_bins
    x_min, x_max = env_size[0]
    y_min, y_max = env_size[1]

    x_bin = int((x - x_min) / (x_max - x_min) * x_bins)
    y_bin = int((y - y_min) / (y_max - y_min) * y_bins)
    return min(max(x_bin, 0), x_bins - 1), min(max(y_bin, 0), y_bins - 1)


def get_local_field_values(fields, column, row):
    return {mol_id: field[column, row] for mol_id, field in fields.items()}


def generate_single_particle_state(config=None):
    config = config or {}
    bounds = config['bounds']
    n_bins = config['n_bins']
    fields = config.get('fields') or {}
    mol_ids = fields.keys()
    mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

    position = config.get('position', tuple(np.random.uniform(low=[0, 0], high=bounds)))
    mass = config.get('mass', np.random.uniform(*mass_range))
    x, y = get_bin_position(position, n_bins, ((0.0, bounds[0]), (0.0, bounds[1])))
    local = get_local_field_values(fields, column=x, row=y)
    exchanges = {f: 0.0 for f in mol_ids}

    return {
        'position': position,
        'local': local,
        'mass': mass,
        'exchange': exchanges
    }


class Particles(Process):
    config_schema = {
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',
        'diffusion_rate': default('float', 1e-1),
        'advection_rate': default('tuple[float,float]', (0, 0)),
        'add_probability': 'float',
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
        return self.inputs()

    def initial_state(self, config=None):
        return {}

    @staticmethod
    def generate_state(config=None):
        config = config or {}
        fields = config.get('fields', {})
        n_bins = config.get('n_bins', (1, 1))
        bounds = config.get('bounds', (1.0, 1.0))
        n_particles = config.get('n_particles', 15)
        mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

        if fields:
            actual_shape = next(iter(fields.values())).shape
            if actual_shape != n_bins:
                raise ValueError(f"Shape mismatch: fields {actual_shape} vs n_bins {n_bins}")

        particles = {}
        for _ in range(n_particles):
            pid = short_id()
            particles[pid] = generate_single_particle_state({
                'bounds': bounds,
                'mass_range': mass_range,
                'n_bins': n_bins,
                'fields': fields,
            })
            particles[pid]['id'] = pid

        return {'particles': particles}

    def update(self, state, interval):
        # Retrieve current particles and field values
        particles = state['particles']
        fields = state['fields']

        # Initialize structures for updated particles and field values
        updated_particles = {'_remove': [], '_add': {}}
        updated_fields = {
            mol_id: np.zeros_like(field)
            for mol_id, field in fields.items()
        }

        # Loop through each particle
        for pid, particle in particles.items():
            # Sample random motion (diffusion + advection)
            dx, dy = np.random.normal(0, self.config['diffusion_rate'], 2) + self.config['advection_rate']
            new_x = particle['position'][0] + dx
            new_y = particle['position'][1] + dy

            # Remove particle if it hits a configured removal boundary
            if self.check_boundary_hit(new_x, new_y):
                updated_particles['_remove'].append(pid)
                continue

            # Keep particles inside the environment bounds with a small buffer
            x_min, x_max = self.env_size[0]
            y_min, y_max = self.env_size[1]
            buffer = 0.0001
            new_x = np.clip(new_x, x_min + buffer, x_max - buffer)
            new_y = np.clip(new_y, y_min + buffer, y_max - buffer)

            # Determine the grid cell (bin) the particle is now in
            new_pos = (new_x, new_y)
            column, row = get_bin_position(new_pos, self.config['n_bins'], self.env_size)

            # Retrieve local field values at the particle's new location
            local = get_local_field_values(fields, column, row)

            # Apply the particle’s exchange values to the environment fields
            exchange = particle['exchange']
            for mol_id, rate in exchange.items():
                updated_fields[mol_id][column, row] += rate

            # Update the particle’s state
            updated_particles[pid] = {
                'position': (dx, dy),  # Store delta; use new_pos if you prefer absolute
                'local': local,  # Updated local concentrations
                'exchange': {mol_id: 0.0 for mol_id in exchange}  # Reset exchanges
            }

        # Randomly add new particles at designated boundaries
        for boundary in self.config['boundary_to_add']:
            if np.random.rand() < self.config['add_probability']:
                position = self.get_boundary_position(boundary)
                new_particle = generate_single_particle_state({
                    'bounds': self.config['bounds'],
                    'n_bins': self.config['n_bins'],
                    'fields': fields,
                    'position': position
                })
                pid = short_id()
                new_particle['id'] = pid
                updated_particles['_add'][pid] = new_particle

        # Return the updated particle states and field updates
        return {
            'particles': updated_particles,
            'fields': updated_fields
        }

    def check_boundary_hit(self, x, y):
        return (
            ('left' in self.config['boundary_to_remove'] and x < self.env_size[0][0]) or
            ('right' in self.config['boundary_to_remove'] and x > self.env_size[0][1]) or
            ('top' in self.config['boundary_to_remove'] and y > self.env_size[1][1]) or
            ('bottom' in self.config['boundary_to_remove'] and y < self.env_size[1][0])
        )

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
    """
    A minimal particle that performs reactions based on Michaelis-Menten kinetics.

    Configuration:
    -------------
    - reactions (dict): {reaction_name: {'reactant': str, 'product': str}}
    - kinetic_params (dict): {reactant: (Km, Vmax)} for each substrate or 'mass'

    Inputs:
    -------
    - mass (float): current mass of the particle
    - substrates (map[positive_float]): concentrations of external substrates

    Outputs:
    --------
    - mass (float): net change in mass
    - substrates (map[float]): net change in substrate concentrations

    Notes:
    ------
    - Supports reactions that use 'mass' as reactant or product (e.g., decay or growth).
    - Reaction rates follow: rate = Vmax * conc / (Km + conc)
    """

    config_schema = {
        'reactions': {
            '_type': 'map[reaction]',
            '_default': {
                'grow': {
                    'reactant': 'glucose',
                    'product': 'mass',
                },
                'release': {
                    'reactant': 'mass',
                    'product': 'detritus',
                }
            }
        },
        'kinetic_params': {
            '_type': 'map[tuple[float,float]]',
            '_default': {
                'glucose': (0.5, 0.01),   # example: Km=0.5, Vmax=0.01
                'mass': (1.0, 0.001)      # decay or export from internal mass
            }
        }
    }

    def initialize(self, config):
        self.reactions = config['reactions']
        self.kinetic_params = config['kinetic_params']

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
        mass = state['mass']

        delta_mass = 0.0
        delta_substrates = {mol_id: 0.0 for mol_id in substrates}

        for reaction in self.reactions.values():
            reactant = reaction['reactant']
            product = reaction['product']

            conc = mass if reactant == 'mass' else substrates.get(reactant, 0.0)

            if reactant not in self.kinetic_params:
                raise ValueError(f"Kinetic parameters not provided for reactant: {reactant}")

            Km, Vmax = self.kinetic_params[reactant]
            rate = Vmax * conc / (Km + conc) if Km + conc > 0 else 0.0

            # update mass
            if reactant == 'mass':
                delta_mass -= rate
            else:
                if reactant not in delta_substrates:
                    delta_substrates[reactant] = 0.0
                delta_substrates[reactant] -= rate

            if product == 'mass':
                delta_mass += rate
            else:
                delta_substrates[product] = delta_substrates.get(product, 0.0) + rate

        return {
            'mass': delta_mass,
            'substrates': delta_substrates
        }
