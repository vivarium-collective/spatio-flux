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
        'id': config.get('id', None),
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
        'add_probability': default('float', 0.0),
        'boundary_to_add': default('list[boundary_side]', ['top']),
        'boundary_to_remove': default('list[boundary_side]', ['left', 'right', 'top', 'bottom']),
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
        mass_range = config.get('mass_range') or INITIAL_MASS_RANGE

        if fields:
            actual_shape = next(iter(fields.values())).shape
            if actual_shape != n_bins:
                raise ValueError(f"Shape mismatch: fields {actual_shape} vs n_bins {n_bins}")

        particles = {}
        for _ in range(n_particles):
            pid = short_id()
            pstate = generate_single_particle_state({
                'bounds': bounds,
                'mass_range': mass_range,
                'n_bins': n_bins,
                'fields': fields,
                'id': pid,
            })
            particles[pid] = pstate

        return {'particles': particles}

    def update(self, state, interval):
        particles = state['particles']
        fields = state['fields']

        updated_particles = {'_remove': [], '_add': {}}
        updated_fields = {mol_id: np.zeros_like(field) for mol_id, field in fields.items()}

        # helpers
        def clamp_in_bounds(x, y):
            x_min, x_max = self.env_size[0]
            y_min, y_max = self.env_size[1]
            buffer = 0.0001
            return (
                float(np.clip(x, x_min + buffer, x_max - buffer)),
                float(np.clip(y, y_min + buffer, y_max - buffer)),
            )

        # main loop
        for pid, particle in particles.items():
            # motion
            dx, dy = np.random.normal(0, self.config['diffusion_rate'], 2) + self.config['advection_rate']
            new_x = particle['position'][0] + dx
            new_y = particle['position'][1] + dy

            # removal by boundary
            if self.check_boundary_hit(new_x, new_y):
                updated_particles['_remove'].append(pid)
                continue

            # keep inside bounds
            new_x, new_y = clamp_in_bounds(new_x, new_y)
            new_pos = (new_x, new_y)

            # env bin + local fields
            col, row = get_bin_position(new_pos, self.config['n_bins'], self.env_size)
            local = get_local_field_values(fields, col, row)

            # write exchange to fields (parent contributes this tick)
            for mol_id, rate in particle.get('exchange', {}).items():
                updated_fields[mol_id][col, row] += rate

            # per-tick particle update (no division)
            updated_particles[pid] = {
                # store delta (dx, dy) to match existing convention; switch to absolute if you prefer
                'position': (dx, dy),
                'local': local,
                'exchange': {m: 0.0 for m in particle.get('exchange', {})},
            }

        # random birth from boundaries
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
