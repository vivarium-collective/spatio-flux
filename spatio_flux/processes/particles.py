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
from bigraph_schema import default
from process_bigraph import Process, default


# Constants
INITIAL_MASS_RANGE = (1E-3, 1.0)
DIVISION_MASS_THRESHOLD = 5.0


def short_id(length=6):
    """Generate a short, URL-safe unique ID string"""
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


class ParticleMovement(Process):
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
                    '_data': 'concentration'
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

        # time step and transport params
        dt = float(interval)
        D = float(self.config['diffusion_rate'])  # diffusion coefficient (len^2 / time)
        vx, vy = self.config['advection_rate']  # advection velocity (len / time)
        sigma = np.sqrt(2.0 * D * dt)  # Brownian std per axis

        # helpers
        def clamp_in_bounds(x, y):
            x_min, x_max = self.env_size[0]
            y_min, y_max = self.env_size[1]
            buffer = 1e-4
            return (
                float(np.clip(x, x_min + buffer, x_max - buffer)),
                float(np.clip(y, y_min + buffer, y_max - buffer)),
            )

        # main loop
        for pid, particle in particles.items():
            # motion: Brownian ~ N(0, 2D dt) + deterministic drift v dt
            dx, dy = np.random.normal(0.0, sigma, 2)
            dx += vx * dt
            dy += vy * dt

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

            # apply exchange to fields
            # If 'exchange' values are per-time fluxes, integrate with dt:
            for mol_id, rate in particle.get('exchange', {}).items():
                updated_fields[mol_id][col, row] += rate * dt  # use '* dt' if rate is per unit time

            # per-tick particle update
            updated_particles[pid] = {
                # store displacement to match your convention
                'position': (dx, dy),
                'local': local,
                'exchange': {m: 0.0 for m in particle.get('exchange', {})},
            }

        # random birth from boundaries (per-tick probability).
        # If you want time-consistent births: let p = 1 - exp(-lambda * dt).
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


class ParticleDivision(Process):
    """
    Stand-alone division process:
      - Tracks particle 'mass' only.
      - If mass >= division_mass_threshold, parent is removed and two children
        are created with half mass each, placed near the parent's position.
    """

    # Same division-related knobs as in Particles, nothing else.
    config_schema = {
        # If <= 0, division is disabled
        'division_mass_threshold': default('float', 0.0),
        # Fraction of a reference length used as jitter radius for child placement.
        # Reference length is inferred from current particle cloud extent (max range of x or y);
        # if not inferable, falls back to 1.0.
        'division_jitter': default('float', 1e-3),
    }

    def initialize(self, config):
        # No environment to set up
        pass

    def inputs(self):
        # Only particles are needed
        return {
            'particles': 'map[particle]',
        }

    def outputs(self):
        # Emit particle deltas in the same convention: _remove, _add, and/or per-id updates
        return {
            'particles': 'map[particle]',
        }

    def initial_state(self, config=None):
        # No default state; upstream composition provides particles
        return {}

    def _infer_ref_length(self, particles):
        """Infer a reference length from particle positions for jitter scaling."""
        xs, ys = [], []
        for p in particles.values():
            pos = p.get('position')
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
        if len(xs) >= 2 and len(ys) >= 2:
            xrange_ = (max(xs) - min(xs))
            yrange_ = (max(ys) - min(ys))
            ref = max(xrange_, yrange_)
            return ref if ref > 0 else 1.0
        return 1.0

    def _make_child(self, parent, new_pos):
        """Create a child particle from parent with half mass and reset exchanges."""
        cid = short_id()
        child = dict(parent)  # shallow copy; particle is assumed flat
        child['id'] = cid
        child['mass'] = max(parent.get('mass', 0.0), 0.0) / 2.0
        child['position'] = (float(new_pos[0]), float(new_pos[1]))
        # carry local as-is if present (no environment management here)
        # reset exchanges if present
        exch = parent.get('exchange', {})
        if isinstance(exch, dict):
            child['exchange'] = {k: 0.0 for k in exch.keys()}
        return cid, child

    def update(self, state, interval):
        particles = state['particles']

        updated_particles = {'_remove': [], '_add': {}}
        thr = float(self.config['division_mass_threshold'])
        if thr <= 0.0 or not particles:
            # Division disabled or nothing to do: no changes
            return {'particles': {}}

        # Pre-compute reference length for jitter radius
        ref_len = self._infer_ref_length(particles)
        r = float(self.config['division_jitter']) * ref_len

        for pid, particle in particles.items():
            mass = float(particle.get('mass', 0.0))
            if mass >= thr:
                # Remove parent
                updated_particles['_remove'].append(pid)

                # Parent position (fallback to (0,0) if missing)
                px, py = (0.0, 0.0)
                pos = particle.get('position')
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    px, py = float(pos[0]), float(pos[1])

                # Symmetric jitter around parent
                if r > 0.0:
                    angle = float(np.random.uniform(0.0, 2.0 * np.pi))
                    ox, oy = r * np.cos(angle), r * np.sin(angle)
                else:
                    ox, oy = 0.0, 0.0

                c1_pos = (px + ox, py + oy)
                c2_pos = (px - ox, py - oy)

                c1_id, c1 = self._make_child(particle, c1_pos)
                c2_id, c2 = self._make_child(particle, c2_pos)

                updated_particles['_add'][c1_id] = c1
                updated_particles['_add'][c2_id] = c2

            # No else: we don't modify non-dividing particles in this process

        # If nothing changed, return empty delta so upstream can skip writes
        if not updated_particles['_remove'] and not updated_particles['_add']:
            return {'particles': {}}

        return {'particles': updated_particles}
