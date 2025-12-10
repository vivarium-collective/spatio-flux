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
from process_bigraph import Process, Step, default


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


import numpy as np

def get_local_field_values(fields, column, row, default=np.nan):
    """
    Safely read per-molecule field values at (column,row).
    - Works with 1D or 2D arrays
    - Clamps indices
    - Returns `default` for empty arrays
    """
    local_values = {}
    for mol_id, field in fields.items():
        arr = np.asarray(field)

        # Empty field → return default (np.nan by default)
        if arr.size == 0:
            local_values[mol_id] = default
            continue

        if arr.ndim == 2:
            # Clamp with shape-aware bounds
            c = int(np.clip(column, 0, arr.shape[0] - 1))
            r = int(np.clip(row,    0, arr.shape[1] - 1))
            local_values[mol_id] = arr[c, r]
        elif arr.ndim == 1:
            c = int(np.clip(column, 0, arr.shape[0] - 1))
            local_values[mol_id] = arr[c]
        else:
            # Try squeezing unusual shapes (e.g., (1, N) or (N, 1))
            arr2 = np.squeeze(arr)
            if arr2.ndim in (1, 2) and arr2.size > 0:
                # Recurse once with the squeezed array
                if arr2.ndim == 2:
                    c = int(np.clip(column, 0, arr2.shape[0] - 1))
                    r = int(np.clip(row,    0, arr2.shape[1] - 1))
                    local_values[mol_id] = arr2[c, r]
                else:
                    c = int(np.clip(column, 0, arr2.shape[0] - 1))
                    local_values[mol_id] = arr2[c]
            else:
                raise ValueError(f"Unsupported field shape {arr.shape} for {mol_id}")
    return local_values



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

    # TODO -- how is particle movement getting field information?
    return {
        'id': config.get('id', None),
        'position': position,
        'local': local,
        'mass': mass,
        'exchange': exchanges
    }


class BrownianMovement(Process):
    config_schema = {
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',
        'diffusion_rate': default('float', 1e-1),
        'advection_rate': default('tuple[float,float]', (0.0, 0.0)),
        'add_probability': default('float', 0.0),
        'boundary_to_add': default('list[boundary_side]', ['top']),
        'boundary_to_remove': default('list[boundary_side]', ['left', 'right', 'top', 'bottom']),
    }

    # ------------------------------------------------------------------
    #                    INITIALIZATION / CONSTANTS
    # ------------------------------------------------------------------

    def initialize(self, config):
        self.env_size = ((0.0, config['bounds'][0]), (0.0, config['bounds'][1]))

        (x_min, x_max), (y_min, y_max) = self.env_size

        # Small buffer to avoid numerical issues at the exact boundary
        buffer = 1e-4
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x_lo, self.x_hi = x_min + buffer, x_max - buffer
        self.y_lo, self.y_hi = y_min + buffer, y_max - buffer

        # Precompute diffusion factor: sigma = sqrt(2 * D * dt) -> we store sqrt(2D)
        D = config['diffusion_rate']
        self._sqrt_2D = np.sqrt(2.0 * D)

        # Advection
        self.vx, self.vy = config['advection_rate']

        # Removal boundary flags
        remove_boundaries = set(config['boundary_to_remove'])
        self.check_left   = 'left'   in remove_boundaries
        self.check_right  = 'right'  in remove_boundaries
        self.check_top    = 'top'    in remove_boundaries
        self.check_bottom = 'bottom' in remove_boundaries

        # Addition boundaries + cached config
        self.add_boundaries = tuple(config['boundary_to_add'])
        self.add_prob = float(config['add_probability'])
        self.bounds = config['bounds']
        self.n_bins = config['n_bins']

    # ------------------------------------------------------------------

    def inputs(self):
        return {'particles': 'map[particle]'}

    def outputs(self):
        return {'particles': 'map[particle]'}

    def initial_state(self, config=None):
        return {}

    @staticmethod
    def generate_state(config=None):
        config = config or {}
        bounds = config.get('bounds', (1.0, 1.0))
        n_bins = config.get('n_bins', (1, 1))
        n_particles = config.get('n_particles', 15)
        mass_range = config.get('mass_range') or INITIAL_MASS_RANGE
        fields = config.get('fields', {})  # optional, used by generate_single_particle_state if provided

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

    # ------------------------------------------------------------------
    #                           MAIN UPDATE
    # ------------------------------------------------------------------

    def update(self, state, interval):
        particles = state['particles']
        updated = {'_remove': [], '_add': {}}

        # Fast path: no particles and no births
        if not particles and self.add_prob <= 0.0:
            return {'particles': updated}

        dt = float(interval)
        # sigma = sqrt(2 * D * dt) but we precomputed sqrt(2D)
        sigma = self._sqrt_2D * np.sqrt(dt)

        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        x_lo, x_hi = self.x_lo, self.x_hi
        y_lo, y_hi = self.y_lo, self.y_hi

        # --------------------------------------------------------------
        #                VECTORIZED PARTICLE POSITION UPDATE
        # --------------------------------------------------------------
        if particles:
            # Convert dict -> parallel arrays (ids, positions)
            items = list(particles.items())
            pids = [pid for pid, _ in items]
            pos = np.array([p['position'] for _, p in items], dtype=float)  # shape (N, 2)

            N = pos.shape[0]

            # Draw all Brownian steps at once
            steps = np.random.normal(loc=0.0, scale=sigma, size=(N, 2))
            # Add advection drift
            steps[:, 0] += self.vx * dt
            steps[:, 1] += self.vy * dt

            new_pos = pos + steps  # shape (N, 2)

            # Vectorized boundary checks
            out_mask = np.zeros(N, dtype=bool)
            if self.check_left:
                out_mask |= (new_pos[:, 0] < x_min)
            if self.check_right:
                out_mask |= (new_pos[:, 0] > x_max)
            if self.check_top:
                out_mask |= (new_pos[:, 1] > y_max)
            if self.check_bottom:
                out_mask |= (new_pos[:, 1] < y_min)

            # Indices for particles that remain in the system
            in_mask = ~out_mask

            # Clamp in-bounds (only for those that remain)
            if np.any(in_mask):
                new_pos[in_mask, 0] = np.clip(new_pos[in_mask, 0], x_lo, x_hi)
                new_pos[in_mask, 1] = np.clip(new_pos[in_mask, 1], y_lo, y_hi)

            # Fill removal list
            if np.any(out_mask):
                remove_idx = np.nonzero(out_mask)[0]
                updated['_remove'] = [pids[i] for i in remove_idx]

            # For remaining particles, store *displacement* (delta) as expected
            if np.any(in_mask):
                keep_idx = np.nonzero(in_mask)[0]
                # displacements = new_pos - old pos
                disp = new_pos[keep_idx] - pos[keep_idx]  # shape (K, 2)
                for i, (dx, dy) in zip(keep_idx, disp):
                    pid = pids[i]
                    updated[pid] = {
                        'position': (dx, dy)
                    }

        # --------------------------------------------------------------
        #                     NEW PARTICLE BIRTHS
        # --------------------------------------------------------------
        if self.add_prob > 0.0:
            for boundary in self.add_boundaries:
                if np.random.rand() < self.add_prob:
                    position = self.get_boundary_position(boundary)
                    pid = short_id()
                    new_p = generate_single_particle_state({
                        'bounds': self.bounds,
                        'n_bins': self.n_bins,
                        'position': position,
                        'id': pid,
                    })
                    updated['_add'][pid] = new_p

        return {'particles': updated}

    # ------------------------------------------------------------------

    def get_boundary_position(self, boundary):
        (x_min, x_max), (y_min, y_max) = self.env_size
        if boundary == 'left':
            return (x_min, np.random.uniform(y_min, y_max))
        if boundary == 'right':
            return (x_max, np.random.uniform(y_min, y_max))
        if boundary == 'top':
            return (np.random.uniform(x_min, x_max), y_max)
        if boundary == 'bottom':
            return (np.random.uniform(x_min, x_max), y_min)




class ParticleExchange(Step):
    config_schema = {
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',
    }

    def initialize(self, config):
        self.env_size = ((0, config['bounds'][0]), (0, config['bounds'][1]))

    def inputs(self):
        return {
            'particles': 'map[particle]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    # '_data': 'concentration'
                    '_data': 'float'
                },
            }
        }

    def outputs(self):
        return self.inputs()

    def initial_state(self, config=None):
        return {}

    def update(self, state):
        particles = state['particles']
        fields = state['fields']

        particle_updates = {}
        # initialize zero-delta arrays for each field
        field_updates = {mol_id: np.zeros_like(array) for mol_id, array in fields.items()}

        for pid, p in particles.items():
            x, y = p['position']
            col, row = get_bin_position((x, y), self.config['n_bins'], self.env_size)

            # ---- local field sampling ----
            local_after = get_local_field_values(fields, col, row)
            local_before = p.get('local', {}) or {}

            if not local_before:
                # first time: add entire local map
                local_update = {'_add': local_after}
            else:
                # subsequent times: store the delta
                local_update = {
                    m: local_after[m] - local_before.get(m, 0.0)
                    for m in local_after
                }

            # ---- exchange: particle ↔ field ----
            exch = p.get('exchange', {}) or {}

            # apply exchange to field bin
            # convention: positive value -> adds to field at this bin
            for mol_id, delta in exch.items():
                if mol_id in field_updates:
                    # you can multiply by dt here if exch is a rate
                    field_updates[mol_id][col, row] += delta

            # update particle's exchange state
            if not exch:
                # initialize exchange map if missing/empty
                exchange_update = {'_add': {m: 0.0 for m in fields.keys()}}
            else:
                # after applying, zero out the exchange (command consumed)
                exchange_update = {mol_id: 0.0 for mol_id in exch}

            particle_updates[pid] = {
                'local': local_update,
                'exchange': exchange_update,
            }

        return {
            'particles': particle_updates,
            'fields': field_updates,
        }


class ParticleDivision(Step):
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

    def update(self, state):
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

        return {
            'particles': updated_particles
        }
