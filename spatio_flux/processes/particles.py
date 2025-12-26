"""
Particles Process
=================

Simulates particle motion in a 2D environment with diffusion, advection,
and probabilistic birth/death at the boundaries. Each particle moves,
reads local field values, and can contribute to the environment.
"""
import base64
import pprint
import uuid
import numpy as np
import math
from bigraph_schema import make_default
from process_bigraph import Process, Step


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


def get_local_field_values(fields, x, y, default=np.nan):
    """
    Safely read per-molecule field values at (x, y).

    Conventions:
      - 2D fields are stored as numpy arrays with shape (ny, nx)
      - Index as arr[y, x]  (row=y, col=x)

    - Works with 1D or 2D arrays
    - Clamps indices
    - Returns `default` for empty arrays
    """
    local_values = {}
    for mol_id, field in fields.items():
        arr = np.asarray(field)

        if arr.size == 0:
            local_values[mol_id] = default
            continue

        if arr.ndim == 2:
            # arr.shape = (ny, nx)
            xi = int(np.clip(x, 0, arr.shape[1] - 1))
            yi = int(np.clip(y, 0, arr.shape[0] - 1))
            local_values[mol_id] = arr[yi, xi]

        elif arr.ndim == 1:
            xi = int(np.clip(x, 0, arr.shape[0] - 1))
            local_values[mol_id] = arr[xi]

        else:
            arr2 = np.squeeze(arr)
            if arr2.size == 0:
                local_values[mol_id] = default
                continue

            if arr2.ndim == 2:
                xi = int(np.clip(x, 0, arr2.shape[1] - 1))
                yi = int(np.clip(y, 0, arr2.shape[0] - 1))
                local_values[mol_id] = arr2[yi, xi]
            elif arr2.ndim == 1:
                xi = int(np.clip(x, 0, arr2.shape[0] - 1))
                local_values[mol_id] = arr2[xi]
            else:
                raise ValueError(f"Unsupported field shape {arr.shape} for {mol_id}")

    return local_values



def generate_single_particle_state(config=None):
    config = config or {}
    bounds = config['bounds']
    mass_range = config.get('mass_range')
    if mass_range is None:
        mass_range = INITIAL_MASS_RANGE
    position = config.get('position',  tuple(np.random.uniform(low=[0, 0], high=bounds)))
    mass = config.get('mass', float(np.random.uniform(*mass_range)))

    return {
        'id': config.get('id', None),
        'position': position,
        'mass': mass,
    }


def generate_multiple_particles_state(config=None):
    """
    Generate an initial particles state using generate_single_particle_state.

    Config options:
      - bounds: (x_max, y_max)
      - n_particles: int
      - mass_range: (min, max)
      - positions: optional iterable of (x,y) positions (len == n_particles)
      - id_prefix: optional prefix for particle ids
    """
    config = config or {}

    bounds = config.get('bounds', (1.0, 1.0))
    n_particles = int(config.get('n_particles', 15))
    mass_range = config.get('mass_range', INITIAL_MASS_RANGE)
    positions = config.get('positions', None)
    id_prefix = config.get('id_prefix', None)

    particles = {}

    for i in range(n_particles):
        pid = short_id() if id_prefix is None else f"{id_prefix}_{i}"

        p_cfg = {
            'bounds': bounds,
            'mass_range': mass_range,
            'id': pid,
        }

        if positions is not None:
            p_cfg['position'] = positions[i]

        pstate = generate_single_particle_state(p_cfg)
        particles[pid] = pstate

    return {'particles': particles}


class BrownianMovement(Process):
    config_schema = {
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',
        'diffusion_rate': make_default('float', 1e-1),
        'advection_rate': make_default('tuple[float,float]', (0.0, 0.0)),
    }

    def initialize(self, config):
        self.env_size = ((0.0, config['bounds'][0]), (0.0, config['bounds'][1]))
        D = config['diffusion_rate']
        self._sqrt_2D = np.sqrt(2.0 * D)
        self.vx, self.vy = config['advection_rate']

    def inputs(self):
        return {'particles': 'map[particle]'}

    def outputs(self):
        return {'particles': 'map[particle]'}

    def update(self, state, interval):
        particles = state['particles']
        if not particles:
            return {'particles': {}}

        dt = float(interval)
        sigma = self._sqrt_2D * np.sqrt(dt)

        items = list(particles.items())
        pids = [pid for pid, _ in items]
        pos = np.array([p['position'] for _, p in items], dtype=float)  # (N,2)

        N = pos.shape[0]

        # Brownian + advection steps
        steps = np.random.normal(loc=0.0, scale=sigma, size=(N, 2))
        steps[:, 0] += self.vx * dt
        steps[:, 1] += self.vy * dt

        # Compute absolute new positions
        new_pos = pos + steps

        # Enforce bounds (clamp)
        (xmin, xmax), (ymin, ymax) = self.env_size
        new_pos[:, 0] = np.clip(new_pos[:, 0], xmin, xmax)
        new_pos[:, 1] = np.clip(new_pos[:, 1], ymin, ymax)

        # Emit absolute positions
        updates = {}
        for pid, (x, y) in zip(pids, new_pos):
            updates[pid] = {'position': (float(x), float(y))}

        return {'particles': updates}


class ManageBoundaries(Step):
    """
    Boundary policies on ABSOLUTE positions.

    Assumptions:
      - Incoming particle state stores ABSOLUTE position: p['position'] = (x, y)
      - This step outputs ABSOLUTE position updates: {'position': (new_x, new_y)}

    Policies:
      - Sides in boundary_to_remove are absorbing (remove particle if it crosses).
      - All other sides reflect.
    """

    config_schema = {
        'bounds': 'tuple[float,float]',

        # births
        'add_rate': make_default('float', 0.0),  # event rate (1/sec)
        'boundary_to_add': make_default('list[boundary_side]', ['top']),

        # absorbing
        'boundary_to_remove': make_default('list[boundary_side]', []),

        # numerical buffer for reflecting safely inside the domain
        'buffer': make_default('float', 1e-6),

        # newborn particle config
        'mass_range': make_default('tuple[float,float]', INITIAL_MASS_RANGE),
    }

    def initialize(self, config):
        self.bounds = tuple(config['bounds'])
        x_max, y_max = float(self.bounds[0]), float(self.bounds[1])
        self.env_size = ((0.0, x_max), (0.0, y_max))

        (x_min, x_max), (y_min, y_max) = self.env_size
        buf = float(config.get('buffer', 1e-6))

        # hard bounds
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # interior bounds for reflection (avoid sticking due to floating precision)
        self.x_lo, self.x_hi = x_min + buf, x_max - buf
        self.y_lo, self.y_hi = y_min + buf, y_max - buf

        remove = set(config.get('boundary_to_remove', []))
        self.remove_left   = 'left'   in remove
        self.remove_right  = 'right'  in remove
        self.remove_top    = 'top'    in remove
        self.remove_bottom = 'bottom' in remove

        self.add_rate = float(config.get('add_rate', 0.0))
        self.add_boundaries = tuple(config.get('boundary_to_add', []))
        self.mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

    def inputs(self):
        return {
            'particles': 'map[particle]',
            'process_interval': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {'particles': 'map[particle]'}

    # -----------------------
    # helpers
    # -----------------------

    def _should_remove(self, x, y):
        # absorbing uses hard bounds
        if self.remove_left and x < self.x_min:
            return True
        if self.remove_right and x > self.x_max:
            return True
        if self.remove_top and y > self.y_max:
            return True
        if self.remove_bottom and y < self.y_min:
            return True
        return False

    @staticmethod
    def _reflect_1d(x, lo, hi):
        """
        Reflect x into [lo, hi], robust to overshoot.
        """
        if hi <= lo:
            return lo
        w = hi - lo
        y = (x - lo) % (2.0 * w)
        return (lo + y) if y <= w else (hi - (y - w))

    def _reflect_xy(self, x, y):
        return (
            float(self._reflect_1d(x, self.x_lo, self.x_hi)),
            float(self._reflect_1d(y, self.y_lo, self.y_hi)),
        )

    @staticmethod
    def _rate_to_interval_prob(rate_per_sec, dt):
        rate = float(rate_per_sec)
        dt = float(dt)
        if rate <= 0.0 or dt <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-rate * dt))

    def get_boundary_position(self, boundary):
        (x_min, x_max), (y_min, y_max) = self.env_size
        if boundary == 'left':
            return (x_min, float(np.random.uniform(y_min, y_max)))
        if boundary == 'right':
            return (x_max, float(np.random.uniform(y_min, y_max)))
        if boundary == 'top':
            return (float(np.random.uniform(x_min, x_max)), y_max)
        if boundary == 'bottom':
            return (float(np.random.uniform(x_min, x_max)), y_min)
        return (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))

    # -----------------------
    # update
    # -----------------------

    def update(self, state):
        particles = state.get('particles', {}) or {}
        dt = float(state.get('process_interval', 1.0))

        out = {}

        # preserve your engine's special keys
        out['_remove'] = []
        out['_add'] = {}

        for pid, p in particles.items():
            ox, oy = p.get('position', (0.0, 0.0))
            x, y = float(ox), float(oy)

            # removal check first (if already out of bounds)
            if self._should_remove(x, y):
                out['_remove'].append(pid)
                continue

            # reflect any out-of-range coordinate
            x, y = self._reflect_xy(x, y)

            # ABSOLUTE position update
            out[pid] = {'position': (x, y)}

        # births (ABSOLUTE)
        if self.add_rate > 0.0 and self.add_boundaries:
            p_birth = self._rate_to_interval_prob(self.add_rate, dt)
            for boundary in self.add_boundaries:
                if np.random.rand() < p_birth:
                    pos = self.get_boundary_position(boundary)
                    pid = short_id()
                    new_p = generate_single_particle_state({
                        'bounds': self.bounds,
                        'mass_range': self.mass_range,
                        'position': pos,
                        'id': pid,
                    })
                    out['_add'][pid] = new_p

        # if nothing happened, emit empty update
        if not out['_remove'] and not out['_add'] and len(out) == 2:
            return {'particles': {}}

        return {'particles': out}


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
            'fields': 'lattice_environment',
            # 'fields': 'node',
            # 'fields': {
            #     '_type': 'lattice',
            #     'substrates': {
            #         '_type': 'substrates',
            #         '_value': {
            #             '_shape': self.config['n_bins'],
            #         }
            #     }
            # }
        }

    def outputs(self):
        return self.inputs()

    def initial_state(self, config=None):
        return {}

    def update(self, state):
        particles = state['particles']
        substrate = state.get('fields', {}).get('substrates')

        particle_updates = {}

        # initialize zero-delta arrays for each field (same shape as stored arrays: (ny, nx))
        field_updates = {mol_id: np.zeros_like(array) for mol_id, array in substrate.items()}

        for pid, p in particles.items():
            x, y = p['position']

            # get_bin_position returns (x_bin, y_bin)
            x_bin, y_bin = get_bin_position((x, y), self.config['n_bins'], self.env_size)

            # ----------------------------
            # local field sampling
            # ----------------------------
            local_after = get_local_field_values(substrate, x=x_bin, y=y_bin)

            local_before = p.get('local', None)
            local_before_present = isinstance(local_before, dict) and len(local_before) > 0

            if not local_before_present:
                local_update = {'_add': local_after}
            else:
                local_update = {
                    m: local_after[m] - local_before.get(m, 0.0)
                    for m in local_after
                }

            # ----------------------------
            # exchange: particle ↔ field
            # ----------------------------
            exch_before = p.get('exchange', None)
            exch_before_present = isinstance(exch_before, dict) and len(exch_before) > 0
            exch = exch_before if exch_before_present else {}

            # Apply exchange into field bin (arrays are indexed [y, x])
            for mol_id, delta in exch.items():
                if mol_id in field_updates:
                    field_updates[mol_id][y_bin, x_bin] += delta

            # update particle's exchange state
            if not exch_before_present:
                exchange_update = {'_add': {m: 0.0 for m in substrate.keys()}}
            else:
                exchange_update = {mol_id: 0.0 for mol_id in exch.keys()}

            particle_updates[pid] = {
                'local': local_update,
                'exchange': exchange_update,
            }

        return {
            'particles': particle_updates,
            'fields': {
                'substrates': field_updates
            },
        }

def prune_instance_containers(obj):
    """
    Recursively remove any dict that contains key 'instance'.
    Returns the pruned object, or None if it should be removed.
    """
    if isinstance(obj, dict):
        if 'instance' in obj:
            return None
        new = {}
        for k, v in obj.items():
            pruned = prune_instance_containers(v)
            if pruned is not None:
                new[k] = pruned
        return new

    elif isinstance(obj, list):
        new = []
        for v in obj:
            pruned = prune_instance_containers(v)
            if pruned is not None:
                new.append(pruned)
        return new

    else:
        return obj


class ParticleDivision(Step):
    """
    Stand-alone division process:
      - Tracks particle 'mass' only.
      - If mass >= division_mass_threshold, parent is removed and two children
        are created with half mass each, placed near the parent's position.
      - Additionally splits particle['sub_mass']['sub_masses'] between daughters
        either equally or randomly (conserving totals).
    """

    config_schema = {
        'division_mass_threshold': make_default('float', 0.0),
        'division_jitter': make_default('float', 1e-3),

        # How to split sub_mass['sub_masses'] across daughters: "equal" or "random"
        'sub_mass_split': make_default('string', 'equal'),
    }

    def initialize(self, config):
        pass

    def inputs(self):
        return {'particles': 'map[particle]'}

    def outputs(self):
        return {'particles': 'map[particle]'}

    def initial_state(self, config=None):
        return {}

    def _infer_ref_length(self, particles):
        xs, ys = [], []
        for p in particles.values():
            pos = p.get('position')
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))

        if len(xs) >= 2 and len(ys) >= 2:
            xrange_ = max(xs) - min(xs)
            yrange_ = max(ys) - min(ys)
            ref = max(xrange_, yrange_)
            return ref if ref > 0 else 1.0

        return 1.0

    def _split_sub_masses(self, parent_sub_masses, mode):
        """
        Split a dict of sub-masses into two dicts, conserving totals.
        parent_sub_masses: dict[str, number]
        mode: "equal" or "random"
        """
        if not isinstance(parent_sub_masses, dict) or not parent_sub_masses:
            return {}, {}

        mode = (mode or "equal").lower().strip()
        c1, c2 = {}, {}

        for k, v in parent_sub_masses.items():
            try:
                m = float(v)
            except (TypeError, ValueError):
                # If it's not numeric, skip or set to 0.0 (choose conservative)
                m = 0.0

            m = max(m, 0.0)

            if mode == "random":
                u = float(np.random.uniform(0.0, 1.0))
                c1[k] = m * u
                c2[k] = m * (1.0 - u)
            else:
                # default: equal
                half = m / 2.0
                c1[k] = half
                c2[k] = half

        return c1, c2

    def _make_child(self, parent, new_pos, child_mass, child_sub_masses):
        cid = short_id()
        child = dict(parent)
        child = prune_instance_containers(child)

        child['id'] = cid
        child['mass'] = max(float(child_mass), 0.0)
        child['position'] = (float(new_pos[0]), float(new_pos[1]))

        # IMPORTANT: do not inherit parent's sub_masses
        child.pop('sub_masses', None)

        # Always write the split values here (top-level)
        child['sub_masses'] = dict(child_sub_masses) if isinstance(child_sub_masses, dict) else {}

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
            return {'particles': {}}

        ref_len = self._infer_ref_length(particles)
        r = float(self.config['division_jitter']) * ref_len

        split_mode = str(self.config.get('sub_mass_split', 'equal')).lower().strip()

        for pid, particle in particles.items():
            mass = float(particle.get('mass', 0.0))
            if mass < thr:
                continue

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

            # Split parent mass equally (as before)
            child_mass = max(mass, 0.0) / 2.0

            # Split sub_masses under sub_mass key
            parent_sub_masses = particle.get('sub_masses', {})
            c1_sub, c2_sub = self._split_sub_masses(parent_sub_masses, split_mode)

            c1_id, c1 = self._make_child(particle, c1_pos, child_mass=child_mass, child_sub_masses=c1_sub)
            c2_id, c2 = self._make_child(particle, c2_pos, child_mass=child_mass, child_sub_masses=c2_sub)

            updated_particles['_add'][c1_id] = c1
            updated_particles['_add'][c2_id] = c2

        if not updated_particles['_remove'] and not updated_particles['_add']:
            return {'particles': {}}

        return {'particles': updated_particles}


class ParticleTotalMass(Step):
    """
    Sum per-particle scalar submasses into a single `mass` for ONE particle.

    Intended to be placed inside each particle's subtree, where `submasses`
    is a map of float-valued contributions.
    """
    config_schema = {}

    def initialize(self, config):
        pass

    def inputs(self):
        return {"sub_masses": "map[mass]"}

    def outputs(self):
        return {"total_mass": "overwrite[mass]"}  #"set_float"}  #"set_float"}  #{"_type": "overwrite[mass]", "_apply": "set"}}

    def initial_state(self, config=None):
        return {}

    def update(self, state):
        submasses = state.get("sub_masses", {}) or {}

        total = 0.0
        for v in submasses.values():
            try:
                total += float(v)
            except Exception:
                # ignore non-numeric values defensively
                continue

        return {"total_mass": total}
