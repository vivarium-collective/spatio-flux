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
        steps = np.random.normal(loc=0.0, scale=sigma, size=(N, 2))
        steps[:, 0] += self.vx * dt
        steps[:, 1] += self.vy * dt

        # Emit displacement only (dx,dy)
        updates = {}
        for pid, (dx, dy) in zip(pids, steps):
            updates[pid] = {'position': (float(dx), float(dy))}

        return {'particles': updates}


class ManageBoundaries(Step):
    """
    Per-side boundary policy.

    Default behavior:
      - All sides reflect (hard walls).
      - Any side listed in boundary_to_remove becomes absorbing (particle removed if it crosses).

    Notes:
      - This step assumes particle state stores ABSOLUTE position in p['position'] = (x, y).
      - It expects incoming motion to be provided as a delta (dx, dy). In the code below, we
        read delta from p.get('position_delta', ...) or fall back to p.get('position') if you
        are still using the "delta in position" convention.
      - If you have a separate movement store, wire that in and replace delta lookup accordingly.
    """

    config_schema = {
        'bounds': 'tuple[float,float]',

        # a per-interval probability using process_interval.
        'add_rate': make_default('float', 0.0),
        'boundary_to_add': make_default('list[boundary_side]', ['top']),

        # these are the absorbing sides; everything else reflects
        'boundary_to_remove': make_default('list[boundary_side]', []),

        'clamp_survivors': make_default('boolean', True),
        'buffer': make_default('float', 1e-4),
        'mass_range': make_default('tuple[float,float]', INITIAL_MASS_RANGE),
    }

    def initialize(self, config):
        self.bounds = tuple(config['bounds'])
        x_max, y_max = float(self.bounds[0]), float(self.bounds[1])
        self.env_size = ((0.0, x_max), (0.0, y_max))

        (x_min, x_max), (y_min, y_max) = self.env_size
        buf = float(config.get('buffer', 1e-4))

        # hard bounds
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # "safe" interior bounds for clamping/reflecting
        self.x_lo, self.x_hi = x_min + buf, x_max - buf
        self.y_lo, self.y_hi = y_min + buf, y_max - buf

        remove = set(config.get('boundary_to_remove', []))
        self.remove_left   = 'left'   in remove
        self.remove_right  = 'right'  in remove
        self.remove_top    = 'top'    in remove
        self.remove_bottom = 'bottom' in remove

        # Treat as event rate (1/sec). Convert to per-interval prob in update().
        self.add_rate = float(config.get('add_rate', 0.0))
        self.add_boundaries = tuple(config.get('boundary_to_add', []))

        self.clamp_survivors = bool(config.get('clamp_survivors', True))
        self.mass_range = config.get('mass_range', INITIAL_MASS_RANGE)

    def inputs(self):
        return {
            'particles': 'map[particle]',
            'process_interval': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {'particles': 'map[particle]'}

    # ---------- reflection helpers ----------

    @staticmethod
    def _reflect_1d(x, lo, hi):
        """
        Reflect x into [lo, hi] by mirroring across boundaries.
        Handles arbitrarily large overshoot via modulo arithmetic.
        """
        if hi <= lo:
            return lo
        w = hi - lo
        y = (x - lo) % (2.0 * w)
        return (lo + y) if y <= w else (hi - (y - w))

    def _apply_reflect(self, x, y):
        return (
            float(self._reflect_1d(x, self.x_lo, self.x_hi)),
            float(self._reflect_1d(y, self.y_lo, self.y_hi)),
        )

    def _should_remove(self, x, y):
        """
        Check if (x,y) crosses an absorbing side.
        Uses the *hard* bounds (x_min/x_max etc.).
        """
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
    def _rate_to_interval_prob(rate_per_sec, dt):
        """
        Convert a Poisson event rate (lambda, 1/sec) into probability of >=1 event in dt seconds.
          P = 1 - exp(-lambda * dt)
        This behaves well for large/small dt and never exceeds 1.
        """
        rate = float(rate_per_sec)
        dt = float(dt)
        if rate <= 0.0 or dt <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-rate * dt))

    # ---------- main update ----------

    def update(self, state):
        particles = state.get('particles', {}) or {}

        # interval in seconds (or whatever your engine uses consistently)
        dt = float(state.get('process_interval', 1.0))

        updated = {'_remove': [], '_add': {}}

        for pid, p in particles.items():
            # ABS position in state (required)
            ox, oy = p.get('position', (0.0, 0.0))

            # DELTA for this tick.
            dx, dy = p.get('position_delta', p.get('delta', p.get('position_delta', (0.0, 0.0))))
            if (dx, dy) == (0.0, 0.0):
                # fallback: some pipelines store delta in position updates
                maybe_dx, maybe_dy = p.get('position', (0.0, 0.0))
                # only use this fallback if it looks like a delta (small-ish)
                if abs(maybe_dx) <= (self.x_max - self.x_min) and abs(maybe_dy) <= (self.y_max - self.y_min):
                    dx, dy = maybe_dx, maybe_dy

            nx = float(ox + dx)
            ny = float(oy + dy)

            # absorbing has priority
            if self._should_remove(nx, ny):
                updated['_remove'].append(pid)
                continue

            # otherwise reflect on any side crossed
            nx, ny = self._apply_reflect(nx, ny)

            # optional numeric clamp (usually redundant with reflect, but safe)
            if self.clamp_survivors:
                nx = float(np.clip(nx, self.x_lo, self.x_hi))
                ny = float(np.clip(ny, self.y_lo, self.y_hi))

            # emit delta update (engine applies to absolute position)
            updated[pid] = {'position': (nx - ox, ny - oy)}

        # births (absolute positions)
        if self.add_rate > 0.0 and self.add_boundaries:
            p_birth = self._rate_to_interval_prob(self.add_rate, dt)
            if p_birth > 0.0:
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
                        updated['_add'][pid] = new_p

        if not updated['_remove'] and not updated['_add'] and len(updated) == 2:
            return {'particles': {}}

        return {'particles': updated}

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
            'fields': 'map[positive_array]',
            # 'fields': {
            #     '_type': 'map',
            #     '_value': {
            #         '_type': 'array',
            #         '_shape': self.config['n_bins'],
            #         # '_data': 'concentration'
            #         # '_data': 'float'
            #     },
            # }
        }

    def outputs(self):
        return self.inputs()

    def initial_state(self, config=None):
        return {}

    def update(self, state):
        particles = state['particles']
        fields = state['fields']

        particle_updates = {}

        # initialize zero-delta arrays for each field (same shape as stored arrays: (ny, nx))
        field_updates = {mol_id: np.zeros_like(array) for mol_id, array in fields.items()}

        for pid, p in particles.items():
            x, y = p['position']

            # get_bin_position returns (x_bin, y_bin)
            x_bin, y_bin = get_bin_position((x, y), self.config['n_bins'], self.env_size)

            # ----------------------------
            # local field sampling
            # ----------------------------
            local_after = get_local_field_values(fields, x=x_bin, y=y_bin)

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
                exchange_update = {'_add': {m: 0.0 for m in fields.keys()}}
            else:
                exchange_update = {mol_id: 0.0 for mol_id in exch.keys()}

            particle_updates[pid] = {
                'local': local_update,
                'exchange': exchange_update,
            }

        return {
            'particles': particle_updates,
            'fields': field_updates,
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
