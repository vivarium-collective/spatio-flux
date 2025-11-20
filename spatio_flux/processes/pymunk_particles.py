"""
TODO: use an actual grow/divide process for demo
"""
import uuid
import random
import math
import random
import pymunk

from bigraph_schema import default
from process_bigraph import Composite, gather_emitter_results, Process
from process_bigraph.emitter import emitter_from_wires
from spatio_flux.plots.multibody_plots import simulation_to_gif
from spatio_flux.library.helpers import get_standard_emitter


def daughter_locations(parent_state, *, gap=1.0, daughter_length=None, daughter_radius=None):
    """
    Compute daughter center locations so they do not overlap, accounting for geometry.
    """
    px, py = parent_state['location']
    angle = float(parent_state.get('angle', 0.0))
    dtype = parent_state.get('type', 'circle')

    if dtype == 'segment':
        Lp = float(parent_state.get('length', 1.0))
        r  = float(parent_state.get('radius', 0.5))
        Ld = float(daughter_length) if daughter_length is not None else max(0.0, 0.5 * Lp)
        rd = float(daughter_radius) if daughter_radius is not None else r

        # Required center spacing to avoid overlap of two capsules
        d_center = Ld + 2.0 * rd + max(0.0, gap)
        hx = 0.5 * d_center * math.cos(angle)
        hy = 0.5 * d_center * math.sin(angle)

        return [[px - hx, py - hy],
                [px + hx, py + hy]]

    else:  # circle or fallback
        r  = float(parent_state.get('radius', 0.5))
        rd = float(daughter_radius) if daughter_radius is not None else r

        # Required center spacing to avoid overlap of two circles
        d_center = 2.0 * rd + max(0.0, gap)
        hx = 0.5 * d_center * math.cos(angle)
        hy = 0.5 * d_center * math.sin(angle)

        return [[px - hx, py - hy],
                [px + hx, py + hy]]

def local_impulse_point_for_shape(shape):
    """Return a random local point on the shape boundary in the body's local coords."""
    if isinstance(shape, pymunk.Circle):
        r = shape.radius
        theta = random.uniform(0, 2 * math.pi)
        return (r * math.cos(theta), r * math.sin(theta))

    if isinstance(shape, pymunk.Segment):
        # segment is from a->b in local coords; choose a random point along it, add small normal offset within radius
        t = random.random()
        ax, ay = shape.a
        bx, by = shape.b
        px, py = ax + t * (bx - ax), ay + t * (by - ay)
        # slight offset toward the “edge” to avoid perfectly central impulses
        nx, ny = -(by - ay), (bx - ax)  # unnormalized normal
        nlen = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nlen, ny / nlen
        return (px + nx * shape.radius, py + ny * shape.radius)

    # fallback: body center
    return (0.0, 0.0)



class PymunkParticleMovement(Process):
    """
    Pymunk-backed version of ParticleMovement.
    """
    config_schema = {
        # Match ParticleMovement
        'bounds': 'tuple[float,float]',             # (x_max, y_max)
        'add_probability': default('float', 0.0),
        'boundary_to_add': default('list[boundary_side]', ['top']),
        'boundary_to_remove': default(
            'list[boundary_side]',
            ['left', 'right', 'top']
        ),

        # Pymunk-specific (mostly optional knobs)
        'substeps': default('integer', 100),
        'gravity': default('float', -9.81),
        'damping_per_second': default('float', 0.98),
        'jitter_per_second': default('float', 1e-2),
        'friction': default('float', 0.8),
        'elasticity': default('float', 0.0),
        'wall_thickness': default('float', 5.0),
        'barriers': default('list[map]', []),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # Domain: (0, x_max) x (0, y_max)
        x_max, y_max = self.config['bounds']
        self.env_size = ((0.0, float(x_max)), (0.0, float(y_max)))

        # Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, float(self.config['gravity']))
        self.substeps = int(self.config['substeps'])
        self.damping_per_second = float(self.config['damping_per_second'])
        self.jitter_per_second = float(self.config['jitter_per_second'])

        self.friction = float(self.config['friction'])
        self.elasticity = float(self.config['elasticity'])
        self.wall_thickness = float(self.config['wall_thickness'])

        # Map: pid -> {'body', 'shape', 'radius', ...}
        self.bodies = {}

        # Build walls based on bounds and boundary_to_remove
        self._build_walls()

        # Optional custom barriers
        for barrier in self.config.get('barriers', []):
            self._add_barrier(barrier)

    # ------------------------------------------------------------------
    # Schema alignment
    # ------------------------------------------------------------------

    def inputs(self):
        # Same as ParticleMovement
        return {'particles': 'map[particle]'}

    def outputs(self):
        # Same as ParticleMovement
        return {'particles': 'map[particle]'}

    def initial_state(self, config=None):
        # Let the upstream composite decide initial particles,
        # or you can seed here if desired.
        return {}

    @staticmethod
    def generate_state(config=None):
        """
        Optional helper for seeding particles, roughly analogous to
        ParticleMovement.generate_state. You can also just ignore this and
        seed particles elsewhere.
        """
        config = config or {}
        bounds = config.get('bounds', (100.0, 100.0))
        n_particles = config.get('n_particles', 10)
        elasticity = config.get('elasticity', 0.0)

        env_size = float(bounds[0])  # assuming square for placement helper
        rng = random.Random(config.get('seed'))

        particles = place_circles(
            rng, env_size, n_particles,
            margin=5.0,
            avoid_overlap=True,
            extra_gap=2.0,
            max_tries=200,
            particle_kwargs=dict(
                elasticity=elasticity,
            ),
        )

        # Convert 'location' -> 'position' to match particle schema
        for pid, p in particles.items():
            p['position'] = p.pop('location')

        return {'particles': particles}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, state, interval):
        """
        Pymunk step that:
        - syncs Pymunk bodies from incoming particles
        - integrates physics
        - emits delta updates in ParticleMovement style:
            {
                'particles': {
                    '_remove': [...],
                    '_add': {pid: full_state, ...},
                    existing_pid: {'position': (dx, dy)},
                    ...
                }
            }
        """
        particles_in = state.get('particles', {}) or {}
        dt_total = float(interval)

        # 1. Sync domain with incoming particles (add/update/remove bodies)
        self._sync_bodies_from_particles(particles_in)

        # 2. Step Pymunk
        n_steps = max(1, int(self.substeps))
        dt = dt_total / n_steps if n_steps > 0 else dt_total
        d_step = self.damping_per_second ** max(dt, 0.0)

        for _ in range(n_steps):
            self.space.damping = d_step
            for body in self.space.bodies:
                self._apply_jitter_force(body, dt)
            self.space.step(dt)

        # 3. Build delta-style output
        (x_min, x_max), (y_min, y_max) = self.env_size
        remove_sides = set(self.config['boundary_to_remove'])

        deltas = {'_remove': [], '_add': {}}

        # Existing particles: compute displacement
        for pid, obj in list(self.bodies.items()):
            body = obj['body']
            x, y = body.position.x, body.position.y

            # Check removal boundaries (if no wall is present on that side)
            kill = (
                ('left' in remove_sides and x < x_min) or
                ('right' in remove_sides and x > x_max) or
                ('bottom' in remove_sides and y < y_min) or
                ('top' in remove_sides and y > y_max)
            )
            if kill:
                deltas['_remove'].append(pid)
                self.space.remove(body, obj['shape'])
                del self.bodies[pid]
                continue

            # If this particle was in the input, give a displacement update
            if pid in particles_in:
                old_x, old_y = particles_in[pid]['position']
                dx = float(x - old_x)
                dy = float(y - old_y)
                deltas[pid] = {'position': (dx, dy)}

        # 4. Births from boundaries (same config semantics as ParticleMovement)
        add_prob = float(self.config['add_probability'])
        if add_prob > 0.0:
            for side in self.config['boundary_to_add']:
                if random.random() < add_prob:
                    pid, pstate = self._spawn_new_particle_at_boundary(side)
                    if pid is not None:
                        deltas['_add'][pid] = pstate

        return {'particles': deltas}

    # ------------------------------------------------------------------
    # Wall + barrier construction
    # ------------------------------------------------------------------

    def _build_walls(self):
        (x_min, x_max), (y_min, y_max) = self.env_size
        t = self.wall_thickness
        remove_sides = set(self.config['boundary_to_remove'])

        def add_segment(a, b):
            seg = pymunk.Segment(self.space.static_body, a, b, t)
            seg.elasticity = self.elasticity
            seg.friction = self.friction
            self.space.add(seg)

        # Only add walls for sides that are NOT removal boundaries
        if 'bottom' not in remove_sides:
            add_segment((x_min, y_min), (x_max, y_min))
        if 'top' not in remove_sides:
            add_segment((x_min, y_max), (x_max, y_max))
        if 'left' not in remove_sides:
            add_segment((x_min, y_min), (x_min, y_max))
        if 'right' not in remove_sides:
            add_segment((x_max, y_min), (x_max, y_max))

    def _add_barrier(self, barrier):
        start_x, start_y = barrier['start']
        end_x, end_y = barrier['end']
        thickness = barrier.get('thickness', self.wall_thickness)

        seg = pymunk.Segment(
            self.space.static_body,
            (start_x, start_y),
            (end_x, end_y),
            thickness,
        )
        seg.elasticity = barrier.get('elasticity', self.elasticity)
        seg.friction = barrier.get('friction', self.friction)
        self.space.add(seg)

    # ------------------------------------------------------------------
    # Sync between particle dicts and Pymunk bodies
    # ------------------------------------------------------------------

    def _sync_bodies_from_particles(self, particles_in):
        """Create/update/remove Pymunk bodies to match incoming particles."""
        incoming_ids = set(particles_in.keys())
        existing_ids = set(self.bodies.keys())

        # Remove stale bodies
        for pid in existing_ids - incoming_ids:
            body = self.bodies[pid]['body']
            shape = self.bodies[pid]['shape']
            self.space.remove(body, shape)
            del self.bodies[pid]

        # Add/update existing
        for pid, attrs in particles_in.items():
            self._upsert_body(pid, attrs)

    def _upsert_body(self, pid, attrs):
        """
        Ensure a Pymunk circle body exists and is in sync with the particle
        state. We treat every particle as a circle here.
        """
        # position: accept either 'position' (ParticleMovement) or 'location' (your utils)
        pos = attrs.get('position', attrs.get('location', (0.0, 0.0)))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        radius = float(attrs.get('radius', 5.0))
        mass = float(attrs.get('mass', 1.0))

        if pid not in self.bodies:
            # create new body+shape
            inertia = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = attrs.get('elasticity', self.elasticity)
            shape.friction = attrs.get('friction', self.friction)
            self.space.add(body, shape)
            self.bodies[pid] = {
                'body': body,
                'shape': shape,
                'radius': radius,
                'mass': mass,
            }
        else:
            obj = self.bodies[pid]
            body = obj['body']
            shape = obj['shape']

            # Update mass/geom if changed
            if abs(obj['radius'] - radius) > 1e-9 or abs(obj['mass'] - mass) > 1e-9:
                # rebuild shape
                self.space.remove(shape)
                inertia = pymunk.moment_for_circle(mass, 0, radius)
                body.mass = mass
                body.moment = inertia
                new_shape = pymunk.Circle(body, radius)
                new_shape.elasticity = attrs.get('elasticity', self.elasticity)
                new_shape.friction = attrs.get('friction', self.friction)
                self.space.add(new_shape)
                obj['shape'] = new_shape
                obj['radius'] = radius
                obj['mass'] = mass

            # Update pose/velocity
            body.position = pos
            body.velocity = (vx, vy)

    # ------------------------------------------------------------------
    # Jitter + births
    # ------------------------------------------------------------------

    def _apply_jitter_force(self, body, dt):
        """
        Small random impulses to avoid perfectly straight motion.
        """
        if not body.shapes:
            return

        shape = next(iter(body.shapes))

        # Simple "random point on circle" jitter
        if isinstance(shape, pymunk.Circle):
            r = shape.radius
            theta = random.uniform(0, 2 * math.pi)
            local_point = (r * math.cos(theta), r * math.sin(theta))
        else:
            local_point = (0.0, 0.0)

        sigma = self.jitter_per_second * math.sqrt(max(dt, 1e-12))
        fx = random.normalvariate(0.0, sigma)
        fy = random.normalvariate(0.0, sigma)
        body.apply_impulse_at_local_point((fx, fy), local_point)

    def _spawn_new_particle_at_boundary(self, side):
        """
        Spawn a new circle near a boundary, return (pid, particle_state)
        in *particle schema* (position, radius, mass, etc.)
        """
        (x_min, x_max), (y_min, y_max) = self.env_size
        margin = 2.0
        radius = 5.0
        mass = 1.0

        if side == 'left':
            x = x_min + margin + radius
            y = random.uniform(y_min + margin, y_max - margin)
        elif side == 'right':
            x = x_max - margin - radius
            y = random.uniform(y_min + margin, y_max - margin)
        elif side == 'bottom':
            x = random.uniform(x_min + margin, x_max - margin)
            y = y_min + margin + radius
        elif side == 'top':
            x = random.uniform(x_min + margin, x_max - margin)
            y = y_max - margin - radius
        else:
            return None, None

        pid = f"p_{uuid.uuid4().hex[:6]}"
        pstate = {
            'id': pid,
            'position': (float(x), float(y)),
            'velocity': (0.0, 0.0),
            'radius': float(radius),
            'mass': float(mass),
            'elasticity': self.elasticity,
        }

        # Also create body immediately so it's present next tick
        self._upsert_body(pid, pstate)
        return pid, pstate




# -------------------------
# Utilities
# -------------------------

def make_id(prefix='id', nhex=6):
    return f"{prefix}_{uuid.uuid4().hex[:nhex]}"

def make_rng(seed=None):
    return random.Random(seed)

# -------------------------
# Mass/geometry conversions
# -------------------------

def circle_mass_from_radius(radius, density):
    # m = ρ π r^2
    return float(density) * math.pi * (float(radius) ** 2)

def circle_radius_from_mass(mass, density):
    # r = sqrt(m / (ρ π))
    return math.sqrt(float(mass) / (float(density) * math.pi))

def capsule_mass_from_length_radius(length, radius, density):
    # Approximate capsule as rectangle length L and diameter 2r:
    # m = ρ * (2r) * L  (ignoring hemispherical ends for simplicity; good for L >> r)
    return float(density) * (2.0 * float(radius)) * float(length)

def capsule_length_from_mass(mass, radius, density):
    # L = m / (ρ * 2r)
    return float(mass) / (float(density) * (2.0 * float(radius)))

# -------------------------
# Primitive builders (single objects)
# - Accept explicit geometry/mass; if one is missing, infer from density.
# - Velocity can be set directly, or drawn from (speed_range, heading).
# -------------------------

def build_particle(
    rng,
    env_size,
    *,
    elasticity=0.0,
    id_prefix='p',
    # position
    x=None, y=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 10.0),
    # geometry / mass (circle)
    radius=None, mass=None, density=0.015
):
    # derive geometry/mass if needed
    if radius is None and mass is None:
        radius = rng.uniform(1.0, 10.0)
    if mass is None:
        mass = circle_mass_from_radius(radius, density)
    if radius is None:
        radius = circle_radius_from_mass(mass, density)

    # position
    if x is None or y is None:
        r = radius
        x = rng.uniform(margin + r, env_size - (margin + r))
        y = rng.uniform(margin + r, env_size - (margin + r))

    # velocity
    if velocity is None:
        speed = rng.uniform(*speed_range)
        theta = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
    else:
        vx, vy = velocity

    return make_id(id_prefix), {
        'type': 'circle',
        'mass': float(mass),
        'radius': float(radius),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

def build_microbe(
    rng,
    env_size,
    *,
    agent_id=None,          # <-- optional explicit id
    elasticity=0.0,
    id_prefix='a',
    # placement & bounds
    x=None, y=None, angle=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 0.4),
    # geometry / mass (capsule segment)
    length=None, radius=None, mass=None, density=0.02,
    length_range=(40.0, 120.0), radius_range=(6.0, 24.0),
):
    if length is None and mass is None:
        length = rng.uniform(*length_range)
    if radius is None:
        radius = rng.uniform(*radius_range)

    if mass is None:
        mass = capsule_mass_from_length_radius(length, radius, density)
    if length is None:
        length = capsule_length_from_mass(mass, radius, density)

    if angle is None:
        angle = rng.uniform(-math.pi, math.pi)

    dx, dy = (length / 2.0) * math.cos(angle), (length / 2.0) * math.sin(angle)
    pad = radius + margin
    if x is None or y is None:
        x = rng.uniform(pad + abs(dx), env_size - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), env_size - (pad + abs(dy)))

    if velocity is None:
        speed = rng.uniform(*speed_range)
        phi = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)
    else:
        vx, vy = velocity

    _id = agent_id or make_id(id_prefix)
    return _id, {
        'id': _id,                 # <-- include id in the object
        'type': 'segment',
        'mass': float(mass),
        'length': float(length),
        'radius': float(radius),
        'angle': float(angle),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

# -------------------------
# Placers (collections)
# -------------------------

def circles_overlap(c1, c2, extra_gap=0.0):
    (x1, y1), r1 = c1['location'], c1['radius']
    (x2, y2), r2 = c2['location'], c2['radius']
    dx, dy = x1 - x2, y1 - y2
    return (dx*dx + dy*dy) < (r1 + r2 + extra_gap) ** 2

def place_circles(
    rng, env_size, n,
    *,
    margin=5.0,
    avoid_overlap=True,
    extra_gap=2.0,
    max_tries=200,
    particle_kwargs=None
):
    particle_kwargs = dict(particle_kwargs or {})
    placed = []
    out = {}
    for _ in range(n):
        if avoid_overlap:
            for _try in range(max_tries):
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                if all(not circles_overlap(cand, prev, extra_gap) for prev in placed):
                    placed.append(cand)
                    out[pid] = cand
                    break
            else:
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                placed.append(cand)
                out[pid] = cand
        else:
            pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
            placed.append(cand)
            out[pid] = cand
    return out

def place_microbes(
    rng, env_size, n,
    *,
    margin=5.0,
    microbe_kwargs=None,
    ids=None,                 # <-- optional list of ids (len == n)
    id_factory=None,          # <-- optional callable -> str
):
    microbe_kwargs = dict(microbe_kwargs or {})
    out = {}
    for i in range(n):
        agent_id = None
        if ids is not None:
            agent_id = ids[i]
        elif id_factory is not None:
            agent_id = id_factory(i)
        aid, obj = build_microbe(rng, env_size, margin=margin, agent_id=agent_id, **microbe_kwargs)
        out[aid] = obj
    return out


# -------------------------
# High-level initializer
# -------------------------

def make_initial_state(
    n_microbes=2,
    n_particles=2,
    env_size=600.0,
    *,
    agents_key='cells',
    particles_key='particles',
    seed=None,
    elasticity=0.0,
    # particle defaults
    particle_radius_range=(1.0, 10.0),
    particle_mass_density=0.015,
    particle_speed_range=(0.0, 10.0),
    # microbe defaults
    microbe_length_range=(40.0, 120.0),
    microbe_radius_range=(6.0, 24.0),
    microbe_mass_density=0.02,
    microbe_speed_range=(0.0, 0.4),
    # placement
    margin=5.0,
    avoid_overlap_circles=True,
    min_gap=2.0,
    max_tries_per_circle=200,
):
    rng = make_rng(seed)

    particles = place_circles(
        rng, env_size, n_particles,
        margin=margin,
        avoid_overlap=avoid_overlap_circles,
        extra_gap=min_gap,
        max_tries=max_tries_per_circle,
        particle_kwargs=dict(
            elasticity=elasticity,
            density=particle_mass_density,
            speed_range=particle_speed_range,
            # you can override radius/mass/velocity here if desired
            # radius=..., mass=..., velocity=(vx, vy), x=..., y=...
        )
    )

    agents = place_microbes(
        rng, env_size, n_microbes,
        margin=margin,
        microbe_kwargs=dict(
            elasticity=elasticity,
            density=microbe_mass_density,
            speed_range=microbe_speed_range,
            length_range=microbe_length_range,
            radius_range=microbe_radius_range,
        ),
        # e.g., fixed IDs:
        # ids=[f"a_seed{i}" for i in range(n_microbes)],
        # or dynamic:
        # id_factory=lambda i: f"a_{uuid.uuid4().hex[:6]}",
    )

    return {agents_key: agents, particles_key: particles}


def run_pymunk_particles():

    # run simulation
    interval = 2000
    config = {
        # 'env_size': 600,
        'gravity': 0,  # -9.81,
        'elasticity': 0.1,
        'bounds': (600.0, 600.0),
    }

    processes = {
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkParticleMovement',
            'config': config,
            'inputs': {
                # 'agents': ['cells'],
                'particles': ['particles'],
            },
            'outputs': {
                # 'agents': ['cells'],
                'particles': ['particles'],
            }
        }
    }

    initial_state = make_initial_state(
        n_particles=100,
        env_size=600,
        elasticity=0.0,
        particle_radius_range=(1, 8),
        microbe_length_range=(50, 100),
        microbe_radius_range=(10, 15)
    )

    # grow and divide schema
    cell_schema = {}
    # cell_schema = get_grow_divide_schema(
    #     core=core,
    #     config={
    #         'agents_key': 'cells',
    #         'rate': 0.02,
    #         'threshold': 80.0,
    #         'mutate': True,
    #     }
    # )

    # complete document
    document = {
        'state': {
            **initial_state,
            **processes,
            # **{'emitter': emitter_state},
        },
        'composition': cell_schema,
    }

    if 'emitter' not in document['state']:
        state_keys = list(document['state'].keys())
        document['state']['emitter'] = get_standard_emitter(state_keys=state_keys)

    # create the composite simulation
    from spatio_flux import core_import
    core = core_import()
    sim = Composite(document, core=core)

    # Save composition JSON
    name = 'pymunk_growth_division'
    sim.save(filename=f'{name}.json', outdir='out')

    # run the simulation
    sim.run(interval)
    results = gather_emitter_results(sim)[('emitter',)]

    print(f'Simulation completed with {len(results)} steps.')

    # make video
    simulation_to_gif(results,
                      filename='circlesandsegments',
                      config=config,
                      color_by_phylogeny=True,
                      # skip_frames=10
                      )


if __name__ == '__main__':
    run_pymunk_particles()
