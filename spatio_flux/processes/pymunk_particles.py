"""
TODO: use an actual grow/divide process for demo
"""
import random
import math
import uuid

import pymunk

from process_bigraph import Composite, gather_emitter_results, Process
from process_bigraph.emitter import emitter_from_wires
from spatio_flux.plots.multibody_plots import pymunk_simulation_to_gif
from spatio_flux.library.tools import get_standard_emitter


def daughter_positions(parent_state, *, gap=1.0, daughter_length=None, daughter_radius=None):
    """
    Compute daughter center position so they do not overlap, accounting for geometry.
    """
    px, py = parent_state['position']
    angle = float(parent_state.get('angle', 0.0))
    dtype = parent_state.get('shape', 'circle')

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
    config_schema = {
        'bounds':         {'_type': 'tuple[float,float]',       '_default': (500.0, 500.0)},
        'n_bins':         {'_type': 'tuple[integer,integer]',   '_default': (1, 1)},
        'diffusion_rate': {'_type': 'float',                    '_default': 1e-1},
        'advection_rate': {'_type': 'tuple[float,float]',       '_default': (0.0, 0.0)},
        'substeps':            {'_type': 'integer', '_default': 100},
        'damping_per_second':  {'_type': 'float',   '_default': 0.98},
        'gravity':             {'_type': 'float',   '_default': -9.81},
        'friction':            {'_type': 'float',   '_default': 0.8},
        'elasticity':          {'_type': 'float',   '_default': 0.0},
        'jitter_per_second':   {'_type': 'float',   '_default': 1e-2},  # base impulse std
        'barriers':            'list[map]',
        'wall_thickness':      {'_type': 'float',   '_default': 100.0},

        # Newborn particle configuration
        'new_particle_radius_range': {'_type': 'tuple[float,float]', '_default': (0.5, 2.0)},
        'new_particle_mass_range': {'_type': 'tuple[float,float]', '_default': (1.0, 1.0)},
        'new_particle_density': {'_type': 'float', '_default': 0.015},
        'new_particle_speed_range': {'_type': 'tuple[float,float]', '_default': (0.0, 0.0)},
    }

    def initialize(self, config=None):
        # convenience
        self.bounds = self.config['bounds']  # (x_max, y_max)
        x_max, y_max = float(self.bounds[0]), float(self.bounds[1])
        self.env_size = ((0.0, x_max), (0.0, y_max))

        self.substeps = int(self.config.get('substeps', 100))
        self.damping_per_second = float(self.config.get('damping_per_second', 0.98))
        self.jitter_per_second = float(self.config.get('jitter_per_second', 1e-2))

        # align with ParticleMovement knobs
        self.diffusion_rate = float(self.config.get('diffusion_rate', 1e-1))
        self.advection_rate = tuple(self.config.get('advection_rate', (0.0, 0.0)))

        # Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, float(self.config['gravity']))

        self.friction = float(self.config['friction'])
        self.elasticity = float(self.config['elasticity'])
        self.wall_thickness = float(self.config.get('wall_thickness', 100.0))

        # internal registry
        self.agents = {}

        # build all walls
        self._build_walls()

        # add custom barriers
        for barrier in self.config.get('barriers', []):
            self.add_barrier(barrier)

    # ------------------------------------------------------------------

    def _build_walls(self):
        (x_min, x_max), (y_min, y_max) = self.env_size
        t = self.wall_thickness
        body = self.space.static_body

        def add_segment(a, b):
            seg = pymunk.Segment(body, a, b, t)
            seg.elasticity = self.elasticity
            seg.friction = self.friction
            self.space.add(seg)

        # Bottom
        add_segment((x_min - t, y_min - t), (x_max + t, y_min - t))
        # Right
        add_segment((x_max + t, y_min - t), (x_max + t, y_max + t))
        # Top
        add_segment((x_max + t, y_max + t), (x_min - t, y_max + t))
        # Left
        add_segment((x_min - t, y_max + t), (x_min - t, y_min - t))

    def add_barrier(self, barrier):
        start_x, start_y = barrier['start']
        end_x, end_y = barrier['end']
        start = pymunk.Vec2d(start_x, start_y)
        end = pymunk.Vec2d(end_x, end_y)
        thickness = barrier.get('thickness', 1)
        segment = pymunk.Segment(self.space.static_body, start, end, thickness)
        segment.elasticity = barrier.get('elasticity', 1.0)
        segment.friction = barrier.get('friction', 0.5)
        self.space.add(segment)

    def inputs(self):
        return {
            'particles': 'map[complex_particle]',
        }

    def outputs(self):
        return {
            'particles': 'map[complex_particle]',
        }

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, inputs, interval):
        """
        Sync particles, step the physics, then emit per-particle *deltas* in position.

        If the particle had a previous position in inputs:
            position = (dx, dy)
        Otherwise:
            position = (x, y)  # absolute
        """
        particles_in = inputs.get('particles', {}) or {}
        particle_ids = set(particles_in)

        # ------- sync bodies -------
        existing_ids = set(self.agents)
        new_ids = particle_ids

        # remove stale (removed by upstream state, not by boundaries)
        for dead_id in existing_ids - new_ids:
            body = self.agents[dead_id]['body']
            shape_instance = self.agents[dead_id]['shape_instance']
            self.space.remove(body, shape_instance)
            del self.agents[dead_id]

        # add/update particles
        for _id, attrs in particles_in.items():
            self.manage_object(_id, attrs)

        # ------- integrate -------
        n_steps = max(1, int(self.config.get('substeps', 100)))
        dt = float(interval) / n_steps
        d_step = self.damping_per_second ** max(dt, 0.0)

        for _ in range(n_steps):
            self.space.damping = d_step
            for body in self.space.bodies:
                self.apply_jitter_force(body, dt)
            self.space.step(dt)

        # ------- emit deltas -------
        particles_out = {}

        for _id, obj in self.agents.items():
            if _id not in particles_in:
                continue

            body = obj['body']
            new_x, new_y = float(body.position.x), float(body.position.y)

            old_state = particles_in.get(_id, {})
            old_pos = old_state.get('position')

            if old_pos is not None:
                old_x, old_y = old_pos
                dx = float(new_x - old_x)
                dy = float(new_y - old_y)
                pos_value = (dx, dy)
            else:
                pos_value = (new_x, new_y)

            if obj['shape'] == 'circle':
                rec = {
                    'shape': obj['shape'],
                    'position': pos_value,
                    'velocity': (float(body.velocity.x), float(body.velocity.y)),
                    'inertia': float(body.moment),
                }
            else:  # 'segment'
                rec = {
                    'shape': obj['shape'],
                    'position': pos_value,
                    'velocity': (float(body.velocity.x), float(body.velocity.y)),
                    'inertia': float(body.moment),
                    'angle': float(body.angle),
                }

            particles_out[_id] = rec

        return {'particles': particles_out}

    # ------------------------------------------------------------------
    # Jitter (diffusion-like) scaled by diffusion_rate
    # ------------------------------------------------------------------

    def apply_jitter_force(self, body, dt):
        shape = next(iter(body.shapes)) if body.shapes else None
        if not shape:
            return

        # simple "random point on circle" jitter
        if isinstance(shape, pymunk.Circle):
            r = shape.radius
            theta = random.uniform(0, 2 * math.pi)
            local_point = (r * math.cos(theta), r * math.sin(theta))
        else:
            local_point = (0.0, 0.0)

        # scale jitter with diffusion_rate
        base_sigma = self.jitter_per_second * math.sqrt(max(dt, 1e-12))
        sigma = base_sigma * math.sqrt(max(self.diffusion_rate, 1e-12))

        fx = random.normalvariate(0.0, sigma)
        fy = random.normalvariate(0.0, sigma)
        body.apply_impulse_at_local_point((fx, fy), local_point)

    def update_bodies(self, agents):
        existing_ids = set(self.agents.keys())
        new_ids = set(agents.keys())

        # Remove objects not in the new state
        for agent_id in existing_ids - new_ids:
            body = self.agents[agent_id]['body']
            shape_instance = self.agents[agent_id]['shape_instance']
            self.space.remove(body, shape_instance)
            del self.agents[agent_id]

        # Add or update existing objects
        for agent_id, attrs in agents.items():
            self.manage_object(agent_id, attrs)

    def manage_object(self, agent_id, attrs):
        agent = self.agents.get(agent_id)
        if not agent:
            self.create_new_object(agent_id, attrs)
            return

        body = agent['body']
        old_shape = agent['shape_instance']
        old_type = agent['shape']

        # robust defaults
        shape_type = attrs.get('shape', old_type or 'circle')
        mass = float(attrs.get('mass', body.mass))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        body.mass = mass
        body.position = pymunk.Vec2d(*attrs.get('position', (body.position.x, body.position.y)))
        body.velocity = pymunk.Vec2d(vx, vy)

        needs_rebuild = False
        if shape_type == 'circle':
            radius = float(attrs['radius'])
            if not isinstance(old_shape, pymunk.Circle) or abs(old_shape.radius - radius) > 1e-9 or old_type != 'circle':
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Circle(body, radius)
                body.moment = pymunk.moment_for_circle(mass, 0, radius)

        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])

            start = pymunk.Vec2d(-length / 2, 0).rotated(angle)
            end = pymunk.Vec2d(length / 2, 0).rotated(angle)

            if not isinstance(old_shape, pymunk.Segment) or abs(old_shape.radius - radius) > 1e-9 or old_type != 'segment':
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Segment(body, start, end, radius)
                body.moment = pymunk.moment_for_segment(mass, start, end, radius)

            body.angle = angle
            body.length = length
            body.width = radius * 2

        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # swap shape if needed
        if needs_rebuild:
            elasticity = attrs.get('elasticity', self.config['elasticity'])
            friction = attrs.get('friction', self.config['friction'])
            new_shape.elasticity = elasticity
            new_shape.friction = friction

            self.space.remove(old_shape)
            self.space.add(new_shape)
            agent['shape_instance'] = new_shape

        # keep dict in sync
        agent['shape'] = shape_type
        agent['mass'] = mass
        if shape_type == 'circle':
            agent['radius'] = radius
            agent['angle'] = None
            agent['length'] = None
        else:
            agent['radius'] = radius
            agent['angle'] = body.angle
            agent['length'] = body.length

    def create_new_object(self, agent_id, attrs):
        shape_type = attrs.get('shape', 'circle')
        mass = float(attrs.get('mass', 1.0))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        pos = attrs.get('position', (0.0, 0.0))

        if shape_type == 'circle':
            radius = float(attrs['radius'])
            inertia = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            shape = pymunk.Circle(body, radius)
            angle = None
            length = None

        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])

            start = (-length / 2, 0)
            end = (length / 2, 0)
            inertia = pymunk.moment_for_segment(mass, start, end, radius)

            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            body.angle = angle
            body.length = length
            body.width = radius * 2

            shape = pymunk.Segment(body, start, end, radius)

        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        shape.elasticity = attrs.get('elasticity', self.config['elasticity'])
        shape.friction = attrs.get('friction', self.config['friction'])

        self.space.add(body, shape)
        self.agents[agent_id] = {
            'body': body,
            'shape_instance': shape,
            'shape': shape_type,
            'mass': mass,
            'radius': radius,
            'angle': angle,
            'length': length,
        }

    def get_state_update(self):
        state = {}
        for agent_id, obj in self.agents.items():
            if obj['shape'] == 'circle':
                state[agent_id] = {
                    'shape': obj['shape'],
                    'position': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'inertia': obj['body'].moment,
                }
            elif obj['shape'] == 'segment':
                state[agent_id] = {
                    'shape': obj['shape'],
                    'position': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'inertia': obj['body'].moment,
                    'angle': obj['body'].angle,
                }
        return state



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
    bounds,
    *,
    elasticity=0.0,
    id_prefix='p',
    # position
    x=None, y=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 0.0),
    # geometry / mass (circle)
    radius=None, mass=None, density=0.015,
    radius_range=(1.0, 10.0),
    mass_range=None,
):
    """
    radius_range and mass_range are mutually exclusive.
    If mass_range is provided, it takes priority.
    """

    # ---------------------------
    # Validate mutually exclusive inputs
    # ---------------------------
    if mass_range is not None and radius_range is not None:
        raise ValueError(
            "Only one of radius_range or mass_range may be provided. "
            "To use mass_range, set radius_range=None."
        )

    # ---------------------------
    # Derive geometry/mass
    # ---------------------------

    # 1. mass_range has priority
    if mass is None and mass_range is not None:
        mass = rng.uniform(*mass_range)
        radius = circle_radius_from_mass(mass, density)

    # 2. Otherwise fall back to radius_range
    elif radius is None and mass is None:
        # Use normal radius_range only if mass_range was not provided
        radius = rng.uniform(*radius_range)
        mass = circle_mass_from_radius(radius, density)

    # 3. If radius given but mass missing
    elif radius is not None and mass is None:
        mass = circle_mass_from_radius(radius, density)

    # 4. If mass given but radius missing
    elif mass is not None and radius is None:
        radius = circle_radius_from_mass(mass, density)

    # At this point both radius and mass must be defined
    # ---------------------------
    # Position sampling
    # ---------------------------
    if x is None or y is None:
        r = radius
        x = rng.uniform(margin + r, bounds[0] - (margin + r))
        y = rng.uniform(margin + r, bounds[1] - (margin + r))

    # ---------------------------
    # Velocity sampling
    # ---------------------------
    if velocity is None:
        speed = rng.uniform(*speed_range)
        theta = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
    else:
        vx, vy = velocity

    # ---------------------------
    # Return particle
    # ---------------------------
    return make_id(id_prefix), {
        'shape': 'circle',
        'mass': float(mass),
        'radius': float(radius),
        'position': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

def build_microbe(
    rng,
    bounds,
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
        x = rng.uniform(pad + abs(dx), bounds[0] - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), bounds[1] - (pad + abs(dy)))

    if velocity is None:
        speed = rng.uniform(*speed_range)
        phi = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)
    else:
        vx, vy = velocity

    _id = agent_id or make_id(id_prefix)
    return _id, {
        'id': _id,                 # <-- include id in the object
        'shape': 'segment',
        'mass': float(mass),
        'length': float(length),
        'radius': float(radius),
        'angle': float(angle),
        'position': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

# -------------------------
# Placers (collections)
# -------------------------

def circles_overlap(c1, c2, extra_gap=0.0):
    (x1, y1), r1 = c1['position'], c1['radius']
    (x2, y2), r2 = c2['position'], c2['radius']
    dx, dy = x1 - x2, y1 - y2
    return (dx*dx + dy*dy) < (r1 + r2 + extra_gap) ** 2

def place_circles(
    rng, bounds, n,
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
                pid, cand = build_particle(rng, bounds, margin=margin, **particle_kwargs)
                if all(not circles_overlap(cand, prev, extra_gap) for prev in placed):
                    placed.append(cand)
                    out[pid] = cand
                    break
            else:
                pid, cand = build_particle(rng, bounds, margin=margin, **particle_kwargs)
                placed.append(cand)
                out[pid] = cand
        else:
            pid, cand = build_particle(rng, bounds, margin=margin, **particle_kwargs)
            placed.append(cand)
            out[pid] = cand
    return out

def place_microbes(
    rng, bounds, n,
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
        aid, obj = build_microbe(rng, bounds, margin=margin, agent_id=agent_id, **microbe_kwargs)
        out[aid] = obj
    return out


# -------------------------
# High-level initializer
# -------------------------

def get_newtonian_particles_state(
    n_particles=2,
    bounds=(600.0, 600.0),
    *,
    seed=None,
    elasticity=0.0,
    # particle defaults
    particle_radius_range=(0.5, 2.0),
    particle_mass_range=None,
    particle_mass_density=0.015,
    particle_speed_range=(0.0, 0.0),
    # placement
    margin=5.0,
    avoid_overlap_circles=True,
    min_gap=2.0,
    max_tries_per_circle=200,
):
    """
     particle_radius_range and particle_mass_range are mutually exclusive.
     If particle_mass_range is provided, it takes priority.
     """

    rng = make_rng(seed)

    particles = place_circles(
        rng, bounds, n_particles,
        margin=margin,
        avoid_overlap=avoid_overlap_circles,
        extra_gap=min_gap,
        max_tries=max_tries_per_circle,
        particle_kwargs=dict(
            elasticity=elasticity,
            density=particle_mass_density,
            speed_range=particle_speed_range,
            radius_range=particle_radius_range,
            mass_range=particle_mass_range,
            # you can override radius/mass/velocity here if desired
            # radius=..., mass=..., velocity=(vx, vy), x=..., y=...
        )
    )

    return particles


def run_pymunk_particles():

    # run simulation
    interval = 100
    config = {
        'gravity': -0.2, #-9.81,
        'elasticity': 0.1,
        'bounds': (100.0, 300.0),
        'boundary_to_remove': [], #['right', 'left'],
        'add_probability': 0.3,
        'new_particle_radius_range': (0.5, 2.5),
        'jitter_per_second': 0.5,
        'damping_per_second': .998,
    }
    n_particles = 200

    processes = {
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkParticleMovement',
            'config': config,
            'inputs': {
                'particles': ['particles'],
            },
            'outputs': {
                'particles': ['particles'],
            }
        }
    }

    initial_state =  {'particles': get_newtonian_particles_state(
        n_particles=n_particles,
        bounds=config['bounds'],
        particle_radius_range=config['new_particle_radius_range'],
    )}

    # complete document
    document = {
        'state': {
            **initial_state,
            **processes,
        },
    }

    if 'emitter' not in document['state']:
        state_keys = list(document['state'].keys())
        document['state']['emitter'] = get_standard_emitter(state_keys=state_keys)

    # create the composite simulation
    from spatio_flux import build_core
    core = build_core()
    sim = Composite(document, core=core)

    # Save composition JSON
    name = 'pymunk_particles'
    sim.save(filename=f'{name}.json', outdir='out')

    # run the simulation
    sim.run(interval)
    results = gather_emitter_results(sim)[('emitter',)]

    print(f'Simulation completed with {len(results)} steps.')

    # make video
    pymunk_simulation_to_gif(results,
                             filename='particles_falling',
                             config=config,
                             color_by_phylogeny=True,
                             agents_key='particles'
                             # skip_frames=10
                             )


if __name__ == '__main__':
    run_pymunk_particles()