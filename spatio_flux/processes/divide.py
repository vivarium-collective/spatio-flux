import base64
import uuid
import numpy as np
from process_bigraph import Process, default


DIVISION_MASS_THRESHOLD = 5.0


def _short_id(length=6):
    raw = uuid.uuid4().bytes[:length]
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')

class ParticleDivision(Process):
    """
    Stand-alone division process:
      - Tracks particle 'mass' only.
      - If mass >= division_mass_threshold, parent is removed and two children
        are created with half mass each, placed near the parent's position.

    No movement, no environment/fields, no boundary handling.
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
        cid = _short_id()
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
