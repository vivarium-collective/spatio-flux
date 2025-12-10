import numpy as np
from scipy.ndimage import convolve
from process_bigraph import Process

# 5-point Laplacian (for dx == dy)
LAPLACIAN_2D = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=float)


class DiffusionAdvection(Process):
    """
    Explicit finite-difference diffusion–advection in 2D, with
    CFL-based substepping for numerical stability.

    - Diffusion: 5-point Laplacian
    - Advection: first-order upwind (directional)
    """

    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        # physical coefficients
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},  # D in L^2 / T
        'diffusion_coeffs': 'map[float]',  # per-species D overrides
        'advection_coeffs': 'map[tuple[float,float]]',  # per-species (vx, vy) in L / T

        # time-stepping / stability controls
        'max_dt': {'_type': 'float', '_default': 1e-1},     # upper bound on dt
        'cfl_adv': {'_type': 'float', '_default': 0.5},     # CFL factor for advection
        'clip_negative': {'_type': 'boolean', '_default': True},
    }

    def initialize(self, config):

        # grid geometry
        nx, ny = self.config['n_bins']
        xmax, ymax = self.config['bounds']
        self.dx = xmax / nx
        self.dy = ymax / ny

        # diffusion coefficients
        self.D_default = self.config['default_diffusion_rate']
        self.D_species = dict(self.config.get('diffusion_coeffs', {}))

        # advection coefficients
        self.v_species = dict(self.config.get('advection_coeffs', {}))

        self.max_dt = self.config['max_dt']
        self.cfl_adv = self.config['cfl_adv']
        self.clip_negative = self.config['clip_negative']

        # Precompute Laplacian scaling assuming dx == dy.
        # If dx != dy, you’d want a more general stencil.
        if not np.isclose(self.dx, self.dy):
            raise ValueError(
                "Current DiffusionAdvection assumes square cells (dx == dy). "
                "Got dx = {:.3g}, dy = {:.3g}".format(self.dx, self.dy)
            )
        self.dx2 = self.dx * self.dx
        self.laplacian_kernel = LAPLACIAN_2D / self.dx2

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                },
            }
        }

    def outputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                },
            }
        }

    # --- public update -----------------------------------------------------

    def update(self, state, interval):
        fields_in = state['fields']
        updated = {}

        # compute global stability constraints from all species present
        dt_stable = self._compute_stable_dt(fields_in)
        n_steps = max(1, int(np.ceil(interval / dt_stable)))
        dt = interval / n_steps

        for species, field in fields_in.items():
            D = self.D_species.get(species, self.D_default)
            vx, vy = self.v_species.get(species, (0.0, 0.0))

            state = field.astype(float, copy=True)
            for _ in range(n_steps):
                state = self._step_diffuse_advect(state, dt, D, vx, vy)

            updated[species] = state - field  # delta for this interval

        return {'fields': updated}

    # --- stability ---------------------------------------------------------

    def _compute_stable_dt(self, fields):
        """
        Compute stable dt from diffusion + advection CFL conditions.
        """
        # diffusion: dt <= 1 / (4 D_max (1/dx^2 + 1/dy^2))
        D_values = [self.D_species.get(s, self.D_default) for s in fields.keys()]
        D_max = max(D_values) if D_values else self.D_default
        if D_max > 0.0:
            dt_diff = 1.0 / (4.0 * D_max * (1.0 / self.dx2 + 1.0 / self.dy**2))
        else:
            dt_diff = np.inf

        # advection CFL: dt <= cfl * min(dx / |vx|max, dy / |vy|max)
        vx_vals = []
        vy_vals = []
        for s in fields.keys():
            vx, vy = self.v_species.get(s, (0.0, 0.0))
            vx_vals.append(abs(vx))
            vy_vals.append(abs(vy))

        vx_max = max(vx_vals) if vx_vals else 0.0
        vy_max = max(vy_vals) if vy_vals else 0.0

        if vx_max > 0.0:
            dt_vx = self.cfl_adv * self.dx / vx_max
        else:
            dt_vx = np.inf

        if vy_max > 0.0:
            dt_vy = self.cfl_adv * self.dy / vy_max
        else:
            dt_vy = np.inf

        dt_adv = min(dt_vx, dt_vy)

        dt_stable = min(dt_diff, dt_adv, self.max_dt)
        if not np.isfinite(dt_stable) or dt_stable <= 0.0:
            dt_stable = self.max_dt

        return dt_stable

    # --- single substep ----------------------------------------------------

    def _step_diffuse_advect(self, state, dt, D, vx, vy):
        """
        One explicit Euler step with diffusion + upwind advection.
        """
        # diffusion term
        if D != 0.0:
            lap = convolve(state, self.laplacian_kernel, mode='reflect')
            diff_term = D * lap
        else:
            diff_term = 0.0

        # advection term (upwind)
        adv_term = 0.0
        if vx != 0.0 or vy != 0.0:
            adv_term = self._upwind_advection(state, vx, vy)

        new_state = state + dt * (diff_term - adv_term)  # note minus: ∂C/∂t + v·∇C

        if self.clip_negative:
            np.maximum(new_state, 0.0, out=new_state)

        return new_state

    # --- upwind advection --------------------------------------------------

    def _upwind_advection(self, state, vx, vy):
        """
        First-order upwind discretization of v · ∇state.
        Returns v·∇C (not multiplied by dt).
        """
        # shift helpers
        roll_xp = np.roll(state, -1, axis=0)
        roll_xm = np.roll(state,  1, axis=0)
        roll_yp = np.roll(state, -1, axis=1)
        roll_ym = np.roll(state,  1, axis=1)

        # dC/dx (upwind)
        if vx > 0:
            dCdx = (state - roll_xm) / self.dx
        elif vx < 0:
            dCdx = (roll_xp - state) / self.dx
        else:
            dCdx = 0.0

        # dC/dy (upwind)
        if vy > 0:
            dCdy = (state - roll_ym) / self.dy
        elif vy < 0:
            dCdy = (roll_yp - state) / self.dy
        else:
            dCdy = 0.0

        return vx * dCdx + vy * dCdy
