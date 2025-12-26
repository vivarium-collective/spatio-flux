import numpy as np
from scipy.ndimage import convolve
from process_bigraph import Process

LAPLACIAN_2D = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=float)


class DiffusionAdvection(Process):
    """
    Explicit finite-difference diffusion–advection in 2D (ny, nx grid).

    Array convention:
      - state has shape (ny, nx) == (rows, cols)
      - x corresponds to columns (axis=1)
      - y corresponds to rows    (axis=0)
    """

    config_schema = {
        # IMPORTANT: n_bins is (ny, nx) to match numpy array shape (rows, cols)
        'n_bins': 'tuple[integer{1},integer{1}]',
        'bounds': 'tuple[float{1.0},float{1.0}]',  # (xmax, ymax)

        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'diffusion_coeffs': 'map[float]',
        'advection_coeffs': 'map[tuple[float,float]]',  # (vx, vy)

        'max_dt': {'_type': 'float', '_default': 1e-1},
        'cfl_adv': {'_type': 'float', '_default': 0.5},
        'clip_negative': {'_type': 'boolean', '_default': True},
    }

    def initialize(self, config):
        # grid geometry (config is (nx, ny))
        nx, ny = self.config['n_bins']
        xmax, ymax = self.config['bounds']

        self.nx = int(nx)  # columns (x)
        self.ny = int(ny)  # rows    (y)

        xmax = float(xmax)
        ymax = float(ymax)

        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"n_bins must be positive, got (nx, ny)=({self.nx}, {self.ny})")
        if xmax <= 0.0 or ymax <= 0.0:
            raise ValueError(f"bounds must be positive, got (xmax, ymax)=({xmax}, {ymax})")

        # cell sizes
        self.dx = xmax / self.nx
        self.dy = ymax / self.ny

        # diffusion coefficients
        self.D_default = float(self.config['default_diffusion_rate'])
        self.D_species = dict(self.config.get('diffusion_coeffs', {}))

        # advection coefficients
        self.v_species = dict(self.config.get('advection_coeffs', {}))

        # stability / stepping
        self.max_dt = float(self.config['max_dt'])
        self.cfl_adv = float(self.config['cfl_adv'])
        self.clip_negative = bool(self.config['clip_negative'])

        # laplacian scaling (your stencil assumes square cells)
        if not np.isclose(self.dx, self.dy):
            raise ValueError(
                "Current DiffusionAdvection assumes square cells (dx == dy). "
                f"Got dx = {self.dx:.6g}, dy = {self.dy:.6g}. "
                "If you want rectangular cells, we need a dx/dy-aware Laplacian stencil."
            )

        self.dx2 = self.dx * self.dx
        self.laplacian_kernel = LAPLACIAN_2D / self.dx2

        # handy: expected numpy array shape for fields
        self.field_shape = (self.ny, self.nx)  # (rows, cols) == (y_bins, x_bins)

    def inputs(self):
        return {
            'substrate_fields': 'substrate_fields',
        }

    def outputs(self):
        return self.inputs()

    def update(self, state, interval):
        fields_in = state['substrate_fields']
        updated = {}

        dt_stable = self._compute_stable_dt(fields_in)
        n_steps = max(1, int(np.ceil(interval / dt_stable)))
        dt = float(interval) / n_steps

        for species, field in fields_in.items():
            D = float(self.D_species.get(species, self.D_default))
            vx, vy = self.v_species.get(species, (0.0, 0.0))
            vx = float(vx)
            vy = float(vy)

            cur = field.astype(float, copy=True)
            for _ in range(n_steps):
                cur = self._step_diffuse_advect(cur, dt, D, vx, vy)

            updated[species] = cur - field  # delta

        return {'substrate_fields': updated}

    def _compute_stable_dt(self, fields):
        D_values = [float(self.D_species.get(s, self.D_default)) for s in fields.keys()]
        D_max = max(D_values) if D_values else self.D_default

        if D_max > 0.0:
            dt_diff = 1.0 / (4.0 * D_max * (1.0 / self.dx2 + 1.0 / (self.dy * self.dy)))
        else:
            dt_diff = np.inf

        vx_max = 0.0
        vy_max = 0.0
        for s in fields.keys():
            vx, vy = self.v_species.get(s, (0.0, 0.0))
            vx_max = max(vx_max, abs(float(vx)))
            vy_max = max(vy_max, abs(float(vy)))

        dt_vx = self.cfl_adv * self.dx / vx_max if vx_max > 0.0 else np.inf
        dt_vy = self.cfl_adv * self.dy / vy_max if vy_max > 0.0 else np.inf

        dt_stable = min(dt_diff, dt_vx, dt_vy, self.max_dt)
        if not np.isfinite(dt_stable) or dt_stable <= 0.0:
            dt_stable = self.max_dt

        return float(dt_stable)

    def _step_diffuse_advect(self, state, dt, D, vx, vy):
        if D != 0.0:
            lap = convolve(state, self.laplacian_kernel, mode='reflect')
            diff_term = D * lap
        else:
            diff_term = 0.0

        adv_term = 0.0
        if vx != 0.0 or vy != 0.0:
            adv_term = self._upwind_advection(state, vx, vy)

        new_state = state + dt * (diff_term - adv_term)

        if self.clip_negative:
            np.maximum(new_state, 0.0, out=new_state)

        return new_state

    def _upwind_advection(self, state, vx, vy):
        """
        v · ∇C with:
          - x along axis=1 (columns)
          - y along axis=0 (rows)
        """
        # x-neighbors (columns)
        roll_xp = np.roll(state, -1, axis=1)  # x+1
        roll_xm = np.roll(state,  1, axis=1)  # x-1

        # y-neighbors (rows)
        roll_yp = np.roll(state, -1, axis=0)  # y+1
        roll_ym = np.roll(state,  1, axis=0)  # y-1

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
