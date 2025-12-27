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

    Boundary conditions are implemented via 1-cell ghost layers,
    so diffusion and advection both respect them.
    """

    config_schema = {
        # IMPORTANT: n_bins is (ny, nx) to match numpy array shape (rows, cols)
        'n_bins': 'tuple[integer{1},integer{1}]',
        'bounds': 'tuple[float{1.0},float{1.0}]',  # (xmax, ymax)

        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'diffusion_coeffs': 'map[float]',
        'advection_coeffs': 'map[tuple[float,float]]',  # (vx, vy)

        # --- boundary conditions ---
        # You can set global defaults and override per species.
        #
        # Format:
        # boundary_conditions = {
        #   "default": {
        #      "x": {"type": "periodic"},                  # shorthand for left/right periodic
        #      "y": {"type": "neumann"},                   # shorthand for bottom/top neumann
        #      # OR explicitly:
        #      "left":   {"type": "dirichlet", "value": 1.0},
        #      "right":  {"type": "neumann"},
        #      "bottom": {"type": "dirichlet", "value": 0.0},
        #      "top":    {"type": "outflow"},
        #   },
        #   "glucose": {"left": {"type":"dirichlet","value":2.0}, "right":{"type":"neumann"}, ...},
        # }
        #
        'boundary_conditions': {'_type': 'map', '_default': {}},

        'max_dt': {'_type': 'float', '_default': 1e-1},
        'cfl_adv': {'_type': 'float', '_default': 0.5},
        'clip_negative': {'_type': 'boolean', '_default': True},
    }

    # -------------------------
    # Initialization
    # -------------------------
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

        # diffusion/advection coefficients
        self.D_default = float(self.config['default_diffusion_rate'])
        self.D_species = dict(self.config.get('diffusion_coeffs', {}))
        self.v_species = dict(self.config.get('advection_coeffs', {}))

        # stability / stepping
        self.max_dt = float(self.config['max_dt'])
        self.cfl_adv = float(self.config['cfl_adv'])
        self.clip_negative = bool(self.config['clip_negative'])

        # laplacian scaling (stencil assumes square cells)
        if not np.isclose(self.dx, self.dy):
            raise ValueError(
                "Current DiffusionAdvection assumes square cells (dx == dy). "
                f"Got dx = {self.dx:.6g}, dy = {self.dy:.6g}."
            )

        self.dx2 = self.dx * self.dx
        self.field_shape = (self.ny, self.nx)

        # --- boundary condition specs ---
        bc_all = dict(self.config.get('boundary_conditions', {}) or {})
        self.bc_default = self._normalize_bc_spec(bc_all.get('default', {}))
        self.bc_species = {k: self._normalize_bc_spec(v) for k, v in bc_all.items() if k != 'default'}

        # sanity: periodic must be paired per axis
        self._validate_periodic_pairing(self.bc_default)
        for sp, spec in self.bc_species.items():
            self._validate_periodic_pairing(spec)

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

    # -------------------------
    # Main update
    # -------------------------
    def update(self, state, interval):
        fields_in = state['fields']
        updated = {}

        dt_stable = self._compute_stable_dt(fields_in)
        n_steps = max(1, int(np.ceil(interval / dt_stable)))
        dt = float(interval) / n_steps

        for species, field in fields_in.items():
            D = float(self.D_species.get(species, self.D_default))
            vx, vy = self.v_species.get(species, (0.0, 0.0))
            vx = float(vx)
            vy = float(vy)

            bc = self._bc_for_species(species)

            cur = np.asarray(field, dtype=float).copy()
            for _ in range(n_steps):
                cur = self._step_diffuse_advect(cur, dt, D, vx, vy, bc)

            updated[species] = cur - field  # delta

        return {'fields': updated}

    # -------------------------
    # Stability dt
    # -------------------------
    def _compute_stable_dt(self, fields):
        D_values = [float(self.D_species.get(s, self.D_default)) for s in fields.keys()]
        D_max = max(D_values) if D_values else self.D_default

        if D_max > 0.0:
            # for square grid: dt <= dx^2/(4D) (2D explicit diffusion)
            dt_diff = (self.dx2) / (4.0 * D_max)
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

    # -------------------------
    # Single substep
    # -------------------------
    def _step_diffuse_advect(self, C, dt, D, vx, vy, bc):
        # apply BC via ghost layer
        G = self._pad_with_bc(C, bc)

        # center and neighbors from ghost-padded array
        Cc  = G[1:-1, 1:-1]
        Cxm = G[1:-1, 0:-2]
        Cxp = G[1:-1, 2:  ]
        Cym = G[0:-2, 1:-1]
        Cyp = G[2:  , 1:-1]

        diff_term = 0.0
        if D != 0.0:
            lap = (Cxp + Cxm + Cyp + Cym - 4.0 * Cc) / self.dx2
            diff_term = D * lap

        adv_term = 0.0
        if vx != 0.0 or vy != 0.0:
            adv_term = self._upwind_advection_from_neighbors(Cc, Cxm, Cxp, Cym, Cyp, vx, vy)

        newC = C + dt * (diff_term - adv_term)

        if self.clip_negative:
            np.maximum(newC, 0.0, out=newC)

        return newC

    # -------------------------
    # Upwind advection using neighbors already BC-consistent
    # -------------------------
    def _upwind_advection_from_neighbors(self, Cc, Cxm, Cxp, Cym, Cyp, vx, vy):
        if vx > 0:
            dCdx = (Cc - Cxm) / self.dx
        elif vx < 0:
            dCdx = (Cxp - Cc) / self.dx
        else:
            dCdx = 0.0

        if vy > 0:
            dCdy = (Cc - Cym) / self.dy
        elif vy < 0:
            dCdy = (Cyp - Cc) / self.dy
        else:
            dCdy = 0.0

        return vx * dCdx + vy * dCdy

    # -------------------------
    # Boundary conditions (ghost cells)
    # -------------------------
    def _bc_for_species(self, species):
        spec = self.bc_species.get(species, {})
        # merge: species overrides default per key
        bc = dict(self.bc_default)
        bc.update(spec)
        return bc

    def _normalize_bc_spec(self, spec):
        """
        Normalize shorthand:
          - allow "x": {...} meaning left/right
          - allow "y": {...} meaning bottom/top
        Ensure keys exist for left/right/bottom/top.
        """
        spec = dict(spec or {})

        if 'x' in spec:
            spec.setdefault('left', spec['x'])
            spec.setdefault('right', spec['x'])
        if 'y' in spec:
            spec.setdefault('bottom', spec['y'])
            spec.setdefault('top', spec['y'])

        # default if missing: neumann (zero-gradient)
        for side in ('left', 'right', 'bottom', 'top'):
            spec.setdefault(side, {'type': 'neumann'})

        # ensure dict + type exists
        out = {}
        for side in ('left', 'right', 'bottom', 'top'):
            bc = spec[side]
            if isinstance(bc, str):
                bc = {'type': bc}
            bc = dict(bc)
            bc.setdefault('type', 'neumann')
            out[side] = bc
        return out

    def _validate_periodic_pairing(self, bc):
        # If either side is periodic, require both sides periodic for that axis.
        lx = bc['left'].get('type', '').lower()
        rx = bc['right'].get('type', '').lower()
        by = bc['bottom'].get('type', '').lower()
        ty = bc['top'].get('type', '').lower()

        if (lx == 'periodic') ^ (rx == 'periodic'):
            raise ValueError("Periodic BC in x must be set on BOTH left and right.")
        if (by == 'periodic') ^ (ty == 'periodic'):
            raise ValueError("Periodic BC in y must be set on BOTH bottom and top.")

    def _pad_with_bc(self, C, bc):
        """
        Create ghost-padded array G with shape (ny+2, nx+2).
        Fill ghosts according to BCs.
        """
        C = np.asarray(C, dtype=float)
        ny, nx = C.shape
        if (ny, nx) != (self.ny, self.nx):
            raise ValueError(f"Field has shape {C.shape}, expected {(self.ny, self.nx)}")

        G = np.empty((ny + 2, nx + 2), dtype=float)
        G[1:-1, 1:-1] = C

        # corners will be filled after sides; initialize corners from interior for safety
        G[0, 0] = C[0, 0]
        G[0, -1] = C[0, -1]
        G[-1, 0] = C[-1, 0]
        G[-1, -1] = C[-1, -1]

        # ----- X direction (left/right) -----
        left_type = bc['left']['type'].lower()
        right_type = bc['right']['type'].lower()

        if left_type == 'periodic' and right_type == 'periodic':
            G[1:-1, 0]  = C[:, -1]   # left ghost from right interior
            G[1:-1, -1] = C[:, 0]    # right ghost from left interior
        else:
            # left ghost
            if left_type in ('neumann', 'outflow'):
                G[1:-1, 0] = C[:, 0]
            elif left_type == 'dirichlet':
                val = float(bc['left'].get('value', 0.0))
                # ghost chosen so that boundary cell center is pinned well; we also pin explicitly later
                G[1:-1, 0] = 2.0 * val - C[:, 0]
            else:
                raise ValueError(f"Unknown left BC type: {left_type}")

            # right ghost
            if right_type in ('neumann', 'outflow'):
                G[1:-1, -1] = C[:, -1]
            elif right_type == 'dirichlet':
                val = float(bc['right'].get('value', 0.0))
                G[1:-1, -1] = 2.0 * val - C[:, -1]
            else:
                raise ValueError(f"Unknown right BC type: {right_type}")

        # ----- Y direction (bottom/top) -----
        bottom_type = bc['bottom']['type'].lower()
        top_type = bc['top']['type'].lower()

        if bottom_type == 'periodic' and top_type == 'periodic':
            G[0, 1:-1]  = C[-1, :]   # bottom ghost from top interior
            G[-1, 1:-1] = C[0, :]    # top ghost from bottom interior
        else:
            if bottom_type in ('neumann', 'outflow'):
                G[0, 1:-1] = C[0, :]
            elif bottom_type == 'dirichlet':
                val = float(bc['bottom'].get('value', 0.0))
                G[0, 1:-1] = 2.0 * val - C[0, :]
            else:
                raise ValueError(f"Unknown bottom BC type: {bottom_type}")

            if top_type in ('neumann', 'outflow'):
                G[-1, 1:-1] = C[-1, :]
            elif top_type == 'dirichlet':
                val = float(bc['top'].get('value', 0.0))
                G[-1, 1:-1] = 2.0 * val - C[-1, :]
            else:
                raise ValueError(f"Unknown top BC type: {top_type}")

        # fill corners consistently (simple average of adjacent ghosts)
        G[0, 0]     = 0.5 * (G[0, 1] + G[1, 0])
        G[0, -1]    = 0.5 * (G[0, -2] + G[1, -1])
        G[-1, 0]    = 0.5 * (G[-2, 0] + G[-1, 1])
        G[-1, -1]   = 0.5 * (G[-2, -1] + G[-1, -2])

        return G


def get_diffusion_advection_process(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
        boundary_conditions=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    if diffusion_coeffs is None:
        diffusion_coeffs = {}
    if advection_coeffs is None:
        advection_coeffs = {}

    # fill in the missing diffusion and advection rates
    diffusion_coeffs_all = {
        mol_id: diffusion_coeffs.get(mol_id, default_diffusion_rate)
        for mol_id in mol_ids
    }
    advection_coeffs_all = {
        mol_id: advection_coeffs.get(mol_id, default_advection_rate)
        for mol_id in mol_ids
    }

    return {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': bounds,
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': diffusion_coeffs_all,
                'advection_coeffs': advection_coeffs_all,
                'boundary_conditions': boundary_conditions,
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }
