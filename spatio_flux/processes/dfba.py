"""
Dynamic FBA simulation
======================

A pluggable dynamic Flux Balance Analysis (dFBA) process.
Performs time-stepped metabolic modeling by combining COBRApy-based
optimization with kinetic uptake constraints.
"""
import os
import pathlib
import warnings
import numpy as np
from pathlib import Path

# Workaround for cobra bug: cobra/core/configuration.py calls
# pathlib.Path.mkdir(parents=True) on its cache directory WITHOUT
# exist_ok=True, so concurrent imports (e.g. multiple Ray actors
# starting simultaneously on the same node) race and the second one
# crashes with FileExistsError. Patch mkdir to always tolerate
# already-existing dirs before importing cobra.
_orig_mkdir = pathlib.Path.mkdir
def _safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    return _orig_mkdir(self, mode=mode, parents=parents, exist_ok=True)
pathlib.Path.mkdir = _safe_mkdir

import cobra
from cobra.io import load_model
from process_bigraph import Process
from spatio_flux.library.tools import build_path


# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")

# Define static bounds for specific models (can customize per model if needed)
default_bounds = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

default_kinetics = {
    'substrate_update_reactions': {
        'glucose': 'EX_glc__D_e',
        'acetate': 'EX_ac_e'
    },
    'kinetic_params': {
        'glucose': (0.5, 1),
        'acetate': (0.5, 2)
    }
}

MODEL_REGISTRY_DFBA = {
    'ecoli core': {
        'model_file': 'textbook',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2),
        },
        'bounds': {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}
        },
        # 'solver': 'glpk',  # optional override; cobra's default is glpk
    },
    'ecoli': {
        'model_file': 'iAF1260.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'formate': 'EX_for_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'formate': (0.5, 2),
        },
    },
    'cdiff': {
        'model_file': 'iCN900.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)
        }
    },
    'pputida': {
        'model_file': 'iJN746.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ammonium': 'EX_nh4_e',
            'glycolate': 'EX_glyclt_e'
        },
        'kinetic_params': {
            'glucose': (1, 2),
            'ammonium': (2, 4),
            'glycolate': (0.5, 1)
        }
    },
    'yeast': {
        'model_file': 'iMM904.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ammonium': 'EX_nh4_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'ammonium': (0.5, 1),
        }
    },
    'llactis': {
        'model_file': 'iNF517.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'glutamate': 'EX_glu__L_e',
            'serine': 'EX_ser__L_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1.25),
            'glutamate': (0.05, 0.1),
            'serine': (0.05, 0.1),
        }
    },
}


def get_dfba_process_from_registry(
    model_id,
    path,
    biomass_id=None,
    i=None,
    j=None,
    interval=1.0,
):
    model_config = MODEL_REGISTRY_DFBA[model_id]
    mol_ids = model_config['substrate_update_reactions'].keys()
    biomass_id = biomass_id or 'biomass'

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": model_config,
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, j, i) for mol_id in mol_ids},  # note j, i order for row,col
            "biomass": build_path(path, biomass_id, j, i)
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, j, i) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, j, i)
        },
        'interval': interval
    }

def validate_model_registry_substrates(model_registry):
    """
    Validate that 'substrate_update_reactions' and 'kinetic_params' fields match for each model.
    Also returns the set of all substrate fields across all models.
    """
    all_fields = set()

    for model_key, model_info in model_registry.items():
        config = model_info.get('config', {})
        reactions = config.get('substrate_update_reactions', {})
        kinetics = config.get('kinetic_params', {})

        fields_reactions = set(reactions.keys())
        fields_kinetics = set(kinetics.keys())

        if fields_reactions != fields_kinetics:
            raise AssertionError(
                f"Mismatch in substrate fields for model '{model_key}':\n"
                f"  In substrate_update_reactions: {sorted(fields_reactions)}\n"
                f"  In kinetic_params:             {sorted(fields_kinetics)}"
            )

        all_fields.update(fields_reactions)

    return sorted(all_fields)


_BASE_MODEL_CACHE = {}


def _load_base_model(model_file):
    """Parse the SBML/named model once and cache the parsed cobra model.

    cobra's ``load_model`` caches the SBML *string*, but still re-parses
    it into a Model on every call. ``read_sbml_model`` doesn't cache at
    all. The reference demo creates one dFBA per particle per division
    and was paying ~170 ms of SBML parsing per daughter.
    """
    cached = _BASE_MODEL_CACHE.get(model_file)
    if cached is not None:
        return cached

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / '..' / 'models'
    full_path = (models_dir / model_file).resolve()

    try:
        if model_file.endswith('.xml'):
            if not full_path.exists():
                raise FileNotFoundError(f"SBML file not found at: {full_path}")
            model = cobra.io.read_sbml_model(str(full_path))
        else:
            # cache=False bypasses cobra.io.web._cached_load, which uses
            # diskcache (pickle deserialization, CVE-affected). Our own
            # _BASE_MODEL_CACHE above already memoizes within the process.
            model = load_model(model_file, cache=False)
    except Exception:
        raise ValueError(
            f"Failed to load model from {model_file}. "
            f"Ensure it is a valid SBML file or registered model name.")

    _BASE_MODEL_CACHE[model_file] = model
    return model


HIGHS_DIRECT_SOLVER = "highs_direct"


def load_fba_model(model_file, bounds, solver=None):
    """
    Load an SBML or named model and apply static bounds.

    Uses a process-wide cache for the parsed base model and copies it
    so each caller gets an independent instance with its own bounds.
    cobra's Model.copy() reuses metabolite/reaction objects more
    efficiently than re-parsing SBML.

    ``solver`` is an optional optlang interface name (e.g. "glpk",
    "hybrid" for HiGHS, "scipy"), or the sentinel ``"highs_direct"`` to
    use the bare-highspy wrapper (handled separately by the caller —
    this function does *not* wrap, only forwards the configured cobra
    solver). When None, cobra's default applies. Set on the *copied*
    model so the cache stays solver-agnostic.
    """
    if model_file in MODEL_REGISTRY_DFBA:
        model_config = MODEL_REGISTRY_DFBA[model_file]
        if solver is None:
            solver = model_config.get('solver')
        model_file = model_config['model_file']

    base = _load_base_model(model_file)
    model = base.copy()

    # cobra doesn't know the highs_direct sentinel — leave its internal
    # solver alone; the caller wraps the model afterward.
    if solver and solver != HIGHS_DIRECT_SOLVER:
        model.solver = solver

    for rxn_id, limits in bounds.items():
        rxn = model.reactions.get_by_id(rxn_id)
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is not None and lower != {}:
            rxn.lower_bound = limits["lower"]
        if upper is not None and upper != {}:
            rxn.upper_bound = limits["upper"]

    return model


def _wrap_with_solver(cobra_model, solver):
    """If ``solver == "highs_direct"`` return a HiGHSFBASolver wrapping
    the cobra model. Otherwise return the cobra model unchanged. The
    wrapper exposes the minimal cobra-Model surface ``run_fba_update``
    uses — bound R/W and ``optimize()`` — but solves via highspy with
    a long-lived warm-started simplex basis."""
    if solver != HIGHS_DIRECT_SOLVER:
        return cobra_model
    from spatio_flux.library.highs_solver import HiGHSFBASolver
    return HiGHSFBASolver(cobra_model)


def run_fba_update(model, config, substrates, biomass, interval):
    """
    Run a single FBA update step using uptake kinetics and biomass growth.

    Units: MM kinetics treat the substrate field value as a concentration
    (mM). FBA returns flux in mmol/gDW/h, so flux × biomass × dt is an
    *amount* (mmol). To write that back into a concentration field we
    divide by ``box_volume_L`` (the per-cell volume in L). The default 1.0
    preserves legacy behavior where the field was effectively dimensionless;
    set it explicitly (e.g. spaceWidth_cm**3 * 1e-3) when comparing against
    a real-units backend like COMETS.
    """

    update_substrates = {}
    delta_biomass = 0.0
    box_volume_L = float(config.get("box_volume_L", 1.0))

    # Set uptake bounds using Michaelis-Menten kinetics, additionally
    # clipped by the substrate budget available in the box. Without this
    # clip, FBA optimizes at the MM rate and returns mu accordingly, but
    # the substrate field can only deliver `c × V_box` mmol — so biomass
    # grows as if uptake succeeded while substrate gets capped to zero.
    # Clipping the FBA *bound* makes mu and the substrate delta consistent
    # with the COMETS convention.
    for substrate, reaction_id in config["substrate_update_reactions"].items():
        Km, Vmax = config["kinetic_params"][substrate]
        substrate_concentration = substrates.get(substrate, 0.0)
        uptake_rate = -1 * Vmax * substrate_concentration / (Km + substrate_concentration)

        # Budget cap: -(amount_mmol) / (biomass × interval) is the most
        # negative uptake the box can support over this step. When biomass
        # or interval is zero, fall back to the MM rate.
        bm_dt = float(biomass) * float(interval)
        if bm_dt > 0.0:
            uptake_budget = -(substrate_concentration * box_volume_L) / bm_dt
            uptake_rate = max(uptake_rate, uptake_budget)

        if model.reactions.get_by_id(reaction_id).upper_bound < uptake_rate:
            model.reactions.get_by_id(reaction_id).upper_bound = uptake_rate
        model.reactions.get_by_id(reaction_id).lower_bound = uptake_rate

    # Run FBA optimization
    solution = model.optimize()

    if solution.status == "optimal":
        mu = solution.objective_value
        delta_biomass = mu * biomass * interval

        for substrate, rxn_id in config["substrate_update_reactions"].items():
            # if substrate not in substrates:
            #     continue

            flux_mmol = solution.fluxes[rxn_id] * biomass * interval
            delta = flux_mmol / box_volume_L  # mmol amount → mM concentration
            delta = max(delta, -substrates[substrate])  # prevent negative concentrations
            update_substrates[substrate] = delta
    else:
        for substrate in config["substrate_update_reactions"]:
            update_substrates[substrate] = 0.0

    return {
        "substrates": update_substrates,
        "biomass": delta_biomass,
    }


class DynamicFBA(Process):
    """
    A dynamic FBA process that integrates FBA optimization with substrate uptake kinetics.

    Configuration:
    -------------
    - model_file (str): Path to the SBML or named model to load.
    - kinetic_params (dict): {substrate: (Km, Vmax)} for uptake kinetics.
    - substrate_update_reactions (dict): {substrate: reaction_id} mapping each substrate to its uptake reaction.
    - bounds (dict): {reaction_id: {'lower': val, 'upper': val}} for setting static bounds.

    Inputs:
    -------
    - substrates (map[concentration]): External concentrations of substrates.
    - biomass (concentration): Current biomass level.

    Outputs:
    --------
    - substrates (map[count]): Changes in substrate concentrations.
    - biomass (count): Change in biomass.

    Notes:
    ------
    - Assumes units are consistent (e.g., mmol/L, gDW).
    - Negative fluxes represent uptake.
    """

    config_schema = {
        "model_file": "string{ecoli core}",
        "kinetic_params": "map[tuple[float,float]]",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
        "solver": "maybe[string]",
        "box_volume_L": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        cobra_model = load_fba_model(
            model_file=config["model_file"],
            bounds=config["bounds"],
            solver=config.get("solver"),
        )
        self.model = _wrap_with_solver(cobra_model, config.get("solver"))

    def inputs(self):
        return {
            "substrates": "map[concentration]",  # external concentrations
            "biomass": "mass",
        }

    def outputs(self):
        return {
            "substrates": "map[count]",   # deltas (not absolute concentrations)
            "biomass": "mass",           # delta biomass
            # "substrates": "map[count]",   # deltas (not absolute concentrations)
            # "biomass": "count",           # delta biomass
        }

    def update(self, inputs, interval):
        update = run_fba_update(
            self.model,
            self.config,
            inputs["substrates"],
            inputs["biomass"],
            interval
        )
        return update


class SpatialDFBA(Process):
    """
    Spatial DFBA with one DFBA instance per grid cell.

    Conventions:
      - config['bounds'] = (xmax, ymax)
      - config['n_bins'] = (nx, ny)  # x bins, y bins
      - all grid arrays are shaped (ny, nx) == (rows, cols)
      - indexing is arr[y, x] (row=y, col=x)
      - model_grid must be shape (ny, nx) if provided
    """

    config_schema = {
        'n_bins': 'tuple[integer{1},integer{1}]',  # (nx, ny)
        'model_file': 'maybe[string{ecoli core}]',
        'models': {
            '_type': 'map',
            '_value': {
                'model_file': 'maybe[string]',
                'kinetic_params': 'map[tuple[float,float]]',
                'substrate_update_reactions': 'map[string]',
                'bounds': 'map[bounds]',
                'solver': 'maybe[string]',
            },
        },
        'model_grid': 'maybe[list[list[string]]]',  # should be (ny, nx)
        'box_volume_L': {'_type': 'float', '_default': 1.0},
    }

    def initialize(self, config):
        # Store config bins and derived array shape
        nx, ny = config['n_bins']  # (x bins, y bins)
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_bins = (self.nx, self.ny)          # config-space
        self.grid_shape = (self.ny, self.nx)      # numpy-space (rows, cols)

        # Containers
        self.models = {}
        self.model_configs = {}
        self.default_model_id = None

        # --- Optional top-level default model_file ---------------------
        top_default_file = config.get('model_file')
        if top_default_file:
            self.default_model_id = "default"
            self.models[self.default_model_id] = load_fba_model(
                model_file=top_default_file,
                bounds={},
            )
            self.model_configs[self.default_model_id] = {
                'model_file': top_default_file,
                'kinetic_params': {},
                'substrate_update_reactions': {},
                'bounds': {},
            }

        # --- Load named FBA models -------------------------------------
        model_configs = config.get('models', {})
        if not model_configs and self.default_model_id is None:
            raise ValueError(
                "SpatialDFBA requires either a non-empty 'models' mapping "
                "or a top-level 'model_file' to define a default model."
            )

        for model_id, model_cfg in model_configs.items():
            if not model_cfg.get('model_file'):
                raise ValueError(f"Model '{model_id}' must define 'model_file'.")

            model_file = model_cfg['model_file']
            bounds = model_cfg.get('bounds', {})

            cobra_model = load_fba_model(
                model_file=model_file,
                bounds=bounds,
                solver=model_cfg.get('solver'),
            )
            self.models[model_id] = _wrap_with_solver(cobra_model, model_cfg.get('solver'))
            self.model_configs[model_id] = dict(model_cfg)

        # --- Build and validate model_grid -----------------------------
        grid_cfg = config.get('model_grid')

        if grid_cfg is None or grid_cfg == []:
            if self.default_model_id is not None:
                model_grid_array = np.full(self.grid_shape, self.default_model_id, dtype='U64')
            else:
                model_grid_array = np.full(self.grid_shape, '', dtype='U64')
        else:
            model_grid_array = np.array(grid_cfg, dtype='U64')
            if model_grid_array.shape != self.grid_shape:
                raise ValueError(
                    f"model_grid shape {model_grid_array.shape} does not match expected grid_shape {self.grid_shape} "
                    f"(ny, nx) derived from n_bins (nx, ny) = {self.n_bins}"
                )

        unique_ids = set(np.unique(model_grid_array))
        unknown_ids = unique_ids - set(self.models.keys()) - {''}
        if unknown_ids:
            raise ValueError(f"Unknown model IDs in model_grid: {unknown_ids}")

        self.model_grid = model_grid_array

    # ------------------------------------------------------------------ #
    # Ports                                                              #
    # ------------------------------------------------------------------ #

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'positive_array',
                    '_shape': self.grid_shape,  # (ny, nx)
                    '_data': 'float',
                },
            },
            'biomass': {
                '_type': 'array',
                '_shape': self.grid_shape,  # (ny, nx)
                '_data': 'float',
            },
        }

    def outputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.grid_shape,  # (ny, nx)
                    '_data': 'float',
                },
            },
            'biomass': {
                '_type': 'array',
                '_shape': self.grid_shape,  # (ny, nx)
                '_data': 'float',
            },
        }

    # ------------------------------------------------------------------ #
    # Update                                                             #
    # ------------------------------------------------------------------ #

    def update(self, state, interval):
        """
        For each grid cell (y, x):
          1) read local substrates and biomass from arrays shaped (ny, nx)
          2) look up model_id in model_grid[y, x]
          3) run dfba
          4) write deltas back into delta arrays at [y, x]
        """
        substrate_fields = state['fields']
        biomass_field = state['biomass']

        # Optional but very helpful shape guard
        if biomass_field.shape != self.grid_shape:
            raise ValueError(f"biomass shape {biomass_field.shape} != expected {self.grid_shape} (ny, nx)")
        for mol_id, arr in substrate_fields.items():
            if arr.shape != self.grid_shape:
                raise ValueError(f"field '{mol_id}' shape {arr.shape} != expected {self.grid_shape} (ny, nx)")

        # Initialize outputs (deltas)
        delta_fields = {
            mol_id: np.zeros(self.grid_shape, dtype=float)
            for mol_id in substrate_fields
        }
        delta_biomass = np.zeros(self.grid_shape, dtype=float)

        # Iterate in physical y,x order but index arrays as [y, x]
        for y in range(self.ny):
            for x in range(self.nx):
                model_id = self.model_grid[y, x]
                if model_id == '' or model_id is None:
                    continue

                model = self.models[model_id]
                model_cfg = self.model_configs[model_id]

                local_substrates = {
                    mol_id: float(substrate_fields[mol_id][y, x])
                    for mol_id in substrate_fields
                }
                local_biomass = float(biomass_field[y, x])

                dfba_config = {
                    'model_file': model_cfg.get('model_file'),
                    'kinetic_params': model_cfg.get('kinetic_params', {}),
                    'substrate_update_reactions': model_cfg.get('substrate_update_reactions', {}),
                    'bounds': model_cfg.get('bounds', {}),
                    'model_id': model_id,
                    'box_volume_L': self.config.get('box_volume_L', 1.0),
                }

                upd = run_fba_update(
                    model,
                    dfba_config,
                    local_substrates,
                    local_biomass,
                    interval,
                )

                # Substrate updates
                for mol_id, delta in upd['substrates'].items():
                    if mol_id not in delta_fields:
                        delta_fields[mol_id] = np.zeros(self.grid_shape, dtype=float)
                    delta_fields[mol_id][y, x] = float(delta)

                # Biomass update
                delta_biomass[y, x] = float(upd['biomass'])

        return {
            'fields': delta_fields,
            'biomass': delta_biomass,
        }


class ShardedDFBA(Process):
    """
    Run dFBA on a configurable list of cells using one shared cobra Model.

    Designed to be hosted on a Ray actor (via ``RayProcess``): each shard
    handles its own cells in a tight loop, so per-shard dispatch costs
    amortize across many cells. The shard is topology-agnostic — the
    cells it owns are an unordered collection of keys, with no implied
    grid structure or neighbor coupling. All spatial coupling lives in
    the diffusion process.

    Config is otherwise identical to ``DynamicFBA``, plus:
      - cell_keys: ordered list of cell identifiers ("c_y_x" by convention).

    Inputs / outputs nest one level deeper: ``cells[<key>] = {substrates,
    biomass}`` so each cell's state can be wired to its own grid path.
    """

    config_schema = {
        "model_file": "string{ecoli core}",
        "kinetic_params": "map[tuple[float,float]]",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
        "solver": "maybe[string]",
        "cell_keys": "list[string]",
        "box_volume_L": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        cobra_model = load_fba_model(
            model_file=config["model_file"],
            bounds=config["bounds"],
            solver=config.get("solver"),
        )
        self.model = _wrap_with_solver(cobra_model, config.get("solver"))
        # cell_keys is per-session, not part of the expensive init.
        # Default to empty so a pool actor can be constructed without
        # knowing its cells yet — Session.reconfigure rebinds them.
        self.cell_keys = list(config.get("cell_keys") or [])

    def reconfigure(self, config):
        """Cheap per-session rebinding — does NOT reload the cobra
        Model or rebuild the LP. Just rebinds the cell list this shard
        will iterate on the next ``update`` call.

        This is what makes ActorPool actually amortize: the cobra +
        LP setup happens once in ``initialize`` (per pool spawn);
        every Session that claims this actor gets a cheap cell_keys
        swap instead of a fresh model load.
        """
        if "cell_keys" in config:
            self.cell_keys = list(config["cell_keys"] or [])
        # Other fields (model, solver, kinetics) are immutable per-pool;
        # changing them requires a different pool (different config_hash).

    def inputs(self):
        return {
            "cells": {
                k: {
                    "substrates": "map[concentration]",
                    "biomass": "mass",
                }
                for k in self.cell_keys
            }
        }

    def outputs(self):
        return {
            "cells": {
                k: {
                    "substrates": "map[count]",
                    "biomass": "mass",
                }
                for k in self.cell_keys
            }
        }

    def update(self, inputs, interval):
        import time as _time
        cells_in = inputs["cells"]
        cells_out = {}
        t_start = _time.monotonic()
        n_cells = len(self.cell_keys)
        for k in self.cell_keys:
            cell = cells_in[k]
            upd = run_fba_update(
                self.model,
                self.config,
                cell["substrates"],
                cell["biomass"],
                interval,
            )
            cells_out[k] = upd
        t_elapsed = _time.monotonic() - t_start
        # Aggregate per-shard tick stats. First-tick print confirms
        # whether per-cell solve time is the bottleneck (~ms is cold-start
        # territory; ~10s of μs is warm-start). Tracks running averages.
        cnt = getattr(self, '_tick_count', 0) + 1
        sum_ms = getattr(self, '_sum_total_ms', 0.0) + t_elapsed * 1000
        self._tick_count = cnt
        self._sum_total_ms = sum_ms
        if cnt <= 2:
            per_cell_us = (t_elapsed / max(1, n_cells)) * 1_000_000
            print(
                f"  ShardedDFBA tick #{cnt}: {n_cells} cells in "
                f"{t_elapsed*1000:.1f}ms ({per_cell_us:.0f}μs/cell)",
                flush=True,
            )
        return {"cells": cells_out}


def get_field_names(model_registry):
    all_fields = set()
    for model_info in model_registry.values():
        config = model_info.get('config', {})
        all_fields.update(config.get('substrate_update_reactions', {}).keys())
        all_fields.update(config.get('kinetic_params', {}).keys())
    return sorted(all_fields)


def restore_bounds_safely(rxn, lb, ub):
    """Restore bounds safely regardless of current invalid state."""
    # Set to dummy permissive bounds first to avoid intermediate errors
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.lower_bound = lb
    rxn.upper_bound = ub


from cobra.medium import minimal_medium

def analyze_fba_model_minimal_media(model_key, config, model_dir, flux_epsilon=1e-10, top_k=10, growth_threshold=1e-6):
    """
    Load model, apply bounds and kinetic constraints, run FBA, and print important exchange reactions,
    including minimal media, alternate nutrient sources, secreted byproducts, and growth-limiting exchanges.
    """
    print(f"\n=== Analyzing model: {model_key} ===")

    # --- Load Model ---
    model_file = config['model_file'].removeprefix('models/')
    if model_file.endswith('.xml'):
        model_path = os.path.join(model_dir, model_file)
        model = load_fba_model(model_path, config.get('bounds', {}))
    else:
        model = load_fba_model(model_file, config.get('bounds', {}))  # named model

    # --- Step 1: Full Media Optimization ---
    print("\n--- Step 1: Optimize with Full Media ---")
    print(f'Objective expression: {model.objective.expression}')
    full_solution = model.optimize()
    if full_solution.status != 'optimal':
        print(f"  ⚠ Optimization not optimal (status: {full_solution.status})")
        return
    baseline_growth = full_solution.objective_value
    print(f"  ✅ Baseline growth rate: {baseline_growth:.6f}")

    # --- Step 2: Active Exchange Reactions (Full Media) ---
    active_exchanges = []
    uptake_reactions = []
    secretion_reactions = []

    print("\n--- Step 2: Active Exchange Reactions ---")
    for rxn in model.exchanges:
        flux = full_solution.fluxes.get(rxn.id, 0.0)
        default_bounds = (-1000.0, 1000.0) if rxn.reversibility else (0.0, 1000.0)
        is_constrained = (rxn.lower_bound, rxn.upper_bound) != default_bounds
        is_active = abs(flux) > flux_epsilon

        if is_constrained or is_active:
            if flux < -flux_epsilon:
                uptake_reactions.append((rxn.id, flux))
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}  [uptake]")
            elif flux > flux_epsilon:
                secretion_reactions.append((rxn.id, flux))
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}  [secretion]")

        if is_active:
            active_exchanges.append((rxn, flux))

    # --- Step 3: Minimal Media Optimization ---
    print("\n--- Step 3: Minimal Media Analysis ---")
    if baseline_growth < growth_threshold:
        print(f"  ⚠ Baseline growth too low for minimal media analysis (growth: {baseline_growth:.6e})")
        min_solution = full_solution
        min_media = {}
    else:
        print("  🧪 Computing minimal media from baseline growth...")
        min_media = minimal_medium(model, baseline_growth * 0.99)

        # Block all uptakes
        for rxn in model.exchanges:
            if rxn.lower_bound < 0 and rxn.upper_bound > 0:
                rxn.lower_bound = 0.0

        # Set only the minimal uptakes
        for rxn_id, flux in min_media.items():
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = -abs(flux)

        # Re-optimize
        min_solution = model.optimize()
        if min_solution.status != 'optimal':
            print("  ❌ Optimization failed under minimal media.")
            return
        print(f"  ✅ Growth with minimal media: {min_solution.objective_value:.6f}")

    # --- Step 4: Minimal Media Uptakes ---
    minimal_uptakes = [
        (rxn_id, flux) for rxn_id, flux in min_solution.fluxes.items()
        if rxn_id in min_media and flux < -flux_epsilon
    ]
    if minimal_uptakes:
        print("\n  📦 Required Minimal Media Uptakes:")
        for rxn_id, flux in sorted(minimal_uptakes, key=lambda x: x[1]):
            print(f"    {rxn_id:20s}  Uptake flux: {flux:10.6f}")
    else:
        print("  ⚠ No active uptake reactions under minimal media.")

    # --- Step 5: Top Uptake Sources ---
    print(f"\n--- Step 4: Top {top_k} Uptake Sources ---")
    top_uptake = sorted(minimal_uptakes, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    for i, (rxn_id, flux) in enumerate(top_uptake, 1):
        print(f"    {i}. {rxn_id:20s}  Flux: {flux:10.6f}")

    # --- Step 6: Alternate Uptake Source Analysis ---
    print(f"\n--- Step 5: Alternate Uptake Sources ---")
    alternate_sources = []

    for rxn_id, flux in top_uptake:
        rxn = model.reactions.get_by_id(rxn_id)
        original_lb, original_ub = rxn.lower_bound, rxn.upper_bound
        restore_bounds_safely(rxn, 0.0, 0.0)  # knockout top uptake

        alt_sol = model.optimize()
        restore_bounds_safely(rxn, original_lb, original_ub)

        if alt_sol.status == 'optimal' and alt_sol.objective_value > growth_threshold:
            # Look for other uptake reactions that now carry flux
            alt_uptakes = [
                (r.id, alt_sol.fluxes.get(r.id, 0.0))
                for r in model.exchanges
                if alt_sol.fluxes.get(r.id, 0.0) < -flux_epsilon and r.id != rxn_id
            ]
            if alt_uptakes:
                alternate_sources.append((rxn_id, alt_sol.objective_value, alt_uptakes))
                print(f"  🔄 Alternate source(s) found for {rxn_id} (growth: {alt_sol.objective_value:.6f}):")
                for alt_id, alt_flux in sorted(alt_uptakes, key=lambda x: x[1]):
                    print(f"    → {alt_id:20s}  Flux: {alt_flux:10.6f}")
            else:
                print(f"  ⚠ {rxn_id}: growth sustained, but no new uptakes found")
        else:
            print(f"  🚫 {rxn_id}: growth not sustained after removal")

    # --- Step 7: Growth-Limiting Reaction Knockouts ---
    print("\n--- Step 6: Growth-Limiting Knockout Analysis ---")
    limiting_rxns = []
    for rxn, flux in active_exchanges:
        original_lb, original_ub = rxn.lower_bound, rxn.upper_bound
        restore_bounds_safely(rxn, 0.0, 0.0)

        perturbed_sol = model.optimize()
        restore_bounds_safely(rxn, original_lb, original_ub)

        if perturbed_sol.status == 'optimal':
            drop = min_solution.objective_value - perturbed_sol.objective_value
            if drop > 1e-12 and perturbed_sol.objective_value > 1e-12:
                limiting_rxns.append({
                    'id': rxn.id,
                    'flux': flux,
                    'drop': drop,
                    'new_growth': perturbed_sol.objective_value
                })
        else:
            print(f"    ⚠ Blocking {rxn.id} made the model infeasible")

    if limiting_rxns:
        limiting_rxns.sort(key=lambda r: r['drop'], reverse=True)
        print(f"\n  🚨 Top {min(top_k, len(limiting_rxns))} Growth-Limiting Exchanges:")
        for i, rxn in enumerate(limiting_rxns[:top_k], 1):
            print(
                f"    {i}. {rxn['id']:20s}  Flux: {rxn['flux']:9.4f}  Growth drop: {rxn['drop']:.6f} → {rxn['new_growth']:.6f}")
    else:
        print("  ⚠ No exchange flux significantly limits growth under minimal media.")


if __name__ == "__main__":
    for model_key, config in MODEL_REGISTRY_DFBA.items():
        analyze_fba_model_minimal_media(model_key, config, MODEL_DIR)



    

    types = {
        'species': 'count_concentration_volume',

        'local': {
            'count': 'count',
            'concentration': 'concentration'},

        'field': {
            'volume': 'float',
            'species': 'map[local]'},

        'fields': 'array[(5|5),field]'}


    state = {
        'fields': {'_type': 'fields'},
        'diffusion': {
            '_type': 'link',
            '_inputs': {
                'fields': 'fields'},
            '_outputs': {
                'fields': 'fields'},
            'inputs': {
                'fields': ['fields']},
            'outputs': {
                'fields': ['fields']}},
        'dfba': {
            '_type': 'link',
            '_inputs': {
                'concentrations': 'map[concentration]'},
            '_outputs': {
                'counts': 'map[count]'},
            'inputs': {
                'concentrations': ['fields', 0, 0, 'species', '*', 'concentration']},
            'outputs': {
                'counts': ['fields', 0, 0, 'species', '*', 'count']}}}
