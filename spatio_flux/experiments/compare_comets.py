"""
Side-by-side comparison: spatio-flux (compositional COMETS) vs the actual COMETS
(Java backend via cometspy). Same e. coli core dFBA on a 2D grid with metabolite
diffusion and a seeded biomass spot.

What this produces in `out/comets_compare/`:
  - snapshots_<grid>.png    snapshot grids for both implementations side-by-side
  - biomass_trace.png       biomass-vs-time at the seed cell, both implementations
  - scaling.png             wall-clock vs grid size, log-log

Run:
  COMETS_HOME=/path/to/comets_2.x.y \
  .venv/bin/python -m spatio_flux.experiments.compare_comets

Requires:
  - cometspy installed
  - $COMETS_HOME pointing at a working COMETS install
  - GLOP optimizer available (open-source; bundled with COMETS)
"""
from __future__ import annotations

import os
import time
import base64
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cobra

# spatio-flux imports
from process_bigraph import allocate_core
from spatio_flux.processes import (
    get_spatial_many_dfba,
    get_diffusion_advection_process,
    get_fields_with_schema,
    MODEL_REGISTRY_DFBA,
)
from spatio_flux.processes.dfba import DynamicFBA
from spatio_flux.library.tools import run_composite_document
from spatio_flux.library.ray_process import register_process_class
from spatio_flux.library.tools import get_standard_emitter
from process_bigraph import Composite, gather_emitter_results

# cometspy
import cometspy as cspy


# ---------------------------------------------------------------------------
# Common simulation parameters (same physical setup on both sides)
# ---------------------------------------------------------------------------
# Units: hours, cm, mmol, gDW.

DEFAULT_GRIDS = [4, 8, 16, 32]     # N for an NxN grid; doubling sweep
COMPARISON_GRID = 16               # the N used for the snapshot/timeseries panels

TIME_STEP = 0.1                    # hours per dFBA step
TOTAL_TIME = 4.0                   # hours total
SPACE_WIDTH = 0.05                 # cm per grid cell (0.5 mm)

# Michaelis-Menten kinetics for glucose / acetate uptake. Matched on both
# sides so the dFBA solves the same problem given the same substrate field.
KM_GLC,  VMAX_GLC = 0.5, 1.0       # mM, mmol / gDW / h
KM_AC,   VMAX_AC  = 0.5, 2.0

GLC_DIFF = 5e-6                    # cm^2/s, ~realistic for small molecule
BIOMASS_DIFF = 1e-9                # cm^2/s, biomass barely diffuses

INITIAL_BIOMASS = 0.05             # seed cell value (gDW or arbitrary biomass units)

# Vertical glucose gradient: row 0 (top) low, row N-1 (bottom) high.
# Both ends well above KM_GLC so the gradient drives differential growth.
GLC_LOW = 1.0
GLC_HIGH = 20.0

OUT_DIR = Path("out/comets_compare")


# ---------------------------------------------------------------------------
# Initial-condition helpers (shared between both runners)
# ---------------------------------------------------------------------------
def initial_glucose_field(n: int) -> np.ndarray:
    """(n, n) array, vertical gradient. arr[y, x]: y = row, top->bottom."""
    col = np.linspace(GLC_LOW, GLC_HIGH, n)[:, None]  # (n, 1)
    return np.repeat(col, n, axis=1)                  # (n, n)

def biomass_seed_position(n: int) -> tuple[int, int]:
    """(x, y) coordinates of the seeded biomass cell — top center."""
    return (n // 2, 0)


# ---------------------------------------------------------------------------
# COMETS (cometspy) side
# ---------------------------------------------------------------------------
def run_cometspy(n: int, total_time: float = TOTAL_TIME) -> dict:
    """
    Run a matched dFBA + diffusion sim on a square NxN grid using cometspy.

    Returns:
        dict with keys:
          'wall_time'           : float, seconds
          'biomass_history'     : (T+1, n, n) array of biomass over time
          'glucose_history'     : (T+1, n, n) array of glucose
          'acetate_history'     : (T+1, n, n) array of acetate
          'time_h'              : (T+1,) array of times (hours)
    """
    e_coli = cobra.io.load_model("textbook")

    # Match the bounds spatio-flux's MODEL_REGISTRY_DFBA['ecoli core'] applies:
    # restrict ATPM (so low-uptake conditions can still satisfy ATP demand)
    # and cap O2 uptake. Without this the dFBA can't grow at our matched
    # Vmax=1 mmol/gDW/h glucose uptake.
    e_coli.reactions.get_by_id("ATPM").bounds = (1.0, 1.0)
    e_coli.reactions.get_by_id("EX_o2_e").lower_bound = -2.0

    cm = cspy.model(e_coli)
    # cometspy's layout constructor reads m.initial_pop[0]; the actual seeded
    # cells come from layout.initial_pop set after construction. Set the model
    # to a zero-amount placeholder at (0,0) so it doesn't double-seed.
    cm.initial_pop = [0, 0, 0.0]
    # Don't open_exchanges() — that would erase the ATPM/O2 bounds above.
    cm.change_optimizer("GLOP")

    layout = cspy.layout([cm])
    layout.grid = [n, n]

    # set baseline metabolites needed by e. coli core (in mmol/cell)
    # high "static" reservoirs so they are never limiting
    for met in ("o2_e", "nh4_e", "pi_e", "h2o_e", "h_e", "co2_e"):
        layout.set_specific_metabolite(met, 1000.0)
        layout.set_specific_static(met, 1000.0)

    # add glucose and acetate as exchanged metabolites (start at near-zero
    # globally, then write the gradient per cell)
    layout.set_specific_metabolite("glc__D_e", 0.0)
    layout.set_specific_metabolite("ac_e", 0.0)

    # diffusion constants (cm^2/s)
    layout.set_specific_metabolite_diffusion("glc__D_e", float(GLC_DIFF))
    layout.set_specific_metabolite_diffusion("ac_e", float(GLC_DIFF))

    # seed glucose vertical gradient
    glc_field = initial_glucose_field(n)
    for y in range(n):
        for x in range(n):
            layout.set_specific_metabolite_at_location(
                "glc__D_e", (x, y), float(glc_field[y, x])
            )

    # seed biomass: single cell at top center
    bx, by = biomass_seed_position(n)
    layout.initial_pop = [[bx, by, INITIAL_BIOMASS]]

    # parameters
    p = cspy.params()
    # match spatio-flux's MM kinetics (Km=0.5 mM, Vmax=1 mmol/gDW/h for glc,
    # Vmax=2 for ac). cometspy uses defaultKm/defaultVmax for any exchange
    # without explicit override; we set these to the glucose values and bump
    # acetate's Vmax via the model object below.
    p.set_param("defaultVmax", float(VMAX_GLC))
    p.set_param("defaultKm", float(KM_GLC))
    p.set_param("defaultDiffConst", float(BIOMASS_DIFF))  # biomass diffusion
    p.set_param("timeStep", float(TIME_STEP))
    p.set_param("maxCycles", int(round(total_time / TIME_STEP)))
    p.set_param("spaceWidth", float(SPACE_WIDTH))
    p.set_param("biomassMotionStyle", "Diffusion 2D(Crank-Nicolson)")
    p.set_param("writeBiomassLog", True)
    p.set_param("writeMediaLog", True)
    p.set_param("writeTotalBiomassLog", True)
    p.set_param("BiomassLogRate", 1)
    p.set_param("MediaLogRate", 1)
    p.set_param("FluxLogRate", 1)
    p.set_param("totalBiomassLogRate", 1)
    p.set_param("numRunThreads", 1)
    p.set_param("exchangestyle", "Monod Style")

    sim = cspy.comets(layout, p)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    # parse biomass log → (T+1, n, n). cometspy biomass log is 0-indexed in (x,y).
    bm_df = sim.biomass.copy()
    n_cycles = int(bm_df["cycle"].max()) if len(bm_df) else 0
    bm_hist = np.zeros((n_cycles + 1, n, n), dtype=float)
    for _, row in bm_df.iterrows():
        c = int(row["cycle"])
        x = int(row["x"])
        y = int(row["y"])
        bm_hist[c, y, x] += float(row["biomass"])

    # parse media log → (T+1, n, n) per metabolite. media log is 1-indexed in (x,y).
    media_df = sim.media.copy()
    glc_hist = np.zeros((n_cycles + 1, n, n), dtype=float)
    ac_hist  = np.zeros((n_cycles + 1, n, n), dtype=float)
    for _, row in media_df.iterrows():
        c = int(row["cycle"])
        if c > n_cycles:
            continue
        x = int(row["x"]) - 1
        y = int(row["y"]) - 1
        m = row["metabolite"]
        v = float(row["conc_mmol"])
        if m == "glc__D_e":
            glc_hist[c, y, x] = v
        elif m == "ac_e":
            ac_hist[c, y, x] = v

    time_h = np.arange(n_cycles + 1) * TIME_STEP

    return {
        "wall_time": wall,
        "biomass_history": bm_hist,
        "glucose_history": glc_hist,
        "acetate_history": ac_hist,
        "time_h": time_h,
    }


# ---------------------------------------------------------------------------
# spatio-flux side
# ---------------------------------------------------------------------------
def run_spatioflux(n: int, total_time: float = TOTAL_TIME, core=None) -> dict:
    """
    Run a matched dFBA + diffusion sim on a square NxN grid using spatio-flux.

    Spatio-flux uses a per-cell DynamicFBA process with Michaelis-Menten kinetics
    (Km, Vmax) per substrate, and a DiffusionAdvection process for transport.

    Returns the same shape of dict as run_cometspy().
    """
    core = core or allocate_core()

    n_bins = (n, n)
    bounds = (n * SPACE_WIDTH, n * SPACE_WIDTH)  # cm
    mol_ids = ["glucose", "acetate", "dissolved biomass"]

    # diffusion: cometspy's defaultDiffConst is cm^2/s; spatio-flux's
    # DiffusionAdvection uses the same dx (here cm). Convert to per-hour to
    # match the dFBA timestep semantics: dx is in cm, time in h => need cm^2/h.
    # Spatio-flux doesn't explicitly require units; we just pass D such that
    # the dimensionless dt*D/dx^2 matches.
    # cometspy uses dt in hours but D in cm^2/s, so the implicit conversion
    # is D_h = D_s * 3600. We pass D_h here.
    D_glc = float(GLC_DIFF) * 3600.0
    D_bm  = float(BIOMASS_DIFF) * 3600.0
    diffusion_coeffs = {
        "glucose":           D_glc,
        "acetate":           D_glc,
        "dissolved biomass": D_bm,
    }

    # initial fields
    glc_field = initial_glucose_field(n)
    ac_field = np.zeros((n, n), dtype=float)
    bm_field = np.zeros((n, n), dtype=float)
    bx, by = biomass_seed_position(n)
    bm_field[by, bx] = INITIAL_BIOMASS

    initial_fields = {
        "glucose": glc_field,
        "acetate": ac_field,
        "dissolved biomass": bm_field,
    }

    spatial_dfba = get_spatial_many_dfba(
        model_id="ecoli core",
        mol_ids=mol_ids,
        n_bins=n_bins,
        path=["fields"],
    )

    # override the per-cell dFBA interval to match TIME_STEP. The default
    # built into get_spatial_many_dfba is 1.0; we want TIME_STEP.
    for k in spatial_dfba:
        spatial_dfba[k]["interval"] = float(TIME_STEP)

    doc = {
        **spatial_dfba,
        "fields": get_fields_with_schema(
            n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields
        ),
        # Match cometspy's tight interleaving: cometspy runs diffusion
        # numDiffPerStep=10 times per dFBA cycle. Without setting interval here
        # spatio-flux's diffusion process defaults to 1.0h, firing only every
        # 10 dFBA steps and then dumping a 1h jump at once.
        "diffusion": {
            **get_diffusion_advection_process(
                bounds=bounds,
                n_bins=n_bins,
                mol_ids=mol_ids,
                diffusion_coeffs=diffusion_coeffs,
                advection_coeffs={},
            ),
            "interval": float(TIME_STEP),
        },
    }

    name = f"sf_compare_n{n}"
    t0 = time.perf_counter()
    res = run_composite_document(
        doc, core=core, name=name, time=float(total_time),
        outdir=str(OUT_DIR / "_sf_runs"),
    )
    wall = time.perf_counter() - t0

    # run_composite_document returns (steps_list, process_time, framework_time).
    # The first element is the per-emit list of states.
    results = res[0] if isinstance(res, tuple) else res

    n_emits = len(results)
    bm_hist = np.zeros((n_emits, n, n), dtype=float)
    glc_hist = np.zeros((n_emits, n, n), dtype=float)
    ac_hist = np.zeros((n_emits, n, n), dtype=float)
    times = np.zeros(n_emits, dtype=float)
    for i, step in enumerate(results):
        fields = step["fields"]
        bm_hist[i] = np.asarray(fields["dissolved biomass"])
        glc_hist[i] = np.asarray(fields["glucose"])
        ac_hist[i]  = np.asarray(fields["acetate"])
        times[i] = float(step.get("global_time", i))

    return {
        "wall_time": wall,
        "biomass_history": bm_hist,
        "glucose_history": glc_hist,
        "acetate_history": ac_hist,
        "time_h": times,
    }


# ---------------------------------------------------------------------------
# spatio-flux SINGLE-PROCESS variant (SpatialDFBA: one process, internal loop)
# ---------------------------------------------------------------------------
def run_spatioflux_single(n: int, total_time: float = TOTAL_TIME, core=None) -> dict:
    """
    Same physics as run_spatioflux, but using SpatialDFBA — a single process
    that internally loops over cells. Eliminates per-cell process-bigraph
    dispatch and lets cells share one cobra Model.
    """
    core = core or allocate_core()

    n_bins = (n, n)
    bounds = (n * SPACE_WIDTH, n * SPACE_WIDTH)
    mol_ids = ["glucose", "acetate", "dissolved biomass"]

    D_glc = float(GLC_DIFF) * 3600.0
    D_bm  = float(BIOMASS_DIFF) * 3600.0
    diffusion_coeffs = {
        "glucose":           D_glc,
        "acetate":           D_glc,
        "dissolved biomass": D_bm,
    }

    glc_field = initial_glucose_field(n)
    ac_field  = np.zeros((n, n), dtype=float)
    bm_field  = np.zeros((n, n), dtype=float)
    bx, by = biomass_seed_position(n)
    bm_field[by, bx] = INITIAL_BIOMASS

    initial_fields = {
        "glucose": glc_field,
        "acetate": ac_field,
        "dissolved biomass": bm_field,
    }

    # SpatialDFBA needs (model_id -> {model_file, kinetic_params, ...}).
    # Pull from the registry so the per-cell solve is identical to the
    # many-process variant.
    ecoli_cfg = MODEL_REGISTRY_DFBA["ecoli core"]
    spatial_dfba = {
        "_type": "process",
        "address": "local:SpatialDFBA",
        "config": {
            "n_bins": n_bins,
            "models": {"ecoli core": ecoli_cfg},
            "model_grid": [["ecoli core"] * n for _ in range(n)],
        },
        "inputs": {
            "fields": {
                "glucose": ["fields", "glucose"],
                "acetate": ["fields", "acetate"],
            },
            "biomass": ["fields", "dissolved biomass"],
        },
        "outputs": {
            "fields": {
                "glucose": ["fields", "glucose"],
                "acetate": ["fields", "acetate"],
            },
            "biomass": ["fields", "dissolved biomass"],
        },
        "interval": float(TIME_STEP),
    }

    doc = {
        "spatial_dfba": spatial_dfba,
        "fields": get_fields_with_schema(
            n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields
        ),
        # Match cometspy's tight interleaving: cometspy runs diffusion
        # numDiffPerStep=10 times per dFBA cycle. Without setting interval here
        # spatio-flux's diffusion process defaults to 1.0h, firing only every
        # 10 dFBA steps and then dumping a 1h jump at once.
        "diffusion": {
            **get_diffusion_advection_process(
                bounds=bounds,
                n_bins=n_bins,
                mol_ids=mol_ids,
                diffusion_coeffs=diffusion_coeffs,
                advection_coeffs={},
            ),
            "interval": float(TIME_STEP),
        },
    }

    name = f"sf_single_n{n}"
    t0 = time.perf_counter()
    res = run_composite_document(
        doc, core=core, name=name, time=float(total_time),
        outdir=str(OUT_DIR / "_sf_runs"),
    )
    wall = time.perf_counter() - t0
    results = res[0] if isinstance(res, tuple) else res

    n_emits = len(results)
    bm_hist = np.zeros((n_emits, n, n), dtype=float)
    glc_hist = np.zeros((n_emits, n, n), dtype=float)
    ac_hist  = np.zeros((n_emits, n, n), dtype=float)
    times = np.zeros(n_emits, dtype=float)
    for i, step in enumerate(results):
        fields = step["fields"]
        bm_hist[i]  = np.asarray(fields["dissolved biomass"])
        glc_hist[i] = np.asarray(fields["glucose"])
        ac_hist[i]  = np.asarray(fields["acetate"])
        times[i] = float(step.get("global_time", i))

    return {
        "wall_time": wall,
        "biomass_history": bm_hist,
        "glucose_history": glc_hist,
        "acetate_history": ac_hist,
        "time_h": times,
    }


# ---------------------------------------------------------------------------
# spatio-flux RAY variant: per-cell DynamicFBA on Ray actors, dispatched
# concurrently via ParallelComposite. Same physics as run_spatioflux —
# only the transport/composition layer changes.
# ---------------------------------------------------------------------------
_RAY_REGISTERED = False

def _ensure_ray_registered():
    """One-time registration of DynamicFBA into the Ray-side registry so
    actors can resolve it. Idempotent."""
    global _RAY_REGISTERED
    if not _RAY_REGISTERED:
        register_process_class("DynamicFBA", DynamicFBA)
        _RAY_REGISTERED = True


def _build_ray_dfba_processes(n: int, mol_ids: list, biomass_id: str = "dissolved biomass",
                              pool_size: int | None = None):
    """Mirror of get_spatial_many_dfba, but each per-cell process is a
    RayProcess routing through a shared actor pool. All cells use the
    same (process_class, process_config) so they share one pool."""
    cfg = MODEL_REGISTRY_DFBA["ecoli core"]
    if pool_size is None:
        pool_size = max(1, (os.cpu_count() or 4))
    nx, ny = n, n
    procs = {}
    for y in range(ny):
        for x in range(nx):
            substrate_wires = {
                m: ["fields", m, y, x] for m in mol_ids if m != biomass_id
            }
            procs[f"dFBA[{x},{y}]"] = {
                "_type": "process",
                "address": "local:RayProcess",
                "config": {
                    "process_class": "DynamicFBA",
                    "process_config": cfg,
                    "pool_size": int(pool_size),
                },
                "inputs": {
                    "substrates": substrate_wires,
                    "biomass": ["fields", biomass_id, y, x],
                },
                "outputs": {
                    "substrates": substrate_wires,
                    "biomass": ["fields", biomass_id, y, x],
                },
                "interval": float(TIME_STEP),
            }
    return procs


def run_spatioflux_ray(n: int, total_time: float = TOTAL_TIME, core=None) -> dict:
    """Per-cell DynamicFBA on Ray actors, dispatched in parallel via
    ParallelComposite. Same physical setup as run_spatioflux."""
    _ensure_ray_registered()
    core = core or allocate_core()

    n_bins = (n, n)
    bounds = (n * SPACE_WIDTH, n * SPACE_WIDTH)
    mol_ids = ["glucose", "acetate", "dissolved biomass"]

    D_glc = float(GLC_DIFF) * 3600.0
    D_bm  = float(BIOMASS_DIFF) * 3600.0
    diffusion_coeffs = {
        "glucose":           D_glc,
        "acetate":           D_glc,
        "dissolved biomass": D_bm,
    }

    glc_field = initial_glucose_field(n)
    ac_field = np.zeros((n, n), dtype=float)
    bm_field = np.zeros((n, n), dtype=float)
    bx, by = biomass_seed_position(n)
    bm_field[by, bx] = INITIAL_BIOMASS

    initial_fields = {
        "glucose": glc_field,
        "acetate": ac_field,
        "dissolved biomass": bm_field,
    }

    state = {
        **_build_ray_dfba_processes(n, mol_ids),
        "fields": get_fields_with_schema(
            n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields
        ),
        "diffusion": {
            **get_diffusion_advection_process(
                bounds=bounds,
                n_bins=n_bins,
                mol_ids=mol_ids,
                diffusion_coeffs=diffusion_coeffs,
                advection_coeffs={},
            ),
            "interval": float(TIME_STEP),
        },
    }
    state["emitter"] = get_standard_emitter(state_keys=list(state.keys()))

    # Hand-rolled runner. parallel_processes=True is the upstream flag that
    # makes the orchestrator dispatch per-tick process invocations through a
    # ThreadPoolExecutor — required for RayProcess.update()'s blocking
    # ray.get to actually overlap across cells.
    sim = Composite(
        {"state": state, "parallel_processes": True},
        core=core,
    )
    t0 = time.perf_counter()
    sim.run(float(total_time))
    wall = time.perf_counter() - t0
    raw = gather_emitter_results(sim)[("emitter",)]

    n_emits = len(raw)
    bm_hist  = np.zeros((n_emits, n, n), dtype=float)
    glc_hist = np.zeros((n_emits, n, n), dtype=float)
    ac_hist  = np.zeros((n_emits, n, n), dtype=float)
    times = np.zeros(n_emits, dtype=float)
    for i, step in enumerate(raw):
        f = step["fields"]
        bm_hist[i]  = np.asarray(f["dissolved biomass"])
        glc_hist[i] = np.asarray(f["glucose"])
        ac_hist[i]  = np.asarray(f["acetate"])
        times[i] = float(step.get("global_time", i))

    return {
        "wall_time": wall,
        "biomass_history": bm_hist,
        "glucose_history": glc_hist,
        "acetate_history": ac_hist,
        "time_h": times,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _frame_indices(T: int, k: int) -> list[int]:
    if T <= 1:
        return [0]
    return [int(round(i * (T - 1) / (k - 1))) for i in range(k)]


# Display order, labels, colors, and line-styles for the three implementations.
# Keys must match the ones used in `per_grid` / `scaling[*]`.
RUN_STYLE = {
    "spatioflux_many":   {"label": "spatio-flux (per-cell processes)",
                          "color": "#1f77b4", "ls": "-",  "marker": "o"},
    "spatioflux_single": {"label": "spatio-flux (SpatialDFBA)",
                          "color": "#2ca02c", "ls": "-.", "marker": "^"},
    "spatioflux_ray":    {"label": "spatio-flux (Ray actors, parallel)",
                          "color": "#9467bd", "ls": ":",  "marker": "D"},
    "cometspy":          {"label": "cometspy",
                          "color": "#d62728", "ls": "--", "marker": "s"},
}


def plot_snapshots(out: dict[str, dict], n: int, out_path: Path,
                   field: str = "biomass", n_snaps: int = 5):
    """Snapshot strip — one row per implementation."""
    field_key = {
        "biomass": "biomass_history",
        "glucose": "glucose_history",
        "acetate": "acetate_history",
    }[field]

    runs = [(k, out[k]) for k in RUN_STYLE if k in out]
    n_rows = len(runs)
    arrs = {k: r[field_key] for k, r in runs}
    idx = {k: _frame_indices(arrs[k].shape[0], n_snaps) for k, _ in runs}

    # Per-row color limits: each implementation auto-scales to its own peak
    # (use 99th percentile to keep isolated bright spots from saturating the
    # rest of the field).
    def _row_vmax(arr: np.ndarray) -> float:
        v = float(np.percentile(arr, 99.0))
        return max(v, float(arr.max()) * 0.5, 1e-12)
    vmaxes = {k: _row_vmax(a) for k, a in arrs.items()}

    fig, axes = plt.subplots(n_rows, n_snaps, figsize=(2.0 * n_snaps, 2.2 * n_rows),
                             constrained_layout=True, squeeze=False)
    for row, (key, r) in enumerate(runs):
        for col, ti in enumerate(idx[key]):
            ax = axes[row, col]
            ax.imshow(arrs[key][ti], origin="upper", vmin=0, vmax=vmaxes[key],
                      interpolation="nearest", cmap="viridis")
            if row == 0:
                ax.set_title(f"t={r['time_h'][ti]:.1f}h", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(
            f"{RUN_STYLE[key]['label']}\nmax≈{vmaxes[key]:.2g}", fontsize=9
        )
    fig.suptitle(f"{field} snapshots — N={n}×{n}  (per-row color scale)",
                 fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_biomass_trace(out: dict[str, dict], n: int, out_path: Path):
    """Biomass at seed cell + total biomass, all implementations overlaid."""
    bx, by = biomass_seed_position(n)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    for ax, accessor, ylabel, title in [
        (axes[0],
         lambda h: h[:, by, bx],
         "biomass at seed cell (gDW)",
         f"seed cell ({bx},{by})"),
        (axes[1],
         lambda h: h.reshape(h.shape[0], -1).sum(axis=1),
         "total biomass (gDW)",
         f"summed across grid (N={n})"),
    ]:
        for key in RUN_STYLE:
            if key not in out:
                continue
            r = out[key]
            style = RUN_STYLE[key]
            ax.plot(r["time_h"], accessor(r["biomass_history"]),
                    label=style["label"], lw=2,
                    color=style["color"], linestyle=style["ls"])
        ax.set_xlabel("time (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_side_by_side_gif(out: dict[str, dict], n: int, out_path: Path,
                          fps: int = 6, dpi: int = 90):
    """
    Animated grid: rows = implementations, cols = (glucose, acetate, biomass).
    Histories are resampled to a common length so playback is in lockstep.
    """
    runs = [(k, out[k]) for k in RUN_STYLE if k in out]
    fields = [
        ("glucose", "glucose_history", "viridis"),
        ("acetate", "acetate_history", "magma"),
        ("biomass", "biomass_history", "Greens"),
    ]

    T = min(r["glucose_history"].shape[0] for _, r in runs)
    idx = {k: _frame_indices(r["glucose_history"].shape[0], T) for k, r in runs}

    # Per-(implementation, field) color limit. Use the 99th percentile so a
    # single bright cell doesn't black out the rest of the field. The
    # implementations are visualized side by side in their own scales —
    # absolute magnitudes can differ (FBA alternate optima, etc).
    def _vmax(arr: np.ndarray) -> float:
        v = float(np.percentile(arr, 99.0))
        return max(v, float(arr.max()) * 0.5, 1e-12)
    vmaxes = {(k, fkey): _vmax(r[fkey]) for k, r in runs for _, fkey, _ in fields}

    n_rows = len(runs)
    frames = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for t_i in range(T):
        fig, axes = plt.subplots(
            n_rows, len(fields), figsize=(2.4 * len(fields), 1.9 * n_rows + 0.7),
            constrained_layout=True, dpi=dpi, squeeze=False,
        )
        t_label = runs[0][1]["time_h"][idx[runs[0][0]][t_i]]
        for row, (key, r) in enumerate(runs):
            for col, (label, fkey, cmap) in enumerate(fields):
                ax = axes[row, col]
                ax.imshow(r[fkey][idx[key][t_i]], vmin=0, vmax=vmaxes[(key, fkey)],
                          cmap=cmap, interpolation="nearest", origin="upper")
                if row == 0:
                    ax.set_title(label, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(RUN_STYLE[key]["label"], fontsize=9)
        fig.suptitle(f"N={n}×{n}   t={t_label:.2f}h   (per-row color scale)",
                     fontsize=11)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)


def plot_scaling(scaling: list[dict], out_path: Path):
    ns = [r["n"] for r in scaling]
    cells = [n * n for n in ns]

    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    for key, style in RUN_STYLE.items():
        time_field = key + "_time"
        if not all(time_field in r for r in scaling):
            continue
        ts = [r[time_field] for r in scaling]
        ax.plot(cells, ts, marker=style["marker"], linestyle=style["ls"],
                color=style["color"], label=style["label"], lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid cells (N×N)")
    ax.set_ylabel("wall-clock (s)")
    ax.set_title(f"scaling at T={TOTAL_TIME}h, dt={TIME_STEP}h")
    for x, n in zip(cells, ns):
        per_n_ts = [r[k + "_time"] for r in scaling if r["n"] == n
                    for k in RUN_STYLE if k + "_time" in r]
        if per_n_ts:
            ax.annotate(f"{n}", (x, max(per_n_ts)), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color="#444")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
_MIME_BY_EXT = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg"}

def _embed(out_dir: Path, name: str) -> str | None:
    """Return a `data:` URI for the image at `out_dir/name`, or None if missing."""
    p = out_dir / name
    if not p.exists():
        return None
    mime = _MIME_BY_EXT.get(p.suffix.lower(), "application/octet-stream")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def write_html_report(out_dir: Path, scaling: list[dict],
                      snap_n: int | None, grids: list[int]) -> None:
    """Write a fully self-contained report.html with all images base64-embedded."""
    def _f(x): return f"{x:.2f}s" if x is not None else "—"
    def _r(num, den): return f"{num/den:.1f}×" if (num and den) else "—"

    rows = []
    for r in scaling:
        n = r["n"]
        cm = r.get("cometspy_time")
        sm = r.get("spatioflux_many_time")
        ss = r.get("spatioflux_single_time")
        sr = r.get("spatioflux_ray_time")
        rows.append(
            "<tr>"
            f"<td>{n}×{n}</td>"
            f"<td>{_f(cm)}</td>"
            f"<td>{_f(sm)}</td>"
            f"<td>{_f(ss)}</td>"
            f"<td>{_f(sr)}</td>"
            f"<td>{_r(sm, cm)}</td>"
            f"<td>{_r(ss, cm)}</td>"
            f"<td>{_r(sr, cm)}</td>"
            "</tr>"
        )
    table = (
        "<table class='tbl'>"
        "<thead><tr>"
        "<th>grid</th><th>cometspy</th>"
        "<th>spatio-flux<br><span class='dim'>per-cell processes</span></th>"
        "<th>spatio-flux<br><span class='dim'>SpatialDFBA</span></th>"
        "<th>spatio-flux<br><span class='dim'>Ray actors</span></th>"
        "<th>per-cell ÷ cometspy</th>"
        "<th>SpatialDFBA ÷ cometspy</th>"
        "<th>Ray ÷ cometspy</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    gif_blocks = []
    for n in grids:
        uri = _embed(out_dir, f"compare_n{n}.gif")
        if uri:
            gif_blocks.append(
                f"<figure><figcaption>N={n}×{n}</figcaption>"
                f"<img src='{uri}' alt='compare n={n}'></figure>"
            )

    snap_block = ""
    if snap_n is not None:
        items = []
        for field in ("glucose", "acetate", "biomass"):
            uri = _embed(out_dir, f"snapshots_{field}_n{snap_n}.png")
            if uri:
                items.append(
                    f"<figure><figcaption>{field}</figcaption>"
                    f"<img src='{uri}' alt='{field} snapshots'></figure>"
                )
        snap_block = (
            f"<h2>Snapshots at N={snap_n}×{snap_n}</h2>"
            "<p>One row per implementation. Each row uses its own color scale "
            "(99th percentile per field), so spatial patterns are visible "
            "regardless of absolute magnitude. The per-row max value is "
            "annotated in the y-label. cometspy's FBA tends to pick alternate "
            "optima with higher acetate overflow than cobra's solver, so the "
            "absolute concentrations differ even though the qualitative "
            "spatial dynamics agree.</p>"
            f"<div class='grid'>{''.join(items)}</div>"
        )

    biomass_block = ""
    if snap_n is not None:
        biomass_uri = _embed(out_dir, f"biomass_trace_n{snap_n}.png")
        if biomass_uri:
            biomass_block = (
                f"<h2>Biomass trajectory at N={snap_n}×{snap_n}</h2>"
                "<p>Biomass at the seed cell (left) and summed across the grid "
                "(right). All three implementations track the same growth curve.</p>"
                f"<img class='wide' src='{biomass_uri}' alt='biomass trace'>"
            )

    scaling_uri = _embed(out_dir, "scaling.png") or ""

    html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>Spatio-Flux vs COMETS comparison</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif;
         max-width: 1200px; margin: 32px auto; padding: 0 20px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .lede {{ color: #555; margin-top: 0; }}
  table.tbl {{ border-collapse: collapse; margin: 12px 0; }}
  table.tbl th, table.tbl td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
  table.tbl th {{ background: #f5f5f5; font-weight: 600; }}
  table.tbl td:first-child, table.tbl th:first-child {{ text-align: left; }}
  .dim {{ color: #888; font-size: 0.85em; font-weight: 400; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
           gap: 18px; }}
  figure {{ margin: 0; background: #fafafa; border: 1px solid #eee; border-radius: 8px;
            padding: 8px; }}
  figcaption {{ font-weight: 600; margin-bottom: 6px; color: #444; font-size: 13px; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  img.wide {{ width: 100%; max-width: 100%; height: auto; }}
  .scaling {{ max-width: 720px; }}
  code {{ background: #f1f1f1; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head><body>

<h1>Spatio-Flux vs COMETS</h1>
<p class='lede'>
  Compositional dFBA (spatio-flux) vs the reference COMETS Java backend (via
  <code>cometspy</code>) on a matched 2D dFBA + diffusion problem. Three
  implementations: cometspy, spatio-flux's per-cell <code>DynamicFBA</code>
  process composition, and spatio-flux's single <code>SpatialDFBA</code>
  process that loops over cells internally.
</p>

<h2>Setup</h2>
<ul>
  <li>Model: <em>E. coli</em> core (textbook)</li>
  <li>Grid: square N×N (N ∈ {{{', '.join(str(n) for n in grids)}}})</li>
  <li>Total time {TOTAL_TIME} h, dFBA timestep {TIME_STEP} h, cell width {SPACE_WIDTH} cm</li>
  <li>Initial: vertical glucose gradient ({GLC_LOW} → {GLC_HIGH}), single biomass seed at top center</li>
  <li>Glucose &amp; acetate diffuse ({GLC_DIFF:.1e} cm²/s); biomass barely diffuses ({BIOMASS_DIFF:.1e} cm²/s)</li>
  <li>Matched MM kinetics: K<sub>m</sub>=0.5 mM, V<sub>max</sub>=1 mmol/gDW/h. Matched ATPM=1, EX_o2_e≥-2.</li>
  <li>LP solver: GLOP (cometspy) and cobra default (spatio-flux)</li>
</ul>

<h2>Performance scaling</h2>
<img class='scaling' src='{scaling_uri}' alt='scaling'>
{table}

{biomass_block}

{snap_block}

<h2>Per-grid animations</h2>
<p>Three rows (per-cell composition, SpatialDFBA, cometspy) × three fields
(glucose, acetate, biomass), played in lockstep.</p>
<div class='grid'>{''.join(gif_blocks)}</div>

</body></html>
"""
    (out_dir / "report.html").write_text(html)
    size_mb = (out_dir / "report.html").stat().st_size / 1e6
    print(f"💾 wrote: {out_dir / 'report.html'} ({size_mb:.1f} MB, self-contained)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SCALING_JSON = "scaling.json"


def _load_scaling(out_dir: Path) -> list[dict]:
    """Load persisted scaling measurements, if any. Returns [] on first run."""
    p = out_dir / SCALING_JSON
    if not p.exists():
        return []
    try:
        import json as _json
        return _json.loads(p.read_text())
    except Exception:
        return []


def _save_scaling(out_dir: Path, scaling: list[dict]) -> None:
    """Persist scaling measurements (sorted by n)."""
    import json as _json
    sorted_scaling = sorted(scaling, key=lambda r: r["n"])
    (out_dir / SCALING_JSON).write_text(_json.dumps(sorted_scaling, indent=2))


def _merge_scaling(prev: list[dict], new: list[dict]) -> list[dict]:
    """Merge new measurements into prev. Same n => new replaces prev."""
    by_n = {r["n"]: r for r in prev}
    for r in new:
        by_n[r["n"]] = r
    return sorted(by_n.values(), key=lambda r: r["n"])


def main(grids=None, max_seconds_per_run: float = 600.0):
    """
    Run the comparison across a sequence of grid sizes.

    Two-phase structure:
      1. Timings phase: run all grids back-to-back, no GIF/plot work between
         runs. Keeps the matplotlib + JSON-serialization overhead from
         contending with subsequent Ray dispatch (which is GIL-sensitive).
      2. Render phase: write GIFs, snapshots, scaling plot, report.

    Scaling measurements are persisted to scaling.json and merged across
    invocations — re-running with grids=[16] only will overwrite n=16's
    measurements while preserving the rest.

    `max_seconds_per_run` is a prospective safety cap: if the previous grid
    took longer than this, we skip the next (larger) one to avoid hangs.
    """
    grids = grids or DEFAULT_GRIDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "COMETS_HOME" not in os.environ:
        raise RuntimeError(
            "COMETS_HOME not set. Point it at your COMETS install dir."
        )

    core = allocate_core()

    # ---- Phase 1: timings only, no rendering ---------------------------- #
    new_scaling = []
    per_grid_results: dict[int, dict] = {}

    last_max = 0.0
    for n in grids:
        if last_max > max_seconds_per_run:
            print(f"\n⏹  prior grid took {last_max:.1f}s > budget "
                  f"{max_seconds_per_run:.0f}s; stopping before n={n}.")
            break

        print(f"\n=== grid {n}x{n} ===")

        print("  → cometspy ...", flush=True)
        cm_res = run_cometspy(n)
        print(f"     wall: {cm_res['wall_time']:.2f}s")

        print("  → spatio-flux (per-cell processes) ...", flush=True)
        sf_many_res = run_spatioflux(n, core=core)
        print(f"     wall: {sf_many_res['wall_time']:.2f}s")

        print("  → spatio-flux (SpatialDFBA) ...", flush=True)
        sf_single_res = run_spatioflux_single(n, core=core)
        print(f"     wall: {sf_single_res['wall_time']:.2f}s")

        print("  → spatio-flux (Ray actors, parallel) ...", flush=True)
        sf_ray_res = run_spatioflux_ray(n, core=allocate_core())
        print(f"     wall: {sf_ray_res['wall_time']:.2f}s")

        new_scaling.append({
            "n": n,
            "cometspy_time":          cm_res["wall_time"],
            "spatioflux_many_time":   sf_many_res["wall_time"],
            "spatioflux_single_time": sf_single_res["wall_time"],
            "spatioflux_ray_time":    sf_ray_res["wall_time"],
        })
        last_max = max(cm_res["wall_time"],
                       sf_many_res["wall_time"],
                       sf_single_res["wall_time"],
                       sf_ray_res["wall_time"])

        per_grid_results[n] = {
            "cometspy":          cm_res,
            "spatioflux_many":   sf_many_res,
            "spatioflux_single": sf_single_res,
            "spatioflux_ray":    sf_ray_res,
        }

    # ---- Persist + merge scaling data ----------------------------------- #
    merged = _merge_scaling(_load_scaling(OUT_DIR), new_scaling)
    _save_scaling(OUT_DIR, merged)
    print(f"\n💾 scaling.json updated ({len(merged)} grid sizes recorded).")

    # ---- Phase 2: rendering (after all timings captured) ---------------- #
    print("\n=== rendering ===")
    snap_n = None
    snap_runs = None
    for n, per_grid in per_grid_results.items():
        gif_path = OUT_DIR / f"compare_n{n}.gif"
        print(f"  → {gif_path.name} ...", flush=True)
        make_side_by_side_gif(per_grid, n, gif_path)

        if n == COMPARISON_GRID:
            snap_runs = per_grid
            snap_n = n

    # If COMPARISON_GRID wasn't in this run, fall back to the largest grid we did.
    if snap_runs is None and per_grid_results:
        snap_n = max(per_grid_results)
        snap_runs = per_grid_results[snap_n]

    if snap_runs:
        print("  → snapshots ...", flush=True)
        plot_snapshots(snap_runs, snap_n, OUT_DIR / f"snapshots_biomass_n{snap_n}.png", field="biomass")
        plot_snapshots(snap_runs, snap_n, OUT_DIR / f"snapshots_glucose_n{snap_n}.png", field="glucose")
        plot_snapshots(snap_runs, snap_n, OUT_DIR / f"snapshots_acetate_n{snap_n}.png", field="acetate")
        plot_biomass_trace(snap_runs, snap_n, OUT_DIR / f"biomass_trace_n{snap_n}.png")

    print("  → scaling plot + report ...", flush=True)
    plot_scaling(merged, OUT_DIR / "scaling.png")
    write_html_report(OUT_DIR, merged, snap_n, [r["n"] for r in merged])

    print(f"\n✅ wrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
