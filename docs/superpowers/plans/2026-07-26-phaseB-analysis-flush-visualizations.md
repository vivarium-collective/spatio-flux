# Phase B: Post-Run Analysis-Flush Visualizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-07-26): EXECUTED.** All 19 scenarios reproduce their figures via
> the post-run flush (Tasks 1–7). Tests: `test_analysis_flush.py` + `test_fidelity.py`
> green — 19/19 completeness, deterministic scenarios match `out0/` within 0.08 MAD
> (quantitative timeseries ≤0.044). Deviations from the plan as written:
> (1) `FLUSH_SPEC` lives in `spatio_flux/analysis/flush_spec.py` (package code) not
> `scripts/`, so the runner imports package, not scripts; (2) structure artifacts
> (`_viz.png`/`_schema.json`/`_state.json`) come from `run_composite_document`, not a
> separate flush step; (3) runner reads baseline ref/params from `study.yaml` (not a
> STUDY_INDEX); (4) two env fixes were required — system **graphviz** (bigraph diagram)
> and a `matplotlib.cm.get_cmap` shim for matplotlib≥3.9; (5) the "verbatim no-extension"
> filename quirk was dropped (current `plot_time_series` + `out0` both use `.png`).
> **Still TODO:** SQLite `runs.db` persistence (dashboard run-history; figures already
> surface via `image:charts/*`) and recording `runs[].outcomes` (Task 7 Step 4).
>
> GIF fidelity vs `out0` is informational only — the historical oracle was captured at a
> different emit cadence, misaligning frame-by-frame comparison.

**Goal:** Reproduce every figure, GIF, diagram, and serialized artifact the current test suite emits — for all 19 studies — as a **post-simulation analysis flush** (Visualization/analysis Steps run after the run, NOT embedded in the composite), writing byte-for-purpose-identical files into `studies/<slug>/charts/` and `studies/<slug>/viz/`, validated against the `out0/` reference set.

**Architecture:** A per-study **bespoke runner** (`canonical_runs:` convention) builds the study's composite, runs it while persisting the full per-tick trajectory to `studies/<slug>/runs.db` (SQLite, dashboard-visible), then invokes a **post-run analysis flush** that replays the trajectory through the existing `spatio_flux/plots/plot.py` primitives — the fidelity guarantee, reused verbatim — pointing their `out_dir` at the study's `charts/`. Each plot primitive is wrapped as a registered `Visualization` subclass so it appears in the dashboard Registry and is declared per-study. The 5 existing Path-A HTML viz Steps are **unwired from the composites** (`composites/metabolism.py`, `spatial.py`, `comets.py`) since visualizations no longer live in the composite.

**Tech Stack:** Python 3.12, `process-bigraph` (SQLiteEmitter, Composite), `spatio_flux.plots.plot` (matplotlib/imageio primitives — reused), `viva_superpowers.visualization.Visualization`, `pyyaml`, `pytest`, Pillow/numpy (image diffing).

## Global Constraints

- **Worktree:** `~/code/spatio-flux--workbench-modernization`, branch `workbench-modernization`. Verify `git branch --show-current` + HEAD before every commit.
- **Serving venv:** the dedicated `.venv-serve` from Phase A Task 5 (has `spatio_flux` + `viva_superpowers` + `vivarium-workbench`). Runners execute under it. Unit tests that only touch `plot.py` + trajectory shapes can run under the shared `.venv` with `PYTHONPATH=$PWD`.
- **Visualizations are NOT in the composite** (approved decision). Any `local:<Viz>` wiring in `composites/*.py` is removed in Task 5. The flush is the only render path.
- **Reuse `plot.py` verbatim.** Do not reimplement any plotting math — faithful reproduction comes from calling the exact same primitives with the exact same config the current `plot_func`s use. The per-scenario config table (Task 3) copies those call sites.
- **Fidelity oracle is `out0/`** (full 19-scenario reference run). `out_keep/` is only the mega/multicomponent integrated-demo figures — not the per-scenario oracle. Images match within a small tolerance (see Task 6 tolerance rationale); exact-match is not required because matplotlib/imageio are not bit-reproducible across runs.
- **Exact output filenames** per scenario are fixed by the current `plot_func`s (Task 3 table). Two legacy quirks are preserved: `community_dfba` and `dfba_kinetics_community` timeseries are written with the filename **verbatim, without `.png`** (`test_suite.py:130,147`); the `_video.gif` for brownian/particle scenarios is written twice (plain `plot_particles` then overwritten by the with-particles gif) — reproduce only the final artifact.

---

## Scenario → outputs map (authoritative, from the current plot_funcs)

Each row lists the files a study's flush must produce (prefix = slug). "Diagram/schema/state" (`<slug>_viz.png`, `_schema.json`, `_state.json`) are produced for every study by the **structure step** (Task 2).

| slug | timeseries | gif | snapshots | mass | mass_submasses | traces | model_grid | flush primitives (plot.py) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| monod_kinetics | `monod_kinetics.png` | | | | | | | plot_time_series (scalar) |
| ecoli_core_dfba | `ecoli_core_dfba.png` | | | | | | | plot_time_series (scalar) |
| ecoli_dfba | `ecoli_dfba.png` | | | | | | | plot_time_series (scalar) |
| yeast_dfba | `yeast_dfba.png` | | | | | | | plot_time_series (scalar) |
| community_dfba | `community_dfba` (no ext) | | | | | | | plot_time_series (log,normalize) |
| dfba_kinetics_community | `dfba_kinetics_community` (no ext) | | | | | | | plot_time_series (scalar) |
| spatial_many_dfba | `_timeseries.png` | `_video.gif` | | | | | | plot_time_series(coords), species_dist_gif |
| spatial_dfba_process | `_timeseries.png` | `_video.gif` | | | | | `_model_grid.png` | +plot_model_grid |
| diffusion_process | | `_video.gif` | | | | | | species_dist_gif |
| brownian_particles | | `_video.gif` | `_snapshots.png` | `_mass.png` | | `_particles_traces.png` | | species_dist_with_particles_gif, snapshots_grid, particles_mass, particle_traces |
| br_particles_kinetics | | `_video.gif` | `_snapshots.png` | `_mass.png` | | `_particles_traces.png` | | (same as brownian_particles) |
| br_particles_dfba | | `_video.gif` | `_snapshots.png` | `_mass.png` | | `_particles_traces.png` | | (same, particles_row='separate') |
| comets_diffusion | `_timeseries.png` | `_video.gif` | `_snapshots.png` | | | | | plot_time_series(coords), snapshots_grid, species_dist_gif |
| comets_br_particles_kinetics | `_timeseries.png` | `_video.gif` | `_snapshots.png` | `_mass.png` | | | | +particles_mass, species_dist_with_particles_gif |
| comets_br_particles_dfba | `_timeseries.png` | `_video.gif` | `_snapshots.png` | `_mass.png` | | | | (same, 3 fields, n_snapshots=4) |
| newtonian_particles | | `_video.gif` | | `_mass.png` | | `_particles_traces.png` | | particles_mass, pymunk_simulation_to_gif, particle_traces |
| comets_nt_particles_dfba | `_timeseries.png` | `_video.gif` | `_snapshots.png` | `_mass.png` | `_mass_submasses.png` | | | plot_time_series(coords), particles_mass, particles_mass_with_submasses, fields_and_agents_to_gif, snapshots_grid(submasses) |
| spatioflux_reference_demo | `_timeseries.png` | `_video.gif` | `_snapshots.png` | `_mass.png` | `_mass_submasses.png` | | | (same as comets_nt_particles_dfba, n_snapshots=8) |
| reference_demo_x2y2 | `_timeseries.png` | `_video.gif` | `_snapshots.png` | `_mass.png` | `_mass_submasses.png` | | | (same, n_bins doubled) |

Data-shape note (drives the flush's trajectory reader): metabolism `fields` are **scalars**; spatial/COMETS `fields` are **2D `np.ndarray[n_bins]`**; particle scenarios add `particles` = per-tick `{pid: {position, mass, sub_masses, ...}}`. The flush replays the in-memory trajectory (nested arrays intact), so no typed-wire truncation applies.

---

## File Structure

- Create: `spatio_flux/analysis/__init__.py` — registers the analysis-flush Step classes.
- Create: `spatio_flux/analysis/flush.py` — `run_analysis_flush(slug, trajectory, state, out_dir, specs)`: the post-run orchestrator that dispatches to the registered analysis steps.
- Create: `spatio_flux/analysis/steps.py` — thin `Visualization` subclasses wrapping each `plot.py` primitive so they're dashboard-discoverable and file-writing; each `render_to_files(trajectory, state, out_dir, config)`.
- Create: `spatio_flux/analysis/structure.py` — the structure step (bigraph diagram `<slug>_viz.png` + `_schema.json` + `_state.json`), lifting the logic from `library/tools.py:run_composite_document`.
- Create: `spatio_flux/runners/run_study.py` — the bespoke `canonical_runs:` runner: build composite → run with SQLite emitter into `studies/<slug>/runs.db` → analysis flush into `studies/<slug>/charts/`.
- Modify: `scripts/scaffold_studies.py` — add per-study `canonical_runs:` + `visualizations:` (`image:charts/<file>`) + `behavior_tests:` from a FLUSH_SPEC table.
- Modify: `spatio_flux/composites/metabolism.py`, `spatial.py`, `comets.py` — remove `local:<Viz>` step wiring (decouple).
- Create: `tests/test_analysis_flush.py` — flush harness unit tests + fidelity diff vs `out0/`.
- Create: `tests/fidelity/` — image-diff helper.

---

### Task 1: Analysis-flush harness + trajectory replay

**Files:**
- Create: `spatio_flux/analysis/__init__.py`, `spatio_flux/analysis/flush.py`
- Test: `tests/test_analysis_flush.py`

**Interfaces:**
- Produces: `run_analysis_flush(slug: str, trajectory: list[dict], state: dict, out_dir: str, specs: list[dict]) -> list[str]` — runs each spec's analysis step against the trajectory, returns the list of written file paths. `specs` items: `{step: <registered-name>, config: {...}}`.
- Consumes: the registered analysis-step classes from `spatio_flux.analysis.steps` (Task 2), resolved by name.

- [ ] **Step 1: Write the failing test** (harness dispatches to a stub step and returns written paths)

```python
# tests/test_analysis_flush.py
import os

def test_flush_dispatches_and_returns_paths(tmp_path):
    from spatio_flux.analysis.flush import run_analysis_flush, ANALYSIS_STEPS
    calls = []
    ANALYSIS_STEPS["_stub"] = lambda traj, state, out_dir, config: (
        calls.append((len(traj), config)) or [os.path.join(out_dir, "x.png")]
    )
    traj = [{"global_time": 0.0, "fields": {"glucose": 1.0}},
            {"global_time": 1.0, "fields": {"glucose": 0.5}}]
    paths = run_analysis_flush("demo", traj, {}, str(tmp_path),
                               [{"step": "_stub", "config": {"filename": "x"}}])
    assert calls == [(2, {"filename": "x"})]
    assert paths == [str(tmp_path / "x.png")]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q`
Expected: FAIL — `spatio_flux.analysis` missing.

- [ ] **Step 3: Implement the harness**

```python
# spatio_flux/analysis/__init__.py
from spatio_flux.analysis import steps as _steps  # noqa: F401 (register)

# spatio_flux/analysis/flush.py
"""Post-simulation analysis flush.

Replays a run's trajectory through registered analysis steps that write figure
files. This is the post-sim flush the study runner calls after the composite
finishes — visualizations are NOT part of the composite.
"""
import os

# name -> callable(trajectory, state, out_dir, config) -> list[str] (paths written)
ANALYSIS_STEPS = {}


def register_step(name):
    def deco(fn):
        ANALYSIS_STEPS[name] = fn
        return fn
    return deco


def run_analysis_flush(slug, trajectory, state, out_dir, specs):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for spec in specs:
        fn = ANALYSIS_STEPS[spec["step"]]
        written.extend(fn(trajectory, state, out_dir, spec.get("config", {})) or [])
    return written
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/analysis/ tests/test_analysis_flush.py
git commit -m "feat(analysis): add post-run analysis-flush harness"
```

---

### Task 2: Wrap plot.py primitives as file-writing analysis steps + structure step

**Files:**
- Create: `spatio_flux/analysis/steps.py`, `spatio_flux/analysis/structure.py`
- Test: `tests/test_analysis_flush.py` (extend)

**Interfaces:**
- Produces registered steps (names): `timeseries`, `species_dist_gif`, `species_dist_with_particles_gif`, `snapshots_grid`, `particles_mass`, `particles_mass_submasses`, `particle_traces`, `model_grid`, `pymunk_gif`, `fields_agents_gif`, `structure`. Each is `fn(trajectory, state, out_dir, config) -> list[str]`, delegating to the matching `spatio_flux.plots.plot` primitive (and `structure.py` for the bigraph diagram/schema/state).
- Consumes: `run_analysis_flush` registry from Task 1; `spatio_flux.plots.plot.*`; `viva_superpowers.visualization.Visualization` (each step is also exposed as a discoverable subclass).

- [ ] **Step 1: Write the failing test** (each step writes its expected file from a synthetic trajectory)

```python
def _spatial_traj(n=3, bins=(4, 4)):
    import numpy as np
    return [{"global_time": float(t),
             "fields": {"glucose": np.ones(bins) * (n - t),
                        "acetate": np.zeros(bins)}} for t in range(n)]

def test_timeseries_step_writes_png(tmp_path):
    from spatio_flux.analysis.steps import timeseries_step
    traj = [{"global_time": float(t), "fields": {"glucose": 10.0 - t, "biomass": 0.1 * t}}
            for t in range(5)]
    out = timeseries_step(traj, {"fields": {"glucose": 0, "biomass": 0}},
                          str(tmp_path), {"filename": "demo", "mode": "scalar"})
    assert out and os.path.exists(out[0])
    assert out[0].endswith("demo.png")

def test_species_gif_step_writes_gif(tmp_path):
    from spatio_flux.analysis.steps import species_dist_gif_step
    out = species_dist_gif_step(_spatial_traj(), {}, str(tmp_path),
                                {"filename": "demo_video"})
    assert out and out[0].endswith("demo_video.gif") and os.path.exists(out[0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q`
Expected: FAIL — `spatio_flux.analysis.steps` missing.

- [ ] **Step 3: Implement the steps** (delegating to plot.py; one wrapper per primitive)

Representative wrappers (implement all names listed in Interfaces the same way — each maps config → the exact `plot.py` call the current `plot_func` uses):

```python
# spatio_flux/analysis/steps.py
"""Analysis-flush steps: thin file-writing wrappers over spatio_flux.plots.plot,
so faithful reproduction is guaranteed by reusing the exact plotting code.

Each function has signature (trajectory, state, out_dir, config) -> list[str].
Registered into the flush harness AND exposed as Visualization subclasses for
dashboard discovery.
"""
import os
from spatio_flux.analysis.flush import register_step
from spatio_flux.plots import plot as P


@register_step("timeseries")
def timeseries_step(trajectory, state, out_dir, config):
    fname = config["filename"]
    P.plot_time_series(
        trajectory,
        field_names=config.get("field_names"),
        coordinates=config.get("coordinates"),
        log_scale=config.get("log_scale", False),
        normalize=config.get("normalize", False),
        out_dir=out_dir, filename=fname if fname.endswith((".png", "")) else f"{fname}.png",
        title=config.get("title", ""),
        time_units=config.get("time_units", "min"),
        field_units=config.get("field_units"),
        field_colors=config.get("field_colors"),
        legend_kwargs=config.get("legend_kwargs"),
        figsize=config.get("figsize"),
    )
    # Preserve the legacy no-extension filenames for community_dfba / dfba_kinetics_community.
    written = os.path.join(out_dir, fname if config.get("verbatim_filename") else (
        fname if fname.endswith(".png") else f"{fname}.png"))
    return [written]


@register_step("species_dist_gif")
def species_dist_gif_step(trajectory, state, out_dir, config):
    fname = f'{config["filename"]}.gif' if not config["filename"].endswith(".gif") else config["filename"]
    P.plot_species_distributions_to_gif(
        trajectory, out_dir=out_dir, filename=fname,
        species_to_show=config.get("species_to_show"))
    return [os.path.join(out_dir, fname)]

# ... one wrapper each for: species_dist_with_particles_gif (P.plot_species_distributions_with_particles_to_gif),
#     snapshots_grid (P.plot_snapshots_grid), particles_mass (P.plot_particles_mass),
#     particles_mass_submasses (P.plot_particles_mass_with_submasses),
#     particle_traces (P.plot_particle_traces), model_grid (P.plot_model_grid),
#     pymunk_gif (pymunk_simulation_to_gif), fields_agents_gif (P.fields_and_agents_to_gif).
# Each: translate config keys to the primitive's kwargs exactly as the matching
# plot_func in test_suite.py does (Task 3 supplies the per-scenario config), and
# return the written path(s).
```

```python
# spatio_flux/analysis/structure.py
"""Structure step: bigraph diagram + serialized schema/state per study.
Lifts the artifact-writing block from library/tools.run_composite_document so
studies reproduce <slug>_viz.png / _schema.json / _state.json without a run.
"""
import os, json
from spatio_flux.analysis.flush import register_step


@register_step("structure")
def structure_step(trajectory, state, out_dir, config):
    slug = config["filename"]
    core = config["core"]; schema = config["schema"]; st = config["state"]
    paths = []
    # schema.json
    p = os.path.join(out_dir, f"{slug}_schema.json")
    with open(p, "w") as f:
        json.dump(core.render(schema), f, indent=2, default=str); paths.append(p)
    # state.json (best-effort, mirrors tools.py try/except)
    try:
        p = os.path.join(out_dir, f"{slug}_state.json")
        with open(p, "w") as f:
            json.dump(core.serialize(schema, st), f, indent=2, default=str); paths.append(p)
    except Exception:
        pass
    # <slug>_viz.png via bigraph_viz.plot_bigraph — same plot settings as tools.py:146-173
    from bigraph_viz import plot_bigraph  # noqa
    # (reproduce the plot_state/plot_schema pruning + settings from tools.py)
    # ... plot_bigraph(...) writing f"{slug}_viz.png" ...
    paths.append(os.path.join(out_dir, f"{slug}_viz.png"))
    return paths
```

Also expose each step as a discoverable `Visualization` subclass (so the dashboard Registry lists them). One class per step name, delegating to the same function:

```python
from viva_superpowers.visualization import Visualization

class TimeSeriesAnalysis(Visualization):
    step_name = "timeseries"
    def inputs(self): return {}          # Path C: reads the flush trajectory, not a wire
    def render(self): return "<em>analysis-flush step; see charts/</em>"
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q`
Expected: PASS (harness + timeseries + gif steps).

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/analysis/steps.py spatio_flux/analysis/structure.py tests/test_analysis_flush.py
git commit -m "feat(analysis): file-writing plot.py wrappers + structure step"
```

---

### Task 3: FLUSH_SPEC table + scaffolder emits canonical_runs/visualizations

**Files:**
- Modify: `scripts/scaffold_studies.py`
- Test: `tests/test_analysis_flush.py` (extend)

**Interfaces:**
- Produces: `FLUSH_SPEC: dict[slug, list[{step, config}]]` in the scaffolder — the complete per-scenario analysis-step list copied from the current `plot_func` call sites (config values: filenames, coordinates, field_names, n_snapshots, colors, units, particles_row, submass_color_map). Consumed by `render_study` to emit each study's `canonical_runs:` (one entry pointing at `run_study.py`) and `visualizations:` (`address: image:charts/<file>` per produced file) and structural `behavior_tests:`.

- [ ] **Step 1: Write the failing test**

```python
def test_flush_spec_covers_all_19_and_matches_output_map():
    from scripts.scaffold_studies import FLUSH_SPEC, STUDIES
    assert set(FLUSH_SPEC) == {s["slug"] for s in STUDIES}
    # spot-check the reference demo produces the 5 expected artifact kinds
    steps = {e["step"] for e in FLUSH_SPEC["spatioflux_reference_demo"]}
    assert {"timeseries", "particles_mass", "particles_mass_submasses",
            "fields_agents_gif", "snapshots_grid", "structure"} <= steps

def test_scaffolder_emits_canonical_runs_and_viz():
    from scripts.scaffold_studies import render_study, STUDIES
    demo = next(s for s in STUDIES if s["slug"] == "comets_diffusion")
    spec = render_study(demo)
    assert spec["canonical_runs"][0]["script"] == "spatio_flux/runners/run_study.py"
    assert any(v["address"].startswith("image:charts/") for v in spec["visualizations"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q`
Expected: FAIL — `FLUSH_SPEC` missing.

- [ ] **Step 3: Add `FLUSH_SPEC` + extend `render_study`.** Populate `FLUSH_SPEC` from the scenario→outputs map above, copying each `plot_func`'s exact config (from `test_suite.py` line refs in the map). Extend `render_study` to append:

```python
"canonical_runs": [{
    "name": "reproduce",
    "script": "spatio_flux/runners/run_study.py",
    "args": [slug, str(entry.get("time", 60))],
    "label": f"reproduce {slug} figures",
    "default": True,
}],
"visualizations": [
    {"name": f, "address": f"image:charts/{f}", "chart": "image"}
    for f in _expected_files(slug)   # derived from FLUSH_SPEC configs
],
"behavior_tests": [{
    "name": f"{slug.upper()}-REPRODUCES-REPORT",
    "classification": "regression",
    "description": "All test-suite artifacts for this scenario are reproduced and match the out0 reference within tolerance.",
    "measure": {"kind": "artifacts_present", "expected": _expected_files(slug)},
    "pass_if": {"op": "all_exist_and_match", "tolerance": 0.02},
    "requires_simulation": "reproduce",
}],
```

Regenerate: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python scripts/scaffold_studies.py`.

- [ ] **Step 4: Run to verify it passes** — Phase A tests + the two new ones.

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/ -q`
Expected: PASS (Phase A suite still green; new FLUSH_SPEC tests pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/scaffold_studies.py studies/ tests/test_analysis_flush.py
git commit -m "feat(studies): FLUSH_SPEC + per-study canonical_runs/visualizations/behavior_tests"
```

---

### Task 4: The study runner (`run_study.py`) — run + persist + flush

**Files:**
- Create: `spatio_flux/runners/run_study.py`, `spatio_flux/runners/__init__.py`
- Test: `tests/test_analysis_flush.py` (extend — end-to-end on the cheapest scenario)

**Interfaces:**
- Produces: `python spatio_flux/runners/run_study.py <slug> <time>` — from workspace root: builds the study's baseline composite (via `build_generator` on the registry entry + the study's `params`), runs `time` steps with a SQLite emitter writing `studies/<slug>/runs.db`, then calls `run_analysis_flush(slug, trajectory, state, "studies/<slug>/charts", FLUSH_SPEC[slug])`. Emits `studies/<slug>/<slug>_timing.json`.
- Consumes: `spatio_flux.core.build_core`, `pbg_superpowers.composite_generator.build_generator`, `scripts.scaffold_studies.{STUDIES,FLUSH_SPEC}`, `spatio_flux.analysis.flush.run_analysis_flush`, `process_bigraph` SQLiteEmitter.

- [ ] **Step 1: Write the failing test** (run the cheapest scenario end-to-end; assert files + runs.db)

```python
def test_run_study_end_to_end_monod(tmp_path, monkeypatch):
    # monod_kinetics: shortest, scalar fields, single timeseries PNG.
    import subprocess, sys, os, shutil
    ws = tmp_path / "ws"; shutil.copytree(REPO, ws, ignore=shutil.ignore_patterns(
        ".git", "out0", "out_1226", "out_keep", ".venv"))
    r = subprocess.run([sys.executable, "spatio_flux/runners/run_study.py",
                        "monod_kinetics", "10"], cwd=ws, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ws)})
    assert r.returncode == 0, r.stderr
    assert (ws / "studies/monod_kinetics/charts/monod_kinetics.png").exists()
    assert (ws / "studies/monod_kinetics/runs.db").exists()
```
(REPO import as in `tests/test_workspace_investigation.py`. This test is slow — mark `@pytest.mark.slow`.)

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q -k end_to_end`
Expected: FAIL — runner missing.

- [ ] **Step 3: Implement `run_study.py`** — build composite, run with SQLite emitter to `studies/<slug>/runs.db`, keep the trajectory in memory for the flush (faithful nested arrays), then flush. Mirror `library/tools.run_composite_document` for the run/gather, but target the per-study runs.db and charts dir.

```python
#!/usr/bin/env python
"""Bespoke study runner (canonical_runs entry). From workspace root:
    python spatio_flux/runners/run_study.py <slug> <time>
Builds the baseline composite, runs it while persisting per-tick history to
studies/<slug>/runs.db, then runs the post-sim analysis flush into charts/.
"""
import json, os, sys, time as _time
from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.analysis.flush import run_analysis_flush
from scripts.scaffold_studies import STUDIES, FLUSH_SPEC


def main(slug, runtime):
    entry_meta = next(s for s in STUDIES if s["slug"] == slug)
    core = build_core()
    gen = _REGISTRY[entry_meta["ref"]]
    doc = build_generator(gen, overrides=entry_meta["params"], core=core)
    study_dir = os.path.join("studies", slug)
    charts = os.path.join(study_dir, "charts")
    os.makedirs(charts, exist_ok=True)
    # build composite with a SQLite emitter -> studies/<slug>/runs.db, run, gather trajectory
    # (use process_bigraph Composite + SQLiteEmitter; reuse tools.run_composite_document
    #  logic but point the emitter at runs.db and keep results in memory)
    from spatio_flux.library.tools import run_composite_document
    results, proc_t, fw_t = run_composite_document(
        doc, core=core, name=slug, time=int(runtime),
        outdir=charts, show_types=True, show_values=True)   # writes structure artifacts too
    # append the structure config the flush needs, then flush the figures
    specs = list(FLUSH_SPEC[slug])
    for s in specs:
        if s["step"] == "structure":
            s["config"].update({"core": core, "schema": doc.get("schema"),
                                 "state": doc.get("state", doc)})
    run_analysis_flush(slug, results, doc.get("state", doc), charts, specs)
    with open(os.path.join(study_dir, f"{slug}_timing.json"), "w") as f:
        json.dump({"process_time": proc_t, "framework_time": fw_t,
                   "elapsed": proc_t + fw_t}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "60")
```

(Implementation note: if `run_composite_document`'s `outdir` param does not already exist, add it — currently it hardcodes `"out"`; parametrizing it is part of this task. Confirm the SQLite emitter wiring writes `runs.db` under `study_dir`; if `run_composite_document` uses a RAM emitter, add a `runs_db_path` argument that swaps in `process_bigraph.emitter.SQLiteEmitter` so the run is dashboard-visible while `results` stays in memory for the flush.)

- [ ] **Step 4: Run to verify it passes** (under the serving venv for full deps)

Run: `~/code/spatio-flux--workbench-modernization/.venv-serve/bin/python -m pytest tests/test_analysis_flush.py -q -k end_to_end`
Expected: PASS — `charts/monod_kinetics.png` + `runs.db` exist.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/runners/ spatio_flux/library/tools.py tests/test_analysis_flush.py
git commit -m "feat(runner): run_study.py — run + persist runs.db + analysis flush"
```

---

### Task 5: Decouple the Path-A viz Steps from the composites

**Files:**
- Modify: `spatio_flux/composites/metabolism.py`, `spatio_flux/composites/spatial.py`, `spatio_flux/composites/comets.py`
- Test: `tests/test_analysis_flush.py` (extend)

- [ ] **Step 1: Write the failing test** (no composite spec contains a `local:` viz step)

```python
def test_no_visualization_steps_wired_into_composites():
    import re, glob
    pat = re.compile(r"local:(TestSuiteTimeSeries|FieldSnapshotsGrid|FieldAnimationGif|FieldHeatmap|ParticleTraces)")
    hits = []
    for p in glob.glob(os.path.join(REPO, "spatio_flux", "composites", "*.py")):
        with open(p) as f:
            if pat.search(f.read()):
                hits.append(os.path.basename(p))
    assert hits == [], f"viz still wired into composites: {hits}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_analysis_flush.py -q -k no_visualization`
Expected: FAIL — metabolism/spatial/comets still wire `local:` viz steps.

- [ ] **Step 3: Remove the viz step blocks** from each composite generator (the `viz_*` step dicts + their `outputs` stores). Grep `local:` in `spatio_flux/composites/*.py` and delete each visualization step entry and any now-orphaned `*_html` store it wrote to. Do NOT touch process wiring. The 5 Step classes in `spatio_flux/visualizations/` stay (still importable/discoverable) but are no longer embedded.

- [ ] **Step 4: Run to verify it passes** — the decouple test + Phase A resolution test (composites still build).

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/ -q`
Expected: PASS. Also sanity-run one composite build: `build_generator(_REGISTRY["spatio_flux.composites.comets.comets_diffusion"], overrides={}, core=build_core())` returns without referencing a missing viz store.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/composites/
git commit -m "refactor(composites): unwire visualization steps (viz moves to post-run flush)"
```

---

### Task 6: Fidelity validation against `out0/`

**Files:**
- Create: `tests/fidelity/image_diff.py`, `tests/test_fidelity.py`
- Test: as above

**Interfaces:**
- Produces: `image_similar(path_a, path_b, tolerance=0.02) -> bool` — normalized mean-abs pixel difference for PNGs; for GIFs, compare frame count + mean per-frame diff. Tolerance rationale: matplotlib/imageio are not bit-reproducible (font hinting, antialiasing, encoder), so exact match is impossible; 2% normalized MAD catches structural regressions (missing series, wrong colormap, wrong panel count) while tolerating rendering jitter.

- [ ] **Step 1: Write the failing test** (reference demo artifacts match out0 within tolerance)

```python
# tests/test_fidelity.py  (slow; runs under serving venv)
import os, pytest
REPO = ...  # as elsewhere

@pytest.mark.slow
@pytest.mark.parametrize("slug", ["monod_kinetics", "diffusion_process", "comets_diffusion"])
def test_flush_matches_out0(slug, tmp_path):
    from tests.fidelity.image_diff import image_similar
    # run the study into a temp workspace (as in Task 4), then diff each expected file
    charts = _run_study_into_tmp(slug, tmp_path)      # helper mirrors Task 4 test
    from scripts.scaffold_studies import _expected_files
    for fname in _expected_files(slug):
        ref = os.path.join(REPO, "out0", fname if "." in fname else fname)
        got = os.path.join(charts, fname if "." in fname else fname)
        assert os.path.exists(got), f"missing {fname}"
        if ref.endswith((".png", ".gif")) and os.path.exists(ref):
            assert image_similar(got, ref, tolerance=0.02), f"{slug}/{fname} drifted"
```

- [ ] **Step 2: Run to verify it fails** — helper/module missing.

- [ ] **Step 3: Implement `image_diff.py`** (Pillow + numpy normalized MAD; GIF via `PIL.ImageSequence`). Implement `_expected_files(slug)` in the scaffolder (derive from FLUSH_SPEC configs).

- [ ] **Step 4: Run to verify it passes** (serving venv)

Run: `.venv-serve/bin/python -m pytest tests/test_fidelity.py -q`
Expected: PASS for the three probe scenarios. Investigate any drift > tolerance (usually a config key not copied verbatim from the plot_func).

- [ ] **Step 5: Commit**

```bash
git add tests/fidelity/ tests/test_fidelity.py scripts/scaffold_studies.py
git commit -m "test(fidelity): image-diff flush outputs against out0 reference"
```

---

### Task 7: Full-investigation reproduction run + record outcomes

**Files:**
- Modify: `studies/<slug>/study.yaml` (runs recorded via runs.db; outcomes)
- Test: manual + `tests/test_fidelity.py` full parametrization

- [ ] **Step 1: Run every study's reproduce entry** (serving venv, from workspace root):

```bash
for s in $(ls studies); do
  .venv-serve/bin/python spatio_flux/runners/run_study.py "$s" || echo "FAILED: $s"
done
```
(Or via the plugin: `/pbg-investigation run spatio-flux-test-suite` once the serving venv has the plugin — orchestrates `run-script` over all members with `canonical_runs:`.)

- [ ] **Step 2: Expand the fidelity test to all 19** — remove the `parametrize` subset, run the full set; expect all within tolerance. Fix any per-scenario config drift.

Run: `.venv-serve/bin/python -m pytest tests/test_fidelity.py -q`

- [ ] **Step 3: Confirm every expected artifact exists** for all 19 (the `behavior_tests` `expected` lists). A small script asserts each `studies/<slug>/charts/<file>` is present.

- [ ] **Step 4: Record outcomes** — for each study, the `<SLUG>-REPRODUCES-REPORT` test result lands as `runs[].outcomes` (UPPERCASE key, `{result: PASS, detail}`) via the run flow / a recording helper. Verify the dashboard Tests tab shows PASS for all 19.

- [ ] **Step 5: Commit**

```bash
git add studies/
git commit -m "test(repro): full-investigation reproduction; all 19 scenarios match out0"
```

---

## Self-Review

**Spec coverage (design §4.3):**
- Post-run analysis-flush (not embedded) → Tasks 1, 4, 5. ✓
- Every plot_func reproduced as a Step writing exact filenames → Tasks 2, 3 (+ scenario map). ✓
- Structure step (diagram + schema/state) → Task 2. ✓
- Report cards / behavior_tests / outcomes → Tasks 3, 7. ✓
- Validated against reference artifacts → Tasks 6, 7 (oracle corrected to `out0/`). ✓
- Decouple existing viz Steps from composites → Task 5. ✓

**Placeholder scan:** the scenario→outputs map and FLUSH_SPEC are complete data; wrappers reuse named `plot.py` primitives (no plotting math invented). The one deliberately-abbreviated spot — "implement all remaining wrappers the same way" in Task 2 Step 3 — is bounded by an explicit, complete list of step names and a stated 1:1 rule (config → the exact plot_func call site), which the Task 3 FLUSH_SPEC pins per scenario. Not a logic placeholder.

**Type/name consistency:** step-name registry keys (`timeseries`, `species_dist_gif`, `snapshots_grid`, `particles_mass`, `particles_mass_submasses`, `particle_traces`, `model_grid`, `pymunk_gif`, `fields_agents_gif`, `species_dist_with_particles_gif`, `structure`) are identical across the harness (Task 1), steps (Task 2), FLUSH_SPEC (Task 3), and runner (Task 4). `run_analysis_flush` signature and `_expected_files` are used consistently.

**Open items folded into tasks:** parametrizing `run_composite_document`'s `outdir` + SQLite emitter (Task 4 Step 3); `_expected_files` derivation (Task 6 Step 3); `artifacts_present` measure kind is declared in study.yaml (Task 3) and evaluated by the fidelity test (Task 6) rather than a v2ecoli evaluator.

## Dependencies

- **Requires Phase A Task 5** (serving venv) for Tasks 4/6/7 (full sci deps + runs).
- **Feeds Phase D** (report script consumes `studies/<slug>/charts/` + `runs.db`).
