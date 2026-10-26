# SpatioFlux Discovery-Engine + Study Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn spatio-flux's reproduction studies into code-evaluated quantitative biology, and extend the `spatio-flux-test-suite` investigation with 6–8 new biology-forward studies wired by a discovery loop.

**Architecture:** Add quantitative evaluators to the workspace's existing `spatio_flux/evaluators.py` hook (read each run's zarr directly with xarray — the same direct-read pattern `artifacts_present` already uses — so no RunReader bridge is needed). Extend the run-store to persist per-particle positions/mass. Then harden the 27 existing studies and author new studies through the `/viva-*` skills, each test computed by the new evaluators.

**Tech Stack:** Python 3.12, process-bigraph ≥1.9, bigraph-schema ≥1.8, xarray + zarr, numpy, pytest; `/viva-study` `/viva-viz` `/viva-report` `/viva-harden-investigation` `/viva-navigate`.

**Spec:** `docs/superpowers/specs/2026-10-25-spatio-flux-discovery-hardening-design.md` (read it alongside this plan).

## Global Constraints

- Worktree only: `~/code/spatio-flux--discovery`, branch `discovery-hardening`. Never commit in the canonical `~/code/spatio-flux`.
- **No AI attribution** on any commit or PR (no `Co-Authored-By`, no "Generated with" footer).
- **No auto-merge.** PR is created; merge only on the user's explicit "merge" in-turn.
- Env: modern repro venv in the worktree — `uv pip install -e . "process-bigraph>=1.9" "bigraph-schema>=1.8" xarray zarr pytest`. Editable install points at the worktree; probe via `PYTHONPATH=<worktree>` if needed.
- **Do NOT regenerate** committed `studies/*/reference/*_schema.json` or `spatio_flux/composites/_snapshots/*.json` — old-locked-stack CI contract. New tests read the run's zarr, never the structural oracle.
- Composite/process source edits are limited to two behavior-preserving additive changes (Task 3 + Task 6a): particle persistence and exposing `comets_spatial_dfba`'s `diffusion_rate` with its current value as default. No other biology changes.
- Every finding must trace to a run + a test (`evidence.from_test` + `provenance.run_ids`). No claim a run didn't measure.

## Review Focus

- **Empty / single-point trajectory** (a run that produced one tick): every evaluator must return a clean FAIL/`needs_rerun`, not raise. Covered in Task 2 tests.
- **Observable field absent from the zarr** (wrong field name, or a composite that never emits it): evaluator returns FAIL with a clear detail, not a KeyError. Covered in Task 2 tests.
- **2D field vs scalar field** (`acetate` may be `(time,y,x)` or `(time,)` depending on composite): the "total field" derived scalar must sum spatial dims when present and pass through when scalar. Covered in Task 2 tests.
- **Variable particle count over time** (division/seeding changes N): per-particle persistence must pad to max-N with NaN and the MSD slope must ignore NaN. Covered in Task 3 tests.
- **New study slug missing from `FLUSH_SPEC`**: `run_study.main` raises `SystemExit` before writing the zarr — each new study's task adds its `FLUSH_SPEC` entry and a smoke step proves the run writes a zarr. Covered per new-study task.

---

### Task 1: Modern repro venv + baseline smoke

**Files:** none (environment + verification only).

- [ ] **Step 1:** Create the venv: `uv venv .venv-repro && source .venv-repro/bin/activate && uv pip install -e . "process-bigraph>=1.9" "bigraph-schema>=1.8" xarray zarr pytest`.
- [ ] **Step 2:** Verify the editable install resolves to the worktree: `python -c "import spatio_flux,os;print(spatio_flux.__file__)"` → path under `spatio-flux--discovery`.
- [ ] **Step 3:** Smoke one existing study end-to-end: `python spatio_flux/runners/run_study.py ecoli_core_dfba 60` → exits 0, writes `studies/ecoli_core_dfba/runs.ecoli_core_dfba-reproduce.zarr`.
- [ ] **Step 4:** Smoke a new-study composite builds: `python -c "from spatio_flux.core import build_core; from pbg_superpowers.composite_generator import build_generator; build_generator('comets_diffusion', build_core())"` → no error.
- [ ] **Step 5:** Commit nothing (env is gitignored); record the venv path in the PR notes.

---

### Task 2: Quantitative evaluators (the norm, code-evaluated)

**Files:**
- Modify: `spatio_flux/evaluators.py` (add evaluators + derived-scalar helpers + extend `register_evaluators`)
- Test: `tests/test_quantitative_evaluators.py` (create)

**Interfaces:**
- Consumes: the zarr schema from `run_store.write_run_zarr` — `ds["<field>"]` dims `(time,)` or `(time,y,x)`, `ds["n_particles"]` `(time,)`, `ds.coords["time"]`; `run_store.zarr_store_path(study_dir, run_id)`.
- Produces (all registered in `register_evaluators`, hook signature `(test, reader, ws_root) -> outcome`):
  - measure kinds `observable_band`, `observable_monotone`, `mass_conserved`, `peak_then_decline`, `observable_flat`.
  - module-level helpers used by the hooks and by new studies:
    - `open_run(ws_root, slug, run_id=None) -> xr.Dataset` (defaults run_id to `<slug>-reproduce`)
    - `series_total(ds, field) -> np.ndarray` (sum over y,x if present, else pass through)
    - `front_position(ds, field, threshold) -> np.ndarray` and `front_velocity(ds, field, threshold) -> float`
    - `radius_of_gyration(ds, field) -> np.ndarray`, `centroid(ds, field) -> np.ndarray` (shape `(time,2)`)
    - `delta_scalar(ds, field) -> float` (final − initial of `series_total`)
  - outcome dict shape identical to the existing hook: `{result, measured_value, evaluated_by:"code", operator, detail}`.

- [ ] **Step 1: Write failing tests** in `tests/test_quantitative_evaluators.py`, building tiny synthetic `xr.Dataset`s written to a tmp zarr via `write_run_zarr`:
```python
def test_observable_monotone_increasing_passes(tmp_path):
    # biomass rising each tick -> PASS for direction "increasing"
def test_observable_monotone_flags_nonmonotone(tmp_path):
    # a dip -> FAIL
def test_mass_conserved_within_tol(tmp_path):
    # total field constant +-eps -> PASS; a leak beyond tol -> FAIL
def test_peak_then_decline_detects_overflow(tmp_path):
    # acetate up then down, peak index in band -> PASS; monotone rise -> FAIL
def test_observable_band_final_value(tmp_path):
    # final total within [lo,hi] -> PASS; outside -> FAIL
def test_missing_field_fails_cleanly(tmp_path):
    # observable names a field not in ds -> result FAIL, detail mentions field, no raise
def test_single_tick_trajectory_no_raise(tmp_path):
    # one-tick zarr -> FAIL/needs_rerun, never an exception
def test_series_total_sums_spatial_dims(tmp_path):
    # (time,y,x) field -> series_total has shape (time,)
```
- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/test_quantitative_evaluators.py -v` → fails (functions/kinds undefined).
- [ ] **Step 3: Implement** the helpers and hooks in `spatio_flux/evaluators.py`. Each hook resolves `study_dir = <ws_root>/studies/<slug>` (slug from `test["measure"]["slug"]` or `requires_simulation`→study), opens the zarr via `open_run`, computes the observable, applies its op with `pass_if.tolerance`, and returns the standard outcome dict (FAIL with a descriptive `detail` on missing field / too-few-ticks). `peak_then_decline` passes when the series rises to a max at an index inside `pass_if.peak_index_band` then ends below `(1-tolerance)*peak`. Extend `register_evaluators` to register the five new kinds alongside `artifacts_present`.
- [ ] **Step 4: Run tests, verify pass** — `pytest tests/test_quantitative_evaluators.py -v` → all PASS.
- [ ] **Step 5: Commit** — `git add spatio_flux/evaluators.py tests/test_quantitative_evaluators.py && git commit -m "feat(evaluators): code-evaluated quantitative measure kinds reading the run zarr"`.

---

### Task 3: Per-particle run-store persistence (enables MSD)

**Files:**
- Modify: `spatio_flux/runners/run_store.py` (`_field_dataarrays` → also persist particle positions/mass)
- Modify: `spatio_flux/evaluators.py` (add `msd_slope(ds) -> float` helper + `observable_msd_linear` kind)
- Test: `tests/test_run_store_particles.py` (create); extend `tests/test_quantitative_evaluators.py`

**Interfaces:**
- Produces in the zarr: `particle_x`, `particle_y`, `particle_mass` with dims `(time, particle)`, NaN-padded to the max particle count across ticks; `particle` is a positional slot coord.
- Produces in evaluators: `msd_slope(ds)` (least-squares slope of mean-squared-displacement-from-initial vs time, over non-NaN slots) and kind `observable_msd_linear` (PASS when slope in `pass_if.band` and R² ≥ `pass_if.min_r2`).

- [ ] **Step 1: Write failing tests** — `test_particles_persisted_padded` (a trajectory whose particle count grows → `particle_x/y/mass` shape `(time, max_n)`, early slots filled, later-born slots NaN before birth); `test_msd_slope_linear_for_brownian` (synthetic diffusive walk → slope ≈ 2·D·dim within tol, ignores NaN).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — in `_field_dataarrays`, when ticks carry `particles`, collect each particle's position + mass per tick keyed by a stable slot (particle id order of first appearance), pad to max N with NaN, emit the three `(time, particle)` DataArrays. Add `msd_slope` + `observable_msd_linear` to `evaluators.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Regression** — re-run Task 1 Step 3 smoke; confirm existing scalar/2D field vars + `n_particles` are unchanged (additive only).
- [ ] **Step 6: Commit** — `git commit -m "feat(run-store): persist per-particle positions+mass; add MSD evaluator"`.

---

### Task 4: Hardening pass — existing studies (pattern + table)

Pattern, applied per study via `/viva-study` (never hand-edit YAML around the skill): **keep** the existing `artifacts_present` test; **add** one code-evaluated quantitative `behavior_tests` entry (new measure kinds from Tasks 2–3) with `requires_simulation: reproduce`; **rewrite** the finding `statement` as a mechanistic claim with `evidence.from_test` + `provenance.run_ids`; set study-level `claim:`. Then run `python spatio_flux/runners/run_study.py <slug>` and `python -m viva_superpowers.study_evaluator` (or `/viva-run`) to confirm the new test is `evaluated_by: code` and PASS. Calibrate each band from that first run.

Per-study assignment (19 test-suite studies; fig studies handled in Task 4b):

| study | observable (field) | kind | claim headline |
|---|---|---|---|
| ecoli_core_dfba | biomass; glucose | monotone↑; monotone↓ | glucose consumed as biomass grows |
| ecoli_dfba | biomass; glucose | monotone↑/↓ | iAF1260 growth depletes glucose |
| yeast_dfba | biomass; glucose | monotone↑/↓ | iMM904 growth on glucose |
| monod_kinetics | biomass; glucose | monotone↑/↓ | Monod uptake drives growth |
| community_dfba | total biomass; acetate | band; peak_then_decline | competition + cross-feeding in a shared pool |
| dfba_kinetics_community | total biomass | monotone↑ | dFBA+Monod coexist on shared substrate |
| diffusion_process | total field mass; band variance | mass_conserved; monotone↑ | closed-BC diffusion conserves mass while spreading |
| spatial_many_dfba | total biomass | monotone↑ | per-site dFBA grows across the lattice |
| spatial_dfba_process | total biomass; glucose | monotone↑/↓ | vectorized dFBA depletes the gradient |
| brownian_particles | MSD slope | observable_msd_linear | Brownian motion is diffusive (MSD∝t) |
| newtonian_particles | n_particles | monotone non-decreasing | particles accumulate under seeding |
| br_particles_kinetics | total biomass; n_particles | monotone↑ | kinetic agents grow + divide on fields |
| br_particles_dfba | total biomass | monotone↑ | internal-dFBA agents grow on a gradient |
| comets_diffusion | total biomass; acetate | monotone↑; peak_then_decline | COMETS fields grow + cross-feed acetate |
| comets_nt_particles_dfba | total biomass | monotone↑ | rigid-body dFBA agents grow among fields |
| comets_br_particles_dfba | total biomass; n_particles | monotone↑ | full COMETS composite grows + divides |
| comets_br_particles_kinetics | total biomass | monotone↑ | kinetic COMETS agents grow |
| reference_demo_x2y2 | total biomass | monotone↑ | reference model grows on a fine lattice |
| spatioflux_reference_demo | total biomass | monotone↑ | flagship reference model grows |

- [ ] **Step 1:** For each study above, apply the pattern via `/viva-study` (add quantitative test; rewrite finding; set `claim`).
- [ ] **Step 2:** Run each study's `run_study.py` + evaluate; confirm new test PASS `evaluated_by: code`. Adjust band from the first run where needed (band-from-first-run is ordinary study-building, not overfitting — document the run id).
- [ ] **Step 3:** `python scripts/reproduce.py` (or the subset touched) stays green; no `reference/*_schema.json` or `_snapshots` regenerated.
- [ ] **Step 4:** Commit in batches of ~5 studies — `git commit -m "feat(studies): quantitative code-evaluated contract for <slugs>"`.

#### Task 4b: Paper-figure studies — honest contracts only

- [ ] For fig-01…08 (`paper-figures` investigation, diagram/methods studies): via `/viva-study`, ensure each has an honest contract — the documented SVG/diagram artifact exists (existing `artifacts_present`-style check) and a finding that states it is a *methods/figure* study, not a quantitative result. Do **not** force quantitative claims. Commit.

---

### Task 5: New studies (exemplar + table)

Build order: **S3, S1** (flagships) → **S2, S4, S7, S8** → **S5, S6** (sweeps). Each single-run study, via `/viva-study` + the runner: (a) create the study under `studies/<slug>/` with a real baseline composite + params; (b) add a `FLUSH_SPEC[slug]` entry (reuse the nearest existing recipe family) so `run_study.py <slug>` runs; (c) `/viva-viz` an interactive Plotly figure for the observable, commit it, reference it in `visualizations:`; (d) add the code-evaluated quantitative `behavior_tests` entry (Tasks 2–3 kinds); (e) run + evaluate (PASS, `evaluated_by: code`); (f) author the finding + `discovery_implications.followup_study_proposals` seeding the next study via `/viva-study propose-followup`.

**Worked exemplar — S3 `acetate_crossfeeding_dynamics`:**
- Baseline: `comets_diffusion`, params `{}` (defaults seed biomass at the top edge).
- Observable: `series_total(ds, "acetate")`. Test kind `peak_then_decline`, `pass_if: {peak_index_band: [lo,hi], tolerance: 0.1}` — calibrate `[lo,hi]` and the drop from the first run.
- Viz: acetate(t) trajectory + the 2D acetate field at peak vs end (Plotly).
- Finding: "overflow metabolism produces an acetate pulse that is re-consumed as cells cross-feed spatially" — `evidence.from_test` the test; `provenance.run_ids` the run.
- Followup proposal → S4 ("is cross-feeding stronger spatial vs well-mixed?").

| study | composite | key knobs | observable → kind | claim / role | seeds |
|---|---|---|---|---|---|
| S1 growth_front_propagation | comets_spatial_dfba | n_bins, advection_rate | front_velocity(dissolved biomass,thr) → observable_band + front monotone | biomass front advances at ~constant velocity | → S2, S5 |
| S2 advection_plume_asymmetry | comets_spatial_dfba | sweep advection magnitude | centroid displacement → observable_monotone | advection bends the front into a plume | → S5 |
| S4 spatial_vs_wellmixed_community | community_dfba + spatial_dfba_process (2 baselines) | — | per-baseline: observable_band on each run's total-biomass yield + glucose depletion | spatial structure changes community outcome | → finer-sweep |
| S7 glucose_limitation_diauxie | ecoli_core_dfba (low vs high glucose, 2 baselines) | glucose | per-baseline: low→acetate peak_then_decline; high→acetate monotone↑ (no decline) | glucose limitation triggers diauxie | → washout follow-up |
| S8 nutrient_starvation_negative_control (`kind: adversarial`) | ecoli_core_dfba / comets_diffusion, glucose=0 | glucose=0 | delta_scalar(biomass) → observable_flat ≈0 | no carbon ⇒ no growth (falsification floor) | rigor |

**Two-baseline studies (S4, S7):** the evaluators are single-run, so each baseline gets its own quantitative test on its own run; the **contrast** (spatial vs well-mixed yield; diauxie present at low glucose, absent at high) lives in the finding's narrative over the two runs' `measured_value`s — not a cross-run op. Both baselines are declared under `conditions.baseline`/variants and each is run via `run_study.py`.

- [ ] **Steps (repeat the a–f procedure per study, in build order); commit each study individually** — `git commit -m "feat(study): <slug> — <claim>"`.

#### Task 5b: Sweep studies S5/S6 (new sweep runner)

**Files:** Create `spatio_flux/runners/run_sweep.py`; `studies/diffusion_growth_phase_diagram/`, `studies/division_seeding_regimes/`.

- [ ] **Step 1:** `run_sweep.py` — a `canonical_runs` entry that loops a param grid, builds+runs the composite per cell, writes a per-cell zarr + a `sweep_summary.json` (grid → the classified metric). Interface: `main(slug, axis_spec)` reading the grid from `study.yaml` `conditions.sweep`.
- [ ] **Step 2:** S5 `diffusion_growth_phase_diagram` — `comets_spatial_dfba`, sweep `diffusion_rate` (Task 6a) × advection magnitude; metric `radius_of_gyration` final; viz heatmap/phase plot; test `observable_monotone` on spread vs diffusion. S6 `division_seeding_regimes` — `comets_br_particles_dfba`, sweep `division_mass_threshold`×`add_rate`; metric final `n_particles`; test monotone decreasing in threshold.
- [ ] **Step 3:** Run (consider `mct` on the mini if a 20×20 grid is slow locally), evaluate, author findings + viz, commit each.

---

### Task 6: Enablers + investigation narrative + rigor

#### Task 6a: Expose `diffusion_rate` on `comets_spatial_dfba` (behavior-preserving)
- [ ] Add `diffusion_rate` to the generator's `parameters={}` and signature with the **current hardcoded value as default**; thread it to the diffusion process config. Test: building with defaults is byte-identical to before (snapshot unchanged); building with a non-default value changes the diffusion config. Commit.

#### Task 6b: Investigation narrative spine
- [ ] Via `/viva-study` + `/viva-investigation set-overview`: rewrite `spatio-flux-test-suite` `executive` (verdict leads with discovered biology), `scientific_argument` (main_claim + evidence_for S1/S3/S5/S7 + caveats + key_figures), and `competing_frameworks` (well-mixed ODE vs spatial FBA; particle vs field biomass). Confirm S8 is `kind: adversarial`. Commit.

#### Task 6c: Reviewer-readiness + rigor sweep
- [ ] Run `/viva-report` (Pass A reviewer-readiness + Pass B lint), `/viva-navigate decisions spatio-flux-test-suite`, and `/viva-harden-investigation`; fix what they surface (uncovered ACs, verdict↔evidence drift, rigor gaps). Re-render. Commit.

---

### Task 7: PR

- [ ] **Step 1:** `python scripts/reproduce.py` green end-to-end (all studies); `pytest -q` green.
- [ ] **Step 2:** Push `discovery-hardening`; `gh pr create` (no AI attribution in title/body). Body: what hardened, the new studies + their claims, the evaluator infra, the two behavior-preserving source touches, and the "do not regenerate the schema oracle" note for CI.
- [ ] **Step 3:** Report the PR + live-preview plan to the user. **Stop — do not merge** until the user says "merge".
