# Novel visually-rich SpatioFlux sims — Implementation Plan (Part A)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native inline). Steps use `- [ ]`.

**Goal:** Replace the 5 tautological discovery-engine studies with 4 novel *qualitative* studies (spatial cross-feeding bridge, range-expansion sectoring, advection washout transition, biofilm settling) — each a real composite + a qualitative code-evaluated contract + an interactive Plotly viz — and reconcile the investigation with the newly-shipped `/viva-investigation reconcile`.

**Architecture:** New workspace evaluators (qualitative, zarr-reading), a `strain` field on the workspace particle type + `particle_strain` persistence, two new composites (2-field cross-feeding; outflow-BC washout) + a strain-seeded variant, Plotly HTML emitted by the runners, and the 4 studies authored on top.

**Tech Stack:** Python 3.12, process-bigraph ≥1.9, bigraph-schema ≥1.8, xarray/zarr/numpy, plotly, pytest; viva-superpowers@main (for `/viva-investigation reconcile`).

**Spec:** `docs/superpowers/specs/2026-10-26-novel-sims-and-investigation-reconcile-design.md` (Part A). Read alongside.

## Global Constraints
- Worktree `~/code/spatio-flux--novel-sims`, branch `novel-spatial-sims` (off `main` f864044). No AI attribution; squash-merge; no auto-merge.
- Sweep/new studies keep a representative `conditions.baseline` mapping (the PR #72 render-gate lesson).
- Composite/particle-type/emitter changes are additive + behavior-preserving for existing studies (the 19 reproduction + 8 figure studies must still pass `reproduce.py`).
- Do NOT regenerate `studies/*/reference/*_schema.json` (old-stack oracle); new modern-stack test modules guard with `pytest.importorskip("xarray"/"zarr")`.
- Anti-goal: never tune a phenomenon into a fake positive. A phenomenon that doesn't appear clearly is reported as an honest open question / negative.

## Review Focus
- **Additive particle-type/emitter change must not break the 19+8 existing studies** — `reproduce.py` stays green; the new `strain`/`particle_strain` default to absent/NaN when unused. Covered Task 2 + Task 8.
- **Qualitative evaluators must FAIL on the null** — `demixing_index` must not "pass" a well-mixed (non-sectoring) run; `threshold_transition` must reject a monotone (no-cliff) curve; `spatial_overlap` must distinguish B-tracks-acetate from B-tracks-glucose. Covered Task 3 tests.
- **New composites must be behavior-real** — cross-feeding B must not grow in an A-absent control; washout must actually lose mass across the boundary (not pile up). Covered Task 4 + the study contracts.
- **Plotly HTML renders in the published snapshot** — `html:` scheme + `include_plotlyjs=True`; `reproduce.py` + the render-check pass. Covered Task 8.
- **Phenomenon honesty** — sectoring/washout: if no clean phenomenon, the study records the negative. Covered Task 6.

---

### Task 1: Env + remove the 5 weak studies
- [ ] Repro venv: `uv venv .venv-repro && uv pip install -e . "process-bigraph>=1.9" "bigraph-schema>=1.8" xarray zarr polars pytest plotly` + `uv pip install "viva-superpowers @ git+https://github.com/vivarium-collective/viva-superpowers.git@main"` (ships `/viva-investigation reconcile`). Verify `python -c "import viva_superpowers.investigation_reconcile"`.
- [ ] `git rm -r studies/{advection_plume_phase,division_seeding_regimes,glucose_limitation_yield,spatial_growth_front,spatial_vs_wellmixed_community}`; remove them from `investigations/spatio-flux-test-suite/investigation.yaml` `studies:`. (Keep `nutrient_starvation_control`; `spatial_overflow_crossfeeding` will be rebuilt as N1 → `git rm` it too.)
- [ ] Smoke `reproduce.py --only ecoli_core_dfba` still green. Commit: `chore(studies): remove the tautological discovery-engine studies`.

### Task 2: `strain` particle field + `particle_strain` persistence (TDD)
**Files:** `spatio_flux/__init__.py` (particle type), `spatio_flux/runners/run_store.py`, `tests/test_run_store_particles.py`.
- [ ] Failing tests: a particle seeded with `strain:"A"` survives `core.serialize('map[particle]', …)`; `write_run_zarr` of a trajectory whose particles carry `strain` persists a `particle_strain` (time,particle) array aligned to the `particle_x/y` slots (object/string dtype; NaN/"" where absent).
- [ ] RED → add `"strain": "string"` to the workspace particle type; extend `_field_dataarrays` to emit `particle_strain` → GREEN.
- [ ] Regression: run an existing particle study (`brownian_particles`) — zarr still has x/y/mass/n_particles; `particle_strain` present (empty strings). Commit.

### Task 3: Qualitative evaluators (TDD)
**Files:** `spatio_flux/evaluators.py`, `tests/test_novel_evaluators.py` (`importorskip` xarray/zarr).
- [ ] `spatial_overlap` — correlation/overlap of two fields; PASS when field-B overlaps the acetate field more than the glucose field (synthetic fixture). 
- [ ] `demixing_index` — from `particle_strain` + `particle_x/y` over time: local same-strain purity at the front rises vs a well-mixed null; PASS on a demixing fixture, FAIL on a stays-mixed fixture.
- [ ] `threshold_transition` — on a metric-vs-axis series: PASS on a sharp step (high→~0 with a localized drop), FAIL on a monotone/no-cliff series.
- [ ] `gradient_correlation` — correlation of a per-particle quantity (mass) with position (y); PASS when correlated, FAIL when flat.
- [ ] Each: RED→GREEN, registered in `register_evaluators`; evaluator suite green. Commit.

### Task 4: New composites (behavior-real)
**Files:** `spatio_flux/composites/spatial.py` or a new module; tests in `tests/test_novel_composites.py`.
- [ ] `spatial_cross_feeding` generator — 2D lattice, strain A (Monod `overflow_metabolism`, glucose→acetate) writing `biomass_A`, strain B (`acetate_only`, acetate→biomass) writing `biomass_B`, shared diffusing `acetate`+`glucose` fields (model on `dfba_kinetics_community`'s two-biomass wiring). Param `seed_separation` (A one side, B other). Test: builds + runs; B grows; an `a_absent` param → B flat.
- [ ] `comets_washout` generator — field dFBA + advection–diffusion of `dissolved biomass` with a downstream `outflow` boundary_conditions; param `advection`. Test: builds + runs; high advection → total dissolved biomass collapses (verified 5.0→0.009), low → sustained.
- [ ] A strain-seeded `comets_br_particles_dfba` variant (param `strain_fractions`) seeding two strain-tagged founder populations. Test: initial particles carry `strain`; division preserves it (round-trips to zarr via Task 2).
- [ ] Commit per composite. (Built with the `/viva-expert --in-workspace` discipline — real processes, no mocks.)

### Task 5: Runner Plotly viz + sweep metric
**Files:** `spatio_flux/runners/run_new_study.py`, `run_sweep.py`, `spatio_flux/analysis/` (a small `plotly_viz.py`).
- [ ] Add a `plotly_viz` helper: animated field heatmap (time slider), particle-movie scatter (color=strain/size=mass), phase portrait — each `fig.to_html(include_plotlyjs=True)` → `studies/<slug>/viz/<name>.html`.
- [ ] `run_new_study.py` + `run_sweep.py` also emit the relevant Plotly HTML.
- [ ] Add `total_field_biomass` to `run_sweep._metric`. Commit.

### Task 6: The 4 studies (author + run + evaluate + viz + findings)
Per study (via the runners + direct study.yaml authoring, same headless loop as PR #72): real composite baseline(s), `conditions.baseline` mapping, `canonical_runs` (run_new_study/run_sweep), the qualitative `behavior_tests` (Task 3 kinds), run → `compute_outcomes` → persist authored outcomes, Plotly viz in `visualizations:` (`html:` scheme), a mechanistic finding, `discovery_implications`, and (sectoring/washout) multiseed + honest-negative handling.
- [ ] **N1 `spatial_cross_feeding_bridge`** (`spatial_cross_feeding`): `peak_then_decline` acetate + `spatial_overlap` (B↔acetate > B↔glucose) + A-absent `observable_flat` control. Viz: 3-panel field movie + B-centroid vs acetate-front.
- [ ] **N2 `range_expansion_sectoring`** (strain variant): `demixing_index` rises vs well-mixed null, ≥3 seeds. Viz: strain-colored particle movie.
- [ ] **N3 `advection_washout_transition`** (`comets_washout` sweep): `threshold_transition` on total_field_biomass vs advection. Viz: survival phase portrait + persist/washout field movies.
- [ ] **N4 `biofilm_settling_metabolism`** (`comets_nt_particles_dfba`): settle (`gradient_correlation` mass↔depth) + population growth, ≥3 seeds. Viz: settling particle movie over the nutrient field.
- [ ] Commit per study. Add all 4 to the investigation `studies:`.

### Task 7: Reconcile the investigation (dogfood Part B)
- [ ] `python -m viva_superpowers.investigation_reconcile spatio-flux-test-suite --ws .` → review drift; `--apply` the mechanical drops; then author the prose reconciliation (executive verdict, scientific_argument.evidence_for, biological_story, competing_frameworks) to the new member set; re-run detector → zero drift. Commit: `docs(investigation): reconcile narrative to the new study set`.

### Task 8: reproduce + PR
- [ ] `reproduce.py` green (all studies, incl. the 4 new) + `pytest -q` (drift oracle deselected) + the render-check locally (install `vivarium-workbench@main`). Then final whole-branch review (fresh subagent, most capable model — this is the bigger/riskier build). Fix Critical/Important. Push; `gh pr create` vs `main` (no AI attribution). Stop — no merge until "merge".
