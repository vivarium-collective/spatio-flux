# SpatioFlux discovery-engine + study hardening — design

**Date:** 2026-10-25
**Workspace:** `~/code/spatio-flux` (modernized vivarium-workbench workspace)
**Target investigation:** `spatio-flux-test-suite` (extend in place — user's call)
**Worktree / branch:** `~/code/spatio-flux--discovery` / `discovery-hardening` (off `origin/main` @ 19849fd)

## Intent & why

spatio-flux is currently a *faithful reproduction* workspace: its 19 test-suite studies and
8 paper-figure studies assert "the artifacts are produced / the schema matches the committed
reference." They prove the code runs; they do not assert **biology**. The user wants to turn
spatio-flux into something that *discovers and narrates new biology*:

1. **Harden** all existing studies so their contracts and findings make real, quantitative,
   code-evaluated claims tied to observables — not just "artifacts present."
2. **Extend** the investigation with 6–8 new biology-forward studies that show phenomena
   spatio-flux can't currently narrate, built from the **real** registered composites.
3. Run the **discovery-engine loop**: each study's findings populate
   `discovery_implications.followup_study_proposals`, which seed the next study, and the
   investigation's narrative spine (executive verdict + scientific_argument) foregrounds the
   **new biology** rather than "reproduces the report."

All new science uses the **real** spatio-flux composites (DFBA via COBRA models, finite-volume
diffusion–advection, Brownian/Newtonian particles). No mocks, no reimplemented math.

## Evaluation machinery — DECIDED: full A (the ecosystem norm)

Workbench studies evaluate `behavior_tests[]` via `viva_superpowers.study_evaluator`: a closed
`measure` (kind + observable store-path + expected band/shape) + `pass_if` (op: `band` /
`monotone` / `flat` / `in_set` / `cv_below` / … + tolerance) + `requires_simulation`. The run is
persisted by an emitter (xarray/zarr) and read back through a **`RunReader`**; the evaluator
resolves the observable series and applies the op → `{result, evaluated_by: "code", operator,
detail}`, written into `runs[].outcomes` (drives `gate_status` + each finding's
`evidence.from_test`). Resolution order: native computed kind → **workspace-registered
evaluator** → else `evaluated_by: "agent"` (fallback) / `needs_rerun`.

The framework's *native* computed kinds are generation/lineage-shaped (built for v2ecoli
whole-cell), which don't fit spatio-flux's single-trajectory runs — so the workspace extends via
the sanctioned hook: `register_evaluators(registry)` + `register_derived_scalars(reg)` in
`spatio_flux/evaluators.py`. **spatio-flux already uses this hook** (its `artifacts_present`
evaluator), so we add sibling quantitative evaluators through the same mechanism — this is the
norm, `evaluated_by: code`, not bespoke infra.

**What we build (one bounded infra stage, up front):**
- A **spatio-flux `RunReader` bridge** so the evaluator can resolve observables from the
  workspace's custom zarr (`write_run_zarr`: `(time,)` scalars, `(time,y,x)` fields,
  `n_particles` count), reusing `viva_emitters.RunReader` where it fits.
- **Quantitative evaluators** registered via `register_evaluators`: `observable_band`,
  `monotone`, `mass_conserved`, `peak_then_decline` (and reuse native `band`/`flat`/`monotone`
  ops where they resolve directly).
- **Derived-scalar computers** registered via `register_derived_scalars`: front position/velocity
  (first lattice row crossing a biomass threshold), biomass radius of gyration / centroid, total
  acetate(t) and its peak time, Δbiomass, MSD slope.
- **Run-store extension (full A):** extend `write_run_zarr`/`run_store` to persist **per-particle
  positions + mass** (additive new arrays — existing arrays and the committed structural-schema
  oracle are untouched), enabling a real Brownian-MSD test and per-particle metrics.

## Goals

- Every existing study carries at least one **quantitative, code-evaluated, falsifiable**
  behavior test with an acceptance band tied to an observable (where the scenario can support
  one), plus a finding whose `statement` is a biological/physical claim, not "reproduces
  artifacts."
- 6–8 **new** studies across four themes (below), each with: a real composite baseline, real runs
  persisted (xarray/zarr), quantitative code-evaluated `behavior_tests`, interactive committed
  viz, and `discovery_implications` that wire the loop.
- Investigation narrative upgraded: `executive` verdict leads with the discovered biology;
  `scientific_argument.main_claim` + evidence_for/against reflect the new studies;
  `competing_frameworks` set; ≥1 `kind: adversarial` study present (rigor floor).
- Everything reproduces deterministically (or band/existence-checks where RNG-bound) and the
  published read-only workbench + report rebuild green.

## Non-goals / anti-goals

- **Not** inventing new biology in composite/process source. New studies are new *configurations
  and sweeps* of existing real composites. Two **behavior-preserving, additive** code touches are
  sanctioned and flagged (NOT silent biology changes): (a) exposing `comets_spatial_dfba`'s
  hardcoded `diffusion_rate` as a composite parameter **with its current value as default**
  (defaults unchanged → snapshots/schema oracle unaffected); (b) persisting per-particle
  positions/mass in the run-store (additive arrays). A genuinely new composite generator, if ever
  needed, is a flagged `/viva-expert --in-workspace` sub-task, not a silent edit.
- **Not** touching the old-locked-stack CI contract: do **not** regenerate committed
  `reference/*_schema.json` or `_snapshots/*.json` under the modern stack (known non-blocking
  drift — modernization memory). Hardening adds *new* tests/findings; it does not rewrite the
  structural-schema oracle. New tests read the run's zarr, not the structural oracle.
- **Not** retiring or restructuring the paper-figures investigation. Fig studies are
  methods/diagram studies (SVGs committed) — hardened only to honest contracts (diagram exists +
  is the documented figure), not forced quantitative claims.
- **Anti-goal:** a study that *claims* biology its run doesn't actually measure. Every finding
  traces to a run + a test that evaluated it (`evidence.from_test` + `provenance.run_ids`).

## The four themes → new studies (real composites + real knobs)

Grounded in the generators in `spatio_flux/composites/` and their `parameters={…}`:

**Theme 1 — Emergent spatial pattern formation**
- **S1 `growth_front_propagation`** — `comets_spatial_dfba` (biomass seeded at top edge,
  vectorized dFBA + advection–diffusion). Claim: a biomass front advances down the lattice at a
  measurable, roughly constant velocity (Fisher-KPP-like). Observable (derived scalar): front
  position (first row crossing a biomass threshold) vs time → fit velocity; test: `monotone`
  advance + velocity `band`. Knobs: `n_bins`, `advection_rate` (+ `diffusion_rate` once exposed).
- **S2 `advection_plume_asymmetry`** — same composite, sweep the scalar magnitude of
  `advection_rate` (2-vector; builder sweeps its magnitude along the seeded axis). Claim:
  advection breaks the symmetric front into a directed plume; biomass centroid displacement
  scales with advection. Observable: centroid displacement vs advection; test: `monotone`.

**Theme 2 — Cross-feeding / community ecology in space**
- **S3 `acetate_crossfeeding_dynamics`** (flagship) — `comets_diffusion` (tracks
  glucose+acetate+dissolved biomass). Claim: overflow metabolism makes total acetate rise then
  fall as it's re-consumed (non-monotonic acetate = spatial cross-feeding signature). Observable:
  total acetate(t); test: `peak_then_decline` with peak time in a `band` (band calibrated from a
  first run).
- **S4 `spatial_vs_wellmixed_community`** — compare `community_dfba` (well-mixed, per-species
  biomass) vs `spatial_dfba_process` (6 species across a glucose gradient, single aggregate
  biomass field). *Redefined observable* (spatial side has no per-species biomass): compare
  **total biomass yield + substrate (glucose/acetate) depletion** well-mixed vs spatial. Claim:
  spatial structure changes the community outcome (yield/depletion differ beyond tolerance). Two
  baselines; test: `band` on the spatial-vs-wellmixed delta.

**Theme 3 — Parameter-sweep phase diagrams**
- **S5 `diffusion_growth_phase_diagram`** — sweep `diffusion_rate` (now exposed) × advection
  magnitude in `comets_spatial_dfba`. Classify each run's spatial-spread metric (biomass radius
  of gyration) into localized vs dispersed → a phase diagram. Multi-run sweep study (new sweep
  runner). Viz = heatmap/phase plot. Test: regime boundary exists (`monotone` spread in
  diffusion).
- **S6 `division_seeding_regimes`** — sweep `division_mass_threshold` × `add_rate` in
  `comets_br_particles_dfba`. Observable: final particle count → phase diagram. Test: `monotone`
  decreasing count in threshold.

**Theme 4 — Perturbation / robustness**
- **S7 `glucose_limitation_diauxie`** — `ecoli_core_dfba` (+ a spatial variant) at low vs high
  glucose. Claim: under glucose limitation growth arrests and acetate is re-consumed (diauxic
  switch). Observable: biomass plateau time + glucose/acetate crossover; test: switch present at
  low glucose, absent at high (perturbation contrast, two baselines).
- **S8 `nutrient_starvation_negative_control`** (`kind: adversarial`) — `ecoli_core_dfba` /
  `comets_diffusion` with **zero** glucose. Falsification: with no carbon source biomass must
  **not** grow (flat within noise). Rigor negative control — passes by *rejecting* growth.
  Observable: Δbiomass; test: `flat` / |final−initial| `band` ≈ 0.

Flagship pair first: **S3** (cross-feeding) + **S1** (front). Then broaden to
S2/S4/S7/S8, then the heavier sweeps S5/S6. Land 6–8.

## Discovery-engine loop (how studies beget studies)

For each study, after it runs + is evaluated, author `discovery_implications`: `resolved` (what
this study settled), `remaining` (open uncertainties), `followup_study_proposals` (each with a
real `motivation` + candidate composite/knob). Proposals seed the next study, e.g.:
- S1 front velocity → "does advection break the front?" → **S2**.
- S3 acetate peak → "is cross-feeding stronger with spatial structure than well-mixed?" → **S4**.
- S5 phase boundary → "where is the boundary sharpest / is it hysteretic?" → finer-sweep follow-up.
- S7 diauxie → "does limitation protect against washout under advection?" → cross-theme follow-up.

Recorded via `/viva-study propose-followup` + `seed-from-followup` (in the workspace, not just prose).

## Hardening pass — all existing studies

For each of the 19 test-suite studies (+ the 8 fig studies, to the extent honest):
1. Keep the existing regression/existence test (don't break the old-stack oracle).
2. **Add** a quantitative, code-evaluated `behavior_tests` entry tied to a real observable from
   the run's zarr, with band + `requires_simulation`. Examples:
   - `ecoli_core_dfba`: glucose(t) `monotone` ↓; biomass(t) `monotone` ↑; final biomass `band`.
   - `diffusion_process`: total field mass `mass_conserved` (closed BCs) ± ε; spatial variance of
     the seeded band increases (`monotone`).
   - particle studies: particle count `monotone` non-decreasing; MSD slope `band` (Brownian,
     via new per-particle persistence).
3. **Rewrite the finding** `statement` as a mechanistic claim, with `evidence.from_test` +
   `provenance.run_ids`, `status`, and `discovery_implications`.
4. Set study-level `claim:` (one-line headline) + `confidence:` where the derived value is wrong.

Pure-integrator studies get an honest quantitative-but-modest test (e.g. Brownian MSD ∝ t)
rather than a forced biological claim.

## Narrative spine (investigation level)

- `executive.verdict`/`verdict_status`: lead with the discovered biology (front velocity, spatial
  cross-feeding acetate peak, diffusion–growth phase boundary, diauxie); reproduction demoted to a
  supporting baseline.
- `scientific_argument`: `main_claim` = the spatial-metabolism synthesis; `evidence_for` =
  S1/S3/S5/S7; `caveats` = single-seed, coarse lattice, model simplifications; `key_figures` = new viz.
- `competing_frameworks`: well-mixed ODE metabolism vs spatial FBA; particle-based vs field-based
  biomass. Set `relation` each.
- ≥1 `kind: adversarial` study (S8).

## How it gets made (builder's calls, settled)

- **Env:** modern-stack repro venv in the worktree:
  `uv pip install -e . "process-bigraph>=1.9" "bigraph-schema>=1.8" xarray zarr pytest`
  (+ `graphviz` present). Run/serve via the worktree venv; editable → worktree, so probe via
  `PYTHONPATH=<worktree>` if needed.
- **Runners:** reuse `spatio_flux/runners/run_study.py` + the `spatio_flux/analysis/` flush for
  single-run studies. **Each new slug needs a `FLUSH_SPEC` entry** (+ any new `$`-context) — part
  of "reuse the runner." Sweep studies (S5/S6) get a small new sweep runner (`canonical_runs`
  entry that loops params, writes per-run zarr + a summary table).
- **Viz:** interactive Plotly via `/viva-viz` (front kymograph, acetate trajectory, phase-diagram
  heatmap); committed to `studies/<slug>/` and referenced in `visualizations:`.
- **Skills:** `/viva-study` (contracts/findings/narrative/followups), `/viva-run` (smoke),
  `/viva-viz`, `/viva-report` (reviewer-readiness), `/viva-navigate decisions` +
  `/viva-harden-investigation` (rigor). Don't hand-edit study.yaml around the skills.
- **Heavy sweeps:** S5/S6 on a 20×20 lattice may run minutes each; option to offload to the mini
  (`mct`) if local is too slow — decided at execution time.

## Staging & merge

1. Worktree + modern repro venv; smoke one existing + one new composite.
2. **Evaluation infra:** spatio-flux RunReader bridge + quantitative evaluators + derived scalars
   (`register_evaluators`/`register_derived_scalars`) + per-particle run-store persistence; unit
   tests green.
3. Hardening pass (test-suite studies) — code-evaluated contracts/findings/claims; `reproduce.py` green.
4. New studies S1, S3 end-to-end with viz; discovery_implications wired.
5. Broaden to S2/S4/S7/S8, then sweeps S5/S6.
6. Investigation narrative spine; `/viva-report` reviewer-readiness; `/viva-harden-investigation`.
7. PR → review → (user-approved) squash-merge → publish rebuilds gh-pages report + workbench.

No auto-merge; no AI attribution on commits/PR. Each stage commits in the worktree only.

## Resolved decisions (was: open questions)

- **Structure:** extend `spatio-flux-test-suite` in place (not a new investigation).
- **Hardening scope:** full pass over all 27 studies.
- **New-study scope:** go broad, 6–8 studies.
- **Evaluation:** full A — code-evaluated quantitative tests via the workspace `register_evaluators`
  hook + per-particle run-store persistence (the ecosystem norm).
- **S4 observable:** redefined to total biomass yield + substrate depletion (spatial has no
  per-species biomass).
- **S5 `diffusion_rate`:** exposed as a behavior-preserving composite parameter (current value as
  default).
