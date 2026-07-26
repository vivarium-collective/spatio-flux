# Spatio-Flux → Vivarium-Workbench Investigation Modernization

**Date:** 2026-07-26
**Branch:** `workbench-modernization`
**Status:** Design approved; ready for implementation planning

---

## 1. Goal

Convert the `spatio-flux` repo into a **vivarium-workbench workspace** whose single
source of truth is one **investigation** composed of **19 studies** — a 1:1 mapping of
the current `spatio_flux/experiments/test_suite.py` `SIMULATIONS` scenarios. The studies
are arranged as a composition DAG: standalone processes first, then pairs, then triples,
up to the reference demos.

Running the investigation must reproduce **every** artifact the current test-suite report
shows — bigraph diagrams, serialized schema/state, time series, snapshot grids, GIFs,
mass plots, particle traces, and per-scenario timing — with no fidelity loss versus the
current outputs.

The current `test_suite.py` runner is **retired** and replaced by the investigation as the
one way to reproduce everything.

Two consumers read the investigation's data:
1. The **workbench-native dashboard** (`vivarium-workbench serve --workspace .`) — free.
2. A **custom report script** that regenerates today's `report.html` look faithfully, so
   the published page at `vivarium-collective.github.io/spatio-flux/report/index.html`
   is preserved while moving to the new paradigm.

Orthogonally, tighten the **types**, add **units**, and add **process contracts/descriptions**
so the bigraph-loom process view has meaningful, unit-bearing type coverage.

### Approved decisions

- **Report strategy:** custom faithful report script **and** native dashboard (both).
- **Old runner:** retire `test_suite.py`; investigation becomes the single source of truth.
- **Graph edges:** composition **ordering/graph structure only** — every study stays
  independently runnable; edges are documentation of composition, not a hard gate.
- **Study count:** **19**, matching the current `SIMULATIONS` dict and the README.
- **Visualization strategy:** visualizations are **NOT embedded in the composite**. They run
  in the **post-simulation analysis flush** — post-run analysis Steps that read the run's
  emitted data (`runs.db` / xarray) and render into `studies/<slug>/charts|viz`, matching the
  workbench's `study_run_post` / `run_study_analyses` path. The existing
  `Visualization(Step)` classes in `spatio_flux/visualizations/` are refactored to run as
  post-run analyses rather than being wired into `composites/metabolism.py` / `spatial.py`.

---

## 2. Current state (audit summary)

### Report pipeline
- `spatio_flux/experiments/test_suite.py` drives everything: a `SIMULATIONS` dict of 19
  scenarios, each resolving a `@composite_generator` from `COMPOSITE_REGISTRY` via
  `build_generator(entry, overrides, core)`, run through
  `run_composite_document(...)`, then a per-scenario `plot_func`.
- `spatio_flux/library/tools.py`:
  - `run_composite_document` writes per scenario: `<name>.json` (full doc),
    `<name>_schema.json` (`core.render`), `<name>_state.json` (`core.serialize`),
    `<name>_viz.png` (bigraph-viz diagram), `<name>_timing.json`.
  - `generate_html_report` builds a single `report.html`: inline CSS, hero header,
    sticky-TOC two-column layout, About/ecosystem/how-to-read sections, and one card per
    scenario (description, runtime line, interactive JSON state viewer, bigraph diagram
    img, PNG plots, GIFs).
- Per-scenario `plot_func`s (in `test_suite.py`) write, to hardcoded `out/`, files named by
  `plot_config['filename']`:
  - `<filename>_timeseries.png`, `<filename>_video.gif`, `<filename>_snapshots.png`,
    `<filename>_mass.png`, `<filename>_mass_submasses.png`,
    `<filename>_particles_traces.png`, `<filename>_model_grid.png`.

### Existing Step-based visualization infra (to extend)
`spatio_flux/visualizations/` already contains `Visualization(Step)` subclasses:
`FieldAnimationGif`, `FieldHeatmap`, `FieldSnapshotsGrid`, `ParticleTraces`,
`TestSuiteTimeSeries` — registered as `local:<name>` and already wired into
`composites/metabolism.py` and `composites/spatial.py`. These subclass
`pbg_superpowers.visualization.Visualization(Step)`.

### Types / units / contracts
- Custom bigraph-schema types exist in `spatio_flux/types/positive.py`
  (`PositiveFloat`, `Concentration`, `Mass`, `Count`, `PositiveArray`, `SetFloat`, `Delta`,
  `Position`) and `spatio_flux/types/counts.py` (`CountConcentrationVolume`), plus composite
  types in `spatio_flux/__init__.py` (`particle`, `complex_particle`, `bounds`, `reaction`,
  `fields`, ...).
- **Weak ports:** `DynamicFBA` / `MonodKinetics` **outputs** are `map[count]` / `map[float]`
  / plain `float` (untyped deltas); `DiffusionAdvection` and `SpatialDFBA` / `ParticleExchange`
  field arrays use generic `array[float]` rather than `positive_array`.
- **No units anywhere on ports** — unit info (mM, gDW, mmol/gDW/h, µm, pg) lives only in
  docstrings.
- **No `describe()` / `description` / contracts on any process** — descriptions exist only
  on the `@composite_generator` decorator at the composite level.

### Reusable ecosystem units convention
`bigraph_schema/units.py` provides pint-backed unit types (`render_units_type`,
`parse_dimensionality`, `get/set_quantity_registry`) and `bigraph_schema/json_codec.py`
round-trips `pint.Quantity` as `{"__pint__": true, "magnitude", "units"}`. v2ecoli's
units-atlas surfaces unit-bearing readouts. **Reuse this**; do not invent a units mechanism.

---

## 3. Target workbench paradigm (reference: v2ecoli, vivarium-workbench)

- **Workspace:** `workspace.yaml` at repo root with a `layout:` map
  (`studies`, `investigations`, `references`) and `runtime.default_emitter`. All path
  resolution goes through `WorkspacePaths` — never hardcode dir names.
- **Investigation:** `investigations/<name>/investigation.yaml` — `schema_version` in
  `[1,2]`; required `name` + `title`; a `studies: [<slug>, ...]` membership list
  (authoritative for graph nodes); narrative spine (`executive`, `scientific_argument`,
  `biological_story`) is lint-gated but only `name`/`title` are schema-required.
- **Study:** `studies/<slug>/study.yaml` (flat, not nested) — `schema_version` in `[3,4]`;
  required `name` + `baseline`. Sibling artifact dirs: `runs.db` (durable run history),
  `charts/*.png`, `viz/*.html`, `viz/report_card/<card>.{html,verdict.json}`.
- **baseline block:** canonical top-level `baseline: [{name, composite, params}]`
  (non-empty array). `composite` is a dotted ref to a registered `@composite_generator`
  factory (must resolve, or the dashboard flags "composite not found").
- **DAG edges:** authoritative field is `pipeline_gate.prerequisites` (list of bare slugs
  or `{study, condition, outputs_used}`); `parent_studies` is the legacy fallback.
  `normalize_dag_edges` is the single read path; `build_investigation_graph` builds nodes
  from the investigation's `studies:` list and edges from `normalize_dag_edges`.
- **Content-addressed artifacts (pull-or-compute):** `lib/artifacts/` — `artifact_id(...)`
  SHA-256 over `composite_id + config + input_ids + commit`; `ArtifactStore` under
  `.pbg/artifacts/<id>/`; `resolve_study(...)` recurses producers then computes-or-reuses.
  **Not used for gating in this design** (edges are ordering-only), but available.
- **Visualizations as Steps:** `Visualization` subclasses with typed ports; discovered via
  `list_visualization_classes`; rendered against a run. `study_run_post` renders declared
  viz into `studies/<slug>/viz/*.html` + `charts/`. A `visualizations[].render:` shell hook
  that drops PNGs into `charts/` is the tolerated escape hatch.
- **Report cards:** `ReportCardStep` subtype → `viz/report_card/<card>.{html,verdict.json}`;
  verdict vocabulary `within_tol | drift | mismatch | ungraded`; self-identifies via
  `<meta name="viv-artifact" content="report-card">`.
- **outcomes shape:** `runs[].outcomes` is a dict keyed by **UPPERCASE** test name, each
  value `{result, detail}`; `result ∈ {PASS, FAIL, SKIP, PARTIAL}`.

---

## 4. Design

### 4.1 Workspace conversion (Phase A)

- Add `workspace.yaml`:
  ```yaml
  layout:
    studies: studies
    investigations: investigations
    references: references
  runtime:
    default_emitter: xarray   # confirm during planning; xarray matches ecosystem default
  ```
- Create `investigations/` and `studies/` dirs.
- The existing `@composite_generator` REGISTRY (19 generators) is unchanged and becomes the
  composite backing each study's `baseline`.
- Delete `test_suite.py`; add a thin `scripts/reproduce.py` that runs the investigation's
  studies (invokes the workbench run path over each study).
- `out*/` scratch dirs stay gitignored. Studies write into
  `studies/<slug>/charts|viz|runs.db`.
- Keep the current `out_keep/` reference artifacts available (outside git or in a fixtures
  dir) as the **fidelity oracle** for Phase B.

### 4.2 Investigation & study graph (Phase A)

One investigation: `investigations/spatio-flux-test-suite/investigation.yaml` with
`studies: [<19 slugs>]` and the narrative spine.

19 studies (slug = current `SIMULATIONS` key), each `schema_version: 4`, with:
- `baseline: [{name: baseline, composite: <dotted ref to existing generator>, params: <current overrides>}]`
- `pipeline_gate.prerequisites: [<upstream slugs>]` (ordering/graph only)
- `behavior_tests` + `runs[].outcomes` (UPPERCASE keys) capturing pass/fail
- `visualizations:` referencing the embedded viz Step outputs (charts/viz)

**Tiering / DAG:**

| Tier | Studies | Prerequisites (composition edges) |
|---|---|---|
| **0 — standalone** | `monod_kinetics` | — |
| | `ecoli_core_dfba` | — |
| | `ecoli_dfba` | — |
| | `yeast_dfba` | — |
| | `diffusion_process` | — |
| | `brownian_particles` | — |
| | `newtonian_particles` | — |
| **1 — pairs** | `community_dfba` | `ecoli_dfba` |
| | `dfba_kinetics_community` | `ecoli_core_dfba`, `monod_kinetics` |
| | `spatial_many_dfba` | `ecoli_core_dfba` |
| | `spatial_dfba_process` | `ecoli_core_dfba`, `diffusion_process` |
| | `comets_diffusion` | `ecoli_core_dfba`, `diffusion_process` |
| | `br_particles_kinetics` | `brownian_particles`, `monod_kinetics` |
| | `br_particles_dfba` | `brownian_particles`, `ecoli_core_dfba` |
| **2 — triples** | `comets_br_particles_kinetics` | `comets_diffusion`, `br_particles_kinetics` |
| | `comets_br_particles_dfba` | `comets_diffusion`, `br_particles_dfba` |
| | `comets_nt_particles_dfba` | `comets_diffusion`, `newtonian_particles`, `br_particles_dfba` |
| **3 — reference demos** | `spatioflux_reference_demo` | `comets_nt_particles_dfba` |
| | `reference_demo_x2y2` | `spatioflux_reference_demo` |

(Exact prerequisite edge sets are finalized in the plan; the tiers are fixed.)

### 4.3 Visualization Steps — post-run analysis flush (Phase B)

Visualizations do **not** run inside the composite. Every `plot_func` in `test_suite.py`
becomes a registered `Visualization(Step)` (extending the existing 5) that runs in the
**post-simulation analysis flush**: after a study's baseline composite runs and emits, the
analysis Steps read the emitted data (the run's `runs.db` / xarray history — the same
`results` list the current `plot_func`s consume) and render the exact current filenames into
`studies/<slug>/charts|viz`:
`<slug>_timeseries.png`, `<slug>_video.gif`, `<slug>_snapshots.png`, `<slug>_mass.png`,
`<slug>_mass_submasses.png`, `<slug>_particles_traces.png`, `<slug>_model_grid.png`.

This means the existing viz Step classes are **decoupled from the composites**
(`composites/metabolism.py` / `spatial.py` no longer wire `local:TestSuiteTimeSeries` etc.
into simulation state) and instead declared per study as post-run analyses, invoked by the
workbench's `study_run_post` / `run_study_analyses` path (or the spatio-flux equivalent).

A **structure step** regenerates, per study, the bigraph-viz diagram (`<slug>_viz.png`) plus
serialized `<slug>_schema.json` and `<slug>_state.json` — the artifacts
`run_composite_document` produces today. Because these derive from the composite spec rather
than the run trajectory, they may be produced by the same post-run analysis pass or directly
by the report script (finalized in the plan).

Each study gets a **ReportCardStep** whose verdict encodes the scenario's pass/fail
(`within_tol` when key metrics match the reference within tolerance).

**Fidelity oracle:** new outputs are diffed against the preserved `out_keep/` reference
artifacts (visual/structural comparison for images; exact/near-exact for schema/state JSON).
Per-scenario plot configuration (colors, coordinates, snapshot counts, units) that currently
lives in `plot_config` moves into each viz Step's config in the study.yaml.

### 4.4 Types, units, contracts (Phase C — orthogonal)

- Tighten weak ports: `DynamicFBA` / `MonodKinetics` outputs → typed delta/`count`/`mass`
  types instead of raw `float`; diffusion/spatial field arrays `array[float]` →
  `positive_array`.
- Attach **pint units** (via `bigraph_schema/units.py`) to the concentration/mass/rate/count
  types, matching the units in docstrings (mM, gDW, mmol/gDW/h, µm, pg).
- Add a process **`describe()` + `description`** to each of the 6 processes (`DynamicFBA`,
  `MonodKinetics`, `DiffusionAdvection`, `BrownianMovement`, `PymunkParticleMovement`, and
  the particle Steps) per the pbg describe() convention, so the bigraph-loom process view
  shows meaningful port descriptions, types, and units.
- This phase is independent of the graph and can land on its own.

### 4.5 Report script + native dashboard (Phase D)

- `scripts/build_report.py` reads the investigation + each study's `charts/viz` artifacts and
  emits the **current-look `report.html`**, reusing today's CSS / JSON-viewer / TOC /
  per-scenario-card structure from `library/tools.py` but driven by study data rather than an
  `out/` directory scan. Published to GitHub Pages as today.
- `vivarium-workbench serve --workspace .` provides the native interactive dashboard with no
  extra code.
- CI publishes both.

---

## 5. Component boundaries

- **Workspace/config** (`workspace.yaml`, dirs) — declares layout; depends on nothing.
- **Composites** (existing `@composite_generator` REGISTRY) — unchanged; referenced by studies.
- **Studies/investigation** (YAML) — declare baseline + DAG + tests; depend on composites.
- **Visualization Steps** (`spatio_flux/visualizations/`) — typed image-emitting Steps run in
  the **post-run analysis flush** (not inside the composite); depend on the run's emitted
  data; consumed by studies and the report.
- **Report script** (`scripts/build_report.py`) — depends on study artifacts; produces
  `report.html`.
- **Types/units/contracts** (`spatio_flux/types/`, process classes) — orthogonal; improve
  loom coverage.

Each unit has one clear purpose and a well-defined interface (YAML schema, Step ports, or
artifact filenames), and can be understood and tested independently.

---

## 6. Testing / verification

- **Phase A:** `vivarium-workbench serve --workspace .` renders the investigation graph
  with 19 nodes and the tier edges; every study's `baseline.composite` resolves (no
  "composite not found" banner).
- **Phase B:** each study, when run, produces the exact set of expected filenames; outputs
  diffed against `out_keep/` reference artifacts within tolerance.
- **Phase C:** loom process view shows units + descriptions on all 6 processes; existing
  `tests/test_composite_generators.py` and `tests/test_demo_visualizations.py` still pass.
- **Phase D:** `scripts/build_report.py` output visually matches the current published
  `report.html` (same sections, cards, artifacts per scenario).

---

## 7. Sequencing

1. **Phase A** — workspace + 19 studies + investigation DAG; native dashboard renders.
2. **Phase B** — Visualization Steps reproduce all outputs; validated vs `out_keep/`.
3. **Phase C** — types/units/contracts (independent; can run in parallel with A/B).
4. **Phase D** — report script matches the current page.

Dependencies: B depends on A; D depends on B; A and C are independent.

---

## 8. Open items for the implementation plan

- Confirm `runtime.default_emitter` (`xarray` vs `parquet`) for these small spatial runs.
- Exact prerequisite edge set per combo study (tiers are fixed; individual edges to finalize).
- Whether the structure step (diagram + schema/state) is one shared Step reused by all
  studies or generated by the report script directly.
- Report-card metric definitions per scenario (what "within tolerance" means for each).
- Location of the `out_keep/` fidelity oracle (fixtures dir vs external).
