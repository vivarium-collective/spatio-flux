# Phase A: Workspace Scaffold + Investigation/Study Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the `spatio-flux` repo into a vivarium-workbench workspace whose one investigation, `spatio-flux-test-suite`, contains 19 studies (1:1 with the current `test_suite.py` scenarios) arranged as a composition DAG (standalone → pairs → triples → reference demos), with every study's baseline composite resolving in the dashboard.

**Architecture:** Add a `workspace.yaml` + a `spatio_flux/core.py::build_core()` so the repo is a valid, runnable workspace. Author `investigations/spatio-flux-test-suite/investigation.yaml` and 19 `studies/<slug>/study.yaml` files (via a data-driven scaffolder over a complete inlined table), each declaring a `baseline` pointing at the existing `@composite_generator` (which resolves through the shared `process_bigraph.composite_spec` registry with no bridging) and `pipeline_gate.prerequisites` encoding the composition edges. Validate with a pytest suite that mirrors the workbench resolver.

**Tech Stack:** Python 3.12, `process-bigraph` / `bigraph-schema` (the composite registry), `pbg_superpowers`/`viva_superpowers` (`@composite_generator`, same registry), `pyyaml`, `pytest`. Dashboard: `vivarium-workbench serve` (manual verification only — not an automated test dependency).

## Global Constraints

- **Worktree:** all work happens in `~/code/spatio-flux--workbench-modernization` on branch `workbench-modernization` (already created). Verify `git branch --show-current` + `git rev-parse --short HEAD` before every commit.
- **Do NOT delete `test_suite.py` in Phase A.** It is the fidelity oracle for Phase B; its retirement + the `reproduce.py` shim happen after B/D land. Phase A leaves it working.
- **Study count is exactly 19** — the current `SIMULATIONS` keys, no more, no less.
- **Composite refs are dotted `<module>.<name>`** exactly as registered (table below); they resolve via `process_bigraph.composite_spec.get(ref)`. A ref that returns `None` is the "composite not found" banner — a test failure.
- **Vocabulary:** the file is `investigation.yaml` with a `studies:` membership list; studies live flat under `studies/<slug>/study.yaml`.
- **Graph edges are ordering/documentation only** (approved decision) — no gating semantics; every study stays independently runnable. Encode them in `pipeline_gate.prerequisites`.
- **Emitter:** `runtime.default_emitter: sqlite` (studies will read back per-step nested field/particle arrays via `runs.db` in Phase B; sqlite is the readable-trajectory shape).
- No `analyses:` blocks (that path requires v2ecoli); visualizations come in Phase B via bespoke `canonical_runs:` runners.

---

## The 19 studies (complete data table)

Composite ref = `spatio_flux.composites.<module>.<name>`. `params` are the current `overrides` from `test_suite.py`. Prereqs are the composition edges. Tier is informational (drives no code; the DAG comes from prereqs).

| slug | module.name | params | tier | prerequisites |
|---|---|---|---|---|
| `monod_kinetics` | `metabolism.monod_kinetics` | `{model_id: overflow_metabolism}` | 0 | — |
| `ecoli_core_dfba` | `metabolism.ecoli_core_dfba` | `{model_id: "ecoli core", glucose: 10.0, acetate: 0.0}` | 0 | — |
| `ecoli_dfba` | `metabolism.ecoli_dfba` | `{model_id: ecoli, glucose: 10.0, formate: 5.0}` | 0 | — |
| `yeast_dfba` | `metabolism.yeast_dfba` | `{model_id: yeast, glucose: 5.0}` | 0 | — |
| `diffusion_process` | `spatial.diffusion_process` | `{}` | 0 | — |
| `brownian_particles` | `particles.brownian_particles` | `{}` | 0 | — |
| `newtonian_particles` | `particles.newtonian_particles` | `{}` | 0 | — |
| `community_dfba` | `metabolism.community_dfba` | `{}` | 1 | `ecoli_dfba` |
| `dfba_kinetics_community` | `metabolism.dfba_kinetics_community` | `{}` | 1 | `ecoli_core_dfba`, `monod_kinetics` |
| `spatial_many_dfba` | `spatial.spatial_many_dfba` | `{model_id: "ecoli core"}` | 1 | `ecoli_core_dfba` |
| `spatial_dfba_process` | `spatial.spatial_dfba_process` | `{}` | 1 | `ecoli_core_dfba`, `diffusion_process` |
| `comets_diffusion` | `comets.comets_diffusion` | `{}` | 1 | `ecoli_core_dfba`, `diffusion_process` |
| `br_particles_kinetics` | `particles.br_particles_kinetics` | `{}` | 1 | `brownian_particles`, `monod_kinetics` |
| `br_particles_dfba` | `particles.br_particles_dfba` | `{particle_model_id: "ecoli core"}` | 1 | `brownian_particles`, `ecoli_core_dfba` |
| `comets_br_particles_kinetics` | `comets.comets_br_particles_kinetics` | `{}` | 2 | `comets_diffusion`, `br_particles_kinetics` |
| `comets_br_particles_dfba` | `comets.comets_br_particles_dfba` | `{}` | 2 | `comets_diffusion`, `br_particles_dfba` |
| `comets_nt_particles_dfba` | `comets.comets_nt_particles_dfba` | `{}` | 2 | `comets_diffusion`, `newtonian_particles`, `br_particles_dfba` |
| `spatioflux_reference_demo` | `reference.spatioflux_reference_demo` | `{n_bins: [10, 10]}` | 3 | `comets_nt_particles_dfba` |
| `reference_demo_x2y2` | `reference.reference_demo_x2y2` | `{n_bins: [20, 20]}` | 3 | `spatioflux_reference_demo` |

(`n_bins` literals: `SQUARE_BINS = (10, 10)`; x2y2 doubles to `(20, 20)`.)

---

## File Structure

- Create `workspace.yaml` — workspace manifest (layout defaults + sqlite emitter).
- Create `spatio_flux/core.py` — `build_core()` for the run/discovery path.
- Create `investigations/spatio-flux-test-suite/investigation.yaml` — narrative + `studies:` list.
- Create `studies/<slug>/study.yaml` × 19 — one per scenario.
- Create `scripts/scaffold_studies.py` — data-driven generator that writes the 19 study.yaml files from an inlined `STUDIES` table + REGISTRY descriptions. Idempotent; re-runnable.
- Create `tests/test_workspace_investigation.py` — validation suite (resolver-mirroring).
- Modify `.gitignore` — ensure `studies/*/runs.db`, `studies/*/charts/`, `studies/*/viz/`, `.pbg/` are ignored (runtime artifacts land there in Phase B).

---

### Task 1: Workspace manifest + `build_core()`

**Files:**
- Create: `workspace.yaml`
- Create: `spatio_flux/core.py`
- Test: `tests/test_workspace_investigation.py`

**Interfaces:**
- Produces: `spatio_flux.core.build_core() -> core` (a bigraph-schema core with spatio-flux types + processes + visualizations + composites registered). Consumed by the workbench run path (`run_runner.execute` imports `<package_path>.core.build_core`) and by later tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workspace_investigation.py
import os, yaml
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_workspace_yaml_present_and_shaped():
    with open(os.path.join(REPO, "workspace.yaml")) as f:
        ws = yaml.safe_load(f)
    assert ws["schema_version"] == 2
    assert ws["name"] == "spatio_flux"
    assert ws["package_path"] == "spatio_flux"
    assert ws["runtime"]["default_emitter"] == "sqlite"

def test_build_core_constructs():
    from spatio_flux.core import build_core
    core = build_core()
    # custom spatio-flux types registered
    assert core.access("concentration") is not None
    assert core.access("particle") is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/code/spatio-flux--workbench-modernization && .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: FAIL — `workspace.yaml` missing / `spatio_flux.core` has no `build_core`.

(Note: the `.venv` is the shared checkout's; run with `PYTHONPATH=~/code/spatio-flux--workbench-modernization` prepended so imports resolve against the worktree, per the worktree rule. Verify with `python -c "import spatio_flux; print(spatio_flux.__file__)"`.)

- [ ] **Step 3: Write `workspace.yaml`**

```yaml
schema_version: 2
name: spatio_flux
package_path: spatio_flux
runtime:
  default_emitter: sqlite
phases: []
observables: []
visualizations: []
simulations: []
datasets: []
references_bib: references/papers.bib
server:
  enabled: true
```

- [ ] **Step 4: Write `spatio_flux/core.py`**

```python
"""Workspace core builder.

The vivarium-workbench run path imports ``<package_path>.core.build_core``
(here ``spatio_flux.core.build_core``) to construct a bigraph-schema core with
this workspace's types, processes, visualizations, and composite generators
registered. ``allocate_core()`` auto-discovers installed process/emitter/
visualization Steps; ``register_types`` adds spatio-flux's custom types; the
composites import fires the ``@composite_generator`` decorators.
"""
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

import spatio_flux                    # exposes register_types; imports composites
import spatio_flux.visualizations     # noqa: F401  (viz Step discovery)


def build_core():
    core = allocate_core()
    spatio_flux.register_types(core)
    core.register_link("RAMEmitter", RAMEmitter)
    return core
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add workspace.yaml spatio_flux/core.py tests/test_workspace_investigation.py
git commit -m "feat(workspace): add workspace.yaml + build_core() for workbench onboarding"
```

---

### Task 2: The study scaffolder + investigation file

**Files:**
- Create: `scripts/scaffold_studies.py`
- Create: `investigations/spatio-flux-test-suite/investigation.yaml`
- Test: `tests/test_workspace_investigation.py` (extend)

**Interfaces:**
- Consumes: `spatio_flux.composites.REGISTRY` (for per-study `description`).
- Produces: `scripts/scaffold_studies.py` with a module-level `STUDIES` list of dicts `{slug, ref, params, tier, prerequisites}` (the complete table above) and `main()` that writes `studies/<slug>/study.yaml` for each. Later tasks/tests import `STUDIES` from it.

- [ ] **Step 1: Write the failing test**

```python
def test_scaffolder_table_complete():
    from scripts.scaffold_studies import STUDIES
    slugs = {s["slug"] for s in STUDIES}
    assert len(STUDIES) == 19
    assert "spatioflux_reference_demo" in slugs
    # every prerequisite references a real slug
    for s in STUDIES:
        for p in s["prerequisites"]:
            assert p in slugs, f"{s['slug']} -> unknown prereq {p}"

def test_investigation_lists_all_19():
    import os, yaml
    with open(os.path.join(REPO, "investigations", "spatio-flux-test-suite",
                           "investigation.yaml")) as f:
        inv = yaml.safe_load(f)
    from scripts.scaffold_studies import STUDIES
    assert inv["name"] == "spatio-flux-test-suite"
    assert set(inv["studies"]) == {s["slug"] for s in STUDIES}
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: FAIL — `scripts.scaffold_studies` missing.

- [ ] **Step 3: Write `scripts/scaffold_studies.py`** (the complete table + renderer)

```python
"""Data-driven scaffolder for the spatio-flux-test-suite investigation.

Writes one studies/<slug>/study.yaml per SIMULATIONS scenario. Idempotent:
re-running overwrites the generated files in place. Studies are faithful
reproductions of the test-suite scenarios, so purpose text is derived from the
composite generator's own description.
"""
import os
import yaml
from spatio_flux.composites import REGISTRY

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Complete table — the 19 scenarios (slug == SIMULATIONS key).
STUDIES = [
    # tier 0 — standalone processes
    {"slug": "monod_kinetics", "ref": "spatio_flux.composites.metabolism.monod_kinetics",
     "params": {"model_id": "overflow_metabolism"}, "tier": 0, "prerequisites": []},
    {"slug": "ecoli_core_dfba", "ref": "spatio_flux.composites.metabolism.ecoli_core_dfba",
     "params": {"model_id": "ecoli core", "glucose": 10.0, "acetate": 0.0}, "tier": 0, "prerequisites": []},
    {"slug": "ecoli_dfba", "ref": "spatio_flux.composites.metabolism.ecoli_dfba",
     "params": {"model_id": "ecoli", "glucose": 10.0, "formate": 5.0}, "tier": 0, "prerequisites": []},
    {"slug": "yeast_dfba", "ref": "spatio_flux.composites.metabolism.yeast_dfba",
     "params": {"model_id": "yeast", "glucose": 5.0}, "tier": 0, "prerequisites": []},
    {"slug": "diffusion_process", "ref": "spatio_flux.composites.spatial.diffusion_process",
     "params": {}, "tier": 0, "prerequisites": []},
    {"slug": "brownian_particles", "ref": "spatio_flux.composites.particles.brownian_particles",
     "params": {}, "tier": 0, "prerequisites": []},
    {"slug": "newtonian_particles", "ref": "spatio_flux.composites.particles.newtonian_particles",
     "params": {}, "tier": 0, "prerequisites": []},
    # tier 1 — pairs
    {"slug": "community_dfba", "ref": "spatio_flux.composites.metabolism.community_dfba",
     "params": {}, "tier": 1, "prerequisites": ["ecoli_dfba"]},
    {"slug": "dfba_kinetics_community", "ref": "spatio_flux.composites.metabolism.dfba_kinetics_community",
     "params": {}, "tier": 1, "prerequisites": ["ecoli_core_dfba", "monod_kinetics"]},
    {"slug": "spatial_many_dfba", "ref": "spatio_flux.composites.spatial.spatial_many_dfba",
     "params": {"model_id": "ecoli core"}, "tier": 1, "prerequisites": ["ecoli_core_dfba"]},
    {"slug": "spatial_dfba_process", "ref": "spatio_flux.composites.spatial.spatial_dfba_process",
     "params": {}, "tier": 1, "prerequisites": ["ecoli_core_dfba", "diffusion_process"]},
    {"slug": "comets_diffusion", "ref": "spatio_flux.composites.comets.comets_diffusion",
     "params": {}, "tier": 1, "prerequisites": ["ecoli_core_dfba", "diffusion_process"]},
    {"slug": "br_particles_kinetics", "ref": "spatio_flux.composites.particles.br_particles_kinetics",
     "params": {}, "tier": 1, "prerequisites": ["brownian_particles", "monod_kinetics"]},
    {"slug": "br_particles_dfba", "ref": "spatio_flux.composites.particles.br_particles_dfba",
     "params": {"particle_model_id": "ecoli core"}, "tier": 1, "prerequisites": ["brownian_particles", "ecoli_core_dfba"]},
    # tier 2 — triples
    {"slug": "comets_br_particles_kinetics", "ref": "spatio_flux.composites.comets.comets_br_particles_kinetics",
     "params": {}, "tier": 2, "prerequisites": ["comets_diffusion", "br_particles_kinetics"]},
    {"slug": "comets_br_particles_dfba", "ref": "spatio_flux.composites.comets.comets_br_particles_dfba",
     "params": {}, "tier": 2, "prerequisites": ["comets_diffusion", "br_particles_dfba"]},
    {"slug": "comets_nt_particles_dfba", "ref": "spatio_flux.composites.comets.comets_nt_particles_dfba",
     "params": {}, "tier": 2, "prerequisites": ["comets_diffusion", "newtonian_particles", "br_particles_dfba"]},
    # tier 3 — reference demos
    {"slug": "spatioflux_reference_demo", "ref": "spatio_flux.composites.reference.spatioflux_reference_demo",
     "params": {"n_bins": [10, 10]}, "tier": 3, "prerequisites": ["comets_nt_particles_dfba"]},
    {"slug": "reference_demo_x2y2", "ref": "spatio_flux.composites.reference.reference_demo_x2y2",
     "params": {"n_bins": [20, 20]}, "tier": 3, "prerequisites": ["spatioflux_reference_demo"]},
]

_TIER_LABEL = {0: "standalone process", 1: "pair composition",
               2: "triple composition", 3: "reference demo"}


def _description_for(ref):
    name = ref.rsplit(".", 1)[1]
    for e in REGISTRY.values():
        if getattr(e, "name", None) == name:
            return getattr(e, "description", "") or ""
    return ""


def render_study(entry):
    slug, ref = entry["slug"], entry["ref"]
    desc = _description_for(ref)
    return {
        "schema_version": 3,
        "name": slug,
        "created": "2026-07-26",
        "status": "planned",
        "phase": "Design",
        "baseline": [{"name": "baseline", "composite": ref, "params": entry["params"]}],
        "purpose": {
            "question": desc or f"Reproduce the {slug} test-suite scenario as a study.",
            "mechanism": f"Runs the {ref} composite ({_TIER_LABEL[entry['tier']]}).",
            "expected_outcome": "Reproduces the current test-suite report artifacts for this scenario.",
        },
        "pipeline_gate": {
            "prerequisites": entry["prerequisites"],
            "enables": [],
            "proceed_condition": "Composition ordering only; this study runs independently.",
        },
        "limitations": [
            "Faithful reproduction of an existing scenario; no new biology.",
            "Graph edges document composition, they do not gate execution.",
        ],
    }


def main():
    for entry in STUDIES:
        d = os.path.join(REPO, "studies", entry["slug"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "study.yaml"), "w") as f:
            yaml.safe_dump(render_study(entry), f, sort_keys=False, default_flow_style=False)
    print(f"wrote {len(STUDIES)} studies")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `investigations/spatio-flux-test-suite/investigation.yaml`**

```yaml
schema_version: 2
name: spatio-flux-test-suite
title: "Spatio-Flux Test Suite — compositional multiscale scenarios"
created: '2026-07-26'
status: running
executive: |
  The spatio-flux reference application reproduced as one investigation: 19
  studies, each a composition scenario from the original test suite, arranged
  standalone processes → pairs → triples → reference demos.
scientific_argument: |
  Independently developed processes (dFBA, Monod kinetics, diffusion-advection,
  Brownian and Newtonian particles) compose through typed shared state. Running
  the standalone processes first, then their combinations, demonstrates that
  composition is additive and inspectable.
biological_story: |
  Metabolism, spatial transport, and particle dynamics coupled on a shared
  lattice reproduce COMETS-style spatial dFBA and particle–field exchange.
studies:
- monod_kinetics
- ecoli_core_dfba
- ecoli_dfba
- yeast_dfba
- diffusion_process
- brownian_particles
- newtonian_particles
- community_dfba
- dfba_kinetics_community
- spatial_many_dfba
- spatial_dfba_process
- comets_diffusion
- br_particles_kinetics
- br_particles_dfba
- comets_br_particles_kinetics
- comets_br_particles_dfba
- comets_nt_particles_dfba
- spatioflux_reference_demo
- reference_demo_x2y2
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: PASS (4 tests). The scaffolder is not yet invoked — only its table + the investigation file are tested here.

- [ ] **Step 6: Commit**

```bash
git add scripts/scaffold_studies.py investigations/ tests/test_workspace_investigation.py
git commit -m "feat(investigation): add scaffolder table + spatio-flux-test-suite investigation.yaml"
```

---

### Task 3: Generate the 19 study.yaml files + resolver-mirroring tests

**Files:**
- Create: `studies/<slug>/study.yaml` × 19 (generated)
- Test: `tests/test_workspace_investigation.py` (extend)

**Interfaces:**
- Consumes: `scripts.scaffold_studies.main`, `STUDIES`.
- Produces: 19 on-disk `study.yaml` files. Later phases read these.

- [ ] **Step 1: Write the failing test** (the core guarantees: discovery, resolution, DAG)

```python
def _load_studies():
    import glob, yaml
    out = {}
    for p in glob.glob(os.path.join(REPO, "studies", "*", "study.yaml")):
        with open(p) as f:
            out[os.path.basename(os.path.dirname(p))] = yaml.safe_load(f)
    return out

def test_all_19_studies_on_disk():
    from scripts.scaffold_studies import STUDIES
    studies = _load_studies()
    assert set(studies) == {s["slug"] for s in STUDIES}

def test_every_baseline_composite_resolves():
    # Mirrors the workbench resolver: process_bigraph.composite_spec.get(ref).
    import spatio_flux.composites  # noqa: F401  (fire @composite_generator)
    from process_bigraph.composite_spec import get as get_spec
    for slug, spec in _load_studies().items():
        ref = spec["baseline"][0]["composite"]
        assert get_spec(ref) is not None, f"{slug}: composite not found: {ref}"

def test_prerequisites_reference_real_slugs_and_are_acyclic():
    studies = _load_studies()
    edges = {slug: spec["pipeline_gate"]["prerequisites"] for slug, spec in studies.items()}
    for slug, prereqs in edges.items():
        for p in prereqs:
            assert p in studies, f"{slug}: unknown prerequisite {p}"
    # topological sort must succeed (acyclic)
    seen, stack = set(), set()
    def visit(n):
        if n in seen:
            return
        assert n not in stack, f"cycle at {n}"
        stack.add(n)
        for p in edges[n]:
            visit(p)
        stack.discard(n); seen.add(n)
    for slug in edges:
        visit(slug)

def test_tier0_have_no_prereqs_and_demos_chain():
    studies = _load_studies()
    for s in ("monod_kinetics", "ecoli_core_dfba", "diffusion_process",
              "brownian_particles", "newtonian_particles"):
        assert studies[s]["pipeline_gate"]["prerequisites"] == []
    assert studies["reference_demo_x2y2"]["pipeline_gate"]["prerequisites"] == ["spatioflux_reference_demo"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: FAIL — no `studies/*/study.yaml` yet.

- [ ] **Step 3: Generate the studies**

Run: `PYTHONPATH=$PWD .venv/bin/python scripts/scaffold_studies.py`
Expected stdout: `wrote 19 studies`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_workspace_investigation.py -q`
Expected: PASS (all tests, including the 19-composite resolution check — the "no composite-not-found banner" guarantee).

- [ ] **Step 5: Update `.gitignore` for runtime artifacts**

Add these lines to `.gitignore` (append; do not remove existing entries):

```gitignore
# workbench runtime artifacts (Phase B+)
studies/*/runs.db
studies/*/parquet-runs/
studies/*/charts/
studies/*/viz/
.pbg/
```

- [ ] **Step 6: Commit**

```bash
git add studies/ tests/test_workspace_investigation.py .gitignore
git commit -m "feat(studies): generate 19 study.yaml files; all baseline composites resolve"
```

---

### Task 4: Manual dashboard verification + plan-doc note

**Files:**
- Modify: `docs/superpowers/plans/2026-07-26-phaseA-workspace-investigation-graph.md` (check off a verification note)

This task has no automated test — it confirms the dashboard renders the workspace end-to-end, which needs the `vivarium-workbench` server (not a spatio-flux dependency).

- [ ] **Step 1: Serve the workspace**

Run (from the worktree root):
```bash
vivarium-workbench serve --workspace . --port 8099
```
If `vivarium-workbench` is not on PATH in this venv, install it into a scratch venv or run from `~/code/vivarium-workbench` with `--workspace ~/code/spatio-flux--workbench-modernization`.

- [ ] **Step 2: Confirm composites resolve via the API**

Run (in another shell):
```bash
curl -s "http://127.0.0.1:8099/api/composite-resolve?id=spatio_flux.composites.reference.spatioflux_reference_demo" | python -m json.tool
```
Expected: JSON with `"wiring_status": "ready"` (not `"unavailable"`, not 404).

- [ ] **Step 3: Confirm the investigation renders 19 studies**

Open `http://127.0.0.1:8099` → Investigations → `spatio-flux-test-suite`. Confirm: 19 study nodes, DAG edges from standalones into their combinations, no "composite not found" banner on any study.

- [ ] **Step 4: Record the result**

If all three pass, note it in this plan file (append a `> Verified <date>: dashboard renders 19 studies, all composites ready.` line under this task) and commit:
```bash
git add docs/superpowers/plans/2026-07-26-phaseA-workspace-investigation-graph.md
git commit -m "docs(phaseA): record dashboard verification of the investigation graph"
```

If any fail: the resolver test in Task 3 should already have caught a bad ref; a dashboard-only failure points at `workspace.yaml`/`core.py` (Task 1) or the env worker not importing the package — re-check `package_path: spatio_flux` and that `spatio_flux` is importable from the worktree (editable install or `PYTHONPATH`).

---

## Self-Review

**Spec coverage (against §4.1–4.2 of the design spec):**
- Workspace conversion (`workspace.yaml`, `investigations/`, `studies/`) → Tasks 1–3. ✓
- `build_core()` for the run path → Task 1. ✓
- One investigation with 19 studies + narrative spine → Task 2. ✓
- Baseline blocks pointing at existing generators → Task 3 (verified resolving). ✓
- DAG edges via `pipeline_gate.prerequisites`, ordering-only → Tasks 2–3. ✓
- Native dashboard renders → Task 4. ✓
- **Deferred (correctly out of Phase A scope):** deleting `test_suite.py` + `reproduce.py` shim (§4.1) — depends on Phase B/D; explicitly held per Global Constraints. `behavior_tests`/`outcomes` (§4.2) — added in Phase B with their evaluators. Visualization Steps (§4.3), types/units/contracts (§4.4), report script (§4.5) — Phases B/C/D.

**Placeholder scan:** no TBD/TODO in generated content; every study value is in the table; `build_core`, `workspace.yaml`, `investigation.yaml`, and the scaffolder are shown in full. ✓

**Type/name consistency:** `STUDIES` dict keys (`slug`, `ref`, `params`, `tier`, `prerequisites`) are used identically in the scaffolder and all tests; `build_core` name matches the workbench import contract (`<package_path>.core.build_core`); composite refs match the extracted `<module>.<name>` registry values. ✓

---

## Follow-on plans (not this document)

- **Phase B** — post-run analysis-flush Visualization Steps reproducing every PNG/GIF/diagram (bespoke `canonical_runs:` runners; Path C `runs.db` reads for nested field/particle arrays); `behavior_tests` + report cards + `outcomes`; validated against `out_keep/`.
- **Phase C** — typed ports + pint units + process `describe()` contracts for loom coverage (orthogonal).
- **Phase D** — the custom report script (`scripts/build_report.py`) reproducing the current `report.html`, retire `test_suite.py`, add `reproduce.py`.
