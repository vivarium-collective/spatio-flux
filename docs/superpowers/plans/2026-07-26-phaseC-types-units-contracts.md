# Phase C: Typed Ports, Units, and Process Contracts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-07-26): EXECUTED (Tasks 1–3, 5).** Units mechanism resolved:
> attach via a `_units` **key on the port schema** (`{'_type':'float','_units':'mM'}`),
> NOT `quantity[float,X]` (leaves `_units` None here) and NO pint registration
> (pint is only invoked by `_compute_unit_scale` for cross-unit wire conversion).
> DFBA/Monod ports carry mM/gDW; all 11 processes/steps have a formal `description`
> surfaced by `Edge.describe()` (verified). Base port types were **preserved** (the
> 19 sims still reproduce) — the riskier float→mass / array→positive_array retyping
> was intentionally NOT done to protect the working flush. Diffusion field arrays left
> generic (heterogeneous molecules). Tests: `tests/test_types_units_contracts.py` green.
> **Task 4 (units_resolver + Visualization.units_resolver): SKIPPED as redundant** —
> the flush already passes `field_units` to the plots, and spatio-flux doesn't render
> through the workbench viz path, so the resolver would be dead code; loom port-units
> come directly from the `_units` schema keys.

**Goal:** Give the bigraph-loom process view meaningful, unit-bearing type coverage for spatio-flux: tighten the weak/generic ports, attach units to the unit-bearing ports, and give every process a formal `description` contract — so the loom inspector shows typed, described, unit-labeled ports and figure axes carry units.

**Architecture:** Three orthogonal, independently-testable changes on the existing processes. (1) A units spike pins how the modern bigraph-schema `quantity[<datatype>,<unit>]` type populates `_units`; (2) re-type unit-bearing ports with `quantity[...]` and tighten generic `float`/`array[float]` outputs to the existing typed `count`/`mass`/`positive_array` types; (3) set a `description` class attribute on each process/Step (surfaced by the workbench's `Edge.describe()` → `_attach_process_docs` path). A `units_resolver.build_units_index()` + `Visualization.units_resolver` wiring (mirroring v2ecoli) turns the port `_units` into figure-axis and inspector labels.

**Tech Stack:** `bigraph-schema` (units.py pint types; `quantity[...]`), `viva_superpowers.visualization.Visualization.units_resolver`, `process_bigraph` (`Edge.describe`/`description`), `pytest`. Runs under the shared `.venv` for type/description unit tests; the loom-view check uses `.venv-serve`.

## Global Constraints

- **Worktree** `~/code/spatio-flux--workbench-modernization`, branch `workbench-modernization`. Verify branch/HEAD before commits.
- **Orthogonal to A/B** — touches only `spatio_flux/processes/*.py`, `spatio_flux/types/`, `spatio_flux/library/units_resolver.py`, `spatio_flux/visualizations/__init__.py`. No study/graph/report changes.
- **Reuse the existing typed types** in `spatio_flux/types/positive.py` (`concentration`, `mass`, `count`, `positive_array`, `set_float`) — don't invent parallel ones. Add units on top via `quantity[...]` only where a real unit applies.
- **Units are best-effort decoration** — a port that can't carry a unit stays as-is; never break composite resolution for a label. (Matches `units_resolver`'s try/except contract.)
- **Contracts are formal descriptions** — the `description` class attribute is the canonical formal (what the process computes), markdown/LaTeX allowed; falls back to docstring. Set it on all 6 processes + the particle Steps + the Phase-B analysis Viz Steps.
- **The 6 processes + Steps:** `DynamicFBA`/`SpatialDFBA`/`ShardedDFBA` (`processes/dfba.py`), `MonodKinetics` (`monod_kinetics.py`), `DiffusionAdvection` (`diffusion_advection.py`), `BrownianMovement`/`ManageBoundaries`/`ParticleExchange`/`ParticleDivision`/`ParticleTotalMass` (`particles.py`), `PymunkParticleMovement` (`pymunk_particles.py`).
- **Documented units** (from current docstrings — the source of truth for which unit goes where): concentration = **mM**; biomass/mass = **gDW** (dFBA) / **pg** (particles); FBA flux = **mmol/gDW/h**; particle position/size = **µm**; box volume = **L**.

## Ports to tighten (from the Phase-A audit)

| Process | Port | Current | Target |
|---|---|---|---|
| DynamicFBA | outputs.substrates | `map[count]` | `map[quantity[count,...]]` (delta) — keep count semantics |
| DynamicFBA | inputs.substrates | `map[concentration]` | `map[quantity[concentration,mM]]` |
| DynamicFBA | inputs/outputs.biomass | `mass` | `quantity[mass,gDW]` |
| MonodKinetics | outputs.biomass / substrates | `float` / `map[float]` | `mass`-delta / `map[concentration]`-delta (typed) |
| DiffusionAdvection | inputs/outputs.fields | `map[array[float]]` | `map[positive_array]` (+ `quantity` unit where mM) |
| SpatialDFBA / ParticleExchange | field arrays | `array[float]` | `positive_array` |
| Pymunk / particles | position / mass / radius | generic floats | `quantity[...,µm]` / `quantity[mass,pg]` where applicable |

---

### Task 1: Units-typing spike — pin `quantity[...]` / `_units` behavior

**Files:**
- Create: `tests/test_types_units_contracts.py`
- Possibly Modify: `spatio_flux/types/positive.py` (register any missing pint units so `_units` populates)

**Why a spike:** `core.access("quantity[float,mM]")` parses but returned `_units == None` in a quick probe; `gDW` is not a standard pint unit. This task pins exactly which unit strings populate `_units`, and registers spatio-flux's units (`gDW`, `mM` if needed) into the bigraph-schema pint registry so the type carries them.

**Interfaces:**
- Produces: a documented set of working unit type strings (e.g. `quantity[float,millimolar]`, `quantity[float,gDW]`) with `_units` populated, plus any registration code in `positive.py`.

- [ ] **Step 1: Write the failing test** (assert `_units` populates for the spatio-flux units)

```python
# tests/test_types_units_contracts.py
from spatio_flux.core import build_core

def test_quantity_types_carry_units():
    core = build_core()
    import spatio_flux.types  # ensures unit registration side-effects ran
    cases = {
        "quantity[float,gDW]": "gDW",
        "quantity[float,mmol/gDW/h]": "mmol/gDW/h",
        "quantity[float,micrometer]": "micrometer",
    }
    for type_str, expected in cases.items():
        node = core.access(type_str)
        assert getattr(node, "_units", None), f"{type_str}: _units empty"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_types_units_contracts.py -q`
Expected: FAIL — `_units` empty for at least `gDW` (unregistered pint unit).

- [ ] **Step 3: Register spatio-flux units** into the bigraph-schema pint registry at type-registration time. In `spatio_flux/types/positive.py` (or a new `units.py` imported by `types/__init__.py`), define the custom units and populate the registry `bigraph_schema.units` uses:

```python
from bigraph_schema.units import get_quantity_registry
_ureg = get_quantity_registry()
# gDW (grams dry weight) and mM are not pint defaults; define them.
_ureg.define("gDW = [biomass]")          # base dimension for dry-weight mass
_ureg.define("millimolar = 1e-3 * mol / liter = mM")
```
(Confirm the exact API for populating `_units` on `quantity[...]` — inspect `bigraph_schema/type_functions.py` where `quantity` is applied; register units so `_unit_from_node` reads a non-empty `_units`.)

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_types_units_contracts.py -q`
Expected: PASS. Record the working unit strings in a comment block for Task 2.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/types/ tests/test_types_units_contracts.py
git commit -m "feat(types): register gDW/mM units so quantity[...] carries _units"
```

---

### Task 2: Re-type unit-bearing + weak ports

**Files:**
- Modify: `spatio_flux/processes/dfba.py`, `monod_kinetics.py`, `diffusion_advection.py`, `particles.py`, `pymunk_particles.py`
- Test: `tests/test_types_units_contracts.py` (extend)

- [ ] **Step 1: Write the failing test** (ports resolve + carry units where expected)

```python
def test_dfba_ports_typed_and_unitful():
    core = build_core()
    from spatio_flux.processes.dfba import DynamicFBA
    inst = DynamicFBA.__new__(DynamicFBA)
    ins, outs = inst.inputs(), inst.outputs()
    # biomass now carries gDW
    node = core.access(ins["biomass"])
    assert getattr(node, "_units", None) == "gDW"
    # outputs.substrates no longer generic — resolves as a typed map
    assert "map" in outs["substrates"]

def test_diffusion_fields_positive_array():
    from spatio_flux.processes.diffusion_advection import DiffusionAdvection
    inst = DiffusionAdvection.__new__(DiffusionAdvection)
    assert "positive_array" in str(inst.inputs()["fields"]) or \
           "positive_array" in str(inst.outputs()["fields"])
```

- [ ] **Step 2: Run to verify it fails.**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_types_units_contracts.py -q -k "typed or positive_array"`
Expected: FAIL.

- [ ] **Step 3: Apply the port retypes** per the "Ports to tighten" table. Edit each process's `inputs()`/`outputs()`/`config_schema` to use `quantity[mass,gDW]`, `quantity[concentration,mM]`, `map[positive_array]`, typed `count`/`mass` deltas. Keep the apply/accumulate semantics (deltas stay deltas). Do NOT change process math.

- [ ] **Step 4: Run to verify it passes** — new tests + a composite-build smoke test (Phase A/B composites still resolve under the retyped ports).

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/ -q`
Then under serving venv, build one composite of each family to confirm no `cannot resolve types` errors:
`.venv-serve/bin/python -c "from spatio_flux.core import build_core; from pbg_superpowers.composite_generator import build_generator,_REGISTRY; c=build_core(); [build_generator(_REGISTRY[r],overrides={},core=c) for r in ('spatio_flux.composites.comets.comets_diffusion','spatio_flux.composites.particles.brownian_particles')]; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/processes/
git commit -m "feat(types): unit-typed + tightened ports across the 6 processes"
```

---

### Task 3: Process contracts (`description` class attribute)

**Files:**
- Modify: all process/Step classes in `spatio_flux/processes/*.py`, `spatio_flux/analysis/steps.py` (Phase B viz Steps if present)
- Test: `tests/test_types_units_contracts.py` (extend)

- [ ] **Step 1: Write the failing test** (every process class yields a non-empty formal description)

```python
def test_all_processes_have_descriptions():
    import inspect
    from spatio_flux.processes import dfba, monod_kinetics, diffusion_advection, particles, pymunk_particles
    from process_bigraph import Process, Step
    mods = [dfba, monod_kinetics, diffusion_advection, particles, pymunk_particles]
    classes = [c for m in mods for _, c in inspect.getmembers(m, inspect.isclass)
               if issubclass(c, (Process, Step)) and c.__module__ == m.__name__]
    assert classes
    for c in classes:
        desc = getattr(c, "description", "")
        assert isinstance(desc, str) and desc.strip(), f"{c.__name__} has no description"
```

- [ ] **Step 2: Run to verify it fails.**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_types_units_contracts.py -q -k descriptions`
Expected: FAIL (no `description` attrs).

- [ ] **Step 3: Add a `description` class attribute** to each process/Step — one formal sentence stating what it computes (e.g. `DynamicFBA.description = "Dynamic FBA: maximizes biomass flux under substrate-limited exchange bounds; updates substrate counts and biomass (gDW) each interval."`). Content per class comes from its existing docstring, promoted to the formal `description`.

- [ ] **Step 4: Run to verify it passes.**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_types_units_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/processes/ spatio_flux/analysis/
git commit -m "feat(contracts): add formal description to every process/step"
```

---

### Task 4: Units resolver + Visualization wiring (labels in figures & inspector)

**Files:**
- Create: `spatio_flux/library/units_resolver.py`
- Modify: `spatio_flux/visualizations/__init__.py`
- Test: `tests/test_types_units_contracts.py` (extend)

**Interfaces:**
- Produces: `build_units_index() -> dict[str, str]` (dotted store-path → unit, walking the registered composite schemas via the port `_units`), `resolve_unit(index, path) -> str | None`, and a resolver object assigned to `Visualization.units_resolver` (mirrors `v2ecoli/visualizations/__init__.py:30`).

- [ ] **Step 1: Write the failing test**

```python
def test_units_index_and_resolver():
    from spatio_flux.library.units_resolver import build_units_index, resolve_unit
    idx = build_units_index()
    # a biomass path resolves to gDW; a concentration path to mM
    assert any(u == "gDW" for u in idx.values())
    from viva_superpowers.visualization import Visualization
    import spatio_flux.visualizations  # sets Visualization.units_resolver
    assert Visualization.units_resolver is not None
```

- [ ] **Step 2: Run to verify it fails.** (module missing / resolver unset)

- [ ] **Step 3: Implement `units_resolver.py`** — mirror v2ecoli's `units_from_schema` walk over the spatio-flux composite schemas (reading port `_units` via the resolved nodes), memoize the index, expose `resolve_unit`. In `visualizations/__init__.py` add:

```python
from viva_superpowers.visualization import Visualization as _V
from spatio_flux.library.units_resolver import build_units_index, resolve_unit
class _SpatioFluxUnitsResolver:
    def __call__(self, path):
        return resolve_unit(build_units_index(), path)
_V.units_resolver = _SpatioFluxUnitsResolver()
```

- [ ] **Step 4: Run to verify it passes** (serving venv, since it needs the composite schemas).

Run: `.venv-serve/bin/python -m pytest tests/test_types_units_contracts.py -q -k units_index`
Expected: PASS. Spot-check a Phase-B figure now shows unit-labeled axes (e.g. `Concentration (mM)`).

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/library/units_resolver.py spatio_flux/visualizations/__init__.py tests/test_types_units_contracts.py
git commit -m "feat(units): units index + Visualization.units_resolver for axis/inspector labels"
```

---

### Task 5: Loom-view verification

- [ ] **Step 1:** Serve from `.venv-serve` (`vivarium-workbench serve --workspace . --port 8137`), open a composite in the loom/inspector, confirm each process box shows its `description` and each port shows its type + unit.
- [ ] **Step 2:** `GET /api/composite-resolve?id=...` and confirm the resolved state's process nodes carry the attached docs (via `_attach_process_docs`). Record the result in this plan file and commit.

---

## Self-Review

**Spec coverage (design §4.4):** tighten weak ports → Task 2; pint units on ports → Tasks 1, 2; `describe()`/description contracts → Task 3; loom coverage (descriptions + units + types surfaced) → Tasks 4, 5. ✓

**Placeholder scan:** Task 1 is an explicit spike with a concrete acceptance test (not a placeholder — it has a pass condition and a fix location). The unit-registration API call (`_ureg.define`) is flagged for confirmation against `type_functions.py` inside the spike, which is the correct place to pin it. ✓

**Type/name consistency:** `build_units_index`/`resolve_unit`/`Visualization.units_resolver` names match across Task 4 and its test; `quantity[<datatype>,<unit>]` form used consistently. ✓

**Dependency:** Task 4 verification + Task 5 need `.venv-serve` (Phase A Task 5). Independent of Phase B, but Task 3 also covers Phase B's analysis Steps if they exist.
