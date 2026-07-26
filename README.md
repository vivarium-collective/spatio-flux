# Spatio-Flux

[![▶ live test report](https://img.shields.io/badge/▶%20live%20report-test%20suite-1b9e77?style=for-the-badge)](https://vivarium-collective.github.io/spatio-flux/report/index.html)
[![▶ read-only workbench](https://img.shields.io/badge/▶%20read--only-workbench-3a5b8f?style=for-the-badge)](https://vivarium-collective.github.io/spatio-flux/workbench/index.html)
[![paper](https://img.shields.io/badge/paper-arXiv%3A2512.23754-3a5b3a?style=flat-square)](https://arxiv.org/abs/2512.23754)
[![ecosystem](https://img.shields.io/badge/part%20of-vivarium--collective-1c4a78?style=flat-square)](https://github.com/vivarium-collective)

A reference application for **compositional multiscale biological modeling** built on the
[Process-Bigraph](https://github.com/vivarium-collective/process-bigraph) framework.
Spatio-Flux composes independently developed processes — metabolism, spatial transport,
particle dynamics, structural rewrites — into a single executable simulation via typed
interfaces and shared orchestration, not tightly coupled solvers.

It's the worked example in *Process Bigraphs and the Architecture of Compositional
Systems Biology* (Agmon & Spangler, [arXiv:2512.23754](https://arxiv.org/abs/2512.23754)).

> ▶ **[Browse the live test suite report →](https://vivarium-collective.github.io/spatio-flux/report/index.html)**
> 19 composite scenarios, each with structure diagrams, time series, and plots.
>
> ▶ **[Open the read-only workbench →](https://vivarium-collective.github.io/spatio-flux/workbench/index.html)**
> The same 19 scenarios as a browsable **investigation** of studies — the
> composition DAG (standalone processes → pairs → triples → reference demos),
> each study's runs, reproduced visualizations, and the process/loom explorer —
> served with no backend.

![Spatio-Flux reference composite](doc/spatioflux_reference_demo_viz.png)

![Spatio-Flux reference demo](doc/spatioflux_reference_demo_video.gif)

---

## What this repo is

Spatio-Flux is a **testbed and reference implementation**, not an optimized domain
simulator. Its purpose is to make model composition explicit and inspectable.

It demonstrates how to:
- compose heterogeneous modeling paradigms (ODEs, dFBA, spatial fields, particles)
- couple mechanisms through shared typed state, not direct process calls
- coordinate multi-timescale execution with reusable orchestration patterns
- swap or recombine processes without modifying surrounding models

---

## The test suite

The heart of the repo is `spatio_flux/experiments/test_suite.py`, which exercises 19
composition patterns and renders the report linked above.

Covered scenarios include:
- Monod and dynamic FBA metabolism (single-strain + multi-strain communities)
- COMETS-style spatial dFBA on a lattice
- Brownian and Newtonian (Pymunk) particle systems
- Particle–field exchange with embedded metabolism
- Event-driven division and boundary handling

Each scenario produces a process-bigraph diagram, serialized schemas and state, and
domain-specific plots or animations.

Each composite is a `@composite_generator`-decorated function under
`spatio_flux/composites/`, discoverable by the
[pbg-superpowers](https://github.com/vivarium-collective/pbg-superpowers) dashboard.

### Reusing composites at different scales

Generators expose their lattice and transport parameters as keyword arguments —
defaults match what the test-suite report uses, but callers (or the dashboard's
parameter form) can override them to reuse the same composite at a different
resolution or with different transport rates. Commonly available knobs:

- `n_bins`, `bounds` — lattice resolution and physical extent (default `SQUARE_BINS` / `SQUARE_BOUNDS`, or `DEFAULT_BINS` / `DEFAULT_BOUNDS` for diffusion-only generators)
- `diffusion_rate`, `advection_rate` — Brownian-particle or field transport coefficients (default `DEFAULT_DIFFUSION`, `DEFAULT_ADVECTION`; some generators use a scaled default such as `DEFAULT_DIFFUSION / 2`)
- `n_particles`, `add_rate`, `division_mass_threshold` — particle population controls
- `initial_min_max` — per-molecule (min, max) ranges used to seed field state
- `field_advection_rate` — field-level advection where it must be disambiguated from particle advection (e.g. `comets_br_particles_dfba`)

Each generator declares only the subset of parameters it actually uses; see the
`parameters={…}` block on each `@composite_generator` for the authoritative
list and types. Build a doc with overrides via:

```python
from pbg_superpowers.composite_generator import build_generator
from spatio_flux.composites import REGISTRY

entry = next(e for e in REGISTRY.values() if e.name == "diffusion_process")
doc = build_generator(entry, overrides={"n_bins": [20, 40], "diffusion_rate": 0.1})
```

---

## Run it locally

```bash
git clone https://github.com/vivarium-collective/spatio-flux.git
cd spatio-flux
uv sync
# Reproduce the whole test suite from the investigation (runs all 19 studies,
# then builds the report) and open it:
uv run python scripts/reproduce.py
open report/index.html
```

`--only <slug>` reproduces a single study. Each scenario is a study under
`studies/<slug>/`; the investigation graph lives in
`investigations/spatio-flux-test-suite/investigation.yaml`. To browse it
interactively, serve the workbench dashboard:

```bash
vivarium-workbench serve --workspace .
```

> The legacy `spatio_flux/experiments/test_suite.py` runner has been superseded by
> the investigation + `scripts/reproduce.py`.

---

## Ecosystem

Spatio-Flux is part of **Vivarium 2.0** — an open-source ecosystem for compositional modeling:

- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) — typed hierarchical schemas
- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — process and composite simulation interfaces
- [bigraph-viz](https://github.com/vivarium-collective/bigraph-viz) — visualization of bigraph structure and data flow
- [pbg-superpowers](https://github.com/vivarium-collective/pbg-superpowers) — convention + dashboard for discoverable composites
- [spatio-flux](https://github.com/vivarium-collective/spatio-flux) — reference multiscale application (this repo)

## Citation

```bibtex
@article{agmon2025spatioflux,
  title  = {Process Bigraphs and the Architecture of Compositional Systems Biology},
  author = {Agmon, Eran and Spangler, Daniel},
  journal = {arXiv preprint arXiv:2512.23754},
  year   = {2025},
}
```
