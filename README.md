# Spatio-Flux: A Reference Application for Compositional Multiscale Simulation

**Spatio-Flux** is a reference application for **compositional multiscale biological modeling** built using the **Process-Bigraph** framework. It demonstrates how independently developed models—metabolism, spatial transport, particle dynamics, and population restructuring—can be composed into a single executable simulation through **typed interfaces and shared orchestration**, rather than tightly coupled solvers.

Spatio-Flux serves as the primary worked example in  
*Process Bigraphs and the Architecture of Compositional Systems Biology* by Agmon & Spangler ([arXiv:2512.23754](https://arxiv.org/abs/2512.23754)).

---

## What Spatio-Flux is

Spatio-Flux is designed as a **testbed and reference implementation**, not as an optimized domain-specific simulator. Its purpose is to make **model composition explicit and inspectable**.

In particular, Spatio-Flux demonstrates how to:
- compose heterogeneous modeling paradigms (ODEs, dFBA, spatial fields, particles),
- couple mechanisms through shared, typed state rather than direct process calls,
- coordinate multi-timescale execution using reusable orchestration patterns,
- swap or recombine processes without modifying surrounding models.

---

## Automated test suite

The heart of Spatio-Flux is its **automated test suite**, which exercises a wide range of
composition patterns using reusable process families.

Covered examples include:
- Monod and dynamic FBA metabolism
- Hybrid microbial communities
- COMETS-style spatial dFBA
- Brownian and Newtonian particle systems
- Particle–field exchange adapters
- Event-driven division and boundary handling

Each test produces:
- a **process-bigraph visualization** of model structure,
- serialized schemas and state for inspection,
- and domain-specific plots or animations.

**Live test report:**  
👉 https://vivarium-collective.github.io/spatio-flux/report/index.html

---

## Reference composite

The figure below shows a representative “mega-composite” integrating metabolic,
spatial, mechanical, and structural processes into a single process bigraph.

![Spatio-Flux reference composite](doc/spatioflux_reference_demo_viz.png)

---

## Example simulation

Motile particles carrying internal metabolic models interact with evolving spatial nutrient fields.

![Spatio-Flux reference demo](doc/spatioflux_reference_demo_video.gif)

---

## Ecosystem

Spatio-Flux is part of **Vivarium 2.0**, an open-source ecosystem for compositional modeling:

- `bigraph-schema` — typed hierarchical schemas
- `process-bigraph` — process and composite simulation interfaces
- `bigraph-viz` — visualization of bigraph structure and data flow
- `spatio-flux` — reference multiscale application and test suite

All components are open source:  
https://github.com/vivarium-collective