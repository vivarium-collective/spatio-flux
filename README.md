# Spatio-Flux: An Example of Compositional Modeling with Process-Bigraphs

**Spatio-Flux** is a modular simulation library for building **spatially explicit, multiscale biological models** using the [Process-Bigraphs](https://github.com/vivarium-collective/process-bigraph) protocol.  
It demonstrates how metabolic, physical, and agent-based processes can be composed into reproducible, scalable simulations of microbial and tissue systems.

---

## Purpose and Context

Spatio-Flux was created to illustrate **compositional modeling principles** for systems biology — how individual processes (e.g., dynamic FBA, diffusion, particle motion) can be connected through well-defined interfaces.  
It serves as both an **educational reference** and a **testbed** for developing interoperable components within the Vivarium ecosystem.

This repository shows how to combine:
- **Dynamic FBA (dFBA)** models based on *COBRApy* for metabolic dynamics.
- **Diffusion–Advection** solvers for spatial metabolite transport.
- **Particle processes** for cell-level motion, growth, and division.
- **Composition and orchestration** via Process-Bigraphs for integrating all processes consistently.

Together, these components demonstrate how **multiscale simulations** can link molecular metabolism to spatial ecology.

---

## What It Demonstrates

| Capability | Description |
|-------------|-------------|
| **Dynamic FBA** | Time-resolved flux balance analysis for multiple species, using uptake kinetics and dynamic exchange. |
| **Spatial Coupling** | 2D diffusion–advection PDEs coupling local metabolism to global field transport. |
| **Agent Dynamics** | Particle movement, division, and field interaction at single-cell resolution. |
| **Process Composition** | Declarative model composition through Process-Bigraphs schema and `Composite` execution engine. |
| **Automation & Visualization** | CI-generated simulations producing movies and time-series reports. |

---

## Running the Demo

You can run the test suite locally to reproduce the GitHub Pages report.

```bash
# install
git clone https://github.com/vivarium-collective/spatio-flux.git
cd spatio-flux
pip install -e .

# run tests and generate a report
python spatio_flux/experiments/test_suite.py --output out

# open the results
open out/report.html
```

Or view the continuously updated online version:  
👉 [**View Simulation Tests Report**](https://vivarium-collective.github.io/spatio-flux/report/index.html)

---

## Citation

Spatio-Flux builds upon the  **Process-Bigraphs** frameworks developed by the Vivarium Collective.  
If you use or adapt this repository, please cite:

> Agmon E., et al. (unpublished). *Process Bigraphs and the Architecture of Compositional Systems Biology.*

---

**Developed and maintained by** the [Vivarium Collective](https://vivarium-collective.github.io)  
**Contact:** [Eran Agmon](https://github.com/eranagmon)
[//]: # (* [Build COMETS with vivarium-interface]&#40;https://vivarium-collective.github.io/spatio-flux/demo/build_comets.html&#41;)
