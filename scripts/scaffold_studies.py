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
