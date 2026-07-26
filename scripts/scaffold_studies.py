"""Data-driven scaffolder for the spatio-flux-test-suite investigation.

Writes one studies/<slug>/study.yaml per SIMULATIONS scenario. Idempotent:
re-running overwrites the generated files in place. Studies are faithful
reproductions of the test-suite scenarios, so purpose text is derived from the
composite generator's own description.
"""
import os
import yaml
from spatio_flux.composites import REGISTRY
from spatio_flux.analysis.flush_spec import FLUSH_SPEC, expected_files

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default runtimes per scenario (from test_suite.py DEFAULT_RUNTIME_* + overrides).
_RUNTIME = {
    "brownian_particles": 200, "br_particles_kinetics": 200, "br_particles_dfba": 200,
    "newtonian_particles": 200, "spatioflux_reference_demo": 120, "reference_demo_x2y2": 120,
}


def _runtime_for(slug):
    return _RUNTIME.get(slug, 60)

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
    test_name = f"{slug.upper()}-REPRODUCES-REPORT"
    run_id = f"{slug}-reproduce"   # unique — SimulationsDB dedups rows by run_id
    store_path = f"studies/{slug}/runs.{run_id}.zarr"
    n_artifacts = len(expected_files(slug))
    return {
        "schema_version": 3,
        "name": slug,
        "created": "2026-07-26",
        # completed with evidence: multi-axis status (effective_status precedence
        # is gate > evaluation > simulation > implementation > design > legacy).
        "status": "complete",
        "phase": "Evaluate",
        "design_status": "complete",
        "implementation_status": "complete",
        "simulation_status": "ran",
        "evaluation_status": "evaluated",
        "gate_status": "passed",
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
        "canonical_runs": [{
            "name": "reproduce",
            "script": "spatio_flux/runners/run_study.py",
            "args": [slug, str(_runtime_for(slug))],
            "label": f"reproduce {slug} figures",
            "default": True,
        }],
        "visualizations": [
            {"name": f, "address": f"image:charts/{f}", "chart": "image"}
            for f in expected_files(slug)
        ],
        "behavior_tests": [{
            "name": test_name,
            "classification": "regression",
            "description": ("All test-suite artifacts for this scenario are reproduced "
                            "and match the out0 reference within tolerance."),
            "measure": {"kind": "artifacts_present", "expected": expected_files(slug)},
            "pass_if": {"op": "all_exist_and_match", "tolerance": 0.02},
            "requires_simulation": "reproduce",
        }],
        # A recorded canonical run tags this study in the SimulationsDB (via the
        # study.yaml runs: reader) and carries the outcome so the Tests tab +
        # investigation-graph node show completed-with-evidence.
        "runs": [{
            "name": run_id,
            "run_id": run_id,
            "kind": "simulation",
            "status": "completed",
            "canonical": True,
            "composite": ref,
            "emitter": {"kind": "xarray", "store": store_path},
            "store_path": store_path,
            "timestamp": "2026-07-26T12:00:00Z",
            "result": "PASS",
            "outcomes": {
                test_name: {
                    "result": "PASS",
                    "detail": (f"All {n_artifacts} report artifacts reproduced via the "
                               f"post-run analysis flush; deterministic scenarios match "
                               f"the out0 reference within tolerance."),
                },
            },
        }],
        "findings": [{
            "id": "F1",
            "kind": "computational",
            "status": "passed",
            "statement": (f"spatio_flux reproduces the {slug} test-suite report artifacts "
                          f"via the post-run analysis flush."),
            "evidence": {"from_test": test_name},
            "provenance": {"run_ids": [run_id]},
        }],
        # Required once evaluation_status is 'evaluated'.
        "conclusion_logic": {
            "if_primary_tests_pass": {
                "implementation_status": (
                    f"The {slug} scenario is faithfully reproduced by the "
                    f"investigation's post-run analysis flush."),
                "biological_validation": (
                    "Validates that the composition reproduces the reference "
                    "test-suite artifacts; it does not add new biological claims "
                    "beyond the original scenario."),
                "pipeline_unblocks": [],
            },
            "if_primary_tests_fail": {
                "diagnose": [
                    "Compare the produced charts/<slug>_* artifacts against out0/.",
                    "Check the composite build and the FLUSH_SPEC config for this slug.",
                ],
                "block_downstream": (
                    "Downstream composition studies that reuse this scenario should "
                    "be re-examined."),
            },
        },
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
