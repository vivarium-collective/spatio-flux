import os
import glob
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- Task 1: workspace manifest + build_core -------------------------------

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


# ---- Task 2: scaffolder table + investigation file -------------------------

def test_scaffolder_table_complete():
    from scripts.scaffold_studies import STUDIES
    slugs = {s["slug"] for s in STUDIES}
    assert len(STUDIES) == 19
    assert "spatioflux_reference_demo" in slugs
    for s in STUDIES:
        for p in s["prerequisites"]:
            assert p in slugs, f"{s['slug']} -> unknown prereq {p}"


def test_investigation_lists_all_19():
    with open(os.path.join(REPO, "investigations", "spatio-flux-test-suite",
                           "investigation.yaml")) as f:
        inv = yaml.safe_load(f)
    from scripts.scaffold_studies import STUDIES
    assert inv["name"] == "spatio-flux-test-suite"
    assert set(inv["studies"]) == {s["slug"] for s in STUDIES}


# ---- Task 3: generated studies resolve + form a valid DAG -------------------

def _load_studies():
    # Scope to the spatio-flux-test-suite investigation only. Other investigations
    # (e.g. paper-figures) also live under studies/ with different conventions
    # (static-spec baselines, no pipeline_gate) — they aren't what these tests check.
    from scripts.scaffold_studies import STUDIES
    slugs = {s["slug"] for s in STUDIES}
    out = {}
    for p in glob.glob(os.path.join(REPO, "studies", "*", "study.yaml")):
        name = os.path.basename(os.path.dirname(p))
        if name not in slugs:
            continue
        with open(p) as f:
            out[name] = yaml.safe_load(f)
    return out


def test_all_19_studies_on_disk():
    from scripts.scaffold_studies import STUDIES
    studies = _load_studies()
    assert set(studies) == {s["slug"] for s in STUDIES}


def test_every_baseline_composite_resolves():
    # Mirrors the workbench resolver / known_composite_ids: the composite
    # registry is keyed by the dotted id `<module>.<name>`. A ref absent from
    # the registry is the dashboard's "composite not found" banner.
    import spatio_flux.composites  # noqa: F401  (fire @composite_generator)
    from pbg_superpowers.composite_generator import _REGISTRY
    known = set(_REGISTRY.keys())
    for slug, spec in _load_studies().items():
        ref = spec["baseline"][0]["composite"]
        assert ref in known, f"{slug}: composite not found: {ref}"


def test_prerequisites_reference_real_slugs_and_are_acyclic():
    studies = _load_studies()
    edges = {slug: spec["pipeline_gate"]["prerequisites"] for slug, spec in studies.items()}
    for slug, prereqs in edges.items():
        for p in prereqs:
            assert p in studies, f"{slug}: unknown prerequisite {p}"
    seen, stack = set(), set()

    def visit(n):
        if n in seen:
            return
        assert n not in stack, f"cycle at {n}"
        stack.add(n)
        for p in edges[n]:
            visit(p)
        stack.discard(n)
        seen.add(n)

    for slug in edges:
        visit(slug)


def test_tier0_have_no_prereqs_and_demos_chain():
    studies = _load_studies()
    for s in ("monod_kinetics", "ecoli_core_dfba", "diffusion_process",
              "brownian_particles", "newtonian_particles"):
        assert studies[s]["pipeline_gate"]["prerequisites"] == []
    assert studies["reference_demo_x2y2"]["pipeline_gate"]["prerequisites"] == ["spatioflux_reference_demo"]
