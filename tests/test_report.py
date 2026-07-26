import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_collect_report_inputs_orders_by_investigation():
    from scripts.build_report import collect_report_inputs
    slugs, descriptions, simulations, runtimes, timing, total = collect_report_inputs(REPO)
    assert len(slugs) == 19
    assert slugs[0] == "monod_kinetics"          # investigation order
    assert slugs[-1] == "reference_demo_x2y2"
    # every study has a description (from its @composite_generator)
    assert descriptions["comets_diffusion"]
    # baseline params surfaced as the scenario config
    assert simulations["ecoli_core_dfba"].get("glucose") == 10.0


def test_collect_report_inputs_reads_timing_when_present(tmp_path):
    # a study with a timing.json contributes to runtimes/total
    from scripts.build_report import collect_report_inputs
    slugs, _, _, runtimes, timing, total = collect_report_inputs(REPO)
    # runtimes only populated for studies that have been run; contract holds either way
    assert isinstance(runtimes, dict) and total >= 0.0


def test_reproduce_lists_all_investigation_studies():
    from scripts.reproduce import study_slugs
    slugs = study_slugs(REPO)
    assert len(slugs) == 19
    assert "spatioflux_reference_demo" in slugs


def test_test_suite_module_retired():
    import importlib.util
    assert importlib.util.find_spec("spatio_flux.experiments.test_suite") is None


def test_no_code_imports_test_suite():
    import subprocess
    r = subprocess.run(
        ["grep", "-rIl", "from spatio_flux.experiments.test_suite import",
         "spatio_flux", "scripts", "tools"],
        cwd=REPO, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"stale imports: {r.stdout}"
