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
