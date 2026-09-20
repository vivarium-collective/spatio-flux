#!/usr/bin/env python
"""Reproducible gate for the spatio-flux-test-suite.

Runs each study's canonical composite, then evaluates its ``artifacts_present``
gate **by code** (via :func:`spatio_flux.evaluators.evaluate_artifacts_present`)
and persists the computed outcome into ``studies/<slug>/study.yaml``. This is
what replaces the previously hard-coded ``result: PASS`` the scaffolder wrote:
the pass is now earned by a run + a structural match against a committed
reference, and is reproducible here and in CI.

Usage (from the repo root)::

    python scripts/verify_reference.py                 # run + evaluate + write all 19
    python scripts/verify_reference.py --no-write      # run + evaluate + print only
    python scripts/verify_reference.py --baseline      # (re)snapshot the schema references
    python scripts/verify_reference.py community_dfba   # restrict to given slugs

Exit code is non-zero if any study fails, so it works as a CI gate.
"""
import argparse
import datetime
import os
import shutil
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from scaffold_studies import STUDIES, _runtime_for           # noqa: E402
from spatio_flux.analysis.flush_spec import expected_files    # noqa: E402
from spatio_flux.evaluators import evaluate_artifacts_present  # noqa: E402
from spatio_flux.runners import run_study                      # noqa: E402

TOLERANCE = 0.0  # schema is deterministic; exact structural match expected


def _now_iso():
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _study_dir(slug):
    return os.path.join(REPO, "studies", slug)


def _baseline(slug):
    """Snapshot the (deterministic) composite schema as the regression reference."""
    sdir = _study_dir(slug)
    ref_dir = os.path.join(sdir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    schema = f"{slug}_schema.json"
    shutil.copyfile(os.path.join(sdir, "charts", schema), os.path.join(ref_dir, schema))
    return schema


def _evaluate(slug):
    sdir = _study_dir(slug)
    return evaluate_artifacts_present(
        expected_files(slug),
        os.path.join(sdir, "charts"),
        os.path.join(sdir, "reference"),
        tolerance=TOLERANCE,
    )


def _honest_description(slug):
    n = len(expected_files(slug))
    return (f"All {n} expected artifacts are produced in charts/, and the composite's "
            f"structural schema ({slug}_schema.json) matches the committed reference "
            "in studies/<slug>/reference/. State/figures embed RNG or non-deterministic "
            "renders, so they are existence-checked.")


def _write_outcome(slug, result, detail):
    """Persist a real, code-evaluated run + outcome into the study.yaml.

    Surgical: touches only gate/outcome/finding fields + the gate's own honesty
    text. Never rewrites `visualizations` or `purpose` (kept as committed so the
    published read-only workbench keeps its curated images).
    """
    path = os.path.join(_study_dir(slug), "study.yaml")
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    test_name = f"{slug.upper()}-REPRODUCES-REPORT"
    run_id = f"{slug}-reproduce"
    store_path = f"studies/{slug}/runs.{run_id}.zarr"
    composite = spec["baseline"][0]["composite"]
    passed = result == "PASS"

    spec["simulation_status"] = "ran"
    spec["evaluation_status"] = "evaluated"
    spec["gate_status"] = "passed" if passed else "failed"

    # Honest gate text (replaces the scaffolder's "match the out0 reference").
    for test in spec.get("behavior_tests", []):
        test["description"] = _honest_description(slug)
        pass_if = test.setdefault("pass_if", {})
        pass_if["tolerance"] = TOLERANCE

    spec["runs"] = [{
        "name": run_id,
        "run_id": run_id,
        "kind": "simulation",
        "status": "completed",
        "canonical": True,
        "composite": composite,
        "emitter": {"kind": "xarray", "store": store_path},
        "store_path": store_path,
        "timestamp": _now_iso(),
        "result": result,
        "outcomes": {
            test_name: {"result": result, "evaluated_by": "code", "detail": detail},
        },
    }]
    for finding in spec.get("findings", []):
        finding["status"] = "passed" if passed else "failed"
        finding["statement"] = (
            f"spatio_flux reproduces the {slug} test-suite artifacts and the "
            "composite's structural schema matches its committed reference.")

    diag = spec.get("conclusion_logic", {}).get("if_primary_tests_fail", {})
    if isinstance(diag.get("diagnose"), list) and diag["diagnose"]:
        diag["diagnose"][0] = (
            "Compare charts/<slug>_schema.json against studies/<slug>/reference/.")

    with open(path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", action="store_true",
                    help="(re)snapshot studies/<slug>/reference/<slug>_schema.json")
    ap.add_argument("--no-write", action="store_true",
                    help="evaluate + print only; do not modify study.yaml")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip the run; evaluate existing charts/ against the reference")
    ap.add_argument("slugs", nargs="*", help="restrict to these study slugs")
    args = ap.parse_args(argv)

    os.chdir(REPO)  # run_study writes to relative studies/<slug>/ paths
    slugs = args.slugs or [e["slug"] for e in STUDIES]
    fails = []
    for slug in slugs:
        if not args.eval_only:
            run_study.main(slug, _runtime_for(slug))
        if args.baseline:
            print(f"baselined {slug}: reference/{_baseline(slug)}")
            continue
        result, detail = _evaluate(slug)
        print(f"{'PASS' if result == 'PASS' else 'FAIL'}  {slug}: {detail}")
        if result != "PASS":
            fails.append(slug)
        if not args.no_write:
            _write_outcome(slug, result, detail)

    if args.baseline:
        print(f"\nbaselined {len(slugs)} schema references.")
        return 0
    if fails:
        print(f"\n{len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print(f"\nAll {len(slugs)} studies PASS (code-evaluated structural gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
