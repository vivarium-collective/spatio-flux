#!/usr/bin/env python
"""Reproduce the whole spatio-flux test suite from the investigation.

Runs every study's ``reproduce`` runner (the post-sim analysis flush), then builds
the report. This REPLACES the retired ``spatio_flux/experiments/test_suite.py`` —
the investigation + its studies are now the single source of truth.

Usage (from workspace root):
    python scripts/reproduce.py                 # all 19 studies + report
    python scripts/reproduce.py --only monod_kinetics
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the repo root is importable so `from scripts.build_report import ...`
# works when invoked as `python scripts/reproduce.py` (which puts scripts/ on
# sys.path, not the repo root).
if REPO not in sys.path:
    sys.path.insert(0, REPO)
INVESTIGATION = os.path.join("investigations", "spatio-flux-test-suite", "investigation.yaml")


def study_slugs(ws_root="."):
    with open(os.path.join(ws_root, INVESTIGATION)) as f:
        return yaml.safe_load(f)["studies"]


def main(only=None, out="report/index.html"):
    slugs = [only] if only else study_slugs()
    failed = []
    for slug in slugs:
        print(f"\n=== {slug} ===")
        r = subprocess.run([sys.executable, "spatio_flux/runners/run_study.py", slug],
                           env={**os.environ, "PYTHONPATH": REPO})
        if r.returncode != 0:
            failed.append(slug)
    from scripts.build_report import main as build_report
    build_report(out)
    if failed:
        print(f"\n⚠️  {len(failed)} studies failed: {failed}")
        return 1
    print(f"\n✅ reproduced {len(slugs)} studies → {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="run a single study slug")
    p.add_argument("--out", default="report/index.html")
    a = p.parse_args()
    sys.exit(main(only=a.only, out=a.out))
