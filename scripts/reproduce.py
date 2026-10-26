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


def _run_study(slug):
    """Run every canonical_runs entry for a study via its declared script+args.

    Old test-suite studies declare run_study.py (FLUSH_SPEC flush); the newer
    discovery-engine studies declare run_new_study.py and may have several
    entries (one per baseline). Falls back to run_study.py when no
    canonical_runs are declared."""
    spec = yaml.safe_load(open(os.path.join("studies", slug, "study.yaml")))
    crs = spec.get("canonical_runs") or []
    cmds = [[sys.executable, cr.get("script", "spatio_flux/runners/run_study.py"),
             *[str(a) for a in (cr.get("args") or [slug])]] for cr in crs]
    if not cmds:
        cmds = [[sys.executable, "spatio_flux/runners/run_study.py", slug]]
    rc = 0
    for cmd in cmds:
        r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": REPO})
        rc = rc or r.returncode
    return rc


def main(only=None, out="report/index.html"):
    slugs = [only] if only else study_slugs()
    failed = []
    for slug in slugs:
        print(f"\n=== {slug} ===")
        if _run_study(slug) != 0:
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
