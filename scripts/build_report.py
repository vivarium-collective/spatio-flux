#!/usr/bin/env python
"""Build the spatio-flux test-suite report from the investigation's study artifacts.

Reproduces the current ``report.html`` look by REUSING the existing renderer
(``spatio_flux.library.tools.generate_html_report``): it merges every study's
``charts/`` into one directory and drives the renderer with study-sourced
descriptions and per-study timing — instead of scanning a monolithic ``out/``.

Usage (from workspace root):
    python scripts/build_report.py --out report/index.html
"""
import argparse
import json
import os
import shutil
import tempfile

import yaml

from spatio_flux.composites import REGISTRY
from spatio_flux.library.tools import generate_html_report

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVESTIGATION = os.path.join("investigations", "spatio-flux-test-suite", "investigation.yaml")


def _description_for(ref):
    name = ref.rsplit(".", 1)[1]
    for e in REGISTRY.values():
        if getattr(e, "name", None) == name:
            return getattr(e, "description", "") or ""
    return ""


def collect_report_inputs(ws_root):
    """Return (ordered slugs, descriptions, simulations, runtimes, timing_details, total)."""
    with open(os.path.join(ws_root, INVESTIGATION)) as f:
        slugs = yaml.safe_load(f)["studies"]
    descriptions, simulations, runtimes, timing = {}, {}, {}, {}
    total = 0.0
    for slug in slugs:
        study_yaml = os.path.join(ws_root, "studies", slug, "study.yaml")
        if not os.path.exists(study_yaml):
            continue
        spec = yaml.safe_load(open(study_yaml))
        base = spec["baseline"][0]
        descriptions[slug] = _description_for(base["composite"])
        simulations[slug] = base.get("params", {})
        tpath = os.path.join(ws_root, "studies", slug, f"{slug}_timing.json")
        if os.path.exists(tpath):
            t = json.load(open(tpath))
            runtimes[slug] = t["elapsed"]
            timing[slug] = (t["process_time"], t["framework_time"])
            total += t["elapsed"]
    return slugs, descriptions, simulations, runtimes, timing, total


def _merge_charts(ws_root, slugs, dest):
    """Copy every study's charts/ into one dir (the renderer scans one dir)."""
    for slug in slugs:
        charts = os.path.join(ws_root, "studies", slug, "charts")
        if not os.path.isdir(charts):
            continue
        for name in os.listdir(charts):
            src = os.path.join(charts, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, name))


def main(out_path="report/index.html", ws_root="."):
    slugs, descriptions, simulations, runtimes, timing, total = collect_report_inputs(ws_root)
    work = tempfile.mkdtemp(prefix="sf-report-")
    try:
        _merge_charts(ws_root, slugs, work)
        # ordered subset that actually has artifacts, in investigation order
        ordered = {s: simulations.get(s, {}) for s in slugs}
        generate_html_report(work, ordered, {s: descriptions.get(s, "") for s in slugs},
                             runtimes, total, timing_details=timing)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.copy2(os.path.join(work, "report.html"), out_path)
        # copy the artifacts alongside the report so <img> src resolve
        for name in os.listdir(work):
            if name != "report.html":
                shutil.copy2(os.path.join(work, name),
                             os.path.join(os.path.dirname(os.path.abspath(out_path)), name))
        print(f"wrote {out_path} ({len(slugs)} scenarios)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/index.html")
    main(p.parse_args().out)
