#!/usr/bin/env python
"""One command to (re)build the paper figures — INCREMENTALLY by default.

Reads the declared build graph (`figures-pipeline.yaml`) + the content-hash
manifest (`figure_pipeline.py`) and rebuilds ONLY the panels whose inputs changed
(a saved loom view, a composite spec, a sim script) and re-stitches ONLY the
figures those panels feed. Save a view -> `build_all_figures.py` re-renders that
one panel and re-stitches that one figure; everything else is skipped.

  python scripts/build_all_figures.py            # incremental: build only stale
  python scripts/build_all_figures.py --dry-run  # show the plan, build nothing
  python scripts/build_all_figures.py --all      # force a full rebuild
  python scripts/build_all_figures.py --only fig07
  python scripts/build_all_figures.py --promote  # capture dashboard saves to git first

Per run: (--promote) -> seed git views into .pbg (deterministic) -> render stale
loom panels -> run stale sim-script panels -> stitch affected studies -> update
the manifest -> append a figure-build event to .pbg/runs.jsonl.

The dashboard serving this workspace must be running (the loom render drives it);
its base URL + loom dir are auto-discovered.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_pipeline as fp  # noqa: E402

WS = fp.WS
PY = sys.executable


def _discover_base() -> str | None:
    reg = Path.home() / ".pbg" / "servers"
    for f in (reg.glob("*.json") if reg.is_dir() else []):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if Path(d.get("path", "")).resolve() == WS and d.get("url"):
            return d["url"]
    return None


def _discover_loom() -> str | None:
    """The serving workbench's loom dir (has node_modules/playwright)."""
    if os.environ.get("VW_LOOM"):
        return os.environ["VW_LOOM"]
    cands = sorted(Path.home().glob("code/vivarium-workbench*/vivarium_workbench/loom"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if (c / "node_modules" / "playwright").exists():
            return str(c)
    return None


def _run(cmd: list[str], *, env: dict | None = None, step: str) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(WS), env=env)
    if r.returncode:
        raise SystemExit(f"[build_all_figures] step failed: {step} (exit {r.returncode})")


def _execute(build: list, seed: bool) -> None:
    """Render/run/stitch one build set (already planned). Reused across re-check
    passes so a view saved mid-build is caught."""
    sync = [PY, str(WS / "scripts" / "sync_loom_views.py")]
    if seed:  # git views -> .pbg so the headless render is deterministic from git
        _run(sync + ["--seed"], step="seed: git views -> .pbg (deterministic render)")

    loom = [n for n in build if n.kind == "loom"]
    if loom:
        base = os.environ.get("_VW_BASE") or _discover_base()
        loomdir = _discover_loom()
        if not base:
            raise SystemExit("[build_all_figures] no dashboard found for this workspace — start one or pass --base")
        if not loomdir:
            raise SystemExit("[build_all_figures] no workbench loom node_modules found — set VW_LOOM")
        jobs = [[n.study, n.composite, n.output_stem, n.config.get("flags", {})] for n in loom]
        jf = WS / ".pbg" / "figures" / "_jobs.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(jobs))
        env = {**os.environ, "VW_BASE": base, "VW_WS": str(WS), "VW_LOOM": loomdir}
        _run(["node", str(WS / "scripts" / "render_loom_svgs.mjs"), "--jobs", str(jf)],
             env=env, step=f"render {len(loom)} loom panel(s)")

    for n in [n for n in build if n.kind == "command"]:
        cmd = n.command.split()
        if cmd and cmd[0] == "python":
            cmd[0] = PY
        env = {**os.environ, "PYTHONPATH": str(WS) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        _run(cmd, env=env, step=f"sim panel: {n.output_stem}")

    for n in [n for n in build if n.kind == "stitch"]:
        _run([PY, str(WS / "scripts" / "build_paper_figures.py"), "--study", n.study],
             step=f"stitch {n.study} -> {n.output_stem}")

    built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest = fp.load_manifest()
    for n in build:
        fp.record(n, manifest, built_at)
    fp.save_manifest(manifest)
    try:
        ev = WS / ".pbg" / "runs.jsonl"
        ev.parent.mkdir(parents=True, exist_ok=True)
        with ev.open("a") as f:
            f.write(json.dumps({"event": "completed", "kind": "figure-build",
                                "nodes": [n.key for n in build], "at": built_at}) + "\n")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="force a full rebuild (ignore the manifest)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; build nothing")
    ap.add_argument("--only", default="", help="restrict to nodes whose key contains this")
    ap.add_argument("--force", action="store_true", help="overwrite outputs edited out-of-band")
    ap.add_argument("--promote", action="store_true", help="capture dashboard .pbg saves into git views first")
    ap.add_argument("--base", default=os.environ.get("VW_BASE", ""), help="dashboard base URL (default: auto)")
    args = ap.parse_args()

    if args.base:
        os.environ["_VW_BASE"] = args.base

    sync = [PY, str(WS / "scripts" / "sync_loom_views.py")]
    if args.promote:
        _run(sync + ["--promote"], step="promote: dashboard .pbg saves -> git views")

    def _plan_set(manifest):
        nodes = fp.load_graph()
        if args.all:
            build = [n for n in nodes if n.kind != "external" and (not args.only or args.only in n.key)]
            reasons = {n.key: "forced (--all)" for n in build}
        else:
            p = fp.plan(nodes, manifest, only=args.only)
            build = [n for n, _ in p["stale"]]
            reasons = {n.key: why for n, why in p["stale"]}
        return nodes, build, reasons

    nodes, build, reasons = _plan_set(fp.load_manifest())
    print("\n=== plan ===")
    for n in build:
        print(f"  BUILD  {n.key:40s} {reasons[n.key]}")
    print(f"  ({len(nodes) - len(build)} up-to-date)")
    if args.dry_run:
        return
    if not build:
        print("\n[build_all_figures] everything up-to-date.")
        return

    # Single-flight: a background auto-build (kicked on Save-as-default) must not
    # stampede with a concurrent one; the running build's re-check loop below
    # picks up any view saved while it runs, so skipping here loses nothing.
    lock = WS / ".pbg" / "figures" / ".build.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and (time.time() - lock.stat().st_mtime) < 1800:
        print("[build_all_figures] another build is running — skipping "
              "(it re-checks for saves made while it runs).")
        return
    lock.write_text(str(time.time()))
    try:
        # Hand-edit guard: never silently clobber an output edited out-of-band.
        if not args.force:
            edited = [n for n in build if fp.hand_edited(n, fp.load_manifest())]
            if edited:
                for n in edited:
                    print(f"  ! {n.primary_output().relative_to(WS)} was edited out-of-band since the pipeline wrote it")
                raise SystemExit("[build_all_figures] refusing to overwrite hand-edited outputs — re-run with --force")

        total = 0
        # Re-check loop: rebuild, then re-plan — a view saved WHILE the build ran
        # (a rapid second Save) becomes stale and is caught on the next pass, so
        # the figures always converge to the latest saves with no manual step.
        for _pass in range(4):
            _execute(build, seed=(_pass == 0))
            total += len(build)
            if args.all:
                break
            _, build, _ = _plan_set(fp.load_manifest())
            if not build:
                break
            print(f"\n  (re-check: {len(build)} node(s) became stale during the build — another pass)")
        print(f"\n[build_all_figures] rebuilt {total} node(s).")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
