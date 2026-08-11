#!/usr/bin/env python
"""One command to remake every paper figure from the git-tracked default views.

This is the streamlined "get the figures out" entrypoint. It is DETERMINISTIC:
the figures are rebuilt from `investigations/paper-figures/loom-views/` (git), not
from whatever machine-local `.pbg` state happens to be lying around — so a fresh
clone, CI, and your laptop all produce the same figures.

Pipeline
--------
  1. (--promote) capture the latest dashboard "Save as default" arrangements from
     .pbg into the git views, so you never lose what you just arranged.
  2. seed the git views back into .pbg (loom-views + per-mode loom-layouts, under
     every id alias) so the headless render resolves to the committed view.
  3. render every loom panel via bigraph-loom (headless) into the study viz dirs.
  4. stitch each figure's panels into figure_<N>.{svg,png} (build_paper_figures).

Sim panels (the matplotlib plots + COMETS/Brownian snapshots) come from actual
runs and are NOT re-run here — they change only when the simulation changes. Pass
--sims to also rebuild those.

Usage
-----
  python scripts/build_all_figures.py                 # remake all, from git views
  python scripts/build_all_figures.py --promote       # capture dashboard saves first
  python scripts/build_all_figures.py --only fig07     # just the fig07 loom panels
  python scripts/build_all_figures.py --base http://127.0.0.1:8091

The dashboard serving THIS workspace must be running (the render drives it). By
default the base URL is auto-discovered from ~/.pbg/servers/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PY = sys.executable


def _discover_base() -> str | None:
    """Find a running dashboard serving THIS workspace (from ~/.pbg/servers)."""
    reg = Path.home() / ".pbg" / "servers"
    if not reg.is_dir():
        return None
    for f in reg.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if Path(d.get("path", "")).resolve() == WS and d.get("url"):
            return d["url"]
    return None


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None, step: str) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd or WS), env=env)
    if r.returncode != 0:
        raise SystemExit(f"[build_all_figures] step failed: {step} (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promote", action="store_true",
                    help="capture the latest dashboard .pbg saves into git views first")
    ap.add_argument("--only", default="", help="render only loom panels whose id/name contains this")
    ap.add_argument("--sims", action="store_true", help="also rebuild the simulation panels (slow)")
    ap.add_argument("--base", default=os.environ.get("VW_BASE", ""),
                    help="dashboard base URL (default: auto-discover)")
    ap.add_argument("--no-render", action="store_true", help="sync + stitch only; skip the loom render")
    args = ap.parse_args()

    sync = [PY, str(WS / "scripts" / "sync_loom_views.py")]

    # 1. Optionally capture dashboard saves into git (source of truth).
    if args.promote:
        _run(sync + ["--promote"], step="promote: dashboard .pbg saves -> git views")

    # 2. Seed git views into .pbg so the render is deterministic from git.
    _run(sync + ["--seed"], step="seed: git views -> .pbg (deterministic render)")

    # 3. Render every loom panel against the running dashboard.
    if not args.no_render:
        base = args.base or _discover_base()
        if not base:
            raise SystemExit(
                "[build_all_figures] no dashboard found for this workspace.\n"
                "  Start one (viva-workbench) or pass --base http://127.0.0.1:<port>.")
        env = {**os.environ, "VW_BASE": base, "VW_WS": str(WS)}
        render = ["node", str(WS / "scripts" / "render_loom_svgs.mjs")]
        if args.only:
            render.append(args.only)
        _run(render, env=env, step=f"render loom panels (base={base}, only={args.only or 'ALL'})")

    # 3b. Optionally rebuild the simulation panels.
    if args.sims:
        for s in ("build_fig07f_snapshots.py", "build_fig07h_traces.py"):
            p = WS / "scripts" / s
            if p.exists():
                _run([PY, str(p)], step=f"rebuild sim panel: {s}")

    # 4. Stitch each figure's panels into figure_<N>.{svg,png}.
    _run([PY, str(WS / "scripts" / "build_paper_figures.py")], step="stitch figures")

    print("\n[build_all_figures] done — figures rebuilt from git default views.")


if __name__ == "__main__":
    main()
