#!/usr/bin/env python
"""Keep the git-tracked loom default-views in lock-step with the machine-local
`.pbg` state the dashboard writes — so the figures are always reproducible from
what you arranged in the dashboard, and a fresh clone renders them identically.

The problem this solves
-----------------------
A loom view lives in THREE places, and they can silently disagree:

  1. investigations/<inv>/loom-views/<id>.json   ← git-tracked source of truth
  2. .pbg/loom-views/<id>.json                    ← dashboard "Save as default"
  3. .pbg/loom-layouts/<id>__<mode>.json          ← every drag/resize (debounced)

The loom's layout effect PREFERS (3) over (2), and (1) is what a headless render
/ a clone / the download folder should use. This tool makes them one truth:

  --promote   .pbg  ->  git   (capture what you just arranged in the dashboard,
              positions taken from the authoritative loom-layouts, settings from
              loom-views; node sizes w/h always carried)
  --seed      git   ->  .pbg  (make the dashboard + every headless render use the
              committed view; run before a deterministic rebuild / on a clone)

Both directions preserve positions AND hand-set node sizes (w/h).

Usage
-----
  python scripts/sync_loom_views.py --promote        # dashboard saves -> git
  python scripts/sync_loom_views.py --seed           # git -> dashboard/.pbg
  python scripts/sync_loom_views.py --promote --id spatio_flux.composites.fig01b-multiscale-composite
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PBG_VIEWS = WS / ".pbg" / "loom-views"
PBG_LAYOUTS = WS / ".pbg" / "loom-layouts"
PIPELINE = WS / "investigations" / "paper-figures" / "figures-pipeline.yaml"


def _safe(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cid)


def _aliases(safe: str) -> list[str]:
    """A figure composite can be BROWSED under a static id (…composites.<name>)
    but RENDERED under a generator id (…composites.figures.<name>) — same figure,
    two ids. Treat them as one view so a dashboard save under either id feeds the
    render under the other. The committed id is canonical; the static form is its
    alias for reading/writing .pbg."""
    out = [safe]
    if ".composites.figures." in safe:
        alt = safe.replace(".composites.figures.", ".composites.")
        if alt not in out:
            out.append(alt)
    return out


def _pbg_view_path(safe: str) -> Path | None:
    for a in _aliases(safe):
        p = PBG_VIEWS / f"{a}.json"
        if p.exists():
            return p
    return None


def _committed_dirs() -> list[Path]:
    """Every investigations/<inv>/loom-views/ dir (git-tracked view homes)."""
    return sorted((WS / "investigations").glob("*/loom-views"))


def _committed_path(safe: str) -> Path:
    """Where the committed view for <safe> lives (or should): the dir that
    already has it, else the sole/first investigation's loom-views dir."""
    dirs = _committed_dirs()
    for d in dirs:
        if (d / f"{safe}.json").exists():
            return d / f"{safe}.json"
    if not dirs:
        raise SystemExit("no investigations/*/loom-views/ dir found — nothing to sync into")
    return dirs[0] / f"{safe}.json"


def _layout_path(safe: str, mode: str) -> Path:
    return PBG_LAYOUTS / f"{safe}__{mode}.json"


def _merge_sizes(positions: dict, layout: dict) -> dict:
    """Overlay authoritative loom-layouts positions (x,y AND w,h) onto the view's
    positions — the layout file is what the dashboard writes on every drag/resize,
    so it holds the freshest placement + hand-set sizes."""
    out = dict(positions)
    for nid, p in layout.items():
        if not isinstance(p, dict):
            continue
        out[nid] = {**out.get(nid, {}), **p}
    return out


def promote_one(safe: str) -> str:
    """.pbg -> git for one composite (canonical = committed id). Positions from
    loom-layouts (authoritative), everything else from the saved default view;
    reads either the committed id or its static alias in .pbg."""
    view_p = _pbg_view_path(safe)
    if view_p is None:
        return f"skip {safe}: no .pbg/loom-views (nothing saved as default)"
    view = json.loads(view_p.read_text())
    mode = view.get("mode", "hierarchy")
    for a in _aliases(safe):
        lay_p = _layout_path(a, mode)
        if lay_p.exists():
            view["positions"] = _merge_sizes(view.get("positions", {}), json.loads(lay_p.read_text()))
            break
    dst = _committed_path(safe)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(view, indent=2) + "\n")
    n = len(view.get("positions", {}))
    sized = sum(1 for p in view.get("positions", {}).values() if p.get("w"))
    src_note = "" if view_p.stem == safe else f" (from alias {view_p.stem})"
    return f"promote {safe}: -> {dst.relative_to(WS)} ({n} nodes, {sized} sized){src_note}"


def seed_one(safe: str) -> str:
    """git -> .pbg for one composite: write the default view AND the per-mode
    layout file (flat positions) under EVERY alias, so the dashboard (static id)
    and the headless render (generator id) both resolve to the committed view."""
    src = _committed_path(safe)
    if not src.exists():
        return f"skip {safe}: no committed view"
    view = json.loads(src.read_text())
    mode = view.get("mode", "hierarchy")
    PBG_VIEWS.mkdir(parents=True, exist_ok=True)
    PBG_LAYOUTS.mkdir(parents=True, exist_ok=True)
    aliases = _aliases(safe)
    for a in aliases:
        (PBG_VIEWS / f"{a}.json").write_text(json.dumps(view))
        # loom-layouts is a FLAT {id: {x,y,w,h}} map for the view's mode.
        _layout_path(a, mode).write_text(json.dumps(view.get("positions", {})))
    tail = f" (+alias {aliases[1]})" if len(aliases) > 1 else ""
    return f"seed    {safe}: git -> .pbg (mode={mode}, {len(view.get('positions', {}))} nodes){tail}"


def _committed_safes() -> set[str]:
    seen: set[str] = set()
    for d in _committed_dirs():
        seen.update(p.stem for p in d.glob("*.json"))
    return seen


def _figure_safes() -> set[str]:
    """Composite ids rendered as FIGURES — read from the declared figure graph
    (`figures-pipeline.yaml` panels' `loom:` ids). A saved view for any of these
    is a figure view worth capturing to git even the FIRST time it's saved, before
    it has a committed view (closes the hole where a figure the user arranged —
    e.g. fig3b — was silently dropped for lacking a committed view)."""
    try:
        import yaml
        spec = yaml.safe_load(PIPELINE.read_text()) or {}
    except (OSError, ImportError, ValueError):
        return set()
    return {_safe(p["loom"]) for p in spec.get("panels", []) if p.get("loom")}


def _all_safes(direction: str) -> list[str]:
    """The set of composite ids to act on, by direction's source."""
    committed = _committed_safes()
    if direction == "promote":
        # Promote composites that already have a committed view OR are a known
        # figure (so a brand-new figure view is captured the first time it's
        # saved) — but never an unrelated composite the user merely browsed.
        candidates = committed | _figure_safes()
        return sorted(s for s in candidates if _pbg_view_path(s) is not None)
    return sorted(committed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--promote", action="store_true", help=".pbg -> git (capture dashboard saves)")
    g.add_argument("--seed", action="store_true", help="git -> .pbg (deterministic render / clone)")
    ap.add_argument("--id", help="only this composite id (default: all)")
    args = ap.parse_args()

    direction = "promote" if args.promote else "seed"
    safes = [_safe(args.id)] if args.id else _all_safes(direction)
    fn = promote_one if args.promote else seed_one
    for safe in safes:
        print(fn(safe))
    print(f"{direction}: {len(safes)} view(s)")


if __name__ == "__main__":
    main()
