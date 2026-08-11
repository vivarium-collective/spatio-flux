#!/usr/bin/env python
"""The figure build graph + content-hash staleness engine.

Reads `investigations/paper-figures/figures-pipeline.yaml` (the declared panel /
sim / stitch graph) and decides, purely from SHA-256 content hashes, which nodes
are STALE — so a build reruns only the panels whose inputs changed (a saved loom
view, a composite spec, a sim script) and re-stitches only the affected figures.

A saved loom view is a real file change in the git working tree
(`investigations/paper-figures/loom-views/<id>.json`, written by the dashboard's
Save-as-default). Hashing — not mtime — is used so a clone / checkout / rebase
that scrambles mtimes without changing content agrees with the laptop that saved.

The memo lives at `.pbg/figures/manifest.json`:
  { <node_key>: {inputs: {relpath: sha256}, config: sha256, output: sha256,
                 built_at: <iso>} }
`output` doubles as a hand-edit guard: if a build product's current hash differs
from the one we last wrote, it was edited out-of-band — the builder refuses to
overwrite it without --force.

Pure + importable: `build_all_figures.py` drives it; it runs no subprocesses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

WS = Path(__file__).resolve().parents[1]
PIPELINE = WS / "investigations" / "paper-figures" / "figures-pipeline.yaml"
VIEWS_DIR = WS / "investigations" / "paper-figures" / "loom-views"
MANIFEST = WS / ".pbg" / "figures" / "manifest.json"
RENDERER = WS / "scripts" / "render_loom_svgs.mjs"


# ── content hashing ──────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file, or None if it doesn't exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── composite id → its spec-input file ───────────────────────────────────────
def spec_input_for(composite_id: str) -> Path:
    """The source a loom panel's content depends on: the static composite json,
    or — for a generator id (…composites.figures.<name>) — the generator source."""
    if ".composites.figures." in composite_id:
        return WS / "spatio_flux" / "paper_figures.py"
    name = composite_id.split(".composites.", 1)[-1]
    return WS / "spatio_flux" / "composites" / f"{name}.composite.json"


def view_input_for(composite_id: str) -> Path:
    return VIEWS_DIR / f"{composite_id}.json"


# ── graph model ──────────────────────────────────────────────────────────────
@dataclass
class Node:
    key: str                       # unique: "<study>/<output>"
    study: str
    kind: str                      # loom | command | external | stitch
    output_stem: str               # e.g. "fig07-1-community-dfba" or "figure_7"
    inputs: list[Path] = field(default_factory=list)   # files this depends on
    config: dict = field(default_factory=dict)         # flags/cmd/module (hashed)
    command: str | None = None
    composite: str | None = None
    module: str | None = None

    def outputs(self) -> list[Path]:
        vd = WS / "studies" / self.study / "visualizations"
        if self.kind in ("loom", "stitch"):
            return [vd / f"{self.output_stem}.svg", vd / f"{self.output_stem}.png"]
        return [vd / f"{self.output_stem}.png"]

    def primary_output(self) -> Path:
        """The output whose hash the stitch depends on + the hand-edit guard."""
        vd = WS / "studies" / self.study / "visualizations"
        return vd / f"{self.output_stem}.png"


def _fig_num(study: str) -> str:
    import re
    m = re.match(r"fig-0*(\d+)", study)
    return m.group(1) if m else study


def load_graph() -> list[Node]:
    """Build the node list from the pipeline yaml. Stitch inputs are derived =
    the primary outputs of every panel of that study (so a re-rendered panel
    makes its stitch stale automatically)."""
    spec = yaml.safe_load(PIPELINE.read_text())
    panels_by_study: dict[str, list[Node]] = {}
    nodes: list[Node] = []
    renderer_hash = sha256_file(RENDERER)

    for p in spec.get("panels", []):
        study, out = p["study"], p["output"]
        n = Node(key=f"{study}/{out}", study=study, output_stem=out, kind="loom")
        if "loom" in p:
            n.kind = "loom"
            n.composite = p["loom"]
            n.config = {"flags": p.get("flags", {}) or {}, "renderer": renderer_hash}
            n.inputs = [view_input_for(n.composite), spec_input_for(n.composite)]
        elif "command" in p:
            n.kind = "command"
            n.command = p["command"]
            n.config = {"command": p["command"]}
            n.inputs = [WS / i for i in (p.get("inputs") or [])]
        elif p.get("external"):
            n.kind = "external"
        nodes.append(n)
        panels_by_study.setdefault(study, []).append(n)

    for s in spec.get("stitch", []):
        study = s["study"]
        n = Node(key=f"{study}/figure_{_fig_num(study)}", study=study,
                 output_stem=f"figure_{_fig_num(study)}", kind="stitch",
                 module=s.get("module"))
        n.config = {"module": s.get("module") or "shelf"}
        # depends on every panel of this study (their primary png outputs)
        n.inputs = [pn.primary_output() for pn in panels_by_study.get(study, [])]
        nodes.append(n)
    return nodes


# ── manifest + staleness ─────────────────────────────────────────────────────
def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True))
    tmp.replace(MANIFEST)


def _input_hashes(node: Node) -> dict[str, str | None]:
    return {str(p.relative_to(WS)): sha256_file(p) for p in node.inputs}


def staleness(node: Node, manifest: dict) -> tuple[bool, str]:
    """(is_stale, reason). External nodes are never built (always up-to-date)."""
    if node.kind == "external":
        return False, "external (not built here)"
    rec = manifest.get(node.key)
    out = node.primary_output()
    if not out.exists():
        return True, "output missing"
    if rec is None:
        return True, "never built by the pipeline"
    if _input_hashes(node) != rec.get("inputs"):
        changed = [k for k, v in _input_hashes(node).items()
                   if v != (rec.get("inputs") or {}).get(k)]
        return True, f"input changed: {', '.join(changed) or '?'}"
    if _sha256_obj(node.config) != rec.get("config"):
        return True, "render config changed"
    return False, "up-to-date"


def hand_edited(node: Node, manifest: dict) -> bool:
    """True if the node's output was edited out-of-band since the pipeline wrote
    it (current hash != recorded) — protects hand-tweaked figures from a clobber."""
    rec = manifest.get(node.key)
    if not rec or not rec.get("output"):
        return False
    cur = sha256_file(node.primary_output())
    return cur is not None and cur != rec["output"]


def record(node: Node, manifest: dict, built_at: str = "") -> None:
    manifest[node.key] = {
        "inputs": _input_hashes(node),
        "config": _sha256_obj(node.config),
        "output": sha256_file(node.primary_output()),
        "built_at": built_at,
    }


def plan(nodes: list[Node], manifest: dict, only: str = "") -> dict:
    """Compute the stale set, PROPAGATING downstream: a panel that will rebuild
    hasn't changed its output file yet at plan time, so its study's stitch must be
    marked stale by dependency, not just by the (still-unchanged) panel-file hash."""
    if only:
        nodes = [n for n in nodes if only in n.key]
    status: dict[str, list] = {}
    for n in nodes:
        st, why = staleness(n, manifest)
        status[n.key] = [st, why, n]
    stale_studies = {n.study for n in nodes if n.kind != "stitch" and status[n.key][0]}
    for n in nodes:
        if n.kind == "stitch" and not status[n.key][0] and n.study in stale_studies:
            status[n.key] = [True, "a panel of this study will rebuild", n]
    stale = [(v[2], v[1]) for v in status.values() if v[0]]
    fresh = [(v[2], v[1]) for v in status.values() if not v[0]]
    return {"stale": stale, "fresh": fresh}


if __name__ == "__main__":
    g = load_graph()
    m = load_manifest()
    p = plan(g, m)
    print(f"{len(p['stale'])} stale, {len(p['fresh'])} up-to-date")
    for n, why in p["stale"]:
        print(f"  STALE  {n.key:40s} {why}")
