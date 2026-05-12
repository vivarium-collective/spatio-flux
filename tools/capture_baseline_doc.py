#!/usr/bin/env python3
"""Capture a baseline composite-generator doc snapshot.

Usage::

    python tools/capture_baseline_doc.py <name>          # one-shot capture
    python tools/capture_baseline_doc.py <name> --update # regenerate from new generator

Capture mode (default) calls the OLD ``doc_func`` (from test_suite.py's
SIMULATIONS dict) with its currently-declared config and writes a
normalized JSON snapshot under
``spatio_flux/composites/_snapshots/<name>.json``.

After a composite has been ported and its old ``doc_func`` deleted, use
``--update`` to regenerate the snapshot from the NEW generator with default
parameters. This is the intentional "I changed the generator, accept the
new shape" knob.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from process_bigraph import allocate_core

from spatio_flux.composites._serialize import normalize_doc

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "spatio_flux" / "composites" / "_snapshots"


def _capture_from_old(name: str) -> dict:
    """Call the legacy doc_func in SIMULATIONS with its declared config."""
    import numpy as np
    np.random.seed(0)
    from spatio_flux.experiments.test_suite import SIMULATIONS  # local import
    sim = SIMULATIONS[name]
    core = allocate_core()
    doc = sim["doc_func"](core=core, config=sim.get("config", {}))
    return doc


def _capture_from_new(name: str) -> dict:
    """Call the registered composite_generator with default parameters."""
    import numpy as np
    np.random.seed(0)
    from spatio_flux.composites import REGISTRY
    from pbg_superpowers.composite_generator import build_generator
    # Find the entry whose name matches.
    matches = [e for e in REGISTRY.values() if e.name == name]
    if not matches:
        raise SystemExit(
            f"no registered composite_generator named '{name}'. "
            f"Known: {sorted(e.name for e in REGISTRY.values())}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"multiple registered generators named '{name}': {matches!r}"
        )
    core = allocate_core()
    # Re-seed after allocate_core so that the RNG state entering the
    # generator is stable regardless of prior framework initialisation.
    # This mirrors the identical pattern in tests/test_composite_generators.py.
    np.random.seed(0)
    return build_generator(matches[0], core=core)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", help="composite name (e.g. ecoli_core_dfba)")
    p.add_argument(
        "--update", action="store_true",
        help="regenerate the snapshot from the NEW generator (after the "
             "old doc_func has been deleted)",
    )
    args = p.parse_args()

    if args.update:
        doc = _capture_from_new(args.name)
    else:
        doc = _capture_from_old(args.name)

    normalized = normalize_doc(doc)
    out_path = SNAPSHOT_DIR / f"{args.name}.json"
    out_path.write_text(json.dumps(normalized, indent=2, sort_keys=True))
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
