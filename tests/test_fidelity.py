"""Fidelity checks: flush outputs vs a committed structural reference.

Two guarantees, both reproducible and environment-stable:

- **Completeness** (all 19): every scenario's flush produces its full expected
  artifact set (the report needs these files).
- **Structural fidelity** (all 19): the composite's schema
  (``<slug>_schema.json``) matches the committed reference in
  ``studies/<slug>/reference/`` — evaluated by
  :func:`spatio_flux.evaluators.evaluate_artifacts_present`, the same gate the
  dashboard and ``scripts/verify_reference.py`` use.

Why structural, not pixel: composite schema is deterministic (verified — up to
``|``-delimited port-set ordering, which the evaluator normalizes). Figures/gifs
are renders and initial state embeds RNG (random particle positions), so they
drift across matplotlib/env versions and cannot be pixel-matched reproducibly.
The previous ``out0/`` pixel oracle was uncommitted *and* stale (it predated the
current bigraph-schema representation), so it silently skipped in CI and failed
the moment a run produced real charts — this replaces it with a committed,
byte-checkable structural reference.

These read already-produced ``studies/<slug>/charts/`` (run the investigation
first: ``python scripts/verify_reference.py``). A slug with no charts is skipped
so the suite stays green before a run.
"""
import glob
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _charts(slug):
    return os.path.join(REPO, "studies", slug, "charts")


def _reference(slug):
    return os.path.join(REPO, "studies", slug, "reference")


def _has_charts(slug):
    d = _charts(slug)
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "*")))


def _all_slugs():
    from scripts.scaffold_studies import STUDIES
    return [s["slug"] for s in STUDIES]


_SLUGS = _all_slugs() if os.path.isdir(os.path.join(REPO, "studies")) else []


@pytest.mark.parametrize("slug", _SLUGS)
def test_expected_artifacts_present(slug):
    if not _has_charts(slug):
        pytest.skip(f"{slug}: not run yet (no charts)")
    from spatio_flux.analysis.flush_spec import expected_files
    charts = _charts(slug)
    missing = [f for f in expected_files(slug) if not os.path.exists(os.path.join(charts, f))]
    assert not missing, f"{slug}: missing artifacts {missing}"


@pytest.mark.parametrize("slug", _SLUGS)
def test_schema_matches_reference(slug):
    if not _has_charts(slug):
        pytest.skip(f"{slug}: not run yet (no charts)")
    ref = os.path.join(_reference(slug), f"{slug}_schema.json")
    if not os.path.exists(ref):
        pytest.skip(f"{slug}: no committed structural reference")
    from spatio_flux.analysis.flush_spec import expected_files
    from spatio_flux.evaluators import evaluate_artifacts_present
    result, detail = evaluate_artifacts_present(
        expected_files(slug), _charts(slug), _reference(slug), tolerance=0.0)
    assert result == "PASS", f"{slug}: {detail}"
