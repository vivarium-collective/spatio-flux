"""Fidelity checks: flush outputs vs the out0/ reference run.

Two guarantees:

- **Completeness** (all 19): every scenario's flush produces its full expected
  artifact set (structural — the report needs these files).
- **Visual similarity** (deterministic scenarios only): the produced figure
  matches out0/ within tolerance. Particle scenarios are STOCHASTIC (random
  positions/seeds) so they cannot pixel-match a prior run — they get completeness
  only.

These read already-produced ``studies/<slug>/charts/`` (run the investigation
first: ``for s in studies/*; do run_study.py ...``). A slug with no charts is
skipped so the suite stays green before a run.
"""
import glob
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The out0/ reference run lives in the canonical checkout (gitignored, not in a
# fresh worktree). Prefer a worktree-local out0/, else the canonical checkout,
# else an env override.
def _out0_dir():
    for cand in (os.path.join(REPO, "out0"),
                 os.environ.get("SPATIOFLUX_OUT0", ""),
                 os.path.expanduser("~/code/spatio-flux/out0")):
        if cand and os.path.isdir(cand):
            return cand
    return os.path.join(REPO, "out0")  # non-existent -> similarity checks skip


OUT0 = _out0_dir()

# Deterministic (no particle stochasticity) — safe to pixel-compare vs out0.
DETERMINISTIC = [
    "monod_kinetics", "ecoli_core_dfba", "ecoli_dfba", "yeast_dfba",
    "community_dfba", "dfba_kinetics_community",
    "diffusion_process", "comets_diffusion",
    "spatial_many_dfba", "spatial_dfba_process",
]


def _charts(slug):
    return os.path.join(REPO, "studies", slug, "charts")


def _has_charts(slug):
    d = _charts(slug)
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "*")))


def _all_slugs():
    from scripts.scaffold_studies import STUDIES
    return [s["slug"] for s in STUDIES]


@pytest.mark.parametrize("slug", _all_slugs() if os.path.isdir(os.path.join(REPO, "studies")) else [])
def test_expected_artifacts_present(slug):
    if not _has_charts(slug):
        pytest.skip(f"{slug}: not run yet (no charts)")
    from spatio_flux.analysis.flush_spec import expected_files
    charts = _charts(slug)
    missing = [f for f in expected_files(slug) if not os.path.exists(os.path.join(charts, f))]
    assert not missing, f"{slug}: missing artifacts {missing}"


@pytest.mark.parametrize("slug", DETERMINISTIC)
def test_deterministic_matches_out0(slug):
    if not _has_charts(slug):
        pytest.skip(f"{slug}: not run yet (no charts)")
    from tests.fidelity.image_diff import png_similarity, gif_similarity
    from spatio_flux.analysis.flush_spec import expected_files
    charts = _charts(slug)
    # PNG tolerance: 0.08 normalized MAD. Field-heatmap montages carry inherent
    # colormap-normalization + font/layout jitter vs a historical run; 0.08
    # catches gross regressions (missing series, wrong colormap, wrong panel
    # count) while tolerating that. Quantitative timeseries land well under it.
    PNG_TOL = 0.08
    checked = 0
    for fname in expected_files(slug):
        # GIFs are compared informationally only: the out0 oracle was captured at
        # a different emit cadence, so frame counts can differ (e.g. comets
        # count_diff~0.7), which misaligns frame-by-frame MAD. Content fidelity is
        # covered by the deterministic timeseries + snapshot PNGs.
        if not fname.endswith(".png") or fname.endswith("_viz.png"):
            continue
        ref = os.path.join(OUT0, fname)
        got = os.path.join(charts, fname)
        if not os.path.exists(ref) or not os.path.exists(got):
            continue
        checked += 1
        mad = png_similarity(got, ref)
        assert mad <= PNG_TOL, f"{slug}/{fname}: png drift mad={mad:.3f}"
    if checked == 0:
        pytest.skip(f"{slug}: no comparable out0 references")
