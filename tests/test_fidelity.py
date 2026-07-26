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
    checked = 0
    for fname in expected_files(slug):
        if not (fname.endswith(".png") or fname.endswith(".gif")):
            continue
        if fname.endswith("_viz.png"):
            continue  # bigraph diagram layout is not pixel-stable
        ref = os.path.join(REPO, "out0", fname)
        got = os.path.join(charts, fname)
        if not os.path.exists(ref) or not os.path.exists(got):
            continue
        checked += 1
        if fname.endswith(".gif"):
            count_diff, frame_mad = gif_similarity(got, ref)
            assert count_diff <= 0.10 and frame_mad <= 0.05, \
                f"{slug}/{fname}: gif drift count={count_diff:.3f} mad={frame_mad:.3f}"
        else:
            mad = png_similarity(got, ref)
            assert mad <= 0.05, f"{slug}/{fname}: png drift mad={mad:.3f}"
    if checked == 0:
        pytest.skip(f"{slug}: no comparable out0 references")
