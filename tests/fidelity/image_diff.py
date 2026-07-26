"""Structural image comparison for fidelity checks.

matplotlib/imageio are not bit-reproducible (fonts, antialiasing, encoders), so
exact-match is impossible. ``image_similar`` compares normalized mean-absolute
pixel difference (MAD) — a ~2% threshold catches structural regressions (missing
series, wrong colormap, wrong panel count) while tolerating rendering jitter.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageSequence


def _to_array(img, size):
    img = img.convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


def _mad(a, b):
    """Normalized mean-absolute difference of two same-shape float arrays in [0,1]."""
    return float(np.mean(np.abs(a - b)))


def png_similarity(path_a, path_b):
    """Return normalized MAD between two PNGs (0.0 = identical)."""
    ia, ib = Image.open(path_a), Image.open(path_b)
    size = (min(ia.width, ib.width), min(ia.height, ib.height))
    return _mad(_to_array(ia, size), _to_array(ib, size))


def gif_similarity(path_a, path_b):
    """Return (frame_count_ratio_diff, mean_frame_MAD) for two GIFs.

    frame_count_ratio_diff = |na - nb| / max(na, nb); mean_frame_MAD averages the
    per-frame MAD over the min frame count on a common size.
    """
    fa = [f.copy() for f in ImageSequence.Iterator(Image.open(path_a))]
    fb = [f.copy() for f in ImageSequence.Iterator(Image.open(path_b))]
    na, nb = len(fa), len(fb)
    count_diff = abs(na - nb) / max(na, nb, 1)
    n = min(na, nb)
    if n == 0:
        return count_diff, 1.0
    w = min(fa[0].width, fb[0].width)
    h = min(fa[0].height, fb[0].height)
    diffs = [_mad(_to_array(fa[i], (w, h)), _to_array(fb[i], (w, h))) for i in range(n)]
    return count_diff, float(np.mean(diffs))


def image_similar(path_a, path_b, tolerance=0.02):
    """True if the two images are structurally similar within ``tolerance``.

    PNG: MAD <= tolerance. GIF: frame-count within 10% and mean-frame MAD <= tolerance.
    """
    if str(path_a).endswith(".gif"):
        count_diff, frame_mad = gif_similarity(path_a, path_b)
        return count_diff <= 0.10 and frame_mad <= tolerance
    return png_similarity(path_a, path_b) <= tolerance
