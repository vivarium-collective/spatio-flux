"""FieldSnapshotsGrid - render N snapshots x M fields as a matplotlib grid.

Stateful ``Visualization`` Step for spatial composites: each declared input
port is a 2D field (numpy ndarray or list-of-lists). On every tick the viz
captures the current field state into ``self._history`` and re-renders a
grid plot whose rows are fields and whose columns are evenly-spaced
snapshots across the trajectory so far. This mirrors
``plot_snapshots_grid`` from ``spatio_flux/plots/plot.py`` that the offline
test-suite report uses.

The rendered PNG is base64-encoded into an ``<img>`` tag so the
Composite-Explorer Run-tab pipeline can serve it as plain HTML.
"""
from __future__ import annotations
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pbg_superpowers.visualization import Visualization


def _to_2d_array(v):
    """Coerce ``v`` to a 2D numpy array; return None if not possible."""
    if v is None:
        return None
    try:
        arr = np.asarray(v.tolist()) if hasattr(v, "tolist") else np.asarray(v)
    except Exception:  # noqa: BLE001
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 0:
        return None
    return arr


class FieldSnapshotsGrid(Visualization):
    """Render ``n_snapshots`` evenly-spaced snapshots of each field as a grid.

    Wire ``inputs`` to one 2D field store per row, e.g.::

        "viz_snapshots": {
            "_type": "step",
            "address": "local:FieldSnapshotsGrid",
            "config": {
                "field_names": ["glucose", "acetate", "dissolved biomass"],
                "n_snapshots": 4,
                "title": "COMETS snapshots",
            },
            "inputs": {
                "glucose": ["fields", "glucose"],
                "acetate": ["fields", "acetate"],
                "dissolved biomass": ["fields", "dissolved biomass"],
            },
            "outputs": {"html": ["viz_snapshots_html"]},
        }
    """

    config_schema = {
        "field_names": {"_type": "list[string]", "_default": []},
        "n_snapshots": {"_type": "integer", "_default": 4},
        "title": {"_type": "string", "_default": "field snapshots"},
        "colormap": {"_type": "string", "_default": "viridis"},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[dict] = []

    def inputs(self):
        fields = (getattr(self, "config", None) or {}).get("field_names") or []
        # Generic array type — store may be ndarray or list-of-lists.
        return {name: {"_type": "array", "_data": "float"} for name in fields}

    def update(self, state):
        if not hasattr(self, "_history") or self._history is None:
            self._history = []
        fields = (getattr(self, "config", None) or {}).get("field_names") or []
        snap = {}
        for name in fields:
            arr = _to_2d_array(state.get(name))
            if arr is not None:
                snap[name] = arr
        if snap:
            self._history.append(snap)
        return {"html": self._render()}

    def _render(self) -> str:
        cfg = getattr(self, "config", None) or {}
        fields = cfg.get("field_names") or []
        n_snap = max(1, int(cfg.get("n_snapshots", 4)))
        title = cfg.get("title", "field snapshots")
        cmap = cfg.get("colormap", "viridis")

        n_steps = len(self._history)
        if n_steps == 0 or not fields:
            return '<p style="color:#888;padding:8px">no field data yet</p>'

        # pick evenly-spaced snapshot indices across the history
        if n_steps <= n_snap:
            idxs = list(range(n_steps))
        else:
            idxs = [
                int(round(i * (n_steps - 1) / (n_snap - 1)))
                for i in range(n_snap)
            ]

        # Per-field global min/max for stable color scaling across columns.
        global_minmax: dict[str, tuple[float, float]] = {}
        for name in fields:
            vals = []
            for snap in self._history:
                arr = snap.get(name)
                if arr is None:
                    continue
                vals.append(arr.flatten())
            if vals:
                stacked = np.concatenate(vals)
                global_minmax[name] = (
                    float(np.min(stacked)), float(np.max(stacked))
                )

        n_rows = len(fields)
        n_cols = len(idxs)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.2 * n_cols + 0.8, 2.0 * n_rows + 0.6),
            squeeze=False,
        )
        for i, name in enumerate(fields):
            vmin, vmax = global_minmax.get(name, (None, None))
            for j, step_idx in enumerate(idxs):
                ax = axes[i][j]
                grid = self._history[step_idx].get(name)
                if grid is None:
                    ax.axis("off")
                    continue
                im = ax.imshow(
                    grid, cmap=cmap, aspect="auto", origin="lower",
                    vmin=vmin, vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(name, rotation=90, fontsize=9)
                if i == 0:
                    ax.set_title(f"step {step_idx}", fontsize=8)
                if j == n_cols - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return (
            '<div style="text-align:center;padding:8px">'
            f'<img alt="{title}" src="data:image/png;base64,{png_b64}" '
            'style="max-width:100%;height:auto;border:1px solid #e5e7eb;'
            'border-radius:4px" />'
            "</div>"
        )

    @classmethod
    def demo(cls):
        return {
            "glucose": [[1.0, 0.8, 0.6], [0.9, 0.7, 0.5]],
            "acetate": [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
        }

    @classmethod
    def is_visualization(cls) -> bool:
        return True


FieldSnapshotsGrid.__pb_kind__ = "visualization"
FieldSnapshotsGrid.__pb_aliases__ = ["FieldSnapshotsGrid"]
