"""FieldAnimationGif — render the time-evolution of N 2D fields as an animated GIF.

Stateful Visualization Step. Accumulates the per-tick field state on each
declared field input port, then re-renders the full animation at every call
to update(). The dashboard receives the latest render via render_results().

Mirrors spatio_flux/plots/plot.py::plot_species_distributions_to_gif but
operates fully in-memory: each frame is a matplotlib PNG buffered through
imageio, then imageio.mimsave is fed an in-memory BytesIO to compose the
animated GIF, which is base64-encoded into a data: URI inside an <img>.
"""
from __future__ import annotations
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import imageio.v2 as imageio

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


class FieldAnimationGif(Visualization):
    """One-row-per-field animated GIF; one column = one timestep.

    Inputs are 2D field grids; each port becomes a column of a single
    composite frame. Captures each step's snapshot into ``self._history``,
    then renders the full animation on every update. Re-render is cheap-ish
    (one GIF encode per call) for the short trajectories the Composite
    Explorer's Run tab uses (typically 5-20 steps).

    Wire ``inputs`` to one 2D field store per column, e.g.::

        "viz_animation": {
            "_type": "step",
            "address": "local:FieldAnimationGif",
            "config": {
                "field_names": ["glucose", "acetate", "dissolved biomass"],
                "title": "COMETS animation",
                "fps": 4,
            },
            "inputs": {
                "glucose": ["fields", "glucose"],
                "acetate": ["fields", "acetate"],
                "dissolved biomass": ["fields", "dissolved biomass"],
            },
            "outputs": {"html": ["viz_animation_html"]},
        }
    """

    config_schema = {
        "field_names": {"_type": "list[string]", "_default": []},
        "title": {"_type": "string", "_default": "field animation"},
        "fps": {"_type": "integer", "_default": 4},
        "cmap": {"_type": "string", "_default": "viridis"},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[dict] = []

    def inputs(self):
        fields = (getattr(self, "config", None) or {}).get("field_names") or []
        return {name: {"_type": "array", "_data": "float"} for name in fields}

    def accumulate(self, state):
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

    def render(self) -> str:
        cfg = getattr(self, "config", None) or {}
        fields = cfg.get("field_names") or []
        title = cfg.get("title", "field animation")
        fps = max(1, int(cfg.get("fps", 4)))
        cmap = cfg.get("cmap", "viridis")
        n_steps = len(self._history)
        if n_steps == 0 or not fields:
            return '<p style="color:#888;padding:8px">No field data yet</p>'

        # Per-field global vmin/vmax for stable colormap across all frames.
        vmin_max: dict[str, tuple[float, float]] = {}
        for name in fields:
            vals = []
            for snap in self._history:
                arr = snap.get(name)
                if arr is None:
                    continue
                arr_np = np.asarray(arr)
                if arr_np.size:
                    vals.append(arr_np.flatten())
            if vals:
                stacked = np.concatenate(vals)
                vmin_max[name] = (float(np.min(stacked)), float(np.max(stacked)))

        n_cols = len(fields)
        frames = []
        for step_idx in range(n_steps):
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(2.4 * n_cols + 0.6, 2.6),
                squeeze=False,
            )
            for j, name in enumerate(fields):
                ax = axes[0][j]
                grid = self._history[step_idx].get(name)
                if grid is None:
                    ax.axis("off")
                    continue
                vmin, vmax = vmin_max.get(name, (None, None))
                im = ax.imshow(
                    grid, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="auto", origin="lower",
                )
                ax.set_title(name, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{title} — step {step_idx}", fontsize=11)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90)
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        # Compose to animated GIF in-memory. Pillow's imageio plugin has
        # deprecated `fps`; use `duration` (per-frame ms) which is the
        # documented replacement.
        gif_buf = io.BytesIO()
        per_frame_ms = max(1, int(round(1000.0 / fps)))
        imageio.mimsave(
            gif_buf, frames, format="GIF",
            duration=per_frame_ms, loop=0,
        )
        gif_bytes = gif_buf.getvalue()
        gif_b64 = base64.b64encode(gif_bytes).decode("ascii")

        return (
            '<div style="text-align:center;padding:8px">'
            f'<img alt="{title}" src="data:image/gif;base64,{gif_b64}" '
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


FieldAnimationGif.__pb_kind__ = "visualization"
FieldAnimationGif.__pb_aliases__ = ["FieldAnimationGif"]
