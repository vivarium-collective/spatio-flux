"""ParticleTraces — static figure of particle trajectories over time.

Stateful Visualization Step. Captures the per-tick ``particles`` state
into ``self._history``, then re-renders a single static matplotlib figure
on every call to update(): one disk per particle per timestep (color
encodes identity, brightness encodes time), with light connecting traces
between consecutive positions.

Wraps spatio_flux/plots/plot.py::plot_particle_traces but operates
fully in-memory and returns a base64 ``<img>`` HTML snippet for the
dashboard's Visualizations panel.
"""
from __future__ import annotations
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbg_superpowers.visualization import Visualization
from spatio_flux.plots.plot import plot_particle_traces


class ParticleTraces(Visualization):
    """Static figure of particle trajectories.

    Wire ``inputs.particles`` to the ``particles`` store. Provide ``bounds``
    in ``config`` so the figure's axes match the simulated domain. Example::

        "viz_traces": {
            "_type": "step",
            "address": "local:ParticleTraces",
            "config": {
                "bounds": [50.0, 50.0],
                "radius_scaling": 0.1,
                "units": "μm",
                "min_brightness": 0.1,
                "legend": False,
            },
            "inputs": {"particles": ["particles"]},
            "outputs": {"html": ["viz_traces_html"]},
        }
    """

    config_schema = {
        "bounds": {"_type": "tuple[float,float]", "_default": (50.0, 50.0)},
        "title": {"_type": "string", "_default": "particle traces"},
        "radius_scaling": {"_type": "float", "_default": 1.0},
        "units": {"_type": "string", "_default": ""},
        "unit_scale": {"_type": "float", "_default": 1.0},
        "min_brightness": {"_type": "float", "_default": 0.15},
        "max_brightness": {"_type": "float", "_default": 1.0},
        "trace_alpha": {"_type": "float", "_default": 0.22},
        "legend": {"_type": "boolean", "_default": True},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[dict] = []

    def inputs(self):
        return {"particles": "map[particle]"}

    def accumulate(self, state):
        if not hasattr(self, "_history") or self._history is None:
            self._history = []
        particles = state.get("particles") or {}
        # Copy so mutations to live state don't corrupt history frames.
        frame = {pid: dict(p) for pid, p in particles.items() if isinstance(p, dict)}
        self._history.append(frame)

    def render(self) -> str:
        cfg = getattr(self, "config", None) or {}
        title = cfg.get("title", "particle traces")
        bounds = cfg.get("bounds") or (50.0, 50.0)
        if not self._history:
            return '<p style="color:#888;padding:8px">No particle data yet</p>'

        try:
            plot_particle_traces(
                history=self._history,
                bounds=bounds,
                out_dir=None,                 # we'll capture from current figure
                show=False,
                radius_scaling=float(cfg.get("radius_scaling", 1.0)),
                min_brightness=float(cfg.get("min_brightness", 0.15)),
                max_brightness=float(cfg.get("max_brightness", 1.0)),
                trace_alpha=float(cfg.get("trace_alpha", 0.22)),
                legend=bool(cfg.get("legend", True)),
                units=cfg.get("units") or None,
                unit_scale=float(cfg.get("unit_scale", 1.0)),
            )
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            return f'<p style="color:#a00;padding:8px">ParticleTraces render error: {type(e).__name__}: {e}</p>'

        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        buf.close()
        return (
            '<div style="text-align:center;padding:8px">'
            f'<img alt="{title}" src="data:image/png;base64,{b64}" '
            'style="max-width:100%;height:auto;border:1px solid #e5e7eb;'
            'border-radius:4px" />'
            "</div>"
        )

    @classmethod
    def demo(cls):
        return {
            "particles": {
                "p0": {"id": "p0", "position": (5.0, 5.0), "mass": 1.0},
                "p1": {"id": "p1", "position": (25.0, 25.0), "mass": 1.0},
            },
        }

    @classmethod
    def is_visualization(cls) -> bool:
        return True


ParticleTraces.__pb_kind__ = "visualization"
ParticleTraces.__pb_aliases__ = ["ParticleTraces"]
