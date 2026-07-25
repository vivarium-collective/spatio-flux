"""TestSuiteTimeSeries - matplotlib timeseries plot embedded in HTML.

Stateful ``Visualization`` Step. Each tick the runtime calls ``update`` with
the latest scalar value on every declared field input port; the viz appends
the snapshot to an internal history list and re-renders a multi-line PNG
showing the trajectory so far. The PNG is base64-encoded and wrapped in
``<img>`` HTML so the dashboard's existing string-html pipeline can serve it.

This mirrors the ``plot_time_series`` helper in
``spatio_flux/plots/plot.py`` that the offline test-suite report uses, so
the Composite Explorer's Run tab shows the same plot conventions as the
test-suite HTML report.

Input ports are config-driven via ``field_names: list[string]`` so the viz
can be reused across composites that track different sets of scalars (e.g.
glucose/acetate/biomass for monod_kinetics; glucose/formate/biomass for
ecoli_dfba). Wire each port to the actual scalar store in the composite.
"""
from __future__ import annotations
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viva_superpowers.visualization import Visualization


# Semantic colors lifted from spatio_flux/experiments/test_suite.py so the
# embedded-PNG plots look the same as the offline test-suite report.
STANDARD_FIELD_COLORS = {
    "glucose": "#1f77b4",
    "acetate": "#ff7f0e",
    "formate": "#9467bd",
    "ammonium": "#bcbd22",
    "glutamate": "#e377c2",
    "biomass": "#2ca02c",
    "dfba_biomass": "#2ca02c",
    "dfba biomass": "#2ca02c",
    "ecoli core biomass": "#2ca02c",
    "ecoli_core_biomass": "#2ca02c",
    "kinetic_biomass": "#1b9e77",
    "kinetic biomass": "#1b9e77",
    "monod_biomass": "#98df8a",
    "monod biomass": "#98df8a",
    "dissolved biomass": "#17becf",
}


def _coerce_float(v):
    """Best-effort scalar coercion; None / non-numeric -> None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class TestSuiteTimeSeries(Visualization):
    """Matplotlib multi-line timeseries embedded as a base64 PNG in HTML.

    Wire ``inputs`` to one scalar store per tracked field, e.g.::

        "viz_timeseries": {
            "_type": "step",
            "address": "local:TestSuiteTimeSeries",
            "config": {
                "field_names": ["glucose", "acetate", "biomass"],
                "title": "Monod kinetics",
                "y_label": "concentration / biomass",
            },
            "inputs": {
                "glucose": ["fields", "glucose"],
                "acetate": ["fields", "acetate"],
                "biomass": ["fields", "biomass"],
            },
            "outputs": {"html": ["viz_timeseries_html"]},
        }
    """

    config_schema = {
        "field_names": {"_type": "list[string]", "_default": []},
        "title": {"_type": "string", "_default": "timeseries"},
        "time_units": {"_type": "string", "_default": "step"},
        "y_label": {"_type": "string", "_default": "value"},
        "log_scale": {"_type": "boolean", "_default": False},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[dict] = []

    def inputs(self):
        fields = (getattr(self, "config", None) or {}).get("field_names") or []
        return {name: "float" for name in fields}

    def accumulate(self, state):
        if not hasattr(self, "_history") or self._history is None:
            self._history = []
        fields = (getattr(self, "config", None) or {}).get("field_names") or []
        snapshot = {name: _coerce_float(state.get(name)) for name in fields}
        self._history.append(snapshot)

    def render(self) -> str:
        cfg = getattr(self, "config", None) or {}
        fields = cfg.get("field_names") or []
        title = cfg.get("title", "timeseries")
        time_units = cfg.get("time_units", "step")
        y_label = cfg.get("y_label", "value")
        log_scale = bool(cfg.get("log_scale"))

        n = len(self._history)
        if n == 0 or not fields:
            return (
                '<p style="color:#888;padding:8px">no data yet</p>'
            )

        times = list(range(n))
        fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=100)
        plotted_any = False
        for name in fields:
            ys = [snap.get(name) for snap in self._history]
            if not any(y is not None for y in ys):
                continue
            color = STANDARD_FIELD_COLORS.get(name)
            ax.plot(times, ys, label=name, color=color, linewidth=1.8)
            plotted_any = True

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"step ({time_units})")
        ax.set_ylabel(y_label)
        if log_scale:
            ax.set_yscale("log")
        if plotted_any:
            ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

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
        return {"glucose": 10.0, "acetate": 0.0, "biomass": 0.1}

    @classmethod
    def is_visualization(cls) -> bool:
        return True


TestSuiteTimeSeries.__pb_kind__ = "visualization"
TestSuiteTimeSeries.__pb_aliases__ = ["TestSuiteTimeSeries"]
