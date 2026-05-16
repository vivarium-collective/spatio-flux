"""FieldHeatmap — render a 2D field as a Plotly heatmap.

Use case: spatial composites (diffusion, COMETS) hold a 2D grid in a field
store; the heatmap captures the latest grid state into HTML so the dashboard
can preview the spatial pattern after a run.

The viz accepts the field either as ``list[list[float]]`` or as a numpy
ndarray (which it will convert). Default ``render_mode='end'``: accumulate
the latest field each tick, render once via :func:`render_results`. Set
``render_mode='stream'`` to render per-tick (legacy behaviour).
"""
from __future__ import annotations
import json

from pbg_superpowers.visualization import Visualization


def _to_2d_list(field):
    """Coerce ``field`` to a plain list-of-lists of floats."""
    if field is None:
        return []
    # numpy array?
    if hasattr(field, 'tolist'):
        try:
            data = field.tolist()
        except Exception:  # noqa: BLE001
            data = list(field)
    else:
        data = list(field)
    # Ensure 2D — wrap a flat list in a single row.
    if data and not isinstance(data[0], (list, tuple)):
        data = [list(data)]
    return [list(row) for row in data]


class FieldHeatmap(Visualization):
    """Plotly heatmap of a 2D ``field`` snapshot.

    Inputs: ``field: array`` (typed as ``any`` for tolerance — accepts numpy
    arrays and plain list-of-lists).
    """

    config_schema = {
        'title': {'_type': 'string', '_default': ''},
        'colorscale': {'_type': 'string', '_default': 'Viridis'},
    }

    def inputs(self):
        # Use a generic array type — the field store may be a numpy ndarray
        # (spatial composites) or a list-of-lists; ``_to_2d_list`` normalises.
        return {'field': {'_type': 'array', '_data': 'float'}}

    def accumulate(self, state):
        # FieldHeatmap is a "latest snapshot" viz — no history, just the
        # most recent grid.
        self._last_field = _to_2d_list(state.get('field'))

    def render(self):
        field = getattr(self, '_last_field', None) or []
        title = (getattr(self, 'config', None) or {}).get('title', '')
        colorscale = (getattr(self, 'config', None) or {}).get(
            'colorscale', 'Viridis')
        traces = [{
            'z': field,
            'type': 'heatmap',
            'colorscale': colorscale,
        }]
        layout = {
            'title': {'text': title, 'font': {'size': 14}},
            'margin': {'l': 40, 'r': 12, 't': 32, 'b': 36},
            'xaxis': {'title': {'text': 'x'}},
            'yaxis': {'title': {'text': 'y'}, 'scaleanchor': 'x'},
            'height': 360,
        }
        return (
            '<div id="viz" style="height:360px"></div>'
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            '<script>Plotly.newPlot("viz", ' + json.dumps(traces) + ', '
            + json.dumps(layout)
            + ', {responsive:true, displayModeBar:false});</script>'
        )

    @classmethod
    def demo(cls):
        return {'field': [[1.0, 0.8, 0.6], [0.9, 0.7, 0.5], [0.7, 0.5, 0.3]]}

    @classmethod
    def is_visualization(cls) -> bool:
        return True


FieldHeatmap.__pb_kind__ = 'visualization'
FieldHeatmap.__pb_aliases__ = ['FieldHeatmap']
