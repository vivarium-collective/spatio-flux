"""Tests for the FieldHeatmap viz that ships with spatio-flux."""
import numpy as np

from spatio_flux.visualizations.field_animation_gif import FieldAnimationGif
from spatio_flux.visualizations.field_heatmap import FieldHeatmap
from pbg_superpowers.visualization import Visualization


def test_field_heatmap_renders_from_list():
    inst = FieldHeatmap.__new__(FieldHeatmap)
    inst.config = {'title': 'glucose', 'colorscale': 'Viridis'}
    out = inst.update({'field': [[1.0, 0.5], [0.3, 0.1]]})
    assert 'html' in out
    assert 'Plotly' in out['html']
    assert 'heatmap' in out['html']
    assert 'glucose' in out['html']


def test_field_heatmap_renders_from_numpy():
    inst = FieldHeatmap.__new__(FieldHeatmap)
    inst.config = {'title': '', 'colorscale': 'Viridis'}
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    out = inst.update({'field': field})
    assert 'html' in out
    assert 'Plotly' in out['html']
    # Values should appear in the JSON-encoded traces.
    assert '6.0' in out['html']


def test_field_heatmap_handles_missing_field():
    inst = FieldHeatmap.__new__(FieldHeatmap)
    inst.config = {'title': '', 'colorscale': 'Viridis'}
    out = inst.update({})
    # Renders a degenerate (empty) heatmap rather than raising.
    assert 'html' in out
    assert 'Plotly' in out['html']


def test_field_heatmap_handles_flat_list():
    inst = FieldHeatmap.__new__(FieldHeatmap)
    inst.config = {'title': '', 'colorscale': 'Viridis'}
    out = inst.update({'field': [1.0, 2.0, 3.0]})
    # _to_2d_list wraps a flat list as a single row.
    assert 'Plotly' in out['html']


def test_field_heatmap_is_a_visualization():
    assert issubclass(FieldHeatmap, Visualization)


def test_field_heatmap_pb_aliases():
    assert 'FieldHeatmap' in FieldHeatmap.__pb_aliases__


def test_field_heatmap_demo_dict():
    demo = FieldHeatmap.demo()
    assert 'field' in demo
    assert isinstance(demo['field'], list)


def test_field_animation_gif_renders_a_gif():
    inst = FieldAnimationGif.__new__(FieldAnimationGif)
    inst.config = {
        'field_names': ['glucose', 'acetate'],
        'title': 'demo animation',
        'fps': 4,
        'cmap': 'viridis',
    }
    inst._history = []
    # Push three timesteps with two fields each.
    for k in range(3):
        out = inst.update({
            'glucose': np.array([[1.0 + k, 0.5], [0.3, 0.1]], dtype=float),
            'acetate': np.array([[0.0, 0.1 + k], [0.2, 0.3]], dtype=float),
        })
    assert 'html' in out
    html = out['html']
    assert 'data:image/gif;base64,' in html
    # Strip the prefix and decode to confirm we produced non-zero GIF bytes.
    b64 = html.split('data:image/gif;base64,', 1)[1].split('"', 1)[0]
    import base64 as _b64
    gif_bytes = _b64.b64decode(b64)
    assert len(gif_bytes) > 0
    # GIF magic header.
    assert gif_bytes[:6] in (b'GIF87a', b'GIF89a')


def test_field_animation_gif_handles_no_data():
    inst = FieldAnimationGif.__new__(FieldAnimationGif)
    inst.config = {
        'field_names': ['glucose'],
        'title': '',
        'fps': 4,
        'cmap': 'viridis',
    }
    inst._history = []
    # Update with no data -> render falls back to placeholder.
    out = inst.update({})
    assert 'html' in out
    assert 'No field data yet' in out['html']
    # No GIF data URI should be emitted.
    assert 'data:image/gif' not in out['html']
