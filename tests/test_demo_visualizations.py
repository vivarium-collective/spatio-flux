"""Tests for the FieldHeatmap viz that ships with spatio-flux.

These vizes now use the new ``accumulate(state)`` + ``render()`` contract
from ``viva_superpowers.visualization.Visualization`` (v0.9.0+), so tests
exercise the two-phase path: accumulate per tick, then render once.
"""
import numpy as np

from spatio_flux.visualizations.field_animation_gif import FieldAnimationGif
from spatio_flux.visualizations.field_heatmap import FieldHeatmap
from viva_superpowers.visualization import Visualization


# --- FieldHeatmap ----------------------------------------------------------

def _heatmap(**config):
    inst = FieldHeatmap.__new__(FieldHeatmap)
    inst.config = config
    return inst


def test_field_heatmap_renders_from_list():
    inst = _heatmap(title='glucose', colorscale='Viridis')
    inst.accumulate({'field': [[1.0, 0.5], [0.3, 0.1]]})
    html = inst.render()
    assert 'Plotly' in html
    assert 'heatmap' in html
    assert 'glucose' in html


def test_field_heatmap_renders_from_numpy():
    inst = _heatmap(title='', colorscale='Viridis')
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    inst.accumulate({'field': field})
    html = inst.render()
    assert 'Plotly' in html
    assert '6.0' in html


def test_field_heatmap_handles_missing_field():
    inst = _heatmap(title='', colorscale='Viridis')
    inst.accumulate({})
    html = inst.render()
    # Renders a degenerate (empty) heatmap rather than raising.
    assert 'Plotly' in html


def test_field_heatmap_handles_flat_list():
    inst = _heatmap(title='', colorscale='Viridis')
    inst.accumulate({'field': [1.0, 2.0, 3.0]})
    html = inst.render()
    # _to_2d_list wraps a flat list as a single row.
    assert 'Plotly' in html


def test_field_heatmap_default_end_mode_emits_empty_html_per_tick():
    """In default ``render_mode='end'``, the per-tick update() must NOT
    materialize HTML — that's what avoided the O(T²) regression."""
    inst = _heatmap(title='glucose', colorscale='Viridis')
    out = inst.update({'field': [[1.0, 0.5], [0.3, 0.1]]})
    assert out == {'html': ''}
    # But state was accumulated, so render() works on demand.
    assert 'Plotly' in inst.render()


def test_field_heatmap_stream_mode_renders_each_tick():
    """``render_mode='stream'`` opts back into the legacy per-tick HTML
    contract for callers that need it."""
    inst = _heatmap(title='glucose', colorscale='Viridis',
                    render_mode='stream')
    out = inst.update({'field': [[1.0, 0.5]]})
    assert 'Plotly' in out['html']


def test_field_heatmap_is_a_visualization():
    assert issubclass(FieldHeatmap, Visualization)


def test_field_heatmap_pb_aliases():
    assert 'FieldHeatmap' in FieldHeatmap.__pb_aliases__


def test_field_heatmap_demo_dict():
    demo = FieldHeatmap.demo()
    assert 'field' in demo
    assert isinstance(demo['field'], list)


# --- FieldAnimationGif -----------------------------------------------------

def _gif(**config):
    inst = FieldAnimationGif.__new__(FieldAnimationGif)
    inst.config = config
    inst._history = []
    return inst


def test_field_animation_gif_renders_a_gif():
    inst = _gif(
        field_names=['glucose', 'acetate'],
        title='demo animation', fps=4, cmap='viridis',
    )
    for k in range(3):
        inst.accumulate({
            'glucose': np.array([[1.0 + k, 0.5], [0.3, 0.1]], dtype=float),
            'acetate': np.array([[0.0, 0.1 + k], [0.2, 0.3]], dtype=float),
        })
    html = inst.render()
    assert 'data:image/gif;base64,' in html
    b64 = html.split('data:image/gif;base64,', 1)[1].split('"', 1)[0]
    import base64 as _b64
    gif_bytes = _b64.b64decode(b64)
    assert len(gif_bytes) > 0
    assert gif_bytes[:6] in (b'GIF87a', b'GIF89a')


def test_field_animation_gif_handles_no_data():
    inst = _gif(field_names=['glucose'], title='', fps=4, cmap='viridis')
    inst.accumulate({})
    html = inst.render()
    assert 'No field data yet' in html
    assert 'data:image/gif' not in html


def test_field_animation_gif_end_mode_skips_per_tick_render():
    """The whole point of this PR: per-tick update() must NOT trigger the
    full GIF re-encode that caused the O(T²) regression."""
    inst = _gif(field_names=['glucose'], title='', fps=4, cmap='viridis')
    for k in range(5):
        out = inst.update({
            'glucose': np.array([[1.0 + k, 0.5]], dtype=float),
        })
        assert out == {'html': ''}, f'tick {k} unexpectedly emitted html'
    # All 5 frames are still accumulated for the final render.
    assert len(inst._history) == 5
    html = inst.render()
    assert 'data:image/gif;base64,' in html
