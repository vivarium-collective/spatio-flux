"""Tests for the FieldHeatmap viz that ships with spatio-flux."""
import numpy as np

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
