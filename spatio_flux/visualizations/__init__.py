"""spatio-flux Visualization Steps.

Importing this package makes its Visualization classes available to
``bigraph_schema.package.discover``; subsequent ``allocate_core()`` /
``build_core()`` calls will register them under their short names in
``core.link_registry`` so composites can address them as ``local:<name>``.
"""
from spatio_flux.visualizations.field_heatmap import FieldHeatmap

__all__ = ["FieldHeatmap"]
