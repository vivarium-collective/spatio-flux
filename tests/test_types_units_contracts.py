from spatio_flux.core import build_core


# ---- Task 1: units mechanism spike (RESOLVED) -------------------------------
# Units attach via a ``_units`` key on the port schema — NOT quantity[float,X]
# (that leaves _units None in this bigraph-schema) and NOT pint registration
# (pint is only invoked by _compute_unit_scale for cross-unit wire conversion,
# so a plain label like 'gDW'/'mM' attaches without being a registered unit).

def test_units_key_attaches_to_schema_node():
    core = build_core()
    for unit in ("mM", "gDW", "mmol/gDW/h", "micrometer"):
        node = core.access({"_type": "float", "_units": unit})
        assert getattr(node, "_units", None) == unit


def test_units_key_works_on_custom_types():
    core = build_core()
    node = core.access({"_type": "concentration", "_units": "mM"})
    assert getattr(node, "_units", None) == "mM"
