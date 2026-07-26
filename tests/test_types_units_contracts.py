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


# ---- Task 2: unit-bearing metabolism ports ---------------------------------

def test_dfba_ports_carry_units():
    from spatio_flux.processes.dfba import DynamicFBA
    inst = DynamicFBA.__new__(DynamicFBA)
    ins, outs = inst.inputs(), inst.outputs()
    assert ins["biomass"] == {"_type": "mass", "_units": "gDW"}
    assert ins["substrates"]["_value"]["_units"] == "mM"
    assert outs["biomass"] == {"_type": "mass", "_units": "gDW"}


def test_monod_ports_carry_units():
    from spatio_flux.processes.monod_kinetics import MonodKinetics
    inst = MonodKinetics.__new__(MonodKinetics)
    ins, outs = inst.inputs(), inst.outputs()
    assert ins["biomass"] == {"_type": "mass", "_units": "gDW"}
    assert ins["substrates"]["_value"]["_units"] == "mM"
    # outputs keep the plain-float delta semantics, just labelled
    assert outs["biomass"]["_units"] == "gDW"
    assert outs["substrates"]["_value"]["_units"] == "mM"


# ---- Task 3: process contracts (description class attribute) ----------------

def _process_classes():
    import inspect
    from process_bigraph import Process, Step
    from spatio_flux.processes import (
        dfba, monod_kinetics, diffusion_advection, particles, pymunk_particles,
    )
    mods = [dfba, monod_kinetics, diffusion_advection, particles, pymunk_particles]
    return [c for m in mods for _, c in inspect.getmembers(m, inspect.isclass)
            if issubclass(c, (Process, Step)) and c.__module__ == m.__name__]


def test_all_processes_have_descriptions():
    classes = _process_classes()
    assert classes, "no process classes discovered"
    missing = [c.__name__ for c in classes
               if not (isinstance(getattr(c, "description", ""), str)
                       and getattr(c, "description", "").strip())]
    assert not missing, f"processes missing a description: {missing}"
