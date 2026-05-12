"""Composite generators — metabolism group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""

from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import get_dfba_process_from_registry


@composite_generator(
    name="ecoli_core_dfba",
    description=(
        "Single-cell metabolism baseline: dynamic FBA for E. coli core with "
        "external glucose/acetate and biomass over time (no space, no particles)."
    ),
    parameters={
        "model_id":       {"type": "string", "default": "ecoli core"},
        "glucose":        {"type": "float",  "default": 10.0},
        "acetate":        {"type": "float",  "default": 0.0},
        "biomass":        {"type": "float",  "default": 0.1},
    },
)
def ecoli_core_dfba(core=None, *, model_id="ecoli core",
                    glucose=10.0, acetate=0.0, biomass=0.1):
    """Port of test_suite.get_dfba_single_doc for model_id='ecoli core'.

    The original function defaulted unspecified substrates to 10.0 by
    walking the dFBA process's substrate list. We preserve that behavior:
    callers can override glucose/acetate explicitly; any other substrate
    the model declares is filled in with 10.0.
    """
    dfba_process = get_dfba_process_from_registry(
        model_id=model_id, biomass_id="biomass", path=["fields"])
    initial_fields = {"glucose": glucose, "acetate": acetate, "biomass": biomass}
    for substrate in dfba_process["inputs"]["substrates"]:
        initial_fields.setdefault(substrate, 10.0)
    return {f"{model_id} dFBA": dfba_process, "fields": initial_fields}
