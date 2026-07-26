"""Composite generators — metabolism group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""

from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import get_dfba_process_from_registry, MODEL_REGISTRY_DFBA, get_field_names
from spatio_flux.processes.monod_kinetics import MODEL_REGISTRY_KINETICS, get_monod_kinetics_process_from_config


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
    return {
        f"{model_id} dFBA": dfba_process,
        "fields": initial_fields,
    }


@composite_generator(
    name="monod_kinetics",
    description=(
        "Field-only baseline: Monod uptake/growth on a well-mixed substrate "
        "pool (no spatial lattice, no particles). Use to sanity-check "
        "kinetics + mass balance."
    ),
    parameters={
        "model_id": {"type": "string", "default": "overflow_metabolism"},
        "interval": {"type": "float",  "default": 0.1},
        "glucose":  {"type": "float",  "default": 10.0},
        "acetate":  {"type": "float",  "default": 0.0},
        "biomass":  {"type": "float",  "default": 0.1},
    },
)
def monod_kinetics(core=None, *, model_id="overflow_metabolism",
                   interval=0.1, glucose=10.0, acetate=0.0, biomass=0.1):
    model_config = MODEL_REGISTRY_KINETICS[model_id]()
    return {
        "monod_kinetics": get_monod_kinetics_process_from_config(
            model_config=model_config, interval=interval),
        "fields": {"glucose": glucose, "acetate": acetate, "biomass": biomass},
    }


@composite_generator(
    name="ecoli_dfba",
    description=(
        "Single-cell metabolism (large model): dynamic FBA using iAF1260 "
        "with tracked extracellular fields (glucose/formate) and biomass. "
        "Stress-tests solver + exchange wiring."
    ),
    parameters={
        "model_id": {"type": "string", "default": "ecoli"},
        "glucose":  {"type": "float",  "default": 10.0},
        "formate":  {"type": "float",  "default": 5.0},
        "biomass":  {"type": "float",  "default": 0.1},
    },
)
def ecoli_dfba(core=None, *, model_id="ecoli",
               glucose=10.0, formate=5.0, biomass=0.1):
    dfba_process = get_dfba_process_from_registry(
        model_id=model_id, biomass_id="biomass", path=["fields"])
    initial_fields = {"glucose": glucose, "formate": formate, "biomass": biomass}
    for substrate in dfba_process["inputs"]["substrates"]:
        initial_fields.setdefault(substrate, 10.0)
    return {
        f"{model_id} dFBA": dfba_process,
        "fields": initial_fields,
    }


@composite_generator(
    name="yeast_dfba",
    description=(
        "Single-cell metabolism (yeast): dynamic FBA using iMM904 with "
        "extracellular glucose and biomass. Cross-model check of the dFBA "
        "pipeline."
    ),
    parameters={
        "model_id": {"type": "string", "default": "yeast"},
        "glucose":  {"type": "float",  "default": 5.0},
        "biomass":  {"type": "float",  "default": 0.1},
    },
)
def yeast_dfba(core=None, *, model_id="yeast", glucose=5.0, biomass=0.1):
    dfba_process = get_dfba_process_from_registry(
        model_id=model_id, biomass_id="biomass", path=["fields"])
    initial_fields = {"glucose": glucose, "biomass": biomass}
    for substrate in dfba_process["inputs"]["substrates"]:
        initial_fields.setdefault(substrate, 10.0)
    return {
        f"{model_id} dFBA": dfba_process,
        "fields": initial_fields,
    }


@composite_generator(
    name="community_dfba",
    description=(
        "Multi-agent well-mixed community: several independent dFBA "
        "instances share the same extracellular pools, creating competition / "
        "cross-feeding dynamics without space."
    ),
    parameters={
        "dt":                {"type": "float",  "default": 1.0},
        "kinetic_model_id":  {"type": "string", "default": "acetate_only"},
        "initial_biomass":   {"type": "float",  "default": 0.1},
        "glucose":           {"type": "float",  "default": 10.0},
        "acetate":           {"type": "float",  "default": 0.0},
    },
)
def community_dfba(core=None, *, dt=1.0, kinetic_model_id="acetate_only",
                   initial_biomass=0.1, glucose=10.0, acetate=0.0):
    model_ids = list(MODEL_REGISTRY_DFBA.keys())
    dfbas = {
        f"{model_id} dFBA": get_dfba_process_from_registry(
            model_id=model_id, biomass_id=model_id, path=["fields"],
            interval=dt)
        for model_id in MODEL_REGISTRY_DFBA
    }
    biomasses = {organism: initial_biomass for organism in model_ids}
    kinetic_biomass_id = "monod biomass"
    biomasses[kinetic_biomass_id] = initial_biomass
    kinetic_model_config = MODEL_REGISTRY_KINETICS[kinetic_model_id]()
    field_names = get_field_names(MODEL_REGISTRY_DFBA)
    more_fields = {m: 0.1 for m in field_names if m not in ("glucose", "acetate")}
    return {
        **dfbas,
        "monod_kinetics": get_monod_kinetics_process_from_config(
            model_config=kinetic_model_config,
            biomass_id=kinetic_biomass_id, interval=dt),
        "fields": {"glucose": glucose, "acetate": acetate,
                   **more_fields, **biomasses},
    }


@composite_generator(
    name="dfba_kinetics_community",
    description=(
        "Hybrid community (well-mixed): mixes Monod-kinetic agents with "
        "dFBA agents in a shared environment. Demonstrates heterogeneous "
        "process composition under one schema."
    ),
    parameters={
        "dfba_model_id":      {"type": "string", "default": "ecoli core"},
        "kinetic_model_id":   {"type": "string", "default": "acetate_only"},
        "dfba_biomass_id":    {"type": "string", "default": "dfba biomass"},
        "kinetic_biomass_id": {"type": "string", "default": "kinetic biomass"},
        "glucose":            {"type": "float",  "default": 10.0},
        "acetate":            {"type": "float",  "default": 0.0},
        "initial_biomass":    {"type": "float",  "default": 0.01},
    },
)
def dfba_kinetics_community(core=None, *,
                            dfba_model_id="ecoli core",
                            kinetic_model_id="acetate_only",
                            dfba_biomass_id="dfba biomass",
                            kinetic_biomass_id="kinetic biomass",
                            glucose=10.0, acetate=0.0, initial_biomass=0.01):
    kinetic_config = MODEL_REGISTRY_KINETICS[kinetic_model_id]()
    return {
        "dFBA": get_dfba_process_from_registry(
            model_id=dfba_model_id, biomass_id=dfba_biomass_id,
            path=["fields"]),
        "monod_kinetics": get_monod_kinetics_process_from_config(
            model_config=kinetic_config, biomass_id=kinetic_biomass_id),
        "fields": {
            "glucose": glucose,
            "acetate": acetate,
            dfba_biomass_id: initial_biomass,
            kinetic_biomass_id: initial_biomass,
        },
    }
