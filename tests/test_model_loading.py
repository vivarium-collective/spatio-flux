"""Tests for dFBA base-model resolution and its failure messages.

Regression guard for the RENCI/HeLx report that spatio-flux composites
"ran" but reported a missing SBML model with no actionable detail. The
genome-scale models are bundled in the wheel as of 1.5.3; when one is
nonetheless absent (old/slim pin, or a bad model_file name) the loader must
fail loud with the resolved path and a hint, not collapse into a vague
"Failed to load model" or a cryptic libSBML error.
"""
import pytest

from spatio_flux.processes.dfba import _load_base_model, MODEL_DIR


def test_missing_sbml_model_raises_actionable_error():
    err = None
    try:
        _load_base_model("does_not_exist.xml")
    except FileNotFoundError as e:
        err = e
    assert err is not None, "expected FileNotFoundError for a missing SBML model"
    msg = str(err)
    # Names the thing that failed and where it looked.
    assert "does_not_exist.xml" in msg
    assert "not found" in msg.lower()
    # Points the caller somewhere useful: either the bundled models that ARE
    # present (normal checkout/>=1.5.3 install) or the upgrade hint (slim pin).
    assert ("Bundled models available" in msg) or ("spatio-flux>=1.5.3" in msg)


def test_bundled_models_are_discoverable():
    """The checkout/wheel ships the genome-scale SBML models the composites use."""
    from pathlib import Path

    models = sorted(p.name for p in Path(MODEL_DIR).glob("*.xml"))
    assert "iAF1260.xml" in models, f"bundled models missing; found {models}"


def test_bad_named_model_raises_actionable_error():
    err = None
    try:
        _load_base_model("definitely_not_a_cobra_builtin")
    except ValueError as e:
        err = e
    except Exception as e:  # network/other cobra failure still must be a clear error
        err = e
    assert err is not None, "expected a clear error for an unknown named model"
    assert "definitely_not_a_cobra_builtin" in str(err)
