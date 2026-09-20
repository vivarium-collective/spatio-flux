"""Unit tests for the spatio-flux-test-suite gate evaluator.

The test-suite studies gate on measure kind ``artifacts_present`` (op
``all_exist_and_match``). These tests pin the behaviour that makes the gate
*real*: all expected artifacts must exist, the composite schema must match a
committed reference, and semantically-irrelevant port-set ordering must not
cause a spurious failure.
"""
import json
import os

from spatio_flux.evaluators import (
    evaluate_artifacts_present,
    register_evaluators,
    _port_set,
)


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _study(tmp_path, schema_produced, schema_reference, extra_files=("x.png",)):
    slug = "demo"
    charts = tmp_path / "studies" / slug / "charts"
    ref = tmp_path / "studies" / slug / "reference"
    _write(str(charts / f"{slug}_schema.json"), schema_produced)
    _write(str(ref / f"{slug}_schema.json"), schema_reference)
    for f in extra_files:
        (charts / f).parent.mkdir(parents=True, exist_ok=True)
        (charts / f).write_text("art")
    expected = list(extra_files) + [f"{slug}_schema.json"]
    return expected, str(charts), str(ref)


def test_pass_when_all_present_and_schema_matches(tmp_path):
    schema = {"a": {"_type": "process", "interval": 0.1}}
    expected, charts, ref = _study(tmp_path, schema, schema)
    result, detail = evaluate_artifacts_present(expected, charts, ref)
    assert result == "PASS", detail


def test_fail_on_missing_artifact(tmp_path):
    schema = {"a": 1}
    expected, charts, ref = _study(tmp_path, schema, schema)
    expected.append("nope.png")
    result, detail = evaluate_artifacts_present(expected, charts, ref)
    assert result == "FAIL"
    assert "nope.png" in detail


def test_fail_on_schema_drift(tmp_path):
    expected, charts, ref = _study(
        tmp_path, {"a": {"interval": 0.1}}, {"a": {"interval": 0.2}})
    result, detail = evaluate_artifacts_present(expected, charts, ref)
    assert result == "FAIL"
    assert "drift" in detail


def test_port_set_order_is_ignored(tmp_path):
    # The emitter _inputs port order is non-deterministic run-to-run but is a
    # set, so a reordering must NOT fail the gate.
    produced = {"emitter": {"_inputs": "global_time:node|fields:node|particles:node"}}
    reference = {"emitter": {"_inputs": "global_time:node|particles:node|fields:node"}}
    expected, charts, ref = _study(tmp_path, produced, reference)
    result, detail = evaluate_artifacts_present(expected, charts, ref)
    assert result == "PASS", detail


def test_port_set_still_catches_changed_ports(tmp_path):
    produced = {"emitter": {"_inputs": "global_time:node|fields:node"}}
    reference = {"emitter": {"_inputs": "global_time:node|particles:node"}}
    expected, charts, ref = _study(tmp_path, produced, reference)
    result, detail = evaluate_artifacts_present(expected, charts, ref)
    assert result == "FAIL"


def test_port_set_helper():
    assert _port_set("b:t|a:t") == _port_set("a:t|b:t")
    assert _port_set("scalar") is None


def test_register_evaluators_exposes_the_kind():
    reg = {}
    register_evaluators(reg)
    assert "artifacts_present" in reg
    assert callable(reg["artifacts_present"])
