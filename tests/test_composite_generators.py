"""Snapshot regression tests for composite generators.

Each registered generator's default-built doc must match its baseline JSON
snapshot under spatio_flux/composites/_snapshots/.
"""
import math
import numpy as np
import pytest

from spatio_flux.composites._serialize import normalize_doc


def test_normalize_doc_passes_scalars_through():
    doc = {"a": 1, "b": "x", "c": True, "d": None, "e": 1.5}
    assert normalize_doc(doc) == doc


def test_normalize_doc_encodes_numpy_arrays():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = normalize_doc({"f": arr})
    assert result == {
        "f": {
            "__numpy__": True,
            "dtype": "float64",
            "shape": [2, 2],
            "values": [[1.0, 2.0], [3.0, 4.0]],
        }
    }


def test_normalize_doc_rounds_floats_to_12_sig_figs():
    # 1/3 has infinite digits; the encoder must round so cross-platform
    # roundtrips are stable.
    arr = np.array([1.0 / 3.0], dtype=np.float64)
    result = normalize_doc({"f": arr})
    rounded = result["f"]["values"][0]
    # 12 sig figs of 0.333... — last digit may differ across machines if
    # we didn't round, but with rounding it's deterministic.
    assert rounded == float(f"{1.0/3.0:.12g}")


def test_normalize_doc_recurses_into_lists():
    arr = np.array([7], dtype=np.int64)
    result = normalize_doc({"xs": [1, "two", arr]})
    assert result["xs"][0] == 1
    assert result["xs"][1] == "two"
    assert result["xs"][2]["__numpy__"] is True


def test_normalize_doc_repr_falls_back_for_unencodable_objects():
    class Weird:
        def __repr__(self):
            return "Weird()"

    result = normalize_doc({"thing": Weird()})
    assert result == {
        "thing": {"__repr__": "Weird", "value": "Weird()"},
    }
