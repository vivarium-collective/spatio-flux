import os
import numpy as np
import pytest
pytest.importorskip("xarray"); pytest.importorskip("plotly")
from spatio_flux.runners.run_store import write_run_zarr
import xarray as xr
from spatio_flux.analysis import plotly_viz as PV


def _ds(tmp, traj):
    sd = os.path.join(tmp, "s"); os.makedirs(sd, exist_ok=True)
    write_run_zarr(sd, "r", traj, provenance={"composite": "t", "run_id": "r"})
    return xr.open_zarr(os.path.join(sd, "runs.r.zarr"))


def test_field_heatmap_movie_writes_html(tmp_path):
    ds = _ds(str(tmp_path), [{"fields": {"glucose": np.ones((4, 4)) * i}} for i in range(3)])
    out = PV.field_heatmap_movie(ds, ["glucose"], str(tmp_path / "v" / "field.html"))
    assert os.path.getsize(out) > 1000 and "plotly" in open(out).read().lower()


def test_particle_movie_writes_html(tmp_path):
    traj = [{"particles": {"p0": {"position": (float(i), 1.0), "mass": 1.0, "strain": "A"}}} for i in range(3)]
    ds = _ds(str(tmp_path), traj)
    out = PV.particle_movie(ds, str(tmp_path / "v" / "parts.html"), color_by="strain")
    assert os.path.getsize(out) > 1000


def test_phase_portrait_writes_html(tmp_path):
    out = PV.phase_portrait([0, 1, 2], [5, 5, 0.1], str(tmp_path / "v" / "phase.html"), annotate_cliff=True)
    assert os.path.getsize(out) > 1000
