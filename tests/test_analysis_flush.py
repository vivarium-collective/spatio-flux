import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- Task 1: analysis-flush harness ----------------------------------------

def test_flush_dispatches_and_returns_paths(tmp_path):
    from spatio_flux.analysis.flush import run_analysis_flush, ANALYSIS_STEPS
    calls = []
    ANALYSIS_STEPS["_stub"] = lambda traj, state, out_dir, config: (
        calls.append((len(traj), config)) or [os.path.join(out_dir, "x.png")]
    )
    traj = [{"global_time": 0.0, "fields": {"glucose": 1.0}},
            {"global_time": 1.0, "fields": {"glucose": 0.5}}]
    paths = run_analysis_flush("demo", traj, {}, str(tmp_path),
                               [{"step": "_stub", "config": {"filename": "x"}}])
    assert calls == [(2, {"filename": "x"})]
    assert paths == [str(tmp_path / "x.png")]


# ---- Task 2: plot.py wrapper steps -----------------------------------------

def _spatial_traj(n=3, bins=(4, 4)):
    import numpy as np
    return [{"global_time": float(t),
             "fields": {"glucose": np.ones(bins) * (n - t),
                        "acetate": np.zeros(bins)}} for t in range(n)]


def test_timeseries_step_writes_png(tmp_path):
    import spatio_flux.analysis  # register steps
    from spatio_flux.analysis.steps import timeseries_step
    traj = [{"global_time": float(t), "fields": {"glucose": 10.0 - t, "biomass": 0.1 * t}}
            for t in range(5)]
    out = timeseries_step(traj, {"fields": {"glucose": 0, "biomass": 0}},
                          str(tmp_path), {"filename": "demo"})
    assert out and out[0].endswith("demo.png") and os.path.exists(out[0])


def test_species_gif_step_writes_gif(tmp_path):
    import spatio_flux.analysis  # register steps
    from spatio_flux.analysis.steps import species_dist_gif_step
    out = species_dist_gif_step(_spatial_traj(), {}, str(tmp_path),
                                {"filename": "demo_video"})
    assert out and out[0].endswith("demo_video.gif") and os.path.exists(out[0])


def test_all_expected_steps_registered():
    import spatio_flux.analysis
    from spatio_flux.analysis.flush import ANALYSIS_STEPS
    expected = {"timeseries", "species_dist_gif", "species_dist_with_particles_gif",
                "snapshots_grid", "particles_mass", "particles_mass_submasses",
                "particle_traces", "model_grid", "pymunk_gif", "fields_agents_gif"}
    assert expected <= set(ANALYSIS_STEPS)


# ---- Task 3: FLUSH_SPEC + scaffolder emits canonical_runs/visualizations ----

def test_flush_spec_covers_all_19_and_uses_registered_steps():
    from spatio_flux.analysis.flush_spec import FLUSH_SPEC
    from spatio_flux.analysis.flush import ANALYSIS_STEPS
    import spatio_flux.analysis  # register
    from scripts.scaffold_studies import STUDIES
    assert set(FLUSH_SPEC) == {s["slug"] for s in STUDIES}
    for slug, specs in FLUSH_SPEC.items():
        for s in specs:
            assert s["step"] in ANALYSIS_STEPS, f"{slug}: unregistered step {s['step']}"
    steps = {e["step"] for e in FLUSH_SPEC["spatioflux_reference_demo"]}
    assert {"timeseries", "particles_mass", "particles_mass_submasses",
            "fields_agents_gif", "snapshots_grid"} <= steps


def test_scaffolder_emits_canonical_runs_and_viz():
    from scripts.scaffold_studies import render_study, STUDIES
    demo = next(s for s in STUDIES if s["slug"] == "comets_diffusion")
    spec = render_study(demo)
    assert spec["canonical_runs"][0]["script"] == "spatio_flux/runners/run_study.py"
    assert any(v["address"].startswith("image:charts/") for v in spec["visualizations"])
    assert spec["behavior_tests"][0]["name"] == "COMETS_DIFFUSION-REPRODUCES-REPORT"


# ---- Task 4: runner runtime-placeholder resolution --------------------------

def test_apply_resolves_placeholders():
    from spatio_flux.runners.run_study import _apply
    ctx = {"$bounds": (50.0, 50.0), "$colors": {"glucose": "#000"}}
    out = _apply({"filename": "x", "bounds": "$bounds",
                  "field_colors": "$colors", "n_snapshots": 5}, ctx)
    assert out["bounds"] == (50.0, 50.0)
    assert out["field_colors"] == {"glucose": "#000"}
    assert out["n_snapshots"] == 5 and out["filename"] == "x"


def test_resolve_runtime_from_state():
    from spatio_flux.runners.run_study import _resolve_runtime
    state = {"fields": {"glucose": 1.0, "acetate": 0.0},
             "diffusion": {"config": {"bounds": (50.0, 50.0), "n_bins": (10, 10)}}}
    ctx = _resolve_runtime(state)
    assert ctx["$bounds"] == (50.0, 50.0)
    assert ctx["$coordinates_corners"] == [[0, 0], [9, 9]]
    assert ctx["$coordinates_comets"] == [[0, 5], [5, 5], [9, 5]]
    assert ctx["$all_fields"] == ["glucose", "acetate"]
