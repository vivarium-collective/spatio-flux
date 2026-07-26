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
