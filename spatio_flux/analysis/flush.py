"""Post-simulation analysis flush.

Replays a run's trajectory through registered analysis steps that write figure
files. This is the post-sim flush the study runner calls after the composite
finishes — visualizations are NOT part of the composite.

An analysis step is ``callable(trajectory, state, out_dir, config) -> list[str]``
returning the paths it wrote. Steps register into ``ANALYSIS_STEPS`` by name via
:func:`register_step`; the study.yaml FLUSH_SPEC picks steps by that name.
"""
import os

# name -> callable(trajectory, state, out_dir, config) -> list[str] (paths written)
ANALYSIS_STEPS = {}


def register_step(name):
    def deco(fn):
        ANALYSIS_STEPS[name] = fn
        return fn
    return deco


def run_analysis_flush(slug, trajectory, state, out_dir, specs):
    """Run each spec's analysis step against the trajectory; return written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for spec in specs:
        fn = ANALYSIS_STEPS[spec["step"]]
        written.extend(fn(trajectory, state, out_dir, spec.get("config", {})) or [])
    return written
