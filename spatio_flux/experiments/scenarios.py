"""Compatibility surface bridging the retired ``test_suite.py`` to the study +
analysis-flush infrastructure.

``figure_A.py`` and ``figure_B.py`` import ``SIMULATIONS`` and a handful of run
helpers from here so they only had to change their *import source*, not their
whole structure. Each ``SIMULATIONS[slug]`` entry carries:

  generator    -- dotted composite ref (from scaffold_studies.STUDIES ``ref``)
  overrides    -- composite params    (from scaffold_studies.STUDIES ``params``)
  time         -- per-slug runtime     (from scaffold_studies._runtime_for)
  plot_func    -- callable(results, state, config) that resolves the slug's
                  FLUSH_SPEC ``$``-placeholders against the run state and runs
                  the post-sim analysis flush (writing the figure PNGs/GIFs).
  plot_config  -- {"out_dir": ...} consumed by ``plot_func`` (default "out").

Callers build the composite document themselves via
``build_generator(_REGISTRY[entry["generator"]], overrides=entry["overrides"],
core=core)`` and unpack ``run_composite_document(...)[0]`` for the trajectory.
"""
from __future__ import annotations

from process_bigraph import allocate_core

from spatio_flux.library.tools import prepare_output_dir, run_composite_document
from spatio_flux.analysis.flush import run_analysis_flush
from spatio_flux.analysis.flush_spec import FLUSH_SPEC
from spatio_flux.runners.run_study import _resolve_runtime, _apply
from scripts.scaffold_studies import STUDIES, _runtime_for

__all__ = [
    "SIMULATIONS",
    "DEFAULT_RUNTIME_LONG",
    "allocate_core",
    "prepare_output_dir",
    "run_composite_document",
]

# Was a module constant in the retired test_suite.py.
DEFAULT_RUNTIME_LONG = 60


def _make_plot_func(slug):
    """Build the (results, state, config) plot callable for one scenario.

    Resolves the slug's FLUSH_SPEC ``$``-placeholders against the run state
    (reusing run_study's resolver) and runs the analysis flush into ``out_dir``.
    """
    def plot_func(results, state, config=None):
        config = config or {}
        out_dir = config.get("out_dir", "out")
        ctx = _resolve_runtime(state)
        specs = [{"step": s["step"], "config": _apply(s["config"], ctx)}
                 for s in FLUSH_SPEC[slug]]
        return run_analysis_flush(slug, results, state, out_dir, specs)

    return plot_func


def _build_simulations():
    sims = {}
    for entry in STUDIES:
        slug = entry["slug"]
        if slug not in FLUSH_SPEC:
            continue
        sims[slug] = {
            "generator": entry["ref"],
            "overrides": entry.get("params", {}),
            "time": _runtime_for(slug),
            "plot_func": _make_plot_func(slug),
            "plot_config": {"out_dir": "out"},
        }
    return sims


SIMULATIONS = _build_simulations()
