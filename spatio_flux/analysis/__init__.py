"""spatio-flux post-run analysis flush.

The flush harness lives in :mod:`spatio_flux.analysis.flush`; the file-writing
analysis steps that wrap ``spatio_flux.plots.plot`` primitives live in
:mod:`spatio_flux.analysis.steps` and register themselves on import.
"""
from spatio_flux.analysis.flush import (  # noqa: F401
    ANALYSIS_STEPS,
    register_step,
    run_analysis_flush,
)
from spatio_flux.analysis import steps as _steps  # noqa: F401  (register steps on import)
