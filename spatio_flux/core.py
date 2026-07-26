"""Workspace core builder.

The vivarium-workbench run path imports ``<package_path>.core.build_core``
(here ``spatio_flux.core.build_core``) to construct a bigraph-schema core with
this workspace's types, processes, visualizations, and composite generators
registered. ``allocate_core()`` auto-discovers installed process/emitter/
visualization Steps; ``register_types`` adds spatio-flux's custom types; the
composites import fires the ``@composite_generator`` decorators.
"""
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

import spatio_flux                    # exposes register_types; imports composites
import spatio_flux.visualizations     # noqa: F401  (viz Step discovery)


def build_core():
    core = allocate_core()
    spatio_flux.register_types(core)
    core.register_link("RAMEmitter", RAMEmitter)
    return core
