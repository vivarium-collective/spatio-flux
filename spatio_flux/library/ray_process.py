"""
[DEPRECATED] Moved upstream to ``process_bigraph.protocols.ray``.

This module re-exports the upstream symbols so existing spatio-flux callers
keep working. New code should import directly from process_bigraph::

    from process_bigraph.protocols.ray import (
        RayProcess, register_process_class, shutdown_pools, pool_stats,
    )

Install the ray extra to use this transport::

    pip install process-bigraph[ray]
"""

from process_bigraph.protocols.ray import (  # noqa: F401  (re-exports)
    RayProcess,
    register_process_class,
    get_registry,
    shutdown_pools,
    pool_stats,
)
