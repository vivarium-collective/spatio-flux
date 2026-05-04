"""
[DEPRECATED] The ``parallel_processes`` flag is now part of upstream
``process_bigraph.Composite``. This shim is kept for compatibility with
existing imports.

Replace::

    from spatio_flux.library.parallel_composite import ParallelComposite
    sim = ParallelComposite({"state": state}, core=core)

With::

    from process_bigraph import Composite
    sim = Composite({"state": state, "parallel_processes": True}, core=core)
"""

from process_bigraph import Composite


class ParallelComposite(Composite):
    """Thin compatibility shim — sets ``parallel_processes=True`` on Composite."""

    def __init__(self, document, *args, **kwargs):
        if isinstance(document, dict) and "parallel_processes" not in document:
            document = {**document, "parallel_processes": True}
        super().__init__(document, *args, **kwargs)
