"""spatio-flux composite generators.

Importing this package triggers ``@composite_generator`` registration for
every composite in every submodule. The dashboard's ``discover_generators``
call relies on this.
"""
from pbg_superpowers.composite_generator import _REGISTRY as REGISTRY

# Trigger decorator side-effects in every submodule.
from . import metabolism  # noqa: F401
from . import spatial     # noqa: F401
from . import particles   # noqa: F401
from . import comets      # noqa: F401
from . import reference   # noqa: F401

__all__ = ["REGISTRY"]
