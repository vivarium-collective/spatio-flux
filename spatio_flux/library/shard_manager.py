"""
ShardManager — explicit-lifecycle pool of dFBA actors for sharded
compositional simulation.

Why this exists (the architectural finding)
-------------------------------------------
``process_bigraph.protocols.ray.RayProcess`` keys its actor pool on
``(process_class, hash(process_config))`` so that many clients with
*identical* config reuse one pool. That works perfectly for the
per-cell DFBA layout in this repo — 4096 cells, 4096 RayProcess
instances, identical config → 1 pool of N actors → cells round-robin.

For a *sharded* layout (each shard owns a different list of cells),
each shard's process_config is unique. Under upstream's keying scheme
this collapses to:

  - N_shards pools instead of one shared pool, each with one actor
    (memory: N×cobra_model — leaks past pool reuse boundaries),
  - serialized actor cold-starts: ``RayProcess.initialize()`` does
    ``ray.get(actor.inputs.remote())`` per shard, which blocks on that
    actor's ``__init__``; with cobra's textbook model loading at
    ~2-3 s/actor, 12 shards = ~30 s of wall-time *before any solve*,
  - module-global ``_POOLS`` never reaped until ``shutdown_pools()`` —
    repeated sweeps in one process accumulate hundreds of MB.

This module fixes that by going around ``RayProcess`` for the shard
path:

  - ``ShardManager`` is a context manager that owns its own actors.
  - Spawning ``N`` actors via ``[Actor.remote(...) for _ in range(N)]``
    is non-blocking; the first ``parallel_processes`` tick then races
    all N ``__init__`` calls in parallel, paying ~3 s once instead of
    ~3N s sequentially.
  - On ``__exit__`` every actor is killed and its registry slot freed.
    No leaked state across sweeps.
  - Each shard surfaces in the Composite as a tiny local Process
    (``_ShardFacade``) that holds an integer ``shard_slot`` and looks
    up its actor in the module registry — no pool dedup in the path.

Usage:

    with ShardManager(model_id="ecoli core", n=64, n_shards=16,
                      solver="highs_direct", mol_ids=mol_ids) as mgr:
        state = {
            **mgr.process_specs(interval=0.1),
            "fields": ...,
            "diffusion": ...,
        }
        sim = Composite({"state": state, "parallel_processes": True}, core=core)
        sim.run(4.0)
        results = gather_emitter_results(sim)
    # actors killed here, regardless of exception
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    import ray
    _RAY_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:
    ray = None  # type: ignore[assignment]
    _RAY_IMPORT_ERROR = _e

from process_bigraph import Process, allocate_core


def _require_ray() -> None:
    if ray is None:
        raise ImportError(
            "ShardManager requires the optional `ray` dependency. "
            "Install with: pip install process-bigraph[ray]"
        ) from _RAY_IMPORT_ERROR


# The ``@ray.remote`` decorator is applied lazily so this module
# imports cleanly even when ray isn't installed.
_ShardActor = None
def _shard_actor_class():
    global _ShardActor
    if _ShardActor is not None:
        return _ShardActor
    _require_ray()

    @ray.remote
    class ShardActor:
        """Long-lived actor owning one ShardedDFBA — single cobra
        model + (optionally) a HiGHS warm-started solver. Receives the
        shard's per-cell state per tick, returns deltas. Persistent
        state across ticks for warm-started simplex bases."""

        def __init__(self, shard_config: dict):
            from spatio_flux.processes.dfba import ShardedDFBA
            self.proc = ShardedDFBA(shard_config, core=allocate_core())

        def update(self, inputs: dict, interval: float) -> dict:
            return self.proc.update(inputs, float(interval))

        def ping(self) -> str:
            """Pre-warm probe — ensures __init__ has completed."""
            return "ready"

    _ShardActor = ShardActor
    return _ShardActor


# Module registry: maps shard_slot (int) → ray actor handle. Manager
# owns the lifecycle of its slots; _ShardFacade looks up its actor by
# slot at initialize time. Globals scoped per Python process.
_SHARD_REGISTRY: dict[int, Any] = {}
_NEXT_SLOT = 0


def _alloc_slot() -> int:
    global _NEXT_SLOT
    s = _NEXT_SLOT
    _NEXT_SLOT += 1
    return s


class _ShardFacade(Process):
    """Local Process delegating ``update()`` to a Ray actor identified
    by integer ``shard_slot``. Bypasses RayProcess and its pool dedup."""

    config_schema = {
        "shard_slot": "integer",
        "cell_keys": "list[string]",
    }

    def initialize(self, config):
        self.cell_keys = list(config["cell_keys"])
        self._slot = int(config["shard_slot"])
        self._actor = _SHARD_REGISTRY.get(self._slot)
        if self._actor is None:
            raise RuntimeError(
                f"_ShardFacade: no actor registered for slot "
                f"{self._slot}. The ShardManager that owns it must be "
                f"alive when the Composite is constructed."
            )

    def inputs(self):
        return {
            "cells": {
                k: {"substrates": "map[concentration]", "biomass": "mass"}
                for k in self.cell_keys
            }
        }

    def outputs(self):
        return {
            "cells": {
                k: {"substrates": "map[count]", "biomass": "mass"}
                for k in self.cell_keys
            }
        }

    def update(self, inputs, interval):
        _require_ray()
        return ray.get(self._actor.update.remote(inputs, float(interval)))


def _stripe_assignment(n: int, n_shards: int) -> list[list[tuple[int, int]]]:
    """Stripe-partition n*n cells across n_shards. Topology-agnostic:
    cells 0, n_shards, 2*n_shards, … land in shard 0; etc. Returns the
    non-empty shards (drops empties when n_shards > n*n)."""
    cells = [(y, x) for y in range(n) for x in range(n)]
    out: list[list[tuple[int, int]]] = [[] for _ in range(n_shards)]
    for i, c in enumerate(cells):
        out[i % n_shards].append(c)
    return [s for s in out if s]


class ShardManager:
    """One-sweep pool of long-lived dFBA Ray actors with explicit
    lifecycle. Use as a context manager.

    Parameters
    ----------
    model_id : str
        Key into MODEL_REGISTRY_DFBA (e.g. "ecoli core").
    n : int
        Grid edge length. Cells are assigned to shards by linear index.
    n_shards : int, optional
        Number of shards (= actors). Defaults to ``min(n*n, cpu_count)``;
        for a remote cluster, set explicitly to the cluster's vCPU count.
    solver : str, optional
        Cobra solver name or "highs_direct" (the bare-highspy wrapper).
    mol_ids : list[str]
        Substrate/biomass field names. Defaults to the e. coli core
        triple ("glucose", "acetate", "dissolved biomass").
    biomass_id : str
        Which mol_id is the biomass field. Default "dissolved biomass".
    ray_address : str, optional
        Ray head address. ``None`` lazy-inits a local instance;
        ``"auto"`` discovers a local one; ``host:port`` connects remote.
    """

    def __init__(
        self,
        model_id: str,
        n: int,
        n_shards: Optional[int] = None,
        solver: Optional[str] = None,
        mol_ids: Optional[list[str]] = None,
        biomass_id: str = "dissolved biomass",
        ray_address: Optional[str] = None,
    ):
        _require_ray()
        if not ray.is_initialized():
            if ray_address:
                ray.init(address=ray_address, log_to_driver=False)
            else:
                ray.init(ignore_reinit_error=True, log_to_driver=False)

        if n_shards is None:
            n_shards = min(n * n, max(1, os.cpu_count() or 4))
        self._shards = _stripe_assignment(n, n_shards)
        self._n_shards = len(self._shards)

        from spatio_flux.processes.dfba import MODEL_REGISTRY_DFBA
        cfg = dict(MODEL_REGISTRY_DFBA[model_id])
        if solver is not None:
            cfg["solver"] = solver
        self._cfg = cfg

        self._mol_ids = list(mol_ids) if mol_ids else [
            "glucose", "acetate", biomass_id,
        ]
        self._biomass_id = biomass_id

        self._slots: list[int] = []
        self._actors: list[Any] = []
        self._entered = False

    # -- lifecycle --------------------------------------------------- #

    def __enter__(self) -> "ShardManager":
        actor_cls = _shard_actor_class()
        # Spawn all actors concurrently. .remote() is non-blocking; the
        # __init__ work happens in parallel on each actor.
        for shard_cells in self._shards:
            keys = [f"c_{y}_{x}" for (y, x) in shard_cells]
            shard_cfg = {**self._cfg, "cell_keys": keys}
            actor = actor_cls.remote(shard_cfg)
            slot = _alloc_slot()
            _SHARD_REGISTRY[slot] = actor
            self._slots.append(slot)
            self._actors.append(actor)
        # Pre-warm: race all __init__'s in parallel so the first
        # sim.run() doesn't pay sequential cold-start. ray.get on a
        # list of futures is the canonical "wait for all in parallel".
        ray.get([a.ping.remote() for a in self._actors])
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for slot, actor in zip(self._slots, self._actors):
            try:
                ray.kill(actor)
            except Exception:
                pass
            _SHARD_REGISTRY.pop(slot, None)
        self._slots = []
        self._actors = []
        self._entered = False
        # Don't suppress exceptions.
        return None

    # -- composite spec ---------------------------------------------- #

    def process_specs(self, interval: float) -> dict:
        """Return the dict of shard-Process specs to splice into the
        Composite document. Each entry wires this shard's cells into
        ``["fields", mol, y, x]`` paths and routes its update to the
        manager's owned actor at that slot."""
        if not self._entered:
            raise RuntimeError(
                "ShardManager.process_specs called before __enter__. "
                "Use 'with ShardManager(...) as mgr:'."
            )
        substrate_mols = [m for m in self._mol_ids if m != self._biomass_id]

        procs = {}
        for i, (slot, shard_cells) in enumerate(zip(self._slots, self._shards)):
            keys = [f"c_{y}_{x}" for (y, x) in shard_cells]
            cells_inputs: dict = {}
            cells_outputs: dict = {}
            for (y, x), key in zip(shard_cells, keys):
                wires = {
                    "substrates": {
                        m: ["fields", m, y, x] for m in substrate_mols
                    },
                    "biomass": ["fields", self._biomass_id, y, x],
                }
                cells_inputs[key] = wires
                cells_outputs[key] = wires

            procs[f"dFBA_shard[{i}]"] = {
                "_type": "process",
                "address": "local:_ShardFacade",
                "config": {
                    "shard_slot": int(slot),
                    "cell_keys": keys,
                },
                "inputs": {"cells": cells_inputs},
                "outputs": {"cells": cells_outputs},
                "interval": float(interval),
            }
        return procs

    @property
    def n_shards(self) -> int:
        return self._n_shards

    @property
    def shards(self) -> list[list[tuple[int, int]]]:
        return list(self._shards)
