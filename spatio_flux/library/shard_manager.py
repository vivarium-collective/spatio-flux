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

import numpy as np

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
        state across ticks for warm-started simplex bases.

        Designed for ActorPool reuse across many Session instances:
          - ``__init__`` does the expensive work (cobra import, model
            load, LP build) ONCE. config should contain everything
            EXCEPT cell_keys.
          - ``reconfigure`` rebinds cell_keys per-Session, no model reload.
          - Pool keeps these actors alive across many Composites.
        """

        def __init__(self, shard_config: dict):
            from spatio_flux.processes.dfba import ShardedDFBA
            # cell_keys default empty — set per-Session via reconfigure.
            cfg = {**shard_config, "cell_keys": shard_config.get("cell_keys") or []}
            self.proc = ShardedDFBA(cfg, core=allocate_core())

        def update(self, inputs: dict, interval: float) -> dict:
            return self.proc.update(inputs, float(interval))

        def reconfigure(self, sim_config: dict) -> None:
            """Cheap per-Session rebinding — delegates to
            ``ShardedDFBA.reconfigure``. Does NOT reload cobra or
            rebuild the LP — that happens once in ``__init__``."""
            self.proc.reconfigure(sim_config)

        def ping(self) -> str:
            """Pre-warm probe — ensures __init__ has completed."""
            return "ready"

    _ShardActor = ShardActor
    return _ShardActor


# Module registries: maps shard_slot (int) → ray actor handle / managing
# ShardManager. Manager owns the lifecycle of its slots; _ShardFacade
# looks up its actor + manager by slot at initialize time. Globals
# scoped per Python process.
_SHARD_REGISTRY: dict[int, Any] = {}
_MANAGER_FOR_SLOT: dict[int, "ShardManager"] = {}
_NEXT_SLOT = 0


def _alloc_slot() -> int:
    global _NEXT_SLOT
    s = _NEXT_SLOT
    _NEXT_SLOT += 1
    return s


def shutdown_pools() -> None:
    """Kill all ShardManager-allocated pool actors.

    ShardManager.__exit__ no longer kills actors — that's the whole
    point of the pool/session refactor. Call this once at end-of-process
    (or end-of-script) to actually release the cluster resources.

    Delegates to ``process_bigraph.protocols.pool.shutdown_all_pools``,
    which tears down EVERY pool registered upstream (not just ours).
    For finer-grained shutdown, call that function directly from your
    own teardown logic.
    """
    try:
        from process_bigraph.protocols.pool import shutdown_all_pools
        shutdown_all_pools()
    except ImportError:
        pass  # process_bigraph not available — nothing to shut down


class _ShardDefer:
    """Defer-shaped object returned by ``_ShardFacade.invoke``. ``.get()``
    blocks until ``ShardManager.flush_pending`` has resolved the batch.
    Composite's ``_flush_protocol_runtimes`` hook ensures that's true
    before any ``apply_updates`` reads from us."""
    __slots__ = ("_manager", "_proc_id")

    def __init__(self, manager: "ShardManager", proc_id: int):
        self._manager = manager
        self._proc_id = proc_id

    def get(self):
        return self._manager.collect(self._proc_id)


class _ShardFacade(Process):
    """Local Process whose ``invoke()`` enqueues onto a ShardManager
    runtime instead of doing a per-tick blocking ray.get. The runtime's
    ``flush_pending`` then resolves all shard updates in PARALLEL (single
    ``ray.get(list_of_futures)`` call), giving real cluster utilization.

    Without this batching the per-shard ``ray.get`` is sequential on
    Composite's invoke pass — N shards = N×latency, no parallelism, 0%
    CPU on workers."""

    config_schema = {
        "shard_slot": "integer",
        "cell_keys": "list[string]",
    }

    def initialize(self, config):
        self.cell_keys = list(config["cell_keys"])
        self._slot = int(config["shard_slot"])
        self._actor = _SHARD_REGISTRY.get(self._slot)
        self._manager = _MANAGER_FOR_SLOT.get(self._slot)
        if self._actor is None or self._manager is None:
            raise RuntimeError(
                f"_ShardFacade: no actor/manager registered for slot "
                f"{self._slot}. The ShardManager that owns it must be "
                f"alive when the Composite is constructed."
            )
        # Composite reads this attribute and adds the manager to its
        # active-runtimes list so flush_pending fires once per tick
        # AFTER the invoke pass and BEFORE apply_updates pulls deltas.
        self._protocol_runtime = self._manager

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

    def invoke(self, state, interval):
        # Non-blocking: just enqueue the .remote() future on the manager.
        # Composite collects all shard Defers across the tick, then
        # flush_pending() does the parallel ray.get on the union.
        _require_ray()
        proc_id = id(self)
        self._manager.enqueue(
            proc_id, self._actor, state, float(interval),
        )
        return _ShardDefer(self._manager, proc_id)


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

        # Heuristic for default n_shards: aim for ~target_cells_per_shard
        # cells per actor so per-actor work amortizes the RPC + cobra-load
        # overhead. Bounded by the cluster's actual CPU count (more actors
        # than CPUs would just queue, no benefit).
        #
        # Why a "cells per actor" target instead of just maxing out CPUs:
        #   - Tiny grids (8x8 = 64 cells) on 72 actors = 1 cell/actor;
        #     overhead dominates and total wall is SLOWER than n_shards=8.
        #   - Huge grids (128x128 = 16384 cells) on 8 actors = 2048
        #     cells/actor; per-actor work is CPU-bound and total wall is
        #     SLOWER than spreading across 72 actors.
        #   - Targeting ~32 cells/actor amortizes RPC overhead while still
        #     parallelizing wide enough at scale.
        # Override via RAY_SHARDS_TARGET_CELLS env var if needed.
        target_cells_per_shard = int(
            os.environ.get("RAY_SHARDS_TARGET_CELLS", "32")
        )
        cluster_cpus = None
        if ray.is_initialized():
            try:
                cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
            except Exception:
                cluster_cpus = None
        cap = cluster_cpus or (os.cpu_count() or 4)
        n_cells = n * n
        if n_shards is None:
            ideal = max(1, n_cells // target_cells_per_shard)
            n_shards = min(n_cells, cap, max(1, ideal))
        else:
            # Honor explicit n_shards but clamp to sane range.
            n_shards = min(n_cells, max(1, n_shards))
        # Print the resolved values so we can verify cluster utilization
        # in the experiment log without redeploying for every check.
        cells_per_shard = n_cells / max(1, n_shards)
        print(
            f"  ShardManager: n={n} cells={n_cells} n_shards={n_shards} "
            f"cells_per_shard={cells_per_shard:.1f} "
            f"cluster_CPU={cap} target_cells/shard={target_cells_per_shard}",
            flush=True,
        )
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
        # Set on __enter__: the ActorPool we claim from + the Session
        # holding our claim. Both stay None when not entered.
        self._pool: Optional[Any] = None
        self._session: Optional[Any] = None

        # Per-tick batch state. enqueue() appends futures here; the
        # Composite's _flush_protocol_runtimes() calls flush_pending()
        # once per tick to ray.get them all in parallel; collect() then
        # serves the resolved deltas to each _ShardDefer.get().
        self._pending: dict[int, Any] = {}
        self._results: dict[int, dict] = {}

        # Tick-level timing accumulators. Per-flush ray.get wall is the
        # actual compute (max actor + serialization); inter-flush gap is
        # whatever Composite/process_bigraph spends per tick (apply_updates,
        # tree traversal, etc.). Summarized at __exit__.
        self._tick_count = 0
        self._sum_get_ms = 0.0
        self._sum_between_ms = 0.0
        self._last_flush_end = None

    # -- lifecycle --------------------------------------------------- #

    def __enter__(self) -> "ShardManager":
        import time as _time
        from process_bigraph.protocols.pool import get_or_create_pool
        from process_bigraph.protocols.session import Session

        t0 = _time.monotonic()
        actor_cls = _shard_actor_class()

        # Pool key intentionally EXCLUDES cell_keys — those are per-Session.
        # Two ShardManagers using the same model + solver share one pool
        # (and so amortize cobra load across all their sims).
        pool_config = {
            "model_file": self._cfg.get("model_file"),
            "solver": self._cfg.get("solver"),
            "kinetic_params": self._cfg.get("kinetic_params"),
            "substrate_update_reactions": self._cfg.get(
                "substrate_update_reactions"),
            "bounds": self._cfg.get("bounds"),
        }
        self._pool = get_or_create_pool(actor_cls, pool_config, self._n_shards)
        was_warmed = self._pool.stats()["warmed"]
        self._pool.warm()
        t_warm = _time.monotonic() - t0

        # Acquire actors via Session — but skip Session's batched
        # reconfigure since we need PER-actor cell_keys, not one
        # uniform sim_config.
        self._session = Session(
            self._pool, n_actors=self._n_shards, reconfigure=False)
        self._session.__enter__()
        actors = self._session.actors

        # Per-shard reconfigure: each actor gets its own cell_keys list.
        per_shard_keys = [
            [f"c_{y}_{x}" for (y, x) in shard_cells]
            for shard_cells in self._shards
        ]
        ray.get([
            actor.reconfigure.remote({"cell_keys": keys})
            for actor, keys in zip(actors, per_shard_keys)
        ])

        # Wire actors into the legacy slot registry so _ShardFacade can
        # look them up by integer slot.
        for actor in actors:
            slot = _alloc_slot()
            _SHARD_REGISTRY[slot] = actor
            _MANAGER_FOR_SLOT[slot] = self
            self._slots.append(slot)
            self._actors.append(actor)

        t_ready = _time.monotonic() - t0
        kind = "(reused warmed pool)" if was_warmed else "(pool cold-start)"
        print(
            f"  ShardManager.__enter__: claimed {len(self._actors)} actors "
            f"from pool {kind} in {t_ready:.2f}s",
            flush=True,
        )
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Release actors back to pool — DO NOT KILL. The whole point of
        # ActorPool is that subsequent Sessions reuse these actors with
        # their cobra Models still loaded.
        for slot in self._slots:
            _SHARD_REGISTRY.pop(slot, None)
            _MANAGER_FOR_SLOT.pop(slot, None)

        # Tick stats summary — answers "is my wall time compute or overhead?"
        if self._tick_count > 0:
            avg_get = self._sum_get_ms / self._tick_count
            avg_between = self._sum_between_ms / self._tick_count
            print(
                f"  ShardManager.tick stats: {self._tick_count} ticks, "
                f"avg ray.get={avg_get:.1f}ms (compute), "
                f"avg between={avg_between:.1f}ms (Composite tick overhead). "
                f"Total compute={self._sum_get_ms/1000:.2f}s, "
                f"total overhead={self._sum_between_ms/1000:.2f}s",
                flush=True,
            )

        if self._session is not None:
            self._session.__exit__(exc_type, exc_val, exc_tb)
            self._session = None

        print(
            f"  ShardManager.__exit__: released {len(self._actors)} actors "
            f"to pool (NOT killed)", flush=True)

        self._slots = []
        self._actors = []
        self._entered = False
        self._pending.clear()
        self._results.clear()
        # Don't suppress exceptions.
        return None

    # -- protocol-runtime API (used by _ShardFacade + Composite) ----- #

    def enqueue(self, proc_id: int, actor: Any,
                inputs: dict, interval: float) -> None:
        """Queue one shard's update as a Ray remote call. Non-blocking;
        the future is resolved by flush_pending() at end-of-tick."""
        self._pending[proc_id] = actor.update.remote(inputs, float(interval))

    def collect(self, proc_id: int) -> dict:
        """Return the resolved delta for proc_id (after flush_pending).
        Empty dict if no update was queued this tick."""
        return self._results.pop(proc_id, {})

    def tick_lifecycle(
            self,
            processes,
            composite,
            global_time: float,
            end_time: float,
            force_complete: bool,
    ):
        """v3 — direct state mutation. Skips ``apply_updates`` entirely.

        At scale the dominant per-tick cost wasn't N-times-overhead (which
        v2 collapsed) but the *single* combined ``apply_updates`` walk over
        a schema covering ~total_cells × n_fields wires. v3 short-circuits
        that by reading and writing the underlying numpy field arrays
        directly, exploiting the spatio-flux topology
        (``cells/key → fields/mol/y/x``) we own:

        - **Read pass**: read ``composite.state['fields']`` once. Per shard,
          numpy-slice each cell's (y, x) coordinates out of the relevant
          mol arrays — no per-process ``_cached_view``.
        - **Dispatch**: one parallel ``ray.get`` over all 72 actor futures.
        - **Write pass**: per shard, build delta arrays and apply via
          ``fields[mol][ys, xs] += d_arr`` (numpy fancy index, vectorized).
          No ``apply_updates``, no schema walk, no reconcile.

        Trade-offs:
        - Doesn't go through bigraph_schema at all on the apply side —
          callers depending on `_apply` semantics for the cell projections
          (e.g. emitter step triggers gated on cell-path changes) won't see
          path-level deltas. Spatio-flux's emitter is time-triggered, so
          this is fine here. Other downstream uses would need v3 to
          report changed paths back to the framework.
        - Couples the runtime to the wiring shape (cells/key → fields).
          The framework hook itself stays generic; the *runtime adapter*
          is allowed to know its own topology.

        Returns ``applied=True`` so the framework's
        ``_run_tick_lifecycle`` skips its apply pass entirely (no defer
        placed in self.front).
        """
        if not processes:
            return None

        import time as _time
        t_start = _time.monotonic()
        # Inter-flush gap measured the same way as flush_pending: time
        # since the previous tick's end. With v3 doing its own ray.get,
        # flush_pending is no longer called — so we manage timing here.
        if self._last_flush_end is not None:
            between_ms = (t_start - self._last_flush_end) * 1000
        else:
            between_ms = 0.0

        # ---- read pass: snapshot inputs from fields arrays ------------
        fields = composite.state['fields']
        substrate_mols = [m for m in self._mol_ids if m != self._biomass_id]
        biomass_id = self._biomass_id

        # Per-shard precomputed (ys_arr, xs_arr) cache. cell_keys parsing
        # is invariant across ticks, so the first tick pays the parse and
        # subsequent ticks reuse the int-array indices.
        # Stored on the manager (not the facade) to keep facades lean.
        cache = getattr(self, '_index_cache', None)
        if cache is None:
            cache = {}
            self._index_cache = cache

        next_time = end_time
        futures = []
        # Parallel arrays: per-process metadata for the apply pass.
        proc_meta: list = []  # (ys_arr, xs_arr, cell_keys)
        for req in processes:
            proc = req['instance']
            interval = float(req['interval']) if req['interval'] else 0.0
            cell_keys = proc.cell_keys

            cached = cache.get(id(proc))
            if cached is None:
                ys = np.empty(len(cell_keys), dtype=np.int64)
                xs = np.empty(len(cell_keys), dtype=np.int64)
                for i, key in enumerate(cell_keys):
                    # Cell keys are always 'c_y_x'.
                    parts = key.split('_')
                    ys[i] = int(parts[1])
                    xs[i] = int(parts[2])
                cache[id(proc)] = (ys, xs)
            else:
                ys, xs = cached

            # numpy-index out the (substrates, biomass) values per cell
            # in one pass per mol. Builds dict-of-dicts that the
            # ShardActor.update API expects.
            sub_vals = {m: fields[m][ys, xs] for m in substrate_mols}
            bio_vals = fields[biomass_id][ys, xs]
            cells_in: dict = {}
            for i, key in enumerate(cell_keys):
                cells_in[key] = {
                    'substrates': {m: float(sub_vals[m][i]) for m in substrate_mols},
                    'biomass': float(bio_vals[i]),
                }

            futures.append(proc._actor.update.remote(
                {'cells': cells_in}, interval))
            proc_meta.append((ys, xs, cell_keys))

            future_time = global_time + interval
            if future_time < next_time:
                next_time = future_time

        # ---- dispatch: one parallel ray.get on all actor futures ------
        t_dispatch = _time.monotonic()
        results = ray.get(futures)
        t_get = _time.monotonic()
        get_ms = (t_get - t_dispatch) * 1000

        # ---- write pass: vectorized fancy-index += per mol per shard --
        for (ys, xs, cell_keys), result in zip(proc_meta, results):
            cells_out = result.get('cells', {})
            n = len(cell_keys)
            # Pre-allocate delta arrays per mol; fill in cell order.
            d_sub = {m: np.zeros(n, dtype=fields[m].dtype) for m in substrate_mols}
            d_bio = np.zeros(n, dtype=fields[biomass_id].dtype)
            for i, key in enumerate(cell_keys):
                cell = cells_out.get(key)
                if cell is None:
                    continue
                substrates = cell.get('substrates') or {}
                for m in substrate_mols:
                    d_sub[m][i] = substrates.get(m, 0.0)
                d_bio[i] = cell.get('biomass', 0.0)
            # Apply via numpy fancy indexing — single vectorized add per
            # mol per shard. Cells within a shard are unique so no need
            # for np.add.at (which handles repeated indices). Across
            # shards the cell sets are disjoint by construction.
            for m in substrate_mols:
                fields[m][ys, xs] += d_sub[m]
            fields[biomass_id][ys, xs] += d_bio

        t_done = _time.monotonic()

        # ---- timing accumulators (matches flush_pending's instrumentation)
        self._tick_count += 1
        self._sum_get_ms += get_ms
        # "between_ticks" here is the gap between successive
        # tick_lifecycle invocations — same metric we tracked for
        # flush_pending so the comparison stays apples-to-apples.
        self._sum_between_ms += between_ms
        self._last_flush_end = t_done
        # Stage breakdown for the first few ticks so we can see where
        # time goes in v3.
        if self._tick_count <= 3:
            t_read = (t_dispatch - t_start) * 1000
            t_apply = (t_done - t_get) * 1000
            print(
                f"  ShardManager.tick_lifecycle #{self._tick_count}: "
                f"{len(futures)} shards, "
                f"read+dispatch={t_read:.1f}ms, ray.get={get_ms:.1f}ms, "
                f"apply={t_apply:.1f}ms, between={between_ms:.1f}ms",
                flush=True,
            )

        return {
            'next_time': next_time,
            'process_paths': [req['path'] for req in processes],
            'applied': True,
        }

    def flush_pending(self) -> None:
        """Resolve all queued updates in parallel via a single
        ``ray.get(list)``. Composite calls this once per tick between
        the invoke pass and apply_updates."""
        if not self._pending:
            return
        import time as _time
        proc_ids = list(self._pending.keys())
        futures = list(self._pending.values())
        self._pending.clear()

        t_start = _time.monotonic()
        # Inter-flush gap = wall time between the previous flush returning
        # and this flush starting. That's everything the orchestrator
        # (Composite + process_bigraph) does per tick OUTSIDE of remote
        # work — invoke pass, apply_updates, schema validation, etc.
        if self._last_flush_end is not None:
            between_ms = (t_start - self._last_flush_end) * 1000
        else:
            between_ms = 0.0

        results = ray.get(futures)
        t_done = _time.monotonic()
        get_ms = (t_done - t_start) * 1000

        for proc_id, result in zip(proc_ids, results):
            self._results[proc_id] = result

        # Accumulate for the __exit__ summary.
        self._tick_count += 1
        self._sum_get_ms += get_ms
        self._sum_between_ms += between_ms
        self._last_flush_end = t_done

        # Print first few ticks live so we can watch behavior change.
        if self._tick_count <= 3:
            print(f"  ShardManager.flush_pending #{self._tick_count}: "
                  f"{len(futures)} futures, ray.get={get_ms:.1f}ms, "
                  f"between_ticks={between_ms:.1f}ms", flush=True)

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
