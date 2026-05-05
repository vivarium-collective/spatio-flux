"""
Bare-highspy FBA solver — bypasses cobra/optlang for the inner LP loop.

Why this exists:
    cobra-via-optlang's GLPK takes ~0.55 ms per E. coli core LP solve;
    cobra-via-optlang's HiGHS ("hybrid") is ~2.8 ms because of the
    Python-side wrapping. Calling highspy directly with warm-start drops
    that to ~46 µs/solve — comparable to COMETS's GLOP.

What this is:
    A minimal cobra-Model-shaped wrapper that intercepts only the surface
    ``run_fba_update`` actually touches:

      - ``model.reactions.get_by_id(rxn_id).{lower_bound, upper_bound}`` (R/W)
      - ``model.optimize()`` returning ``.status``, ``.objective_value``, ``.fluxes[rxn_id]``

    The LP structure (S, c) is built once at construction; subsequent
    bound updates touch only the changed columns; the underlying
    ``highspy.Highs`` instance keeps its simplex basis between solves so
    each re-solve is a warm restart.

Usage (set the solver sentinel in MODEL_REGISTRY_DFBA or DynamicFBA config)::

    cfg = dict(MODEL_REGISTRY_DFBA["ecoli core"], solver="highs_direct")

    # internally, DynamicFBA does roughly:
    cobra_model = load_fba_model(cfg["model_file"], cfg["bounds"])
    self.model = HiGHSFBASolver(cobra_model)

The wrapper holds a reference to the underlying cobra model so other
cobra-aware code paths (e.g. test inspection) still work, but they're
not required.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:
    import highspy
    _HIGHSPY_IMPORT_ERROR = None
except ImportError as _e:
    highspy = None  # type: ignore[assignment]
    _HIGHSPY_IMPORT_ERROR = _e


def _require_highspy():
    if highspy is None:
        raise ImportError(
            "HiGHSFBASolver requires highspy. Install with: pip install highspy"
        ) from _HIGHSPY_IMPORT_ERROR


class _RxnProxy:
    """Per-reaction view for ``model.reactions.get_by_id(...)``.

    Bound writes mutate the parent solver's bound arrays and mark the
    column dirty so the next ``optimize()`` flushes the change to highspy.
    Reads come from those same arrays — fast, no highspy round-trip.
    """
    __slots__ = ("_p", "_j", "id")

    def __init__(self, parent: "HiGHSFBASolver", j: int, rxn_id: str):
        self._p = parent
        self._j = j
        self.id = rxn_id

    @property
    def lower_bound(self) -> float:
        return float(self._p._lb[self._j])

    @lower_bound.setter
    def lower_bound(self, v: float) -> None:
        if self._p._lb[self._j] != v:
            self._p._lb[self._j] = float(v)
            self._p._dirty.add(self._j)

    @property
    def upper_bound(self) -> float:
        return float(self._p._ub[self._j])

    @upper_bound.setter
    def upper_bound(self, v: float) -> None:
        if self._p._ub[self._j] != v:
            self._p._ub[self._j] = float(v)
            self._p._dirty.add(self._j)


class _RxnLookup:
    """Tiny ``model.reactions``-shaped lookup. ``get_by_id`` only."""
    __slots__ = ("_p",)

    def __init__(self, parent: "HiGHSFBASolver"):
        self._p = parent

    def get_by_id(self, rxn_id: str) -> _RxnProxy:
        try:
            j = self._p._rxn_idx[rxn_id]
        except KeyError as e:
            raise KeyError(f"reaction id not in HiGHSFBASolver: {rxn_id!r}") from e
        return _RxnProxy(self._p, j, rxn_id)


class _FluxView:
    """``solution.fluxes[rxn_id]`` ⇒ scalar. Backed by the col_value vector."""
    __slots__ = ("_v", "_idx")

    def __init__(self, v: np.ndarray, idx: dict[str, int]):
        self._v = v
        self._idx = idx

    def __getitem__(self, key: str) -> float:
        return float(self._v[self._idx[key]])


class _Solution:
    __slots__ = ("status", "objective_value", "fluxes")

    def __init__(self, status: str, obj: float, fluxes: _FluxView):
        self.status = status
        self.objective_value = obj
        self.fluxes = fluxes


# Statuses that mean "optimal solution available" in highspy. Anything
# else maps to "infeasible" for run_fba_update's purposes — it just wants
# to know whether to read fluxes or zero out the deltas.
_OPTIMAL_STATUSES: tuple[Any, ...] = ()


def _populate_optimal_statuses() -> None:
    """Lazy: highspy needs to be importable before we can name its enums."""
    global _OPTIMAL_STATUSES
    if _OPTIMAL_STATUSES or highspy is None:
        return
    s = (highspy.HighsModelStatus.kOptimal,)
    # Some HiGHS builds also expose kSolveError / kUnbounded etc. — only
    # kOptimal counts as "fluxes are meaningful".
    _OPTIMAL_STATUSES = s


class HiGHSFBASolver:
    """Drop-in cobra-Model-shaped FBA solver backed by highspy.

    Built once from a cobra Model, then maintains its own bound state and
    a long-lived ``highspy.Highs`` instance for warm-started re-solves.
    Use one solver per process (or per shard actor); not threadsafe across
    threads sharing the same instance.
    """

    def __init__(self, cobra_model: Any):
        _require_highspy()
        _populate_optimal_statuses()

        rxns = list(cobra_model.reactions)
        mets = list(cobra_model.metabolites)
        self._n_rxns = len(rxns)
        self._n_mets = len(mets)
        self._rxn_idx: dict[str, int] = {r.id: i for i, r in enumerate(rxns)}
        self._met_idx: dict[str, int] = {m.id: i for i, m in enumerate(mets)}

        self._lb = np.array([r.lower_bound for r in rxns], dtype=np.float64)
        self._ub = np.array([r.upper_bound for r in rxns], dtype=np.float64)
        self._dirty: set[int] = set()

        c = np.array([r.objective_coefficient for r in rxns], dtype=np.float64)

        # Sparse stoichiometric matrix in CSC (highspy's column-wise format).
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for j, rxn in enumerate(rxns):
            for met, coef in rxn.metabolites.items():
                rows.append(self._met_idx[met.id])
                cols.append(j)
                vals.append(float(coef))
        # CSC indptr from per-column bucketing (avoids scipy dep).
        col_counts = np.zeros(self._n_rxns + 1, dtype=np.int32)
        for j in cols:
            col_counts[j + 1] += 1
        indptr = np.cumsum(col_counts).astype(np.int32)
        indices = np.empty(len(vals), dtype=np.int32)
        data = np.empty(len(vals), dtype=np.float64)
        write_at = indptr[:-1].copy()
        for r, c_, v in zip(rows, cols, vals):
            k = write_at[c_]
            indices[k] = r
            data[k] = v
            write_at[c_] += 1

        lp = highspy.HighsLp()
        lp.num_col_ = self._n_rxns
        lp.num_row_ = self._n_mets
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = c
        lp.col_lower_ = self._lb.copy()
        lp.col_upper_ = self._ub.copy()
        # FBA is steady-state: S v = 0
        lp.row_lower_ = np.zeros(self._n_mets, dtype=np.float64)
        lp.row_upper_ = np.zeros(self._n_mets, dtype=np.float64)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = indptr
        lp.a_matrix_.index_ = indices
        lp.a_matrix_.value_ = data

        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", False)
        self._h.passModel(lp)

        self.reactions = _RxnLookup(self)
        # keep a back-reference; some external code may want it. Optional.
        self._cobra_model = cobra_model

    # cobra Model.copy() is called by load_fba_model — but at the point
    # we build the wrapper, the cobra model has already been copied and
    # had bounds applied. The HiGHS solver itself is a fresh object that
    # doesn't need cobra-style copying; processes that need a private
    # solver should construct a new HiGHSFBASolver from the cobra model.
    def copy(self) -> "HiGHSFBASolver":
        return HiGHSFBASolver(self._cobra_model)

    # cobra exposes `.solver` so external code (and tests) can probe it;
    # mirror with an informative string.
    @property
    def solver(self) -> str:
        return "highs_direct"

    def _flush(self) -> None:
        if not self._dirty:
            return
        # changeColBounds is a single-col call; for the typical dFBA case
        # only 1-3 reactions change per tick (substrate uptake bounds) so
        # the per-call overhead is negligible. Bulk path
        # (changeColsBounds) exists if profiles ever push us toward it.
        for j in self._dirty:
            self._h.changeColBounds(int(j), float(self._lb[j]), float(self._ub[j]))
        self._dirty.clear()

    def optimize(self) -> _Solution:
        self._flush()
        self._h.run()
        status = self._h.getModelStatus()
        if status not in _OPTIMAL_STATUSES:
            # Match cobra's "infeasible" code path: fluxes vector is
            # not meaningful but keep the shape so callers can index.
            return _Solution(
                "infeasible",
                0.0,
                _FluxView(np.zeros(self._n_rxns), self._rxn_idx),
            )
        info = self._h.getInfo()
        sol = self._h.getSolution()
        return _Solution(
            "optimal",
            float(info.objective_function_value),
            _FluxView(np.asarray(sol.col_value, dtype=np.float64), self._rxn_idx),
        )
