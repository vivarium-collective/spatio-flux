"""
MAPK signalling cycle as a Bigraphical Reactive System
======================================================

This module pairs a generic simulation engine
(:class:`BigraphicalReactiveSystem`) with a worked example that
re-casts a textbook compartmentalised phosphorylation cycle as a
structural-rewriting bigraph. References for both the formalism
and the biology are collected in ``brs_mapk_references.md``.

The biology
-----------
The MAP-kinase cascade is the canonical signal-transduction relay
in mammalian cells. We model just its last catalytic step:

    free MEK + free ERK   →   MEK·pERK   →   free MEK + free pERK

* **MEK1** (MAP2K1) is a 43 kDa serine/threonine kinase that sits
  in the cytoplasm and dual-phosphorylates ERK on its activation
  loop (Thr-Glu-Tyr motif).  PDB **3EQH** (Ohren et al. 2004).
* **ERK2** (MAPK1) has the classic bilobal kinase fold.  PDB
  **1ERK** (Zhang et al. 1994, inactive form).  After dual
  phosphorylation it adopts the active conformation **2ERK**
  (Canagarajah et al. 1997).  In our cycle ERK is treated strictly
  as MEK's substrate — its own kinase activity is downstream.
* **NPC** (nuclear pore complex) is the gateway to the nucleus
  for phospho-ERK; we draw it with the canonical eight-fold
  symmetry (Lin & Hoelz 2019).
* Crucially MEK is *active-site limited*: it has one catalytic
  cleft, so it can be in complex with at most one ERK at a time.

Mapping the biology onto a bigraph (Milner 2009)
------------------------------------------------
A bigraph has two graphs sharing a common node set: a **place
graph** for spatial nesting, and a **link graph** for
connectivity.  We use one node per molecule / compartment, with a
``_type`` label (Milner's *control* in the bigraph signature, but
written here in the same field the typesystem uses so signature
labels and schema types share one namespace):

================  =================================================
Sort (``_type``)  Biology
================  =================================================
``Cell``          The whole cell — outer container.
``Compartment``   Subcellular volume — cytoplasm, nucleus, ER lumen.
``Cytoplasm``     Compartment-kind tag, child of cytoplasm.
``Nucleus``       Compartment-kind tag, child of nucleus.
``ERLumen``       Compartment-kind tag, child of er_lumen.
``NPC``           Nuclear-pore-complex-like channel at every
                  compartment boundary.
``MEK``           The kinase; lives in the cytoplasm.
``ERK``           Free substrate (unphosphorylated).
``pERK``          Phospho-substrate (carries the Thr183/Tyr185
                  phosphates).
================  =================================================

Place graph: ``Cell ⊃ Compartments ⊃ {NPC, MEK, ERK, pERK}``.
Link graph: a single shared edge between MEK's catalytic port and
the substrate's docking port whenever they're in complex.

Reaction rules (Gillespie SSA, propensity = ``rate × |matches|``):

* ``phosphorylate``      (k = 2.0)  — free ERK + free MEK in the
  same compartment → MEK·pERK with a fresh shared edge.  The redex
  marks both ports ``Absent``, encoding the active-site
  stoichiometry: MEK cannot be double-bound.
* ``dissociate``         (k = 0.5)  — MEK·pERK in a compartment →
  free MEK + free pERK, both staying in that compartment; the
  shared edge is destroyed.
* ``translocate_erk``      (k = 1.0)  — free ERK hops between
  sibling compartments (symmetric diffusion).
* ``translocate_perk_in``  (k = 2.0)  — free pERK,
  cytoplasm → nucleus.  Fast, importin-mediated active import
  (Plotnikov et al. 2011).
* ``translocate_perk_out`` (k = 0.1)  — free pERK,
  nucleus → cytoplasm.  Slow leak; in real biology this is
  driven by nuclear MAPK phosphatases that dephosphorylate
  pERK before export, but we collapse it into a first-order
  rate.  The 20-fold in/out asymmetry is the structural
  origin of nuclear pERK accumulation.

Why bigraphs?
-------------
*Compartment-only* models capture *where* a molecule is, but not
*what it is bound to*: they cannot say which copy of MEK is
currently in complex with which copy of ERK.  *Reaction-network*
models capture binding (S + E ⇌ SE) but collapse the spatial
dimension — "MEK in cytoplasm vs. nucleus" is invisible.

Bigraphs are the structure that has *both*.  A single redex can
constrain place *and* link in the same step::

    phosphorylate redex ≅
        Compartment.( MEK[outputs absent]
                    | ERK[outputs absent]
                    | rest )

— "co-located AND no prior bond" reads off directly.  This also
gives a clean place to encode active-site stoichiometry (via
``Absent`` on the catalytic port) and a clean place to mint or
destroy bonds (via ``LinkVar``).  Neither pure place-graph nor
pure reaction-network formalisms handle this well.
"""
import copy
import math
import random

from process_bigraph.composite import Process

from bigraph_schema.schema import Site
from bigraph_schema.assembly import (
    ReactionRule, LinkVar, Absent, find_matches, fire_rule)


# =====================================================================
# BRS Process (biology-agnostic engine)
# =====================================================================

class BigraphicalReactiveSystem(Process):
    """A Process that fires Milner-style reaction rules on each tick.

    Whereas ``ReactionStep`` is a Step (fires only when its inputs
    change, until quiescence), this Process fires on a regular time
    interval — making it suitable for time-stepped traces where rules
    have no fixed point (e.g. molecules continually wandering between
    compartments).

    Config:
        rules: List of ``ReactionRule`` objects. Each rule's
            ``rate`` field is treated as a *propensity coefficient*
            in the chemical-master-equation sense: the propensity
            of rule ``R`` is ``rule.rate * |matches(R)|``, so the
            firing distribution maps to mass-action over the match
            multiset.
        mode: One of:

            - ``'deterministic'`` — first matching rule wins; one
              firing per tick.
            - ``'stochastic'`` — one firing per tick, picked
              Gillespie-style by per-match propensity.
            - ``'gillespie'`` — proper SSA τ-leap: sample
              exponential waits with parameter ``λ = Σ k_i·|m_i|``
              until ``t ≥ interval``. Fires zero or more rules
              per tick depending on the propensity.

        seed: RNG seed (stochastic / gillespie modes).
        max_per_tick: Cap on firings per tick (default 1 for the
            non-Gillespie modes; Gillespie defaults to no cap).
    """

    config_schema = {}

    def initialize(self, config):
        self.rules = config.get('rules', [])
        self.mode = config.get('mode', 'deterministic')
        self.rng = random.Random(config.get('seed', 0))
        default_cap = (
            10**9 if self.mode == 'gillespie' else 1)
        self.max_per_tick = int(config.get('max_per_tick', default_cap))
        self.fired_log = []  # (sim_time, rule_label, match_path)

    def inputs(self):
        return {'state': 'tree[node]'}

    def outputs(self):
        return {'state': 'overwrite[tree[node]]'}

    def update(self, state, interval):
        subtree = state.get('state', {})
        if self.mode == 'gillespie':
            new_subtree, fired_any = self._gillespie_step(
                subtree, interval)
            return {'state': new_subtree} if fired_any else {}

        any_fired = False
        for _ in range(self.max_per_tick):
            new_subtree, label, path = self._fire_one(subtree)
            if label is None:
                break
            self.fired_log.append((interval, label, path))
            subtree = new_subtree
            any_fired = True
        if not any_fired:
            return {}
        return {'state': subtree}

    def _enumerate_candidates(self, subtree):
        candidates = []
        for rule in self.rules:
            matches = find_matches(subtree, rule.redex)
            rate = rule.rate if rule.rate is not None else 1.0
            for i, m in enumerate(matches):
                candidates.append((rule, i, m, rate))
        return candidates

    def _pick_candidate(self, candidates):
        if not candidates:
            return None
        total = sum(r for _, _, _, r in candidates)
        if total <= 0:
            return None
        pick = self.rng.random() * total
        cum = 0.0
        for rule, idx, match, rate in candidates:
            cum += rate
            if cum >= pick:
                return rule, idx, match
        rule, idx, match, _ = candidates[-1]
        return rule, idx, match

    def _fire_one(self, subtree):
        if self.mode == 'stochastic':
            candidates = self._enumerate_candidates(subtree)
            picked = self._pick_candidate(candidates)
            if picked is None:
                return subtree, None, None
            rule, idx, match = picked
            new_state, _ = fire_rule(subtree, rule, match_index=idx)
            return new_state, rule.label, match.path
        else:
            for rule in self.rules:
                new_state, match = fire_rule(subtree, rule)
                if match is not None:
                    return new_state, rule.label, match.path
            return subtree, None, None

    def _gillespie_step(self, subtree, interval):
        t = 0.0
        fired_any = False
        steps = 0
        while t < interval and steps < self.max_per_tick:
            candidates = self._enumerate_candidates(subtree)
            if not candidates:
                break
            total = sum(r for _, _, _, r in candidates)
            if total <= 0:
                break
            u = max(self.rng.random(), 1e-12)
            dt = -math.log(u) / total
            if t + dt > interval:
                break
            t += dt
            picked = self._pick_candidate(candidates)
            if picked is None:
                break
            rule, idx, match = picked
            subtree, _ = fire_rule(subtree, rule, match_index=idx)
            self.fired_log.append((t, rule.label, match.path))
            fired_any = True
            steps += 1
        return subtree, fired_any


# =====================================================================
# MAPK signalling example
# =====================================================================
#
# Bigraph signature for the MAPK model. Each name is a sort that
# tags a place-graph node via ``_type`` — sharing one namespace
# with the typesystem's schema registry, so these labels can later
# carry typed methods (serialization, value-update behaviour, etc.)
# without changing the matcher.
MAPK_SORTS = (
    'Cell', 'Compartment', 'NPC', 'MEK', 'ERK', 'pERK',
    'Cytoplasm', 'Nucleus', 'ERLumen')

# Backward-compat alias (older callers may still import this name).
MAPK_CONTROLS = MAPK_SORTS

# Minimal schema entries — each sort inherits the base ``node`` type,
# which is a permissive tree-shaped schema with no value semantics.
# Future work can replace these with richer schemas that attach typed
# methods (e.g. a ``MEK`` schema with a catalytic-rate update_method)
# without changing rule application — the matcher only reads the
# label, not the schema body.
MAPK_TYPE_SCHEMAS = {sort: {'_inherit': 'node'} for sort in MAPK_SORTS}


def register_mapk_types(core):
    """Register the MAPK bigraph signature as schema types on ``core``.

    The matcher works on string equality alone and does not require
    these registrations, but registering them makes the signature a
    first-class citizen of the typesystem: future ``update_method``,
    serialization, and validation hooks can be attached to e.g. the
    ``MEK`` type and will compose with the BRS rules without further
    plumbing.
    """
    core.register_types(MAPK_TYPE_SCHEMAS)
    return core


def _erk(name):
    """Free ERK substrate: an ``ERK``-control node carrying its
    individual identity in the ``name`` field. No ``outputs``
    means no docking bond — i.e. unbound."""
    return {'_type': 'ERK', 'name': name}


# ── Reaction rules ──────────────────────────────────────────────────


def rule_phosphorylate():
    """Free MEK + free ERK in the same compartment form the
    Michaelis complex.

    Biology:
        Pre:   MEK (free, no substrate in active site)
             + ERK (free, unphosphorylated)
        Post:  MEK·pERK   (covalently transferred phosphate, drawn
                          here as a shared link-graph edge)

    Bigraph encoding:
        - The redex restricts both ports with ``Absent()`` — so
          the rule cannot fire on a MEK that's already in complex
          (active-site stoichiometry).
        - The reactum re-introduces both ports wired to a single
          ``LinkVar('bond')``.  ``e`` is unbound on the redex side,
          so ``instantiate`` mints a *fresh* anchor path under
          ``_edges`` and links both ports to it.
        - The substrate's control flips ``ERK → pERK``.
    """
    return ReactionRule(
        redex={
            'compartment': {
                '_type': 'Compartment',
                'enzyme': {
                    '_type': 'MEK',
                    'outputs': Absent()},
                'substrate': {
                    '_type': 'ERK',
                    'name': Site(),
                    'outputs': Absent()},
                'bystanders': Site()}},
        reactum={
            'compartment': {
                '_type': 'Compartment',
                'enzyme': {
                    '_type': 'MEK',
                    'outputs': {'enzyme_port': LinkVar('bond')}},
                'substrate': {
                    '_type': 'pERK',
                    'name': Site(),
                    'outputs': {'substrate_port': LinkVar('bond')}},
                'bystanders': Site()}},
        instantiation={'bystanders': 'bystanders', 'name': 'name'},
        rate=2.0,
        label='phosphorylate')


def rule_dissociate():
    """The MEK·pERK complex dissociates inside its compartment.

    Biology:
        Pre:   MEK·pERK
        Post:  free MEK   +   free pERK
                              (both stay in this compartment;
                               pERK can subsequently translocate)

    Bigraph encoding:
        - The redex requires both ports wired to the same edge
          (``LinkVar('bond')``) — so the rule fires exactly when MEK
          and a pERK are in complex.
        - The reactum drops the ``outputs`` keys on both nodes,
          so the bond's two endpoints are gone — the edge is
          effectively destroyed.
    """
    return ReactionRule(
        redex={
            'compartment': {
                '_type': 'Compartment',
                'enzyme': {
                    '_type': 'MEK',
                    'outputs': {'enzyme_port': LinkVar('bond')}},
                'substrate': {
                    '_type': 'pERK',
                    'name': Site(),
                    'outputs': {'substrate_port': LinkVar('bond')}},
                'bystanders': Site()}},
        reactum={
            'compartment': {
                '_type': 'Compartment',
                'enzyme': {'_type': 'MEK'},
                'substrate': {'_type': 'pERK', 'name': Site()},
                'bystanders': Site()}},
        instantiation={'bystanders': 'bystanders', 'name': 'name'},
        rate=0.5,
        label='dissociate')


def rule_dephosphorylate():
    """A nuclear MAP-kinase phosphatase removes the phosphate from
    a free pERK in the nucleus, returning it to ERK.

    Biology:
        Pre:   pERK (free, NOT in MEK complex) inside the nucleus
        Post:  ERK in the same nucleus

    Real biology: nuclear DUSP/MKP phosphatases (DUSP1, DUSP5, …)
    dephosphorylate the Thr-X-Tyr motif and reset the kinase to its
    inactive form. We collapse the phosphatase machinery into a
    single first-order rate. Only nuclear pERK is dephosphorylated;
    cytoplasmic pERK is shielded by ongoing MEK activity.

    The dephosphorylated ERK can then leave via
    ``translocate_erk_out``, get re-phosphorylated by MEK in the
    cytoplasm, and re-import — closing the signalling cycle. This
    is the rule that keeps events firing past steady state."""
    return ReactionRule(
        redex={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'substrate': {
                        '_type': 'pERK',
                        'name': Site(),
                        'outputs': Absent()},
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        reactum={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'substrate': {'_type': 'ERK', 'name': Site()},
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        instantiation={
            'outer_rest': 'outer_rest',
            'inner_rest': 'inner_rest',
            'name': 'name'},
        rate=0.4,
        label='dephosphorylate')


def rule_translocate_erk_in():
    """A free unphosphorylated ERK descends from cytoplasm into a
    child compartment (nucleus or ER lumen).

    Biology:
        Pre:   ERK in cytoplasm body, child compartment present
        Post:  ERK inside that child compartment

    A coarse-grained model of facilitated diffusion across the
    nuclear envelope or ER membrane. The redex doesn't restrict
    the child's kind — it fires for any nested Compartment, and
    the matcher will independently find each child.
    """
    return ReactionRule(
        redex={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'substrate': {
                    '_type': 'ERK',
                    'name': Site()},
                'inner': {
                    '_type': 'Compartment',
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        reactum={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'inner_rest': Site(),
                    'substrate': {'_type': 'ERK', 'name': Site()}},
                'outer_rest': Site()},
        },
        instantiation={
            'outer_rest': 'outer_rest',
            'inner_rest': 'inner_rest',
            'name': 'name'},
        rate=1.0,
        label='translocate_erk_in')


def rule_translocate_erk_out():
    """A free unphosphorylated ERK ascends from a child compartment
    back into cytoplasm.

    Biology:
        Pre:   ERK inside a child compartment (nucleus / ER lumen)
        Post:  ERK in cytoplasm body

    Symmetric counterpart of ``translocate_erk_in``; the same rate
    in both directions makes raw ERK distribution diffusive.
    """
    return ReactionRule(
        redex={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'substrate': {
                        '_type': 'ERK',
                        'name': Site()},
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        reactum={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'inner_rest': Site()},
                'substrate': {'_type': 'ERK', 'name': Site()},
                'outer_rest': Site()},
        },
        instantiation={
            'outer_rest': 'outer_rest',
            'inner_rest': 'inner_rest',
            'name': 'name'},
        rate=1.0,
        label='translocate_erk_out')


def rule_translocate_perk_in():
    """Active nuclear import of free phospho-ERK.

    Biology:
        Pre:   pERK in cytoplasm body (NOT in complex), nucleus
               present as a child compartment of cytoplasm
        Post:  pERK inside the nucleus

    Phospho-ERK accumulates in the nucleus where it
    phosphorylates downstream transcription factors. Import is
    importin-mediated and biased: the steady state has most pERK
    inside the nucleus (Plotnikov et al. 2011).  We encode this
    as a single fast forward rule, with a slow leak rule
    (``translocate_perk_out``) for the reverse direction.

    Bigraph encoding:
        - The nesting (Cytoplasm ⊃ Nucleus) is matched structurally
          via the place-graph parent/child relation, not a string
          property.
        - ``outputs: Absent()`` keeps bound complexes pinned in the
          cytoplasm (MEK is not nuclear).
    """
    return ReactionRule(
        redex={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'substrate': {
                    '_type': 'pERK',
                    'name': Site(),
                    'outputs': Absent()},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        reactum={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'inner_rest': Site(),
                    'substrate': {'_type': 'pERK', 'name': Site()}},
                'outer_rest': Site()},
        },
        instantiation={
            'outer_rest': 'outer_rest',
            'inner_rest': 'inner_rest',
            'name': 'name'},
        rate=2.0,
        label='translocate_perk_in')


def rule_translocate_perk_out():
    """Slow nuclear export / leak of free phospho-ERK.

    Biology:
        Pre:   pERK in nucleus (NOT in complex), nucleus is a
               child of cytoplasm
        Post:  pERK back in cytoplasm body

    Real export typically requires MAPK dephosphorylation in the
    nucleus followed by ERK translation back, but at this
    abstraction level we encode it directly as a slow first-order
    process. The asymmetry between the import (k=2.0) and export
    (k=0.1) rates is what produces the steady-state nuclear
    accumulation that defines the biological signal.
    """
    return ReactionRule(
        redex={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'substrate': {
                        '_type': 'pERK',
                        'name': Site(),
                        'outputs': Absent()},
                    'inner_rest': Site()},
                'outer_rest': Site()},
        },
        reactum={
            'outer': {
                '_type': 'Compartment',
                'kind': {'_type': 'Cytoplasm'},
                'inner': {
                    '_type': 'Compartment',
                    'kind': {'_type': 'Nucleus'},
                    'inner_rest': Site()},
                'substrate': {'_type': 'pERK', 'name': Site()},
                'outer_rest': Site()},
        },
        instantiation={
            'outer_rest': 'outer_rest',
            'inner_rest': 'inner_rest',
            'name': 'name'},
        rate=0.1,
        label='translocate_perk_out')


def mapk_rules():
    """All seven rules together — the MAPK BRS we simulate.

    With nesting (cytoplasm ⊃ {nucleus, ER lumen}), translocation
    is expressed structurally as parent ↔ child compartment moves;
    nucleus ↔ ER lumen direct transit becomes inexpressible.

    The cycle keeps moving because nuclear pERK is dephosphorylated
    back to ERK, exported, and re-phosphorylated by MEK — i.e.,
    a steady stream of events rather than a one-shot transient.
    The asymmetry between ``translocate_perk_in`` (k=2.0) and
    ``translocate_perk_out`` (k=0.1) still produces a nuclear pERK
    accumulation at steady state — the actual biological *point*
    of the signalling cycle."""
    return [
        rule_phosphorylate(),       # MEK + ERK → MEK·pERK
        rule_dissociate(),          # MEK·pERK → free MEK + free pERK
        rule_dephosphorylate(),     # nuclear pERK → nuclear ERK
        rule_translocate_erk_in(),  # cyto → child compartment
        rule_translocate_erk_out(), # child → cyto
        rule_translocate_perk_in(), # cyto → nucleus, fast
        rule_translocate_perk_out(),# nucleus → cyto, slow leak
    ]


# ── Initial-state builder ──────────────────────────────────────────


def initial_mapk_state(seed=0):
    """Nested-compartment cell with MEK in the cytoplasm and an
    assortment of ERK / pERK copies seeded to drive the cycle.

    Place graph (nested — nucleus and ER lumen are spatially
    contained inside cytoplasm, matching cell biology)::

        Cell
        └── cytoplasm    (MEK, erk1, erk2, erk0)
            ├── nucleus  (empty initially; pERK accumulates here)
            └── er_lumen (erk3)

    A pre-existing free ``pERK`` (``erk0``) is included in the
    cytoplasm so that the initial bigraph diagram shows at least
    one node of every sort — including phospho-substrate. It will
    translocate into the nucleus on the first import event.

    NPCs are not separate entities — they live in the nuclear
    envelope and are rendered as part of the nucleus visualization.

    Link graph: empty initially. Bonds are minted by
    ``phosphorylate`` and destroyed by ``dissociate``.
    """
    return {
        '_type': 'Cell',
        'cytoplasm': {
            '_type': 'Compartment',
            'kind': {'_type': 'Cytoplasm'},
            'mek':  {'_type': 'MEK'},
            'erk0': {'_type': 'pERK', 'name': 'erk0'},
            'erk1': _erk('erk1'),
            'erk2': _erk('erk2'),
            'nucleus': {
                '_type': 'Compartment',
                'kind': {'_type': 'Nucleus'},
            },
            'er_lumen': {
                '_type': 'Compartment',
                'kind': {'_type': 'ERLumen'},
                'erk3': _erk('erk3'),
            },
        },
    }


# =====================================================================
# Helpers for analysis (substrate counts, structure traversal)
# =====================================================================


def list_substrates(state):
    """Return ``[(name, compartment, control, bound)]`` for every
    ERK / pERK node in the cell. ``bound`` is True iff the node
    has any ``outputs`` wire."""
    out = []

    def walk(node, comp_name=None):
        if not isinstance(node, dict):
            return
        ctrl = node.get('_type', '')
        if ctrl in ('ERK', 'pERK'):
            name = node.get('name', '?')
            outs = node.get('outputs')
            bound = isinstance(outs, dict) and bool(outs)
            out.append((name, comp_name, ctrl, bound))
            return
        for k, v in node.items():
            if isinstance(v, dict):
                next_comp = (
                    k if v.get('_type') == 'Compartment' else comp_name)
                walk(v, comp_name=next_comp)

    walk(state)
    return out


def count_substrates_per_compartment(state):
    """Return ``{compartment_name: (free_count, bound_count, perk_count)}``
    summary for the state — convenient for plotting trajectories."""
    counts = {}
    for name, comp, ctrl, bound in list_substrates(state):
        if comp is None:
            continue
        free, perk_free, bound_count = counts.get(comp, (0, 0, 0))
        if ctrl == 'ERK':
            free += 1
        elif ctrl == 'pERK' and bound:
            bound_count += 1
        elif ctrl == 'pERK':
            perk_free += 1
        counts[comp] = (free, perk_free, bound_count)
    return counts


# =====================================================================
# Spatio-flux test-suite integration
# =====================================================================


def get_brs_mapk_doc(core=None, config=None):
    """Build a process-bigraph document for the MAPK BRS example.

    The document wires a single ``BigraphicalReactiveSystem``
    Process against a typed ``cell`` store. Each tick advances
    simulation time by ``interval`` and fires zero or more rule
    matches (under Gillespie SSA). An explicit emitter is included
    since the spatio-flux standard emitter only knows about the
    generic ``fields`` / ``particles`` / ``lattice`` stores.

    If ``core`` is provided, the MAPK bigraph signature is also
    registered onto it as schema types — making the controls
    first-class typesystem citizens that can later carry methods.
    """
    from process_bigraph.emitter import emitter_from_wires
    config = config or {}
    if core is not None:
        register_mapk_types(core)
    return {
        'schema': {'cell': 'tree[node]'},
        'state': {
            'cell': initial_mapk_state(),
            'brs': {
                '_type': 'process',
                'address': (
                    'local:!spatio_flux.processes.bigraph_reactive_system'
                    '.BigraphicalReactiveSystem'),
                'config': {
                    'rules': mapk_rules(),
                    'mode': config.get('mode', 'gillespie'),
                    'seed': config.get('seed', 42),
                    'max_per_tick': config.get('max_per_tick', 10**6),
                },
                'inputs': {'state': ['cell']},
                'outputs': {'state': ['cell']},
                'interval': config.get('interval', 1.0),
            },
            'emitter': emitter_from_wires(
                {'global_time': ['global_time'], 'cell': ['cell']}),
        }}


# =====================================================================
# Plotting (molecular cartoons + smooth animation)
# =====================================================================

RULE_COLORS = {
    'phosphorylate':        '#7570b3',
    'dissociate':           '#d95f02',
    'dephosphorylate':      '#a6611a',
    'translocate_erk_in':   '#1b9e77',
    'translocate_erk_out':  '#5fbf94',
    'translocate_perk_in':  '#117a59',
    'translocate_perk_out': '#66c2a5',
}


def infer_firing(before, after):
    """Compare two consecutive emitted states and infer which rule
    fired between them.

    For our six-rule nested MAPK system:

    - ``phosphorylate``        : ``ERK → pERK`` AND becomes bound.
    - ``dissociate``           : stays ``pERK``, same compartment,
                                  but ``outputs`` disappears.
    - ``translocate_erk_in``   : ``ERK`` moves cytoplasm → child
                                  compartment (nucleus / ER lumen).
    - ``translocate_erk_out``  : ``ERK`` moves child → cytoplasm.
    - ``translocate_perk_in``  : free ``pERK`` moves
                                  cytoplasm → nucleus.
    - ``translocate_perk_out`` : free ``pERK`` moves
                                  nucleus → cytoplasm.
    """
    pre = {n: (c, ctrl, b) for n, c, ctrl, b in list_substrates(before)}
    post = {n: (c, ctrl, b) for n, c, ctrl, b in list_substrates(after)}
    for name, (cm_post, ctrl_post, b_post) in post.items():
        if name not in pre:
            continue
        cm_pre, ctrl_pre, b_pre = pre[name]
        if ctrl_pre == 'ERK' and ctrl_post == 'pERK' and b_post and not b_pre:
            return 'phosphorylate'
        if ctrl_pre == 'pERK' and ctrl_post == 'pERK' \
                and cm_pre == cm_post and b_pre and not b_post:
            return 'dissociate'
        if ctrl_pre == 'pERK' and ctrl_post == 'ERK' \
                and cm_pre == cm_post and cm_post == 'nucleus':
            return 'dephosphorylate'
        if ctrl_pre == 'ERK' and ctrl_post == 'ERK' and cm_pre != cm_post:
            if cm_pre == 'cytoplasm':
                return 'translocate_erk_in'
            if cm_post == 'cytoplasm':
                return 'translocate_erk_out'
            return 'translocate_erk_in'
        if ctrl_pre == 'pERK' and ctrl_post == 'pERK' \
                and cm_pre != cm_post and not b_pre and not b_post:
            if cm_pre == 'cytoplasm' and cm_post == 'nucleus':
                return 'translocate_perk_in'
            if cm_pre == 'nucleus' and cm_post == 'cytoplasm':
                return 'translocate_perk_out'
            return 'translocate_perk_in'
    return None


def plot_brs_mapk(results, state, config=None):
    """Plot a five-view summary of the MAPK BRS trace, all using
    suggestive bilobal-kinase silhouettes for MEK/ERK/pERK and an
    eight-fold symmetric NPC for the pore:

    0. **Initial bigraph + biology explainer** (``<filename>_viz.png``)
    1. **Population trajectories** (``<filename>_timeseries.png``)
    2. **Structural snapshots** (``<filename>_snapshots.png``)
    3. **Transition trace** (``<filename>_trace.png``)
    4. **Smooth animation** (``<filename>_animation.gif``) — frame-
       interpolated walk through every firing with cubic ease-in-out.
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    config = config or {}
    filename = config.get('filename', 'brs_mapk')
    out_dir = config.get('out_dir', 'out')
    os.makedirs(out_dir, exist_ok=True)

    times = [step['global_time'] for step in results]
    states = [step['cell'] for step in results]

    compartment_names = []
    for s in states:
        for name, _ in _iter_compartments(s):
            if name not in compartment_names:
                compartment_names.append(name)

    # Per-compartment time series, split by control state
    erk_series = {c: [] for c in compartment_names}
    perk_free_series = {c: [] for c in compartment_names}
    perk_bound_series = {c: [] for c in compartment_names}
    for s in states:
        per_comp = count_substrates_per_compartment(s)
        for c in compartment_names:
            free, perk_free, bound = per_comp.get(c, (0, 0, 0))
            erk_series[c].append(free)
            perk_free_series[c].append(perk_free)
            perk_bound_series[c].append(bound)

    firings = []
    for i in range(1, len(states)):
        rule = infer_firing(states[i - 1], states[i])
        if rule is not None:
            firings.append((times[i], rule, states[i - 1], states[i]))

    # ── 0. Initial bigraph diagram (re-render the composite-level
    # _viz.png so it shows the full place graph of the cell tree)
    # plus standalone explainer markdown plus an honest state
    # JSON for the report's interactive viewer. ────────────────────
    _replot_initial_bigraph(filename, out_dir, states[0])
    _save_explainer_markdown(filename, out_dir)
    _save_state_json(filename, out_dir, states[0])

    # ── 1. Population trajectories ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, c in enumerate(compartment_names):
        col = palette[i % len(palette)]
        total = [
            e + pf + pb
            for e, pf, pb
            in zip(erk_series[c], perk_free_series[c], perk_bound_series[c])]
        ax.plot(times, total, color=col, label=f'{c} (total)', linewidth=2)
        bound_or_perk = [
            pf + pb
            for pf, pb
            in zip(perk_free_series[c], perk_bound_series[c])]
        if any(bound_or_perk):
            ax.plot(times, bound_or_perk, color=col, linestyle=':',
                    label=f'{c} (pERK)', linewidth=1.2)
    if firings:
        ymin, ymax = ax.get_ylim()
        tick_top = ymax + 0.05 * (ymax - ymin) if ymax > ymin else 0.5
        tick_height = 0.06 * (ymax - ymin) if ymax > ymin else 0.1
        for t, rule, _, _ in firings:
            color = RULE_COLORS.get(rule, '#888')
            ax.vlines(t, tick_top, tick_top + tick_height,
                      color=color, linewidth=2.5)
        ax.set_ylim(ymin, tick_top + tick_height + 0.15 * (ymax - ymin))
        rule_handles = [
            mpatches.Patch(color=RULE_COLORS[r], label=r)
            for r in RULE_COLORS]
        ax.legend(
            handles=(
                [plt.Line2D([0], [0], color=palette[i % len(palette)],
                            linewidth=2, label=f'{c} (total)')
                 for i, c in enumerate(compartment_names)]
                + [plt.Line2D([0], [0], color=palette[i % len(palette)],
                              linestyle=':', linewidth=1.2,
                              label=f'{c} (pERK)')
                   for i, c in enumerate(compartment_names)
                   if any(pf + pb for pf, pb in zip(
                       perk_free_series[c], perk_bound_series[c]))]
                + rule_handles),
            fontsize=7, loc='upper right', ncol=3)
    else:
        ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_xlabel('time (ticks)')
    ax.set_ylabel('substrates per compartment')
    ax.set_title('MAPK BRS — substrate populations '
                 '(top: rule firings)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{filename}_timeseries.png'),
                dpi=150)
    plt.close(fig)

    # ── 2. Structural snapshots ────────────────────────────────────
    n_snapshots = config.get('n_snapshots', 6)
    if len(states) >= n_snapshots:
        idx = [round(i * (len(states) - 1) / (n_snapshots - 1))
               for i in range(n_snapshots)]
    else:
        idx = list(range(len(states)))
    n_cols = config.get('snapshot_cols', 3)
    n_rows = (len(idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows),
        squeeze=False)
    flat_axes = [a for row in axes for a in row]
    for ax, j in zip(flat_axes, idx):
        _draw_cell_snapshot(
            ax, states[j], compartment_names,
            title=f't={times[j]:.0f}')
    # Hide unused axes if grid has more cells than snapshots
    for ax in flat_axes[len(idx):]:
        ax.set_visible(False)
    legend_handles = _legend_handles()
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        'MAPK BRS — kinase, substrate, phospho-substrate across compartments',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{filename}_snapshots.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3. Transition trace ────────────────────────────────────────
    if firings:
        import matplotlib.image as mpimg
        # Show one example of each rule rather than the first N
        # firings — that way every rule in ``mapk_rules()`` appears
        # exactly once with a real before/after state, and the
        # exhaustive set is visible at a glance even when the
        # chronological order would only sample some of them.
        seen = set()
        unique_firings = []
        for entry in firings:
            rule_label = entry[1]
            if rule_label in seen:
                continue
            seen.add(rule_label)
            unique_firings.append(entry)
        # Preserve the rule-list order so the trace reads top-to-
        # bottom in the same order as ``mapk_rules()``.
        rule_order = {r.label: i for i, r in enumerate(mapk_rules())}
        unique_firings.sort(
            key=lambda e: rule_order.get(e[1], len(rule_order)))
        trace_firings = unique_firings
        n_pairs = len(trace_firings)

        # Pre-render each unique rule's redex / reactum once and
        # cache the PNG path. Cheaper than re-running plot_bigraph
        # per firing, and the resulting images can be re-used in
        # any documentation that needs the rule itself as a figure.
        rule_pattern_imgs = {}
        rules_by_label = {r.label: r for r in mapk_rules()}
        for _, rule_label, _, _ in trace_firings:
            if rule_label in rule_pattern_imgs:
                continue
            rule = rules_by_label.get(rule_label)
            if rule is None:
                rule_pattern_imgs[rule_label] = (None, None)
                continue
            redex_path = _render_rule_pattern(
                rule_label, rule.redex, out_dir, 'redex')
            reactum_path = _render_rule_pattern(
                rule_label, rule.reactum, out_dir, 'reactum')
            rule_pattern_imgs[rule_label] = (redex_path, reactum_path)

        # Layout per firing (one row in the outer grid):
        #   redex → reactum   ┊   before → after
        # Seven columns: redex, arrow, reactum, gap,
        # before, arrow, after. The rule's label sits as
        # a small caption above each of the two arrows.
        fig = plt.figure(figsize=(15.0, 3.2 * n_pairs))
        gs = fig.add_gridspec(
            n_pairs, 7,
            width_ratios=[1, 0.28, 1, 0.45, 1, 0.28, 1],
            wspace=0.06, hspace=0.4)
        for i, (t_after, rule, before, after) in enumerate(trace_firings):
            color = RULE_COLORS.get(rule, '#444')

            # Left half: redex → reactum.
            ax_redex = fig.add_subplot(gs[i, 0])
            ax_arrow_rule = fig.add_subplot(gs[i, 1])
            ax_reactum = fig.add_subplot(gs[i, 2])
            redex_path, reactum_path = rule_pattern_imgs.get(
                rule, (None, None))
            for ax_p, path, kind_label in (
                    (ax_redex, redex_path, 'redex'),
                    (ax_reactum, reactum_path, 'reactum')):
                ax_p.set_xticks([])
                ax_p.set_yticks([])
                for spine in ax_p.spines.values():
                    spine.set_visible(False)
                if path:
                    try:
                        img = mpimg.imread(path)
                        ax_p.imshow(img)
                    except Exception:
                        pass
                ax_p.set_title(
                    kind_label, fontsize=9, color='#444',
                    style='italic')
            _draw_rule_arrow(ax_arrow_rule, color, label=rule)

            # Right half: before → after cell snapshots.
            ax_b = fig.add_subplot(gs[i, 4])
            ax_arrow_state = fig.add_subplot(gs[i, 5])
            ax_a = fig.add_subplot(gs[i, 6])
            _draw_cell_snapshot(
                ax_b, before, compartment_names,
                title=f't={t_after - 1:.0f} (before)')
            _draw_cell_snapshot(
                ax_a, after, compartment_names,
                title=f't={t_after:.0f} (after)')
            _draw_rule_arrow(ax_arrow_state, color, label=rule)

        fig.suptitle(
            f'MAPK BRS — rule catalog (one example per rule, '
            f'{n_pairs} of {len(rule_order)} rules covered by the '
            f'{len(firings)}-firing run). Left: rule redex → reactum. '
            'Right: state before → after.',
            fontsize=11, y=0.995)
        fig.savefig(os.path.join(out_dir, f'{filename}_trace.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── 4. Smooth animation across rule firings ────────────────────
    if len(states) > 1:
        _make_smooth_animation(
            filename, out_dir, states, times, compartment_names,
            firings)


# ── State traversal helpers ────────────────────────────────────────


def _iter_compartments(state):
    """Yield ``(name, compartment_dict)`` for every Compartment in
    the cell tree, including nested ones (cytoplasm contains
    nucleus and er_lumen as children)."""
    if not isinstance(state, dict):
        return
    for k, v in state.items():
        if isinstance(v, dict):
            if v.get('_type') == 'Compartment':
                yield k, v
            yield from _iter_compartments(v)


def _compartment_parent_map(state):
    """Return ``{compartment_name: parent_name_or_None}`` so layout
    code can position nested compartments relative to their parent."""
    parents = {}

    def walk(node, parent):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            if v.get('_type') == 'Compartment':
                parents[k] = parent
                walk(v, k)
            else:
                walk(v, parent)

    walk(state, None)
    return parents


# ── Legend helpers ─────────────────────────────────────────────────


def _legend_handles():
    """Build legend handles whose icons match the snapshot cartoons."""
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    return [
        mpatches.Patch(color='#9ecae1', label='MEK  (kinase)'),
        mpatches.Patch(color='#7fbf7b', label='ERK  (substrate)'),
        mpatches.Patch(color='#ef8a62', label='pERK  (phospho-substrate)'),
        mpatches.Patch(facecolor='#fff7e6', edgecolor='#7d5b3a',
                       label='nuclear envelope (with NPCs)'),
        mpatches.Patch(facecolor='#fde4cf', edgecolor='#a96a3a',
                       label='ER lumen'),
        mlines.Line2D([0], [0], color='#2ca02c', linewidth=2.4,
                      label='shared edge  (MEK·pERK bond)'),
    ]


# ── Molecular-cartoon entity drawing ────────────────────────────────
#
# We draw each protein as a "halfway-realistic" silhouette: above
# the box-and-label level, but coarser than an atomic structure.
# Both ERK (PDB 1ERK / 2ERK) and MEK (PDB 3EQH) share the classic
# bilobal protein-kinase fold: a smaller N-lobe (β-sheet rich) and
# a larger C-lobe (α-helix rich), with the active-site cleft
# between them.  We render the silhouette as two overlapping
# ellipses, plus a small protrusion suggesting the activation loop
# where the phosphates sit.


# ── Cell-anatomy layout (figure-coord positions) ───────────────────
#
# The same coordinates are used by every figure (snapshots, trace,
# animation) so an entity's place on the page is consistent across
# panels.  Coordinates are subpanel-relative (each axes uses 0..1).

CELL_BOUNDS = (0.04, 0.06, 0.96, 0.92)
NUCLEUS_CENTER = (0.30, 0.54)
NUCLEUS_RADIUS = 0.17
ER_CENTER = (0.72, 0.55)
ER_HALF_W = 0.18
ER_HALF_H = 0.22


def _draw_cell(ax):
    """The plasma membrane: an organic rounded rectangle drawn with
    a double contour (suggesting the phospholipid bilayer) and a
    pale cytosolic fill — every other organelle is drawn on top of
    this, so the visible interior IS the cytoplasm."""
    import matplotlib.patches as mpatches
    x0, y0, x1, y1 = CELL_BOUNDS
    # Inner cytosol fill
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.018,rounding_size=0.06',
        linewidth=0, facecolor='#f4f7f0', zorder=0))
    # Outer leaflet of the bilayer (slightly outset from inner)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - 0.004, y0 - 0.004),
        x1 - x0 + 0.008, y1 - y0 + 0.008,
        boxstyle='round,pad=0.018,rounding_size=0.06',
        linewidth=1.4, edgecolor='#3a5b3a', facecolor='none',
        zorder=1))
    # Inner leaflet
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.018,rounding_size=0.06',
        linewidth=1.0, edgecolor='#5a7a5a', facecolor='none',
        zorder=1))


def _draw_nucleus(ax, label='nucleus'):
    """Nucleus as a circle with a double envelope (inner + outer
    nuclear membrane), embedded NPCs around the perimeter, and a
    nucleolus dot inside. Lin & Hoelz 2019."""
    import matplotlib.patches as mpatches
    cx, cy = NUCLEUS_CENTER
    r = NUCLEUS_RADIUS
    # Pale nucleoplasm fill
    ax.add_patch(mpatches.Circle(
        (cx, cy), r, facecolor='#fff7e6',
        edgecolor='none', zorder=2))
    # Outer nuclear membrane
    ax.add_patch(mpatches.Circle(
        (cx, cy), r,
        facecolor='none', edgecolor='#7d5b3a', linewidth=1.5,
        zorder=3))
    # Inner nuclear membrane (slightly inset → double-envelope cue)
    ax.add_patch(mpatches.Circle(
        (cx, cy), r - 0.012,
        facecolor='none', edgecolor='#a8835a', linewidth=0.8,
        zorder=3))
    # NPCs embedded in the envelope at six positions around the
    # circle. Each NPC is a small annulus with eight-fold symmetry.
    n_pores = 6
    for k in range(n_pores):
        a = 2 * math.pi * k / n_pores + math.pi / 8
        nx = cx + r * math.cos(a)
        ny = cy + r * math.sin(a)
        _draw_npc_glyph(ax, nx, ny, radius=0.022)
    # Nucleolus — small darker spot, slightly off-centre
    ax.add_patch(mpatches.Circle(
        (cx + 0.025, cy - 0.018), 0.024,
        facecolor='#d4a673', edgecolor='#7d5b3a',
        linewidth=0.6, zorder=3, alpha=0.7))
    # Label sits just below the nucleus
    ax.text(cx, cy - r - 0.020, label,
            ha='center', va='top', fontsize=6.6, style='italic',
            color='#7d5b3a', zorder=6)


def _draw_er_lumen(ax, label='ER lumen'):
    """ER lumen rendered as two stacked, slightly tilted cisternae
    (flat membrane sacs) with thin tubular connectors — the
    canonical ER cartoon."""
    import matplotlib.patches as mpatches
    cx, cy = ER_CENTER
    # Two cisternae (tilted ellipses), one upper one lower
    for dy, tilt in [(+0.07, 8), (-0.07, -8)]:
        # outer membrane
        ax.add_patch(mpatches.Ellipse(
            (cx, cy + dy), 2 * ER_HALF_W * 0.95, 0.075,
            angle=tilt, linewidth=1.2,
            edgecolor='#a96a3a', facecolor='#fde4cf', zorder=2))
        # inner membrane (the lumen contour)
        ax.add_patch(mpatches.Ellipse(
            (cx, cy + dy), 2 * ER_HALF_W * 0.78, 0.045,
            angle=tilt, linewidth=0.6,
            edgecolor='#c98a5a', facecolor='none', zorder=3))
    # Tubular connector between the two cisternae
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - 0.018, cy - 0.05),
        0.036, 0.10,
        boxstyle='round,pad=0.0,rounding_size=0.018',
        linewidth=1.0, edgecolor='#a96a3a',
        facecolor='#fde4cf', zorder=2))
    # A few suggested ribosomes (rough-ER) — small dark dots on the
    # outer faces of the cisternae.
    for dy, tilt in [(+0.07, 8), (-0.07, -8)]:
        for k in range(4):
            t = -1 + k * 0.66
            rx = cx + (ER_HALF_W * 0.85) * t
            ry = cy + dy + 0.030 * math.cos(t * math.pi)
            ax.add_patch(mpatches.Circle(
                (rx, ry), 0.005,
                facecolor='#5a3a20', edgecolor='none',
                zorder=4))
    ax.text(cx, cy - ER_HALF_H + 0.005, label,
            ha='center', va='top', fontsize=6.6, style='italic',
            color='#a96a3a', zorder=6)


def _draw_npc_glyph(ax, cx, cy, radius=0.022):
    """Compact NPC glyph for embedding in the nuclear envelope —
    smaller version of the eight-fold annulus, no label."""
    import matplotlib.patches as mpatches
    r = radius
    ax.add_patch(mpatches.Circle(
        (cx, cy), r,
        linewidth=0.8, edgecolor='#555', facecolor='#cfcfcf',
        zorder=4))
    ax.add_patch(mpatches.Circle(
        (cx, cy), r * 0.42,
        linewidth=0.6, edgecolor='#555', facecolor='#fdfdf9',
        zorder=5))
    for k in range(8):
        a = k * math.pi / 4
        ax.plot(
            [cx + r * 0.42 * math.cos(a), cx + r * math.cos(a)],
            [cy + r * 0.42 * math.sin(a), cy + r * math.sin(a)],
            color='#888', linewidth=0.4, zorder=4)


def _draw_kinase_bilobal(
        ax, cx, cy, label, *,
        face_n='#9ecae1', face_c='#74a9cf', edge='#1c4a78',
        scale=1.0, p_in_cleft=False, label_color='#1c4a78',
        label_weight='bold'):
    """Generic bilobal-kinase silhouette: a smaller upper N-lobe
    and a larger lower C-lobe, with an active-site cleft on the
    right side. Used for both MEK and (free / phospho-) ERK; the
    callers vary face colors and overlay decorations to identify
    the particular protein."""
    import matplotlib.patches as mpatches
    s = scale
    # C-lobe (larger, lower) — α-helix-rich substructure
    ax.add_patch(mpatches.Ellipse(
        (cx - 0.002 * s, cy - 0.018 * s),
        0.085 * s, 0.058 * s, angle=-6,
        linewidth=1.0, edgecolor=edge, facecolor=face_c, zorder=3))
    # N-lobe (smaller, upper) — β-sheet substructure
    ax.add_patch(mpatches.Ellipse(
        (cx - 0.005 * s, cy + 0.024 * s),
        0.058 * s, 0.043 * s, angle=12,
        linewidth=1.0, edgecolor=edge, facecolor=face_n, zorder=3))
    # Tiny "hinge" highlight where the two lobes meet on the left
    ax.plot(
        [cx - 0.030 * s, cx - 0.020 * s],
        [cy + 0.005 * s, cy + 0.000 * s],
        color=edge, linewidth=0.6, alpha=0.5, zorder=3)
    # When the active site is occupied, draw a phosphate disc in
    # the cleft (right side, between lobes).
    if p_in_cleft:
        px = cx + 0.038 * s
        py = cy + 0.005 * s
        ax.add_patch(mpatches.Circle(
            (px, py), 0.012 * s,
            edgecolor='#7c4a00', facecolor='#fde047',
            linewidth=0.8, zorder=4))
        ax.text(px, py, 'P', ha='center', va='center',
                fontsize=4.6, weight='bold', color='#5a3500',
                zorder=5)
    if label:
        ax.text(cx, cy - 0.072 * s, str(label),
                ha='center', va='top', fontsize=6.6,
                weight=label_weight,
                color=label_color, zorder=5)


def _draw_mek(ax, cx, cy, label, bound=False, scale=1.0):
    """MEK1 — the kinase. Bilobal kinase fold; cleft is occupied
    when MEK is in complex with a substrate."""
    _draw_kinase_bilobal(
        ax, cx, cy, label,
        face_n='#9ecae1', face_c='#74a9cf', edge='#1c4a78',
        scale=scale, p_in_cleft=bound,
        label_color='#1c4a78', label_weight='bold')


def _draw_erk(ax, cx, cy, label, phosphorylated=False, scale=1.0):
    """ERK2 (and pERK) — drawn as a bilobal kinase silhouette,
    smaller and rotated slightly compared to MEK so the two are
    visually distinct in compartment views.  Phospho-ERK shows two
    P discs on its activation loop (Thr183 / Tyr185) — the
    canonical TXY motif highlighted by Canagarajah et al. 1997
    (PDB 2ERK)."""
    import matplotlib.patches as mpatches
    s = 0.7 * scale  # ERK drawn ~70% the size of MEK
    face_n = '#abdb98' if not phosphorylated else '#f6a285'
    face_c = '#7fbf7b' if not phosphorylated else '#ef8a62'
    edge = '#3a7d3a' if not phosphorylated else '#a04020'
    # C-lobe
    ax.add_patch(mpatches.Ellipse(
        (cx + 0.003 * s, cy - 0.014 * s),
        0.075 * s, 0.052 * s, angle=8,
        linewidth=1.0, edgecolor=edge, facecolor=face_c, zorder=3))
    # N-lobe
    ax.add_patch(mpatches.Ellipse(
        (cx + 0.006 * s, cy + 0.021 * s),
        0.052 * s, 0.038 * s, angle=-12,
        linewidth=1.0, edgecolor=edge, facecolor=face_n, zorder=3))
    # Hinge highlight (other side from MEK to keep them visually distinct)
    ax.plot(
        [cx + 0.028 * s, cx + 0.018 * s],
        [cy + 0.004 * s, cy + 0.000 * s],
        color=edge, linewidth=0.6, alpha=0.5, zorder=3)
    if phosphorylated:
        # Activation-loop "tail" sticks out the lower-right edge,
        # carrying two P discs (T-X-Y dual phosphorylation).
        for i, (dx, dy) in enumerate([(0.038, 0.005), (0.044, -0.012)]):
            px = cx + dx * s
            py = cy + dy * s
            ax.add_patch(mpatches.Circle(
                (px, py), 0.010 * s,
                edgecolor='#7c4a00', facecolor='#fde047',
                linewidth=0.7, zorder=4))
            ax.text(px, py, 'P', ha='center', va='center',
                    fontsize=4.0, weight='bold', color='#5a3500',
                    zorder=5)
    if label:
        ax.text(cx, cy - 0.060 * s, str(label),
                ha='center', va='top', fontsize=6.3,
                color='#222', zorder=5)


def _draw_npc(ax, cx, cy, label, radius=0.038, scale=1.0):
    """Nuclear pore complex: an annulus with the canonical
    eight-fold rotational symmetry of the NPC's nucleoporin
    arrangement (Lin & Hoelz 2019)."""
    import matplotlib.patches as mpatches
    r = radius * scale
    ax.add_patch(mpatches.Circle(
        (cx, cy), r,
        linewidth=1.0, edgecolor='#555', facecolor='#cfcfcf',
        zorder=2))
    ax.add_patch(mpatches.Circle(
        (cx, cy), r * 0.42,
        linewidth=0.8, edgecolor='#555', facecolor='#fdfdf9',
        zorder=3))
    for k in range(8):
        a = k * math.pi / 4
        ax.plot(
            [cx + r * 0.42 * math.cos(a), cx + r * math.cos(a)],
            [cy + r * 0.42 * math.sin(a), cy + r * math.sin(a)],
            color='#888', linewidth=0.5, zorder=2)
    if label:
        ax.text(cx, cy - r * 1.6, str(label),
                ha='center', va='top', fontsize=5.7, color='#555',
                style='italic', zorder=5)


def _draw_bond(ax, x1, y1, x2, y2, alpha=1.0):
    """Shared edge = MEK·pERK noncovalent bond. A dark-green
    parallel double-line so it reads as a chemical interaction
    rather than a wire."""
    L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if L < 1e-6:
        return
    ox = -(y2 - y1) / L * 0.005
    oy = (x2 - x1) / L * 0.005
    for sgn in (1, -1):
        ax.plot([x1 + sgn * ox, x2 + sgn * ox],
                [y1 + sgn * oy, y2 + sgn * oy],
                color='#2ca02c', linewidth=1.5, zorder=4,
                alpha=alpha, solid_capstyle='round')


# ── Layout: deterministic per-entity slot positions ────────────────


def _position_in_compartment(comp_name, entity_name, control):
    """Absolute (x, y) for an entity sitting in the named compartment.

    Slot positions are stable per (compartment, entity-name) so
    animation interpolates smoothly: when erk1 leaves cytoplasm
    for nucleus its destination slot is fixed in advance.

    The slots inside each compartment are arranged to fit visually
    — three positions inside the small nucleus circle, three along
    the ER cisternae, three in the cytosolic margins around them."""
    if control == 'MEK':
        # MEK lives in the cytoplasm, in the narrow strip between
        # nucleus (left) and ER lumen (right) — the natural focal
        # point of the cell cartoon.
        return (0.50, 0.50)

    if comp_name == 'cytoplasm':
        slots = [
            (0.50, 0.18),  # bottom-centre
            (0.50, 0.84),  # top-centre
            (0.10, 0.50),  # far left
        ]
    elif comp_name == 'nucleus':
        cx, cy = NUCLEUS_CENTER
        slots = [
            (cx - 0.078, cy + 0.005),  # left of nucleolus
            (cx - 0.020, cy + 0.075),  # upper centre
            (cx - 0.020, cy - 0.075),  # lower centre
        ]
    elif comp_name == 'er_lumen':
        ex, ey = ER_CENTER
        slots = [
            (ex - 0.085, ey + 0.06),   # upper-left of cisterna
            (ex + 0.085, ey + 0.05),   # upper-right
            (ex - 0.060, ey - 0.07),   # lower-left
        ]
    else:
        return (0.5, 0.5)

    name_to_slot = {'erk1': 0, 'erk2': 1, 'erk3': 2}
    idx = name_to_slot.get(entity_name, abs(hash(entity_name)) % 3)
    return slots[idx]


def _layout_state(state, compartment_names=None):
    """Compute ``{label: (cx, cy, control, wire_or_None, comp_name)}``
    per entity for a given state. Used by both the static snapshot
    drawer and the smooth animation.

    Two-pass: stable per-entity slots first, then bound substrates
    are *snapped into the kinase's active-site cleft* so the
    Michaelis complex visually reads as 'substrate inside enzyme'
    rather than 'two adjacent shapes connected by a line'."""
    layout = {}

    def walk(node, comp_name):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            ctrl = v.get('_type', '')
            if ctrl == 'Compartment':
                walk(v, comp_name=k)
                continue
            if ctrl in ('MEK', 'ERK', 'pERK'):
                label = v.get('name', k)
                cx, cy = _position_in_compartment(comp_name, label, ctrl)
                outs = v.get('outputs')
                wire = None
                if isinstance(outs, dict) and outs:
                    first = next(iter(outs.values()))
                    if isinstance(first, list):
                        wire = tuple(first)
                layout[label] = (cx, cy, ctrl, wire, comp_name)
                continue
            # kind tags or other markers — recurse but don't position
            walk(v, comp_name)

    walk(state, comp_name=None)

    # Cleft-snap: any (MEK, pERK) pair sharing an edge gets the
    # substrate moved into the kinase's active-site position.
    # This makes binding visually unmistakable.
    by_wire = {}
    for nm, (cx, cy, ctrl, wire, _) in layout.items():
        if wire is None:
            continue
        by_wire.setdefault(wire, []).append((nm, cx, cy, ctrl))
    for endpoints in by_wire.values():
        if len(endpoints) != 2:
            continue
        mek = next(((n, x, y) for n, x, y, c in endpoints if c == 'MEK'),
                   None)
        sub = next(((n, x, y) for n, x, y, c in endpoints if c == 'pERK'),
                   None)
        if mek is None or sub is None:
            continue
        sub_name = sub[0]
        mek_cx, mek_cy = mek[1], mek[2]
        # MEK's cleft sits to the right of its centre; place the
        # substrate just outside the cleft, slightly nestled.
        cleft_x = mek_cx + 0.045
        cleft_y = mek_cy - 0.005
        old = layout[sub_name]
        layout[sub_name] = (
            cleft_x, cleft_y, old[2], old[3], old[4])
    return layout


def _draw_cell_snapshot(ax, state, compartment_names=None, title=''):
    """Draw a biology-style cell cartoon at a given state: the
    plasma membrane forms the cell outline; cytoplasm is the visible
    interior; the nucleus and ER lumen are nested organelles drawn
    on top.  Molecular cartoons (MEK, ERK, pERK) are placed in slots
    inside whichever compartment they currently occupy. Bonds appear
    as a double-line chemical-bond glyph between cartoons."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)

    # Plasma membrane / cytoplasm
    _draw_cell(ax)
    ax.text(0.5, 0.97, 'cell', ha='center', va='bottom',
            fontsize=8, color='#3a5b3a', style='italic', zorder=6)

    # Inner organelles — only draw the ones actually present in the
    # state, but for our model both are always present.
    have_nucleus = any(
        c.get('kind', {}).get('_type') == 'Nucleus'
        for _, c in _iter_compartments(state))
    have_er = any(
        c.get('kind', {}).get('_type') == 'ERLumen'
        for _, c in _iter_compartments(state))
    if have_nucleus:
        _draw_nucleus(ax)
    if have_er:
        _draw_er_lumen(ax)

    # Molecules at their slot positions
    layout = _layout_state(state)
    bound_names = set()
    for label, (cx, cy, ctrl, wire, _) in layout.items():
        bound = wire is not None
        if bound and ctrl != 'MEK':
            bound_names.add(label)
        if ctrl == 'MEK':
            _draw_mek(ax, cx, cy, label, bound=bound)
        elif ctrl == 'ERK':
            _draw_erk(ax, cx, cy, label, phosphorylated=False)
        elif ctrl == 'pERK':
            _draw_erk(ax, cx, cy, label, phosphorylated=True)

    # Bond glyphs between paired wire endpoints
    by_wire = {}
    for label, (cx, cy, ctrl, wire, _) in layout.items():
        if wire is None:
            continue
        by_wire.setdefault(wire, []).append((cx, cy))
    for endpoints in by_wire.values():
        if len(endpoints) < 2:
            continue
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                (ax_, ay_), (bx, by) = endpoints[i], endpoints[j]
                _draw_bond(ax, ax_, ay_, bx, by)


# ── Initial-state diagram + biology-first explainer ─────────────────


_BRS_EXPLAINER = (
    r"$\bf{Biology}$" + "\n"
    "MAP-kinase signalling: the cytoplasmic kinase MEK1\n"
    "(PDB 3EQH) phosphorylates ERK2 (PDB 1ERK / 2ERK) on\n"
    "its Thr-X-Tyr activation motif. Phospho-ERK then\n"
    "translocates through nuclear pore complexes (NPCs;\n"
    "Lin & Hoelz 2019) into the nucleus, where it\n"
    "phosphorylates downstream transcription factors. MEK\n"
    "is active-site limited — at most one ERK may occupy\n"
    "its catalytic cleft at a time.\n"
    "\n"
    r"$\bf{Bigraph\ encoding}$" + "\n"
    "  Place graph (nested):\n"
    "    Cell ⊃ Cytoplasm ⊃ {Nucleus, ERLumen, MEK, ERK, pERK}\n"
    "  Link graph: one shared edge per MEK·pERK complex.\n"
    "  Sorts (_type): Cell, Compartment, NPC, MEK, ERK, pERK,\n"
    "                Cytoplasm, Nucleus, ERLumen  (kind tags)\n"
    "\n"
    r"$\bf{Rules}$" + "  (Gillespie SSA, propensity = k × |matches|):\n"
    "  • phosphorylate        (k=2.0): ERK + MEK co-located,\n"
    "      both unbound (Absent), bind via a fresh edge.\n"
    "  • dissociate           (k=0.5): MEK·pERK → free MEK\n"
    "      + free pERK (bond destroyed).\n"
    "  • dephosphorylate      (k=0.4): nuclear pERK → ERK\n"
    "      (DUSP-style phosphatase; closes the cycle).\n"
    "  • translocate_erk_in   (k=1.0): free ERK,\n"
    "      cytoplasm → child compartment.\n"
    "  • translocate_erk_out  (k=1.0): free ERK,\n"
    "      child → cytoplasm (symmetric diffusion).\n"
    "  • translocate_perk_in  (k=2.0): free pERK,\n"
    "      cytoplasm → nucleus  (active import).\n"
    "  • translocate_perk_out (k=0.1): free pERK,\n"
    "      nucleus → cytoplasm  (slow leak).\n"
    "  The 20-fold in/out asymmetry is the structural\n"
    "  source of nuclear pERK accumulation. Nesting also\n"
    "  makes nucleus ↔ ER direct transit inexpressible.\n"
    "\n"
    r"$\bf{Why\ bigraphs?}$" + "\n"
    "Compartment-only models capture which pool a molecule\n"
    "is in, but not which kinase is bound to which\n"
    "substrate. Reaction-network (mass-action) models\n"
    "capture binding (S+E ⇌ SE) but lose the spatial\n"
    "dimension. Bigraphs unify both: a single redex\n"
    "constrains place AND link in one step — e.g.\n"
    "phosphorylate requires co-location AND no prior bond.\n"
    "\n"
    "See brs_mapk_references.md for structural and biological\n"
    "citations (Milner 2009; Archibald et al. 2024; Zhang et al.\n"
    "1994; Canagarajah et al. 1997; Ohren et al. 2004; Lin &\n"
    "Hoelz 2019; Plotnikov et al. 2011)."
)


_BRS_EXPLAINER_MARKDOWN = """\
### Biology

MAP-kinase signalling: the cytoplasmic kinase **MEK1** (PDB
`3EQH`) phosphorylates **ERK2** (PDB `1ERK` / `2ERK`) on its
Thr-X-Tyr activation motif. Phospho-ERK then translocates
through nuclear pore complexes (NPCs; Lin & Hoelz 2019) into
the nucleus, where it phosphorylates downstream transcription
factors. MEK is active-site limited — at most one ERK may
occupy its catalytic cleft at a time.

### Bigraph encoding

- **Place graph** (nested):
  `Cell ⊃ Cytoplasm ⊃ {Nucleus, ERLumen, MEK, ERK, pERK}`
- **Link graph**: one shared edge per MEK·pERK complex.
- **Sorts** (`_type`): `Cell`, `Compartment`, `MEK`, `ERK`,
  `pERK`, plus the compartment-kind tags `Cytoplasm`,
  `Nucleus`, `ERLumen`. Each sort is registered as a schema
  type so it can later carry typed methods that compose with
  the rewrite rules.

### Rules (Gillespie SSA, propensity = `k × |matches|`)

- **`phosphorylate`** (k = 2.0): ERK + MEK co-located in the
  cytoplasm, both unbound (`Absent` preimage), bind via a
  fresh shared edge.
- **`dissociate`** (k = 0.5): MEK·pERK → free MEK + free
  pERK; the bond is destroyed.
- **`dephosphorylate`** (k = 0.4): free pERK in the nucleus
  → free ERK. Models nuclear MAP-kinase phosphatases (DUSPs)
  resetting the kinase. Closes the cycle so the system keeps
  firing instead of stalling at "all pERK in nucleus".
- **`translocate_erk_in`** (k = 1.0): free ERK,
  cytoplasm → child compartment.
- **`translocate_erk_out`** (k = 1.0): free ERK,
  child → cytoplasm. Symmetric diffusion.
- **`translocate_perk_in`** (k = 2.0): free pERK,
  cytoplasm → nucleus. Active import.
- **`translocate_perk_out`** (k = 0.1): free pERK,
  nucleus → cytoplasm. Slow leak.

The **20-fold in/out asymmetry** for pERK is the structural
source of nuclear pERK accumulation — the signal itself.
Nesting also makes nucleus ↔ ER direct transit
inexpressible: it isn't a parent/child pair, so no rule
redex names it.

### Why bigraphs?

Compartment-only models capture which pool a molecule is in,
but not which kinase is bound to which substrate.
Reaction-network (mass-action) models capture binding
(S + E ⇌ SE) but lose the spatial dimension. Bigraphs unify
both: a single redex constrains place AND link in one step
— e.g. `phosphorylate` requires co-location AND no prior
bond.

See `spatio_flux/processes/brs_mapk_references.md` for
structural and biological citations (Milner 2009; Archibald
et al. 2024; Zhang et al. 1994; Canagarajah et al. 1997;
Ohren et al. 2004; Lin & Hoelz 2019; Plotnikov et al. 2011).
"""


def _save_explainer_markdown(filename, out_dir):
    """Write the biology + bigraph encoding + rules explainer to a
    standalone markdown file alongside the figures, so the report
    can render it as proper text instead of pixels in an image."""
    import os
    path = os.path.join(out_dir, f'{filename}_overview.md')
    with open(path, 'w') as f:
        f.write(_BRS_EXPLAINER_MARKDOWN)


def _save_state_json(filename, out_dir, state):
    """Overwrite ``<filename>_state.json`` with the actual nested
    cell tree.  The composite-level serializer collapses our
    ``tree[node]`` cell slot to ``{}`` because each child looks
    like a node leaf to it; this preserves the structure for the
    report's JSON viewer."""
    import os
    import json
    path = os.path.join(out_dir, f'{filename}_state.json')
    with open(path, 'w') as f:
        json.dump({'cell': state}, f, indent=2, default=str)


_BIGRAPH_FILL_COLORS = {
    'MEK':  '#9ecae1',          # kinase blue
    'ERK':  '#abdb98',          # substrate green
    'pERK': '#f6a285',          # phospho orange
    'Site': '#d8d8d8',          # rule-pattern placeholder
    'Compartment_Cytoplasm': '#e3edd7',  # cytosolic pale green
    'Compartment_Nucleus':   '#f5e6c4',  # nucleoplasm pale beige
    'Compartment_ERLumen':   '#f4d3a8',  # ER pale peach
    'Cell':                  '#f5efde',  # plasma membrane cream
}

# Neutral fill for any node we haven't given a specific colour
# (intermediate dict keys like ``outputs``, port names like
# ``cleft``, name leaves, rule role-labels, etc.). Light enough not
# to compete with the molecule fills, dark enough to read as
# coloured rather than as the bare-white "no-fill" default.
_BIGRAPH_DEFAULT_FILL = '#dde2e8'

# Light yellow for wire-path leaves (the link-graph endpoints) so
# they read as paired with the gold dashed link edge that
# bigraph-viz now draws between them.
_BIGRAPH_WIRE_FILL = '#fff3b0'


def _bigraph_fill_colors_for_state(state, prefix=()):
    """Walk the entire tree and emit ``{path: color}`` for every
    place-graph node, so that ``plot_bigraph`` renders nothing as
    bare-white-with-thick-outline. Typed nodes get their type
    colour; compartments are tinted by their ``kind`` child; wire
    endpoints get the link-graph yellow; every remaining node gets
    a neutral medium-gray."""
    fills = {}

    def walk(node, path):
        # Leaves: scalars and wire paths.
        if isinstance(node, (list, tuple)):
            if (path and len(node) >= 2 and node and node[0] == '_edges'):
                fills[path] = _BIGRAPH_WIRE_FILL
            return
        if not isinstance(node, dict):
            if path:
                fills[path] = _BIGRAPH_DEFAULT_FILL
            return

        ctrl = node.get('_type', '')
        color = None
        if ctrl == 'Compartment':
            kind = node.get('kind', {}).get('_type', '')
            color = _BIGRAPH_FILL_COLORS.get(f'Compartment_{kind}')
        elif ctrl in _BIGRAPH_FILL_COLORS:
            color = _BIGRAPH_FILL_COLORS[ctrl]
        if path:
            fills[path] = color or _BIGRAPH_DEFAULT_FILL

        for k, v in node.items():
            if isinstance(k, str) and not k.startswith('_'):
                walk(v, path + (k,))

    walk(state, prefix)
    return fills


def _draw_rule_arrow(ax, color, label=None):
    """A simple horizontal arrow used between redex/reactum and
    between before/after panels in the trace figure. If ``label``
    is given, it sits above the arrow as a small caption."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate(
        '', xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(arrowstyle='->', linewidth=2, color=color))
    if label:
        ax.text(
            0.5, 0.62, label, ha='center', va='bottom',
            fontsize=9, weight='bold', color=color)


def _redex_pattern_to_state(node, edge_anchors=None):
    """Convert a redex/reactum pattern (which mixes plain dicts with
    Site / LinkVar / Absent placeholders) into a state dict that
    ``plot_bigraph`` can render. Sites become a labelled ``Site``
    leaf, each ``LinkVar('bond')`` becomes a wire path
    ``['_edges', '~e']`` — bigraph-viz's place-graph rendering
    then draws a real link-graph edge between every port whose
    value points to the same anchor. Absents are dropped
    (they're a precondition, not a structure)."""
    if edge_anchors is None:
        edge_anchors = set()
    if isinstance(node, Site):
        return {'_type': 'Site'}
    if isinstance(node, LinkVar):
        edge_anchors.add(node.name)
        # Drop the '~' anchor-fresh marker for display: the rule
        # patterns are static, so the edge label can use the
        # LinkVar name directly ('bond' rather than '~bond').
        return ['_edges', node.name]
    if isinstance(node, Absent):
        return None
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if isinstance(v, Absent):
                continue
            if isinstance(k, str) and k == '_type':
                out[k] = v
                continue
            converted = _redex_pattern_to_state(v, edge_anchors)
            if converted is None:
                continue
            out[k] = converted
        return out
    if isinstance(node, list):
        return [_redex_pattern_to_state(v, edge_anchors) for v in node]
    return node


def _render_rule_pattern(label, pattern, out_dir, suffix):
    """Render a redex or reactum pattern to ``<out>/_rule_{label}_
    {suffix}.png`` via ``plot_bigraph``. Returns the absolute path.

    Each LinkVar in the pattern becomes a wire path that
    bigraph-viz's link-graph renderer draws as an actual edge
    between paired endpoints (e.g. MEK·cleft and pERK·docking
    sharing ``LinkVar('bond')``)."""
    try:
        from bigraph_viz import plot_bigraph
        from process_bigraph import allocate_core
    except ImportError:
        return None
    import os
    edge_anchors = set()
    state = _redex_pattern_to_state(pattern, edge_anchors)
    if not isinstance(state, dict):
        return None
    core = allocate_core()
    register_mapk_types(core)
    fname = f'_rule_{label}_{suffix}'
    fills = _bigraph_fill_colors_for_state(state)
    plot_bigraph(
        state=state,
        core=core,
        out_dir=out_dir,
        filename=fname,
        dpi='120',
        show_values=False,
        show_compiled_state=False,
        node_fill_colors=fills)
    return os.path.join(out_dir, f'{fname}.png')


def _replot_initial_bigraph(filename, out_dir, state):
    """Re-render the standard ``<filename>_viz.png`` so it shows the
    full place graph of the initial state — Cell ⊃ Cytoplasm ⊃
    {Nucleus, ERLumen, MEK, ERK, pERK} — coloured to match the
    snapshot palette: MEK blue, ERK green, pERK orange, cytoplasm
    pale green, nucleus pale beige, ER pale orange."""
    try:
        from bigraph_viz import plot_bigraph
        from process_bigraph import allocate_core
    except ImportError:
        return
    core = allocate_core()
    register_mapk_types(core)
    fills = _bigraph_fill_colors_for_state(state, prefix=('cell',))
    plot_bigraph(
        state={'cell': state},
        core=core,
        out_dir=out_dir,
        filename=f'{filename}_viz',
        dpi='150',
        show_values=True,
        show_compiled_state=False,
        node_fill_colors=fills)


# ── Smooth animation with cubic ease-in-out ────────────────────────


def _ease_in_out_cubic(t):
    """Smoothstep cubic easing on [0, 1]: zero derivative at the
    endpoints, so motion eases into and out of each snapshot
    transition."""
    return 3 * t * t - 2 * t * t * t


def _make_smooth_animation(
        filename, out_dir, states, times, compartment_names, firings,
        n_intermediate=8, fps=10):
    """Generate a continuous animation by interpolating each
    entity's snapshot positions with cubic ease-in-out.

    For each snapshot pair (state[i], state[i+1]) we render
    ``n_intermediate`` frames whose entity positions are linear
    blends between the two layouts (after applying the easing
    function to the blend parameter). Control-state changes
    (e.g. ERK → pERK after a phosphorylate firing) and bond
    appearances/disappearances cross-fade in alpha across the
    middle of the transition.

    The output GIF is saved to ``<filename>_animation.gif``."""
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Pre-compute layouts and rule-firing markers per snapshot.
    layouts = [_layout_state(s) for s in states]
    # Track whether each state has nucleus / ER lumen so the
    # animation's per-frame draw doesn't render a missing organelle.
    has_nuc = [
        any(c.get('kind', {}).get('_type') == 'Nucleus'
            for _, c in _iter_compartments(s))
        for s in states]
    has_er = [
        any(c.get('kind', {}).get('_type') == 'ERLumen'
            for _, c in _iter_compartments(s))
        for s in states]
    # Map snapshot index → list of rule labels that fired in
    # the transition from that index to the next.
    firing_at = {}
    for t_after, rule, _, _ in firings:
        for i, t in enumerate(times):
            if abs(t - t_after) < 1e-6:
                firing_at[i - 1] = rule
                break

    # Build the frame schedule: each (i, k) corresponds to the
    # k-th of n_intermediate frames in the i→i+1 transition.
    schedule = []
    for i in range(len(states) - 1):
        for k in range(n_intermediate):
            schedule.append((i, k))
    # Hold the final frame for half a second
    for _ in range(max(n_intermediate // 2, 2)):
        schedule.append((len(states) - 1, 0))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    def _interp_layout(i, tau_eased):
        """Blend layouts[i] and layouts[i+1] at eased parameter
        ``tau_eased`` ∈ [0, 1]. Position: linear in tau_eased.
        Control: switches at tau_eased >= 0.5 (modelled as the
        moment a rule "fires" in this transition).
        Wire (bond): held until tau_eased >= 0.5, then switched."""
        la = layouts[i]
        lb = layouts[i + 1] if i + 1 < len(layouts) else layouts[i]
        out = {}
        names = set(la.keys()) | set(lb.keys())
        for nm in names:
            a = la.get(nm)
            b = lb.get(nm)
            if a is None and b is None:
                continue
            if a is None:
                a = b
            if b is None:
                b = a
            xa, ya, ca, wa, _ = a
            xb, yb, cb, wb, _ = b
            x = xa + tau_eased * (xb - xa)
            y = ya + tau_eased * (yb - ya)
            ctrl = cb if tau_eased >= 0.5 else ca
            wire = wb if tau_eased >= 0.5 else wa
            out[nm] = (x, y, ctrl, wire)
        return out

    def _draw_at(layout, label_top='', frame_idx=None):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Plasma membrane / cytoplasm
        _draw_cell(ax)
        ax.text(0.5, 0.97, 'cell', ha='center', va='bottom',
                fontsize=8, color='#3a5b3a', style='italic', zorder=6)
        # Inner organelles — toggle on the *target* state's presence
        # so they appear/disappear cleanly across firings (here both
        # always present, but kept general).
        idx = frame_idx if frame_idx is not None else 0
        idx = min(idx, len(has_nuc) - 1)
        if has_nuc[idx]:
            _draw_nucleus(ax)
        if has_er[idx]:
            _draw_er_lumen(ax)
        # Entities
        for nm, (cx, cy, ctrl, wire) in layout.items():
            if ctrl == 'MEK':
                _draw_mek(ax, cx, cy, nm, bound=(wire is not None))
            elif ctrl == 'ERK':
                _draw_erk(ax, cx, cy, nm, phosphorylated=False)
            elif ctrl == 'pERK':
                _draw_erk(ax, cx, cy, nm, phosphorylated=True)
        # Bonds
        by_wire = {}
        for nm, (cx, cy, ctrl, wire) in layout.items():
            if wire is None:
                continue
            by_wire.setdefault(wire, []).append((cx, cy))
        for endpoints in by_wire.values():
            if len(endpoints) < 2:
                continue
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    (ax_, ay_), (bx, by) = endpoints[i], endpoints[j]
                    _draw_bond(ax, ax_, ay_, bx, by)
        ax.set_title(label_top, fontsize=10)

    def render(item):
        i, k = item
        if i >= len(states) - 1:
            # Final hold frames: just draw the last layout.
            last = layouts[-1]
            simple = {nm: (cx, cy, ctrl, wire)
                      for nm, (cx, cy, ctrl, wire, _) in last.items()}
            _draw_at(simple, label_top=f't = {times[-1]:.1f}',
                     frame_idx=len(states) - 1)
            return
        tau = (k + 1) / (n_intermediate + 1)  # avoid 0 and 1
        eased = _ease_in_out_cubic(tau)
        layout = _interp_layout(i, eased)
        # Frame label: which rule fired in this transition
        rule = firing_at.get(i)
        t = times[i] + tau * (times[i + 1] - times[i])
        if rule:
            color = RULE_COLORS.get(rule, '#444')
            label = f't ≈ {t:.1f}    →  {rule}'
            ax_title = label
        else:
            ax_title = f't ≈ {t:.1f}'
        idx_for_organelles = i + 1 if eased >= 0.5 else i
        _draw_at(layout, label_top=ax_title, frame_idx=idx_for_organelles)
        # Rule-color tint on the title font when a firing is in progress
        if rule:
            ax.set_title(
                ax_title, fontsize=10, color=color, weight='bold')

    anim = FuncAnimation(
        fig, render, frames=schedule,
        interval=int(1000 / fps), repeat=True)
    gif_path = os.path.join(out_dir, f'{filename}_animation.gif')
    try:
        anim.save(gif_path, writer=PillowWriter(fps=fps))
    except Exception as e:
        print(f'⚠ animation save failed: {e}')
    plt.close(fig)


# =====================================================================
# Backward-compatibility aliases
# =====================================================================
#
# A handful of name aliases kept around so callers that imported
# the old "smart-building" identifiers don't break. Prefer the
# MAPK-named symbols above.

building_rules = mapk_rules
initial_building_state = initial_mapk_state
get_brs_building_doc = get_brs_mapk_doc
plot_brs_building = plot_brs_mapk
