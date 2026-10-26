# Novel visually-rich SpatioFlux sims + viva-superpowers investigation-reconcile — design

**Date:** 2026-10-26
**Two coupled projects, two repos / two PRs:**
- **Part A** — replace the weak discovery-engine studies in `~/code/spatio-flux` with genuinely novel, visually-rich, complexity-raising simulations.
- **Part B** — a reusable `viva-superpowers` capability (`~/code/viva-superpowers`) that makes **reconciling an investigation after a structural edit** (studies removed / added / merged) easy and systematic. Built **first**, then dogfooded on Part A's restructuring.

Feasibility was probed against the real composites (see the probe findings table below); nothing here is unverified hand-waving.

---

## Why

The first discovery-engine round (PR #72) added 7 studies, but most asserted a **monotone response to a knob** ("more advection → plume goes further") — an engineering check that the composite runs as wired, not a biological discovery. This round removes those and builds simulations whose results are **qualitative / emergent** — a phenomenon the parameter name does not predict — each with a **great interactive visualization** (the composites are inherently spatial/temporal, so viz = animation, not line plots).

Separately, restructuring an investigation (this edit removes 5, rebuilds 1, adds 4) forces a large manual rewrite of the narrative spine and cross-references. That reconciliation should be a **known viva-superpowers operation**, not ad-hoc hand-editing — hence Part B.

## Goals

- Remove the 5 tautological studies; rebuild 1; add 4 novel studies — each with a **qualitative** code-evaluated contract (not a monotone-knob check) and an **interactive Plotly** visualization committed to the study, rendering in the published read-only workbench.
- Any new phenomenon that does **not** appear clearly is reported honestly (the study states it as an open question / negative result) — never tuned into a fake positive.
- A `/viva-investigation reconcile <inv>` capability in viva-superpowers that **deterministically detects** investigation drift after a structural edit and **guides the narrative rewrite**, following the existing AI-free-helper + agent-authoring split.
- Both land green (CI + the publish render-check gate), no AI attribution, squash-merge, no auto-merge.

## Non-goals / anti-goals

- **Anti-goal:** forcing a phenomenon. If the washout cliff isn't sharp, or sectors don't form cleanly, the study says so (honest negative / open question) — do not retune until a number passes.
- Not touching the 19 hardened reproduction studies or the 8 figure studies (this round is only the new-biology set).
- Part B does not auto-rewrite prose silently — it detects drift deterministically and the agent authors the rewrite (same contract as `study_narrative` / `findings`). No AI content is written by the dashboard/helper.
- Not reworking the whole investigation schema — reconcile operates on the existing v2/v3/v4 narrative fields.

---

## Feasibility probe findings (verified 2026-10-26, modern repro venv)

| Direction | Verdict | Evidence / what it needs |
|---|---|---|
| **Physics+metabolism biofilm** | READY (existing composite) | `comets_nt_particles_dfba`: particles settle under gravity (mean y 18.7→7.5), feed (mass grows), divide (3→4). Pymunk rigid-body + per-particle dFBA + fields. |
| **Sectoring** | Needs schema + emitter change (CORRECTED) | `child = dict(parent)` carries a `strain` tag *only in transient state* — `core.serialize('map[particle]', …)` **strips** it (not in the workspace particle type), and `run_store` persists only position/mass, so strain never reaches the zarr. So N2 needs: add `strain` to the workspace particle type, persist a `particle_strain` (time,particle) array in `run_store`, and seed strains via a composite param. All additive workspace changes (not bigraph-schema core). |
| **Spatial cross-feeding** | Confirmed buildable + ran | `community_dfba` (acetate_only pop) shows acetate rise-then-fall (0→0.18@14→0) and the acetate-only biomass grows. `biomass_id` is a per-process param and `dfba_kinetics_community` already wires **two** biomass fields sharing one acetate field — so the 2-field spatial composite is a wiring exercise (builder confirmed it builds + runs). |
| **Washout transition** | CONFIRMED (field biomass) | CORRECTED: `dissolved biomass` **is** an advected field — with `boundary_conditions: outflow`/`dirichlet` + advection it washes out (builder verified total **5.0→0.009**, a real cliff candidate). Use the FIELD representation (not particles). Needs a `total_field_biomass` metric added to `run_sweep`. |

---

## Part A — the studies

Removal set (delete): `advection_plume_phase`, `division_seeding_regimes`, `glucose_limitation_yield`, `spatial_growth_front`, `spatial_vs_wellmixed_community`. **Keep** `nutrient_starvation_control` (adversarial rigor floor). **Rebuild** `spatial_overflow_crossfeeding` → the spatial cross-feeding study below.

### New study N1 — `spatial_cross_feeding_bridge` (rebuilds the overflow study)
- **Composite (NEW):** `spatial_cross_feeding` — a 2D lattice with strain A (overflow: glucose→acetate) seeded on one side and strain B (`acetate_only`: acetate→biomass) seeded on the other, sharing a diffusing acetate field; separate `biomass_A` / `biomass_B` fields. Built from the existing Monod/dFBA processes + diffusion (wiring, not new science).
- **Novel phenomenon:** B grows **only where A's acetate has diffused** — an emergent spatial partnership. B's biomass centroid tracks A's acetate plume; B cannot grow without A (shown by an A-absent control).
- **Contract (qualitative):** (a) acetate shows `peak_then_decline` (produced then consumed); (b) B's biomass grows **and** its spatial overlap with the acetate field exceeds its overlap with glucose (a **spatial-separation / bridge** metric — new evaluator `spatial_overlap`); (c) A-absent control: B does not grow (`observable_flat`).
- **Viz:** interactive Plotly — animated 3-panel field movie (glucose | acetate bridge | biomass_A+biomass_B) with a time slider; a second panel tracing B-centroid vs acetate-front position.

### New study N2 — `range_expansion_sectoring`
- **Enabling changes (additive, workspace-level):** add `"strain": "string"` to the workspace particle type (`SIMPLE_PARTICLE_TYPE` in `spatio_flux/__init__.py`) so serialize/emission preserves it; persist a `particle_strain` (time,particle) array in `run_store._field_dataarrays` (parallel to `particle_x/y/mass`); add a `strains`/`strain_fractions` seeding param to the particle composite (there is no "variant" concept — only `parameters`). Division already carries the field via `dict(parent)`.
- **Composite:** a thin `comets_br_particles_dfba` variant seeded with **two strain-tagged particle populations** at a founding patch.
- **Novel phenomenon:** as the colony expands, neutral strains **demix into sectors** (genetic-drift-like domains) rather than staying mixed — a hallmark of range expansions.
- **Contract (qualitative):** a **demixing / sector metric** (new evaluator `demixing_index`) — local strain purity at the front rises over time vs a well-mixed null (strains that start interleaved segregate). Robust over ≥3 seeds (stochastic).
- **Viz:** interactive Plotly particle-movie — scatter of particle positions colored by strain, size = mass, animated over time; the sector structure is visible as colored wedges at the expanding front.

### New study N3 — `advection_washout_transition`
- **Composite (NEW):** `comets_washout` — field dFBA + advection–diffusion of the **`dissolved biomass` field** with an **outflow/dirichlet boundary** at the downstream edge, so the advected biomass field exits the domain (builder verified total 5.0→0.009). (Field representation, not particles — `dissolved biomass` genuinely advects.)
- **Novel phenomenon:** a **critical advection (Péclet) threshold** — below it the colony grows and persists; above it growth is outrun by flow and the population **washes out** (survival collapses). A qualitative transition, not a monotone trend.
- **Contract (qualitative):** sweep advection; the final-biomass-vs-advection curve has a **sharp cliff** — new evaluator `threshold_transition` (detects a step/critical point: sustained-high below, near-zero above, with a localized drop). Add a `total_field_biomass` metric to `run_sweep._metric` (currently only `n_particles`/`centroid_y`). If no sharp cliff appears, report it honestly as "no clean transition at these parameters" (open question).
- **Viz:** interactive Plotly — survival phase portrait (final biomass vs advection, cliff annotated) + two side-by-side field movies (persist vs washout) across the transition.

### New study N4 — `biofilm_settling_metabolism`
- **Composite:** `comets_nt_particles_dfba` (existing, READY).
- **Novel phenomenon:** rigid-body cells **settle and pack under gravity while feeding**, forming a dense basal layer where metabolism couples to mechanics — a nutrient gradient develops through the packed cells (cells at the nutrient-facing edge grow faster → spatially structured growth from physics, not imposed).
- **Contract (qualitative):** (a) cells settle (mean particle y decreases, gravity); (b) a **growth gradient** correlates with height (mass gain vs y position) — a `correlation`/gradient evaluator; (c) population grows + packs (count ↑, vertical extent saturates). Multiseed.
- **Viz:** interactive Plotly particle-movie — cells as circles sized by mass, positioned in 2D over the nutrient field heatmap, animated as they settle + grow + divide.

### New composites (flagged, behavior-additive — the "complexity bar-raising")
- `spatial_cross_feeding` (N1), `comets_washout` (N3): new `@composite_generator`s wiring existing processes (Monod/dFBA/diffusion) in new spatial arrangements (two biomass fields; outflow BC). `range_expansion_sectoring` (N2): a thin `comets_br_particles_dfba` variant PLUS the additive strain enablers (workspace particle-type field + `run_store` `particle_strain` array + strain-seeding param — see N2). These are built via the `/viva-expert --in-workspace` discipline (real processes, no mocks).

### New evaluators (so contracts stay qualitative, not monotone)
Added to `spatio_flux/evaluators.py` via `register_evaluators`, read the zarr directly, TDD with synthetic fixtures:
- `spatial_overlap` — correlation/overlap of two fields (B-biomass vs acetate vs glucose) → cross-feeding bridge.
- `demixing_index` — strain-purity / sector metric from the new `particle_strain` + `particle_x/y` zarr arrays over time (requires the N2 schema+emitter change above).
- `threshold_transition` — detects a sharp critical point in a metric-vs-axis sweep series (cliff, not monotone).
- `gradient_correlation` — correlation of a per-particle quantity (mass gain) with position (height) → physics-coupled growth.

### Interactive Plotly viz (the bar)
Self-contained Plotly HTML committed under `studies/<slug>/viz/<name>.html`, generated by the study runner from the run zarr: animated field heatmaps (time slider + play), particle-movie scatter animations (color = strain / size = mass), phase portraits. **Self-contained via `fig.to_html(..., include_plotlyjs=True)`** (inline Plotly, not an absolute CDN tag — the publish inliner only self-contains a relative loader). Referenced in `visualizations:` with the canonical scheme: `address: html:viz/<name>.html`, `chart: html` (a bare `.html` is lint-blocked). The renderer/inliner (iframe `srcdoc`) lives in the `vivarium-workbench` repo — no change needed there, just emit valid declared HTML.

### Runners
Extend `run_new_study.py` (single/multi-run) and `run_sweep.py` (multiseed) to also emit the Plotly HTML. `reproduce.py` already dispatches each study's `canonical_runs` script.

---

## Part B — `viva-superpowers` investigation-reconcile capability

**Where:** `~/code/viva-superpowers` (its own PR). **Build on `origin/main` first** — the local checkout is 56 commits behind and `study_narrative.py` was DELETED (#219, moved to workbench). Mirror the live AI-free/agent-authoring analogues `finding_observations.py` + `seed_from_followup.py`, and write YAML via `investigation_canonicalize._ruamel()` (comment-preserving) — NOT `study_io.dump_yaml` (strips the scaffold's TODO-comment narrative sections). Also builds on `report_linter.py` + `investigation_import.investigation_members()`.

**New:** `/viva-investigation reconcile <inv>` subcommand + `viva_superpowers.investigation_reconcile` helper.

**AI-free drift detection** (`investigation_reconcile.detect_drift(ws, inv) -> report`), deterministic, no writes — targeting the REAL schema (verified against the dogfood target):
- Membership resolved via the existing `investigation_import.investigation_members()` (unions `studies:` AND `members:` — the canonicalizer renames `studies`→`members`; the detector must not read one key only). Report members missing on disk (removed) and on-disk studies not listed (orphan).
- **Prose name-matching** for references to a removed study (the real investigation carries these as FREE TEXT with embedded names, not structured `{study:}` refs): scan `executive.verdict`/`what_is_this`, `scientific_argument.evidence_for`/`evidence_against`/`key_figures` (incl. embedded `studies/<slug>/...png` chart paths), `biological_story`. Substring match on the removed slug/title → flag (accept false-positive risk; the agent confirms before editing, never auto-deletes prose).
- Structured refs where present: `acceptance_criteria[].behavior` (key is `behavior`, holding a `behavior_tests[].name` — not prose), `at_a_glance` (tolerate BOTH shapes: dict-with-`studies` and list-of-`{study,role}`), `pipeline_gate.prerequisites` (the canonical per-study prereq edge — `parent_studies` is a legacy alias, not an investigation-graph edge).
- Tolerate MISSING sections: the dogfood target has no `at_a_glance`/`acceptance_criteria`/`glossary` — absent sections are "nothing to reconcile," not errors.
- Stale rollup: `executive.verdict`/`verdict_status` mentioning gone studies or not matching the surviving member set.
- Each drift item: `{kind, where (dotted path), detail, suggested_action}`.

**Agent-guided reconciliation** (the skill prose encodes "what to do"):
1. Run `detect_drift` → show the drift report grouped by kind.
2. For each item, the skill tells the agent the fix: drop removed-study entries from `studies`/`at_a_glance`/`acceptance_criteria`; rewrite `scientific_argument.evidence_for` to the surviving + new studies; refresh `executive.verdict`/`verdict_status` to reflect the new member set; reconcile `biological_story` mentions; re-point/prune `parent_studies` edges; add `at_a_glance` roles for new studies. Writes go through the existing YAML-direct path (ruamel, comment-preserving), same as `study_narrative`.
3. Re-run `detect_drift` → confirm zero residual drift.
4. Optional `--merged a,b->c` hint: when two studies are reconciled into one, record provenance (`reconciled_from: [a,b]`) so the lineage is visible.

**Skill doc:** a `reconcile` section in `skills/viva-investigation/SKILL.md` with the checklist above + when to run it (after any add/remove/merge of member studies). Tests in `viva-superpowers/tests/` pin `detect_drift` on fixtures (removed-study-referenced-in-evidence, orphan study, dangling graph edge).

**Dogfood:** after Part B merges, run `/viva-investigation reconcile spatio-flux-test-suite` as the mechanism that reconciles the narrative during Part A's restructuring — proving the capability on the real edit.

---

## Sequencing & integration

1. **Part B first** (viva-superpowers PR): build + test `investigation_reconcile` + the skill `reconcile` section. Review → user-approved squash-merge.
2. **Part A** (spatio-flux PR): remove the 5 studies; build the new composites + evaluators (TDD) + the 4 studies + Plotly viz; use the newly-shipped `/viva-investigation reconcile` to reconcile the investigation narrative; `reproduce.py` + the render-check pass locally. Review → user-approved squash-merge → gh-pages republish.

Both: dedicated worktrees, no AI attribution, no auto-merge, verify-before-complete. The spatio-flux sweep/new studies keep the `conditions.baseline` representative-arm requirement (render-check gate) learned in PR #72.

## Resolved decisions
- Directions: all four (user). Viz: interactive Plotly (user). Removal: the 5, keep `nutrient_starvation_control`, rebuild overflow (user).
- Phenomenon-risk: **washout is now CONFIRMED** (field biomass 5.0→0.009); the remaining phenomenon-risk is **sectoring** (needs a dense enough front for distinct sectors) — handled honestly per the anti-goal.

## Calls I made after the builder-check (veto any)
1. **N2 sectoring needs real enabling work, not a param:** add `strain` to the workspace particle type + persist `particle_strain` in `run_store` + a strain-seeding param. All additive workspace changes. **I'm proceeding with this** (it's the bar-raising for sectoring); say so if you'd rather drop N2 instead of touching the particle type/emitter.
2. **Washout uses the FIELD biomass representation** (`dissolved biomass` + outflow BC), confirmed to wash out — not particles. Add a `total_field_biomass` sweep metric.
3. **Part B builds on `origin/main`** (local is 56 behind; `study_narrative.py` deleted) and mirrors `finding_observations.py`/`seed_from_followup.py` + `investigation_canonicalize._ruamel()` comment-preserving writes.
4. **detect_drift targets the real schema:** membership via `investigation_members()`; prose name-matching (agent confirms, never auto-deletes) for free-text narrative; tolerate both `at_a_glance` shapes and missing sections.
5. **Viz:** `address: html:viz/<name>.html` + `chart: html`, self-contained via `include_plotlyjs=True`.

## Open questions for the user
- None blocking. The one worth a conscious nod is call #1 (touching the workspace particle type + emitter to enable sectoring) — otherwise the plan is settled.
