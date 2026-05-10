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
