# References — MAPK signalling as a Bigraphical Reactive System

This bibliography accompanies the BRS-MAPK example in
`bigraph_reactive_system.py` (and the corresponding
`brs_mapk` entry in `experiments/test_suite.py`). It tracks the
formalism papers we build on and the experimental / structural
references for the biological system we encode.

The example models a simplified compartmentalised
phosphorylation cycle: the dual-specificity kinase MEK1
phosphorylates ERK2 on its Thr183-Glu184-Tyr185 activation
motif in the cytoplasm; the resulting phospho-ERK2 dissociates
and can translocate through nuclear pore complexes to the
nucleus, where it phosphorylates downstream transcription
factors.

## Bigraphs and Bigraphical Reactive Systems

1. **Milner, R.** *The Space and Motion of Communicating
   Agents.* Cambridge University Press, 2009.
   — Foundational monograph defining bigraphs, place + link
   graphs, parametric reaction rules, and BRS.

2. **Archibald, B., Calder, M., Sevegnani, M.** "Practical
   Modelling with Bigraphs." arXiv:2405.20745, 2024.
   — Tutorial paper whose smart-building running example
   originally inspired this BRS demo (which we have now
   re-cast as a kinase-substrate cycle).

3. **Sevegnani, M., Calder, M.** "BigraphER: rewriting and
   analysis engine for bigraphs." *CAV 2016*, LNCS 9780,
   pp. 494–501. — Reference implementation of bigraph
   matching and rewriting; we follow similar redex/reactum
   semantics in `bigraph_schema.assembly`.

## ERK (MAPK1/3) — substrate

4. **Zhang, F., Strand, A., Robbins, D., Cobb, M.H.,
   Goldsmith, E.J.** "Atomic structure of the MAP kinase ERK2
   at 2.3 Å resolution." *Nature* **367**, 704–711 (1994).
   — PDB **1ERK**. First structure of ERK2; defines the
   classic bilobal kinase fold used in our cartoon
   silhouette.

5. **Canagarajah, B.J., Khokhlatchev, A., Cobb, M.H.,
   Goldsmith, E.J.** "Activation mechanism of the MAP kinase
   ERK2 by dual phosphorylation." *Cell* **90**, 859–869
   (1997). — PDB **2ERK**. Shows the dual-phosphorylated
   active conformation; basis for our `pERK` cartoon (P
   markers on the activation loop).

6. **Roskoski, R., Jr.** "ERK1/2 MAP kinases: structure,
   function, and regulation." *Pharmacol. Res.* **66**,
   105–143 (2012). — Comprehensive review covering ERK
   structure, activation mechanism, substrate specificity,
   and nuclear translocation.

## MEK1 (MAP2K1) — kinase

7. **Ohren, J.F., Chen, H., Pavlovsky, A. et al.**
   "Structures of human MAP kinase kinase 1 (MEK1) and MEK2
   describe novel noncompetitive kinase inhibition."
   *Nat. Struct. Mol. Biol.* **11**, 1192–1197 (2004).
   — PDB **3EQH** / **3EQB**. Source of the MEK kinase fold
   silhouette in our drawing.

8. **Akella, R., Moon, T.M., Goldsmith, E.J.** "Unique MAP
   kinase binding sites." *Biochim. Biophys. Acta* **1784**,
   48–55 (2008). — D-site / DEF-site docking interactions
   between MAPKs and their substrates / activators; the
   kind of noncovalent contact our `phosphorylate` rule
   represents as a single shared link-graph edge.

## Nuclear pore complex and ERK translocation

9. **Lin, D.H., Hoelz, A.** "The Structure of the Nuclear
   Pore Complex (An Update)." *Annu. Rev. Biochem.* **88**,
   725–783 (2019). — Review of NPC architecture; basis for
   the eight-fold symmetric annular cartoon.

10. **Plotnikov, A., Zehorai, E., Procaccia, S., Seger, R.**
    "The MAPK cascades: signaling components, nuclear roles
    and mechanisms of nuclear translocation." *Biochim.
    Biophys. Acta* **1813**, 1619–1633 (2011). — Reviews
    how phospho-ERK is shuttled into the nucleus through
    NPCs; the import/export asymmetry that drives nuclear
    pERK accumulation is the biological basis for the
    asymmetric `translocate_perk_in` (k=2.0) and
    `translocate_perk_out` (k=0.1) rates in our model.

## Notes on the abstraction

The model is intentionally coarse-grained:

- **Substrate copies** (`erk1`, `erk2`, `erk3`) are
  identical molecules; we name them only so individual
  trajectories are traceable in plots and animations.
- **Stoichiometry**: MEK has *one* active site, so the
  `phosphorylate` redex requires both endpoints unbound
  (Absent) and the `dissociate` rule destroys the bond — a
  faithful encoding of active-site-limited catalysis.
- **Translocation** is asymmetric for `pERK` and symmetric
  for `ERK`. We encode compartment identity structurally
  via a `kind` tag inside each `Compartment` (sorts
  `Cytoplasm`, `Nucleus`, `ERLumen`), so
  `translocate_perk_in` matches only
  `Cytoplasm → Nucleus` and `translocate_perk_out` only
  `Nucleus → Cytoplasm`. The 20-fold rate asymmetry produces
  nuclear pERK accumulation — the signal itself. Real biology has
  importin-mediated, NLS / NES-dependent transport plus
  nuclear MAPK phosphatases that drive the export step
  via dephosphorylation; we collapse that into a slow
  first-order leak.
- **Compartments**: we include three (cytoplasm, nucleus,
  ER lumen) so the diffusion dynamics are visible. Real
  ERK signalling has many more relevant locales (plasma
  membrane, mitochondrion, late endosomes, cytoskeleton).

## PDB entries cited above (quick reference)

| ID    | Description                                  |
|-------|----------------------------------------------|
| 1ERK  | ERK2 inactive form                           |
| 2ERK  | ERK2 dual-phosphorylated (active)            |
| 3EQH  | MEK1 ternary complex with allosteric inhibitor |
| 3EQB  | MEK1 ATP-bound                                |
