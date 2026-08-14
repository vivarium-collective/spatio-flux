"""Illustrative per-node figures for the Fig-1 composites.

Each entry is a small, original inline-SVG icon (no third-party assets) keyed by
the node's name as it appears in the fig-1a / 1b / 1c states. `attach(state)`
walks a composite state and stamps ``_figure`` onto each matching process / store
node; the loom renders it inside the card when Detail → Figures is on (Auto shows
it whenever a node carries one).

Design notes: every icon lives in a 72×48 viewBox and is wrapped in a group that
defaults to ``fill:none`` + rounded line caps/joins, so the strokes read soft and
consistent. Gradient / filter ids are prefixed per-icon (``g_dna`` …) because all
icons are inlined into ONE loom document, where ids are global — a shared id would
cross-wire the wrong gradient.
"""
from __future__ import annotations

import math


def _svg(body: str, w: int = 72, h: int = 48) -> str:
    # The wrapping <g> gives every child rounded caps/joins + fill:none by default;
    # filled shapes set their own `fill`, so this only softens the line work.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<g fill="none" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>')


# palette — saturated ink for line work, matched light tints for fills
_DNA = "#2563eb"; _RNA = "#0891b2"; _PROT = "#7c3aed"; _MET = "#0d9488"
_ENE = "#f59e0b"; _CELL = "#db2777"; _FIELD = "#3b82f6"; _GREEN = "#16a34a"
_INK = "#334155"
_DNA_T = "#dbeafe"; _RNA_T = "#cffafe"; _PROT_T = "#ede9fe"; _MET_T = "#ccfbf1"
_ENE_T = "#fde68a"; _CELL_T = "#fce7f3"; _FIELD_T = "#dbeafe"; _GREEN_T = "#dcfce7"
_INK_T = "#e2e8f0"


# A clean vertical double helix centred on x=cx (two half-period-offset strands
# that cross on the axis, plus three rungs). Reused by DNA / gene_expression /
# transcription so the "helix" motif reads identically everywhere.
def _helix(cx: int, color: str) -> str:
    a = 8  # amplitude
    l, r = cx - a, cx + a
    return (
        f'<path d="M{cx} 7 Q{r} 13 {cx} 19 Q{l} 25 {cx} 31 Q{r} 37 {cx} 43" stroke="{color}" stroke-width="2.6"/>'
        f'<path d="M{cx} 7 Q{l} 13 {cx} 19 Q{r} 25 {cx} 31 Q{l} 37 {cx} 43" stroke="{color}" stroke-width="2.6"/>'
        f'<g stroke="{color}" stroke-width="1.7" opacity="0.5">'
        f'<line x1="{l+1}" y1="13" x2="{r-1}" y2="13"/>'
        f'<line x1="{l+1}" y1="25" x2="{r-1}" y2="25"/>'
        f'<line x1="{l+1}" y1="37" x2="{r-1}" y2="37"/></g>')


ICONS: dict[str, str] = {
    # ── fig-1c workflow additions ───────────────────────────────────────────
    "emitter": _svg(  # broadcast waves captured into a storage cylinder
        f'<g fill="none" stroke="{_FIELD}" stroke-width="2.2" stroke-linecap="round">'
        f'<path d="M13 17 A11 11 0 0 1 13 31"/><path d="M8 12 A17 17 0 0 1 8 36"/></g>'
        f'<circle cx="17" cy="24" r="3.2" fill="{_FIELD}"/>'
        f'<g fill="#f8fafc" stroke="{_INK}" stroke-width="1.9">'
        f'<ellipse cx="47" cy="14" rx="10" ry="3.6"/><path d="M37 14 v19 a10 3.6 0 0 0 20 0 v-19"/></g>'
        f'<path d="M37 23.5 a10 3.6 0 0 0 20 0" fill="none" stroke="{_INK}" stroke-width="1.4" opacity="0.5"/>'),
    "emitter_data": _svg(  # the stored emitter output — a data-log cylinder
        f'<g fill="#f8fafc" stroke="{_INK}" stroke-width="1.9">'
        f'<ellipse cx="36" cy="11" rx="13" ry="4.5"/><path d="M23 11 v24 a13 4.5 0 0 0 26 0 v-24"/></g>'
        f'<g fill="none" stroke="{_INK}" stroke-width="1.4" opacity="0.5">'
        f'<path d="M23 19 a13 4.5 0 0 0 26 0"/><path d="M23 27 a13 4.5 0 0 0 26 0"/></g>'),
    "load_results": _svg(  # read a store (left) into a results table (right)
        f'<g fill="#f8fafc" stroke="{_INK}" stroke-width="1.8">'
        f'<ellipse cx="14" cy="15" rx="8" ry="3"/><path d="M6 15 v14 a8 3 0 0 0 16 0 v-14"/></g>'
        f'<path d="M26 24 H37" stroke="{_GREEN}" stroke-width="2.4" stroke-linecap="round"/>'
        f'<path d="M33 20 l5 4 -5 4" fill="none" stroke="{_GREEN}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<rect x="42" y="10" width="24" height="28" rx="3" fill="#f8fafc" stroke="{_INK}" stroke-width="1.8"/>'
        f'<path d="M42 18 H66 M54 10 V38" stroke="{_INK}" stroke-width="1.3" opacity="0.5"/>'),
    # ── processes ──────────────────────────────────────────────────────────
    "gene_expression": _svg(
        _helix(17, _DNA)
        + f'<path d="M34 24 H52" stroke="{_INK}" stroke-width="2.4"/>'
        + f'<path d="M46 18 L54 24 L46 30" stroke="{_INK}" stroke-width="2.4"/>'
        + f'<circle cx="61" cy="24" r="4.5" fill="{_PROT_T}" stroke="{_PROT}" stroke-width="2"/>'),
    "transcription": _svg(
        _helix(16, _DNA)
        + f'<path d="M30 24 Q40 13 50 24 Q58 32 64 26" stroke="{_RNA}" stroke-width="2.8"/>'
        + f'<circle cx="64" cy="26" r="2.4" fill="{_RNA}"/>'),
    "translation": _svg(
        f'<path d="M6 36 Q16 28 26 36 Q34 42 42 37" stroke="{_RNA}" stroke-width="2.6"/>'
        f'<circle cx="33" cy="26" r="11" fill="{_PROT_T}" stroke="{_PROT}" stroke-width="2.2"/>'
        f'<path d="M22 20 A11 11 0 0 1 44 20 Z" fill="#ddd6fe" stroke="{_PROT}" stroke-width="2.2"/>'
        f'<g fill="{_PROT}"><circle cx="50" cy="17" r="2.6"/><circle cx="56" cy="12" r="2.6"/><circle cx="63" cy="9" r="2.6"/></g>'
        f'<path d="M45 19 L62 9" stroke="{_PROT}" stroke-width="1.6" opacity="0.6"/>'),
    "rna_degradation": _svg(
        f'<path d="M6 24 Q13 16 20 24 Q27 32 34 24" stroke="{_RNA}" stroke-width="2.6"/>'
        f'<circle cx="43" cy="15" r="3.4" fill="none" stroke="{_INK}" stroke-width="2"/>'
        f'<circle cx="43" cy="33" r="3.4" fill="none" stroke="{_INK}" stroke-width="2"/>'
        f'<path d="M45.5 17.5 L66 30 M45.5 30.5 L66 18" stroke="{_INK}" stroke-width="2.2"/>'),
    "metabolism": _svg(
        f'<path d="M20 13 L14 31 L33 37 L45 20 L20 13 M45 20 L55 33" stroke="{_MET}" stroke-width="2"/>'
        f'<g fill="{_MET}"><circle cx="20" cy="13" r="4.4"/><circle cx="14" cy="31" r="4.4"/>'
        f'<circle cx="33" cy="37" r="4.4"/><circle cx="45" cy="20" r="4.4"/><circle cx="55" cy="33" r="4.4"/></g>'),
    "morphogen_gradient": _svg(
        f'<defs><radialGradient id="g_mg" cx="34%" cy="42%" r="72%">'
        f'<stop offset="0%" stop-color="{_FIELD}"/><stop offset="100%" stop-color="{_FIELD_T}"/></radialGradient></defs>'
        f'<rect x="7" y="7" width="58" height="34" rx="9" fill="url(#g_mg)"/>'
        f'<g stroke="#ffffff" stroke-width="1.6" opacity="0.7"><circle cx="26" cy="21" r="6"/><circle cx="26" cy="21" r="11.5"/></g>'
        f'<circle cx="26" cy="21" r="2.4" fill="#ffffff"/>'),
    "diffusion": _svg(
        f'<circle cx="14" cy="24" r="7.5" fill="{_FIELD}"/>'
        f'<g fill="{_FIELD}"><circle cx="29" cy="16" r="3.4" opacity="0.9"/><circle cx="31" cy="33" r="3" opacity="0.8"/>'
        f'<circle cx="43" cy="20" r="2.6" opacity="0.6"/><circle cx="47" cy="32" r="2.2" opacity="0.5"/>'
        f'<circle cx="57" cy="24" r="1.8" opacity="0.35"/></g>'),
    "multicellular_interactions": _svg(
        f'<g stroke="{_CELL}" stroke-width="1.7" stroke-dasharray="3 3" opacity="0.75">'
        f'<line x1="18" y1="16" x2="46" y2="15"/><line x1="18" y1="16" x2="32" y2="35"/><line x1="46" y1="15" x2="32" y2="35"/></g>'
        f'<g fill="{_CELL_T}" stroke="{_CELL}" stroke-width="2.2"><circle cx="18" cy="16" r="7.5"/><circle cx="46" cy="15" r="7.5"/><circle cx="32" cy="35" r="7.5"/></g>'
        f'<g fill="{_CELL}"><circle cx="18" cy="16" r="2.4"/><circle cx="46" cy="15" r="2.4"/><circle cx="32" cy="35" r="2.4"/></g>'),
    "abm": _svg(
        f'<g stroke="{_CELL}" stroke-width="1.7" stroke-dasharray="3 3" opacity="0.75">'
        f'<line x1="18" y1="16" x2="46" y2="15"/><line x1="18" y1="16" x2="32" y2="35"/><line x1="46" y1="15" x2="32" y2="35"/></g>'
        f'<g fill="{_CELL_T}" stroke="{_CELL}" stroke-width="2.2"><circle cx="18" cy="16" r="7"/><circle cx="46" cy="15" r="7"/><circle cx="32" cy="35" r="7"/></g>'
        f'<g fill="{_CELL}"><circle cx="18" cy="16" r="2.2"/><circle cx="46" cy="15" r="2.2"/><circle cx="32" cy="35" r="2.2"/></g>'),
    "structural_packing": _svg(
        f'<defs><radialGradient id="g_pack" cx="35%" cy="30%" r="75%">'
        f'<stop offset="0%" stop-color="#f5f3ff"/><stop offset="100%" stop-color="{_PROT_T}"/></radialGradient></defs>'
        f'<g fill="url(#g_pack)" stroke="{_PROT}" stroke-width="1.9">'
        f'<circle cx="25" cy="19" r="8.5"/><circle cx="42" cy="21" r="7.5"/><circle cx="31" cy="34" r="7.5"/><circle cx="47" cy="34" r="6"/></g>'),
    "growth": _svg(
        f'<path d="M36 43 V19" stroke="{_GREEN}" stroke-width="2.8"/>'
        f'<path d="M36 27 Q24 25 20 15 Q33 15 36 27 Z" fill="{_GREEN_T}" stroke="{_GREEN}" stroke-width="2"/>'
        f'<path d="M36 31 Q48 30 53 21 Q40 20 36 31 Z" fill="{_GREEN_T}" stroke="{_GREEN}" stroke-width="2"/>'
        f'<path d="M36 19 l-4 4 M36 19 l4 4" stroke="{_GREEN}" stroke-width="2.6"/>'),
    "division": _svg(
        f'<g fill="{_CELL_T}" stroke="{_CELL}" stroke-width="2.4"><circle cx="25" cy="24" r="11"/><circle cx="47" cy="24" r="11"/></g>'
        f'<line x1="36" y1="12" x2="36" y2="36" stroke="{_CELL}" stroke-width="1.7" stroke-dasharray="3 3"/>'
        f'<g fill="{_CELL}"><circle cx="25" cy="24" r="2.6"/><circle cx="47" cy="24" r="2.6"/></g>'),
    "preprocess": _svg(
        # a checklist document — validate / map / check (green ticks + rule lines)
        f'<path d="M20 7 H44 L52 15 V41 H20 Z" fill="#f8fafc" stroke="{_GREEN}" stroke-width="1.9"/>'
        f'<path d="M44 7 V15 H52" fill="none" stroke="{_GREEN}" stroke-width="1.6"/>'
        f'<g stroke="{_GREEN}" stroke-width="2"><path d="M25 21 l2.4 2.4 4.2 -5.2"/><path d="M25 30 l2.4 2.4 4.2 -5.2"/></g>'
        f'<g stroke="{_INK}" stroke-width="1.8" opacity="0.4"><line x1="34" y1="21" x2="47" y2="21"/><line x1="34" y1="30" x2="47" y2="30"/></g>'),
    "analysis": _svg(
        f'<line x1="12" y1="40" x2="60" y2="40" stroke="{_INK}" stroke-width="1.8" opacity="0.6"/>'
        f'<g stroke-width="0">'
        f'<rect x="14" y="27" width="8" height="13" rx="2" fill="{_FIELD}"/>'
        f'<rect x="26" y="18" width="8" height="22" rx="2" fill="{_GREEN}"/>'
        f'<rect x="38" y="23" width="8" height="17" rx="2" fill="{_ENE}"/>'
        f'<rect x="50" y="12" width="8" height="28" rx="2" fill="{_CELL}"/></g>'
        f'<path d="M18 24 L30 15 L42 20 L54 9" stroke="{_INK}" stroke-width="2" opacity="0.55"/>'),
    "neural_dynamics": _svg(
        # a 2 → 3 → 2 fully-connected neural network (the ML formalism)
        f'<g stroke="{_PROT}" stroke-width="1" opacity="0.35">'
        f'<path d="M16 16 L34 10 M16 16 L34 24 M16 16 L34 38 M16 32 L34 10 M16 32 L34 24 M16 32 L34 38"/>'
        f'<path d="M34 10 L56 16 M34 10 L56 32 M34 24 L56 16 M34 24 L56 32 M34 38 L56 16 M34 38 L56 32"/></g>'
        f'<g fill="{_PROT}"><circle cx="16" cy="16" r="3.4"/><circle cx="16" cy="32" r="3.4"/>'
        f'<circle cx="34" cy="10" r="3.4"/><circle cx="34" cy="24" r="3.4"/><circle cx="34" cy="38" r="3.4"/>'
        f'<circle cx="56" cy="16" r="3.4"/><circle cx="56" cy="32" r="3.4"/></g>'),
    "analyses": _svg(
        f'<line x1="12" y1="40" x2="60" y2="40" stroke="{_INK}" stroke-width="1.8" opacity="0.6"/>'
        f'<g stroke-width="0">'
        f'<rect x="14" y="27" width="8" height="13" rx="2" fill="{_FIELD}"/>'
        f'<rect x="26" y="18" width="8" height="22" rx="2" fill="{_GREEN}"/>'
        f'<rect x="38" y="23" width="8" height="17" rx="2" fill="{_ENE}"/>'
        f'<rect x="50" y="12" width="8" height="28" rx="2" fill="{_CELL}"/></g>'),
    "visualizations": _svg(
        # a framed spatial view: a heatmap patch + a small scatter/network
        f'<rect x="14" y="10" width="44" height="28" rx="3" fill="#f8fafc" stroke="{_FIELD}" stroke-width="1.9"/>'
        f'<rect x="18" y="25" width="15" height="10" rx="1.5" fill="{_GREEN}" opacity="0.45"/>'
        f'<path d="M40 17 L48 22 L43 31 Z" fill="none" stroke="{_CELL}" stroke-width="1.4" opacity="0.7"/>'
        f'<g fill="{_CELL}"><circle cx="40" cy="17" r="2.2"/><circle cx="48" cy="22" r="2.2"/><circle cx="43" cy="31" r="2.2"/></g>'
        f'<circle cx="24" cy="17" r="2" fill="{_ENE}"/>'),
    "tests": _svg(
        # a green check inside a ring (pass)
        f'<circle cx="36" cy="24" r="15" fill="#f0fdf4" stroke="{_GREEN}" stroke-width="2.4"/>'
        f'<path d="M28 24 l5.5 5.5 11 -13" fill="none" stroke="{_GREEN}" stroke-width="2.8"/>'),

    # ── stores ─────────────────────────────────────────────────────────────
    "datasets": _svg(
        # a spreadsheet / table (header row + grid of cells)
        f'<rect x="16" y="9" width="40" height="30" rx="3" fill="#f8fafc" stroke="{_FIELD}" stroke-width="1.9"/>'
        f'<rect x="17.5" y="10.5" width="37" height="7.5" fill="{_FIELD}" opacity="0.22"/>'
        f'<path d="M16 18 H56 M16 28.5 H56 M29.3 18 V39 M42.7 18 V39" stroke="{_FIELD}" stroke-width="1.2" opacity="0.55"/>'
        f'<g fill="{_FIELD}"><rect x="20" y="12.5" width="6" height="3" rx="1"/><rect x="33" y="12.5" width="6" height="3" rx="1"/><rect x="46" y="12.5" width="5" height="3" rx="1"/></g>'),
    "model_specification": _svg(
        f'<rect x="18" y="7" width="36" height="34" rx="4" fill="#f8fafc" stroke="{_INK}" stroke-width="1.9"/>'
        f'<path d="M30 13 q-4 0 -4 5 q0 5 -4 6 q4 1 4 6 q0 5 4 5" fill="none" stroke="{_PROT}" stroke-width="2"/>'
        f'<path d="M42 13 q4 0 4 5 q0 5 4 6 q-4 1 -4 6 q0 5 -4 5" fill="none" stroke="{_PROT}" stroke-width="2"/>'),
    "analysis_results": _svg(
        # a data document: rule lines with leading dots (tables / metrics)
        f'<path d="M22 7 H44 L52 15 V41 H22 Z" fill="#f8fafc" stroke="{_GREEN}" stroke-width="1.9"/>'
        f'<path d="M44 7 V15 H52" fill="none" stroke="{_GREEN}" stroke-width="1.6"/>'
        f'<g stroke="{_GREEN}" stroke-width="1.8" opacity="0.75"><line x1="29" y1="22" x2="47" y2="22"/><line x1="29" y1="29" x2="47" y2="29"/><line x1="29" y1="36" x2="42" y2="36"/></g>'
        f'<g fill="{_GREEN}"><circle cx="26" cy="22" r="1.4"/><circle cx="26" cy="29" r="1.4"/><circle cx="26" cy="36" r="1.4"/></g>'),
    "figures": _svg(
        # a bar chart (plots / diagrams)
        f'<line x1="16" y1="38" x2="57" y2="38" stroke="{_INK}" stroke-width="1.6" opacity="0.5"/>'
        f'<g stroke-width="0">'
        f'<rect x="19" y="26" width="7.5" height="12" rx="1.5" fill="{_FIELD}"/>'
        f'<rect x="30.5" y="17" width="7.5" height="21" rx="1.5" fill="{_FIELD}"/>'
        f'<rect x="42" y="22" width="7.5" height="16" rx="1.5" fill="{_FIELD}"/></g>'),
    "test_report": _svg(
        # a report document with a pass badge (HTML / PDF / notebooks)
        f'<path d="M20 7 H42 L50 15 V41 H20 Z" fill="#f8fafc" stroke="{_GREEN}" stroke-width="1.9"/>'
        f'<path d="M42 7 V15 H50" fill="none" stroke="{_GREEN}" stroke-width="1.6"/>'
        f'<g stroke="{_INK}" stroke-width="1.7" opacity="0.4"><line x1="25" y1="20" x2="40" y2="20"/><line x1="25" y1="26" x2="40" y2="26"/></g>'
        f'<circle cx="41" cy="34" r="6.5" fill="#f0fdf4" stroke="{_GREEN}" stroke-width="1.8"/>'
        f'<path d="M37.5 34 l2.4 2.4 4 -4.8" fill="none" stroke="{_GREEN}" stroke-width="2"/>'),
    "DNA": _svg(_helix(36, _DNA)),
    "mRNA": _svg(
        f'<path d="M7 24 Q17 11 27 24 Q37 37 47 24 Q57 11 65 22" stroke="{_RNA}" stroke-width="2.8"/>'
        f'<g stroke="{_RNA}" stroke-width="1.6" opacity="0.55"><line x1="17" y1="17" x2="19" y2="21"/><line x1="37" y1="30" x2="35" y2="27"/><line x1="55" y1="17" x2="57" y2="21"/></g>'),
    "protein": _svg(
        f'<path d="M22 30 Q14 22 22 15 Q30 9 40 13 Q50 17 48 27 Q46 37 35 37 Q26 37 22 30 Z" fill="{_PROT_T}" stroke="{_PROT}" stroke-width="2.2"/>'
        f'<path d="M27 26 q4 -5 8 0 t8 0" stroke="{_PROT}" stroke-width="2"/>'
        f'<circle cx="27" cy="26" r="1.8" fill="{_PROT}"/><circle cx="43" cy="26" r="1.8" fill="{_PROT}"/>'),
    "metabolites": _svg(
        f'<path d="M36 10 L48 17 V31 L36 38 L24 31 V17 Z" fill="{_MET_T}" stroke="{_MET}" stroke-width="2.2"/>'
        f'<path d="M28 19 V29 M44 19 V29" stroke="{_MET}" stroke-width="1.6" opacity="0.6"/>'
        f'<g fill="{_MET}"><circle cx="36" cy="10" r="2.6"/><circle cx="48" cy="17" r="2.6"/><circle cx="48" cy="31" r="2.6"/>'
        f'<circle cx="36" cy="38" r="2.6"/><circle cx="24" cy="31" r="2.6"/><circle cx="24" cy="17" r="2.6"/></g>'),
    "energy": _svg(
        f'<defs><linearGradient id="g_ene" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#fbbf24"/><stop offset="100%" stop-color="#f59e0b"/></linearGradient></defs>'
        f'<path d="M42 6 L22 27 H34 L30 42 L50 20 H37 Z" fill="url(#g_ene)" stroke="{_ENE}" stroke-width="1.6"/>'),
    "reg_signals": _svg(
        f'<circle cx="17" cy="24" r="4.5" fill="{_ENE}"/>'
        f'<g stroke="{_ENE}" stroke-width="2.4" opacity="0.9"><path d="M27 15 Q34 24 27 33"/></g>'
        f'<g stroke="{_ENE}" stroke-width="2.2" opacity="0.6"><path d="M36 11 Q47 24 36 37"/></g>'
        f'<g stroke="{_ENE}" stroke-width="2" opacity="0.35"><path d="M45 8 Q60 24 45 40"/></g>'),
    "fields": _svg(
        f'<defs><linearGradient id="g_field" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{_FIELD}"/><stop offset="55%" stop-color="#93c5fd"/><stop offset="100%" stop-color="{_FIELD_T}"/></linearGradient></defs>'
        f'<rect x="8" y="8" width="56" height="32" rx="5" fill="url(#g_field)"/>'
        f'<g stroke="#ffffff" stroke-width="1" opacity="0.55">'
        f'<line x1="8" y1="18.7" x2="64" y2="18.7"/><line x1="8" y1="29.3" x2="64" y2="29.3"/>'
        f'<line x1="26.7" y1="8" x2="26.7" y2="40"/><line x1="45.3" y1="8" x2="45.3" y2="40"/></g>'),
    "cell_population": _svg(
        f'<g fill="{_CELL_T}" stroke="{_CELL}" stroke-width="2">'
        f'<circle cx="18" cy="18" r="6.5"/><circle cx="35" cy="14" r="6.5"/><circle cx="52" cy="19" r="6"/>'
        f'<circle cx="26" cy="32" r="6.5"/><circle cx="44" cy="32" r="6"/></g>'
        f'<g fill="{_CELL}" opacity="0.8"><circle cx="18" cy="18" r="2"/><circle cx="35" cy="14" r="2"/><circle cx="52" cy="19" r="2"/><circle cx="26" cy="32" r="2"/><circle cx="44" cy="32" r="2"/></g>'),
    "mass": _svg(
        f'<path d="M28 18 Q29 9 36 9 Q43 9 44 18" stroke="{_INK}" stroke-width="2.4"/>'
        f'<path d="M20 17 H52 L48 41 H24 Z" fill="{_INK_T}" stroke="{_INK}" stroke-width="2.2"/>'
        f'<text x="36" y="34" font-family="Inter,system-ui,sans-serif" font-size="10" font-weight="700" fill="{_INK}" text-anchor="middle" stroke="none">m</text>'),
    "volume": _svg(
        f'<path d="M36 8 L52 17 L36 26 L20 17 Z" fill="#eef2ff" stroke="{_INK}" stroke-width="1.9"/>'
        f'<path d="M20 17 V33 L36 42 V26 Z" fill="#c7d2fe" stroke="{_INK}" stroke-width="1.9"/>'
        f'<path d="M52 17 V33 L36 42 V26 Z" fill="#a5b4fc" stroke="{_INK}" stroke-width="1.9"/>'),
    "phase": _svg(
        f'<g fill="none" stroke-width="4.2">'
        f'<path d="M36 11 A13 13 0 0 1 49 24" stroke="{_GREEN}"/>'
        f'<path d="M49 24 A13 13 0 0 1 36 37" stroke="{_FIELD}"/>'
        f'<path d="M36 37 A13 13 0 0 1 23 24" stroke="{_ENE}"/>'
        f'<path d="M23 24 A13 13 0 0 1 36 11" stroke="{_CELL}"/></g>'
        f'<circle cx="36" cy="24" r="2.6" fill="{_INK}"/><path d="M36 24 L36 15" stroke="{_INK}" stroke-width="2"/>'),
    "structure": _svg(
        f'<path d="M22 15 L50 15 L36 38 Z M22 15 L36 26 L50 15 M36 26 L36 38" stroke="{_PROT}" stroke-width="2"/>'
        f'<g fill="{_PROT_T}" stroke="{_PROT}" stroke-width="2">'
        f'<circle cx="22" cy="15" r="4.6"/><circle cx="50" cy="15" r="4.6"/><circle cx="36" cy="38" r="4.6"/><circle cx="36" cy="26" r="4.6"/></g>'),
    "raw_data": _svg(
        f'<rect x="12" y="8" width="48" height="32" rx="4" fill="#f8fafc" stroke="{_INK}" stroke-width="1.9"/>'
        f'<path d="M12 18 H60 M28 18 V40 M44 18 V40" stroke="{_INK}" stroke-width="1.2" opacity="0.45"/>'
        f'<rect x="12" y="8" width="48" height="10" rx="4" fill="{_FIELD_T}" stroke="{_INK}" stroke-width="1.9"/>'
        f'<g fill="{_FIELD}"><circle cx="20" cy="27" r="1.7"/><circle cx="36" cy="27" r="1.7"/><circle cx="52" cy="27" r="1.7"/>'
        f'<circle cx="20" cy="34" r="1.7"/><circle cx="36" cy="34" r="1.7"/><circle cx="52" cy="34" r="1.7"/></g>'),
    "results": _svg(
        f'<rect x="13" y="7" width="46" height="34" rx="4" fill="#f8fafc" stroke="{_INK}" stroke-width="1.9"/>'
        f'<path d="M20 34 V16 M20 34 H52" stroke="{_INK}" stroke-width="1.6" opacity="0.5"/>'
        f'<path d="M22 30 L31 22 L38 26 L52 13" stroke="{_FIELD}" stroke-width="2.4"/>'
        f'<circle cx="52" cy="13" r="3" fill="{_GREEN_T}" stroke="{_GREEN}" stroke-width="2"/>'),
}


# ── Fig-1a: a RED-themed icon set (panel A is a red monochrome theme) ────────
# Same five formalisms as the fig-1a cards, drawn as clean single-colour red
# glyphs (like the reference): a DNA helix, a metabolic hub, a morphogen scatter,
# a cell cluster, and a brain. Kept separate from ICONS so the SHARED keys
# (metabolism …) stay multi-colour everywhere else (e.g. fig-1b).
_A_RED = "#c0392b"


def _hub(cx: int, cy: int, r: int, color: str) -> str:
    pts = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 360, 60)]
    spokes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}"/>' for px, py in pts)
    outer = "".join(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4"/>' for px, py in pts)
    return (f'<g stroke="{color}" stroke-width="1.7">{spokes}</g>'
            f'<circle cx="{cx}" cy="{cy}" r="4.6" stroke="{color}" stroke-width="2.2"/>'
            f'<g stroke="{color}" stroke-width="1.9">{outer}</g>')


FIG1A_ICONS: dict[str, str] = {
    "gene_expression": _svg(
        f'<g stroke="{_A_RED}" stroke-width="2.4">'
        f'<path d="M8 11 C20 11 20 37 32 37 C44 37 44 11 56 11"/>'
        f'<path d="M8 37 C20 37 20 11 32 11 C44 11 44 37 56 37"/></g>'
        f'<g stroke="{_A_RED}" stroke-width="1.5" opacity="0.7">'
        f'<line x1="14" y1="16" x2="14" y2="32"/><line x1="32" y1="13" x2="32" y2="35"/><line x1="50" y1="16" x2="50" y2="32"/></g>'),
    "metabolism": _svg(_hub(36, 24, 15, _A_RED)),
    "morphogen_gradient": _svg(
        f'<g fill="{_A_RED}">'
        f'<circle cx="12" cy="16" r="2.4"/><circle cx="12" cy="30" r="2.4"/><circle cx="18" cy="23" r="2.2"/>'
        f'<circle cx="18" cy="12" r="1.8"/><circle cx="18" cy="35" r="1.8"/><circle cx="25" cy="18" r="1.8"/>'
        f'<circle cx="25" cy="30" r="1.8"/><circle cx="32" cy="24" r="1.6" opacity="0.85"/>'
        f'<circle cx="33" cy="14" r="1.4" opacity="0.7"/><circle cx="34" cy="33" r="1.4" opacity="0.7"/>'
        f'<circle cx="42" cy="20" r="1.3" opacity="0.55"/><circle cx="44" cy="30" r="1.2" opacity="0.5"/>'
        f'<circle cx="52" cy="24" r="1.1" opacity="0.4"/><circle cx="58" cy="19" r="1" opacity="0.3"/></g>'),
    "multicellular_interactions": _svg(
        f'<g stroke="{_A_RED}" stroke-width="1.9">'
        f'<circle cx="29" cy="16" r="5.6"/><circle cx="41" cy="16" r="5.6"/><circle cx="23" cy="26" r="5.6"/>'
        f'<circle cx="35" cy="26" r="5.6"/><circle cx="47" cy="26" r="5.6"/><circle cx="29" cy="36" r="5.6"/><circle cx="41" cy="36" r="5.6"/></g>'
        f'<g fill="{_A_RED}"><circle cx="29" cy="16" r="1.5"/><circle cx="41" cy="16" r="1.5"/><circle cx="23" cy="26" r="1.5"/>'
        f'<circle cx="35" cy="26" r="1.5"/><circle cx="47" cy="26" r="1.5"/><circle cx="29" cy="36" r="1.5"/><circle cx="41" cy="36" r="1.5"/></g>'),
    "neural_dynamics": _svg(
        f'<path d="M36 9 C26 9 20 14 21 21 C15 23 15 33 23 35 C24 41 33 42 36 39 C39 42 48 41 49 35 '
        f'C57 33 57 23 51 21 C52 14 46 9 36 9 Z" stroke="{_A_RED}" stroke-width="2.2"/>'
        f'<path d="M36 11 V39" stroke="{_A_RED}" stroke-width="1.4" opacity="0.55"/>'
        f'<g stroke="{_A_RED}" stroke-width="1.3" opacity="0.6">'
        f'<path d="M28 18 q4 3 0 6"/><path d="M44 18 q-4 3 0 6"/><path d="M27 28 q5 2 0 5"/><path d="M45 28 q-5 2 0 5"/></g>'),
}


# ── Fig-1b: a BLUE-themed icon set (panel B is a blue monochrome theme) ──────
# Clean single-colour blue glyphs in the same style as the fig-1a red set, for
# every process + store in the multiscale composite. Kept separate from ICONS so
# the shared keys stay multi-colour elsewhere.
_B_BLUE = "#2b5bd0"


def _b(color=_B_BLUE):
    return color


FIG1B_ICONS: dict[str, str] = {
    # ── processes ──
    "diffusion": _svg(
        f'<circle cx="14" cy="24" r="6.5" fill="{_B_BLUE}"/>'
        f'<g fill="{_B_BLUE}"><circle cx="28" cy="17" r="3" opacity="0.85"/><circle cx="31" cy="31" r="2.6" opacity="0.7"/>'
        f'<circle cx="42" cy="21" r="2.2" opacity="0.55"/><circle cx="47" cy="32" r="1.9" opacity="0.45"/>'
        f'<circle cx="56" cy="25" r="1.6" opacity="0.32"/></g>'),
    "multicellular_interactions": _svg(
        f'<g stroke="{_B_BLUE}" stroke-width="1.7" stroke-dasharray="3 3" opacity="0.7">'
        f'<line x1="20" y1="17" x2="46" y2="16"/><line x1="20" y1="17" x2="33" y2="35"/><line x1="46" y1="16" x2="33" y2="35"/></g>'
        f'<g stroke="{_B_BLUE}" stroke-width="2"><circle cx="20" cy="17" r="6"/><circle cx="46" cy="16" r="6"/><circle cx="33" cy="35" r="6"/></g>'
        f'<g fill="{_B_BLUE}"><circle cx="20" cy="17" r="1.8"/><circle cx="46" cy="16" r="1.8"/><circle cx="33" cy="35" r="1.8"/></g>'),
    "transcription": _svg(
        _helix(16, _B_BLUE) + f'<path d="M30 24 Q40 13 50 24 Q58 32 64 26" stroke="{_B_BLUE}" stroke-width="2.8"/>'
        f'<circle cx="64" cy="26" r="2.4" fill="{_B_BLUE}"/>'),
    "translation": _svg(
        f'<path d="M6 36 Q16 28 26 36 Q34 42 42 37" stroke="{_B_BLUE}" stroke-width="2.6"/>'
        f'<circle cx="33" cy="26" r="11" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<path d="M22 20 A11 11 0 0 1 44 20" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<g fill="{_B_BLUE}"><circle cx="50" cy="17" r="2.6"/><circle cx="56" cy="12" r="2.6"/><circle cx="63" cy="9" r="2.6"/></g>'),
    "rna_degradation": _svg(
        f'<path d="M6 24 Q13 16 20 24 Q27 32 34 24" stroke="{_B_BLUE}" stroke-width="2.6"/>'
        f'<circle cx="43" cy="15" r="3.4" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<circle cx="43" cy="33" r="3.4" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<path d="M45.5 17.5 L66 30 M45.5 30.5 L66 18" stroke="{_B_BLUE}" stroke-width="2.2"/>'),
    "metabolism": _svg(_hub(36, 24, 15, _B_BLUE)),
    "structural_packing": _svg(
        f'<g stroke="{_B_BLUE}" stroke-width="1.9">'
        f'<circle cx="25" cy="19" r="8.5"/><circle cx="42" cy="21" r="7.5"/><circle cx="31" cy="34" r="7.5"/><circle cx="47" cy="34" r="6"/></g>'),
    "growth": _svg(
        f'<path d="M36 43 V19" stroke="{_B_BLUE}" stroke-width="2.8"/>'
        f'<path d="M36 27 Q24 25 20 15 Q33 15 36 27 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<path d="M36 31 Q48 30 53 21 Q40 20 36 31 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<path d="M36 19 l-4 4 M36 19 l4 4" stroke="{_B_BLUE}" stroke-width="2.6"/>'),
    "division": _svg(
        f'<g stroke="{_B_BLUE}" stroke-width="2.4"><circle cx="25" cy="24" r="11"/><circle cx="47" cy="24" r="11"/></g>'
        f'<line x1="36" y1="12" x2="36" y2="36" stroke="{_B_BLUE}" stroke-width="1.7" stroke-dasharray="3 3"/>'
        f'<g fill="{_B_BLUE}"><circle cx="25" cy="24" r="2.6"/><circle cx="47" cy="24" r="2.6"/></g>'),
    # ── stores ──
    "fields": _svg(
        f'<rect x="8" y="8" width="56" height="32" rx="5" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<g stroke="{_B_BLUE}" stroke-width="1.2" opacity="0.55">'
        f'<line x1="8" y1="18.7" x2="64" y2="18.7"/><line x1="8" y1="29.3" x2="64" y2="29.3"/>'
        f'<line x1="26.7" y1="8" x2="26.7" y2="40"/><line x1="45.3" y1="8" x2="45.3" y2="40"/></g>'
        f'<g fill="{_B_BLUE}" opacity="0.5"><rect x="12" y="12" width="10.5" height="5.5"/><rect x="49" y="30.5" width="11" height="5.5"/></g>'),
    "cell_population": _svg(
        f'<g stroke="{_B_BLUE}" stroke-width="2">'
        f'<circle cx="18" cy="18" r="6.5"/><circle cx="35" cy="14" r="6.5"/><circle cx="52" cy="19" r="6"/>'
        f'<circle cx="26" cy="32" r="6.5"/><circle cx="44" cy="32" r="6"/></g>'
        f'<g fill="{_B_BLUE}"><circle cx="18" cy="18" r="2"/><circle cx="35" cy="14" r="2"/><circle cx="52" cy="19" r="2"/><circle cx="26" cy="32" r="2"/><circle cx="44" cy="32" r="2"/></g>'),
    "tissue": _svg(  # a bounded sheet of cells (organ/tissue)
        f'<rect x="9" y="8" width="54" height="32" rx="10" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<g stroke="{_B_BLUE}" stroke-width="1.9" fill="none">'
        f'<circle cx="26" cy="20" r="5.5"/><circle cx="42" cy="17" r="5"/><circle cx="34" cy="31" r="5.5"/><circle cx="50" cy="30" r="4.5"/></g>'
        f'<g fill="{_B_BLUE}"><circle cx="26" cy="20" r="1.7"/><circle cx="42" cy="17" r="1.6"/>'
        f'<circle cx="34" cy="31" r="1.7"/><circle cx="50" cy="30" r="1.4"/></g>'),
    "cell": _svg(  # a single cell — membrane + nucleus + organelles
        f'<circle cx="36" cy="24" r="15" fill="none" stroke="{_B_BLUE}" stroke-width="2.4"/>'
        f'<circle cx="36" cy="24" r="6.5" fill="{_B_BLUE}" opacity="0.9"/>'
        f'<g fill="{_B_BLUE}" opacity="0.5"><circle cx="25" cy="18" r="2.3"/><circle cx="47" cy="29" r="2.1"/>'
        f'<circle cx="45" cy="16" r="1.8"/><circle cx="27" cy="31" r="1.8"/></g>'),
    "molecules": _svg(  # a molecule — central atom bonded to three others
        f'<g stroke="{_B_BLUE}" stroke-width="2.3"><line x1="23" y1="19" x2="36" y2="27"/>'
        f'<line x1="36" y1="27" x2="50" y2="17"/><line x1="36" y1="27" x2="39" y2="39"/></g>'
        f'<circle cx="36" cy="27" r="6" fill="none" stroke="{_B_BLUE}" stroke-width="2.3"/>'
        f'<circle cx="36" cy="27" r="2.1" fill="{_B_BLUE}"/>'
        f'<g fill="{_B_BLUE}"><circle cx="23" cy="19" r="5"/><circle cx="50" cy="17" r="5.5"/><circle cx="39" cy="39" r="4.5"/></g>'),
    "DNA": _svg(_helix(36, _B_BLUE)),
    "mRNA": _svg(
        f'<path d="M7 24 Q17 11 27 24 Q37 37 47 24 Q57 11 65 22" stroke="{_B_BLUE}" stroke-width="2.8"/>'
        f'<g stroke="{_B_BLUE}" stroke-width="1.6" opacity="0.55"><line x1="17" y1="17" x2="19" y2="21"/><line x1="37" y1="30" x2="35" y2="27"/><line x1="55" y1="17" x2="57" y2="21"/></g>'),
    "protein": _svg(
        f'<path d="M22 30 Q14 22 22 15 Q30 9 40 13 Q50 17 48 27 Q46 37 35 37 Q26 37 22 30 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<path d="M27 26 q4 -5 8 0 t8 0" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<circle cx="27" cy="26" r="1.8" fill="{_B_BLUE}"/><circle cx="43" cy="26" r="1.8" fill="{_B_BLUE}"/>'),
    "metabolites": _svg(
        f'<path d="M36 10 L48 17 V31 L36 38 L24 31 V17 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<path d="M28 19 V29 M44 19 V29" stroke="{_B_BLUE}" stroke-width="1.6" opacity="0.6"/>'
        f'<g fill="{_B_BLUE}"><circle cx="36" cy="10" r="2.6"/><circle cx="48" cy="17" r="2.6"/><circle cx="48" cy="31" r="2.6"/>'
        f'<circle cx="36" cy="38" r="2.6"/><circle cx="24" cy="31" r="2.6"/><circle cx="24" cy="17" r="2.6"/></g>'),
    "energy": _svg(f'<path d="M42 6 L22 27 H34 L30 42 L50 20 H37 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2.4" stroke-linejoin="round"/>'),
    "reg_signals": _svg(
        f'<circle cx="17" cy="24" r="4.5" fill="{_B_BLUE}"/>'
        f'<g stroke="{_B_BLUE}" stroke-width="2.4" opacity="0.9"><path d="M27 15 Q34 24 27 33"/></g>'
        f'<g stroke="{_B_BLUE}" stroke-width="2.2" opacity="0.6"><path d="M36 11 Q47 24 36 37"/></g>'
        f'<g stroke="{_B_BLUE}" stroke-width="2" opacity="0.35"><path d="M45 8 Q60 24 45 40"/></g>'),
    "structure": _svg(
        f'<path d="M22 15 L50 15 L36 38 Z M22 15 L36 26 L50 15 M36 26 L36 38" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<g fill="none" stroke="{_B_BLUE}" stroke-width="2">'
        f'<circle cx="22" cy="15" r="4.6"/><circle cx="50" cy="15" r="4.6"/><circle cx="36" cy="38" r="4.6"/><circle cx="36" cy="26" r="4.6"/></g>'),
    "mass": _svg(
        f'<path d="M28 18 Q29 9 36 9 Q43 9 44 18" stroke="{_B_BLUE}" stroke-width="2.4"/>'
        f'<path d="M20 17 H52 L48 41 H24 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<text x="36" y="34" font-family="Inter,system-ui,sans-serif" font-size="10" font-weight="700" fill="{_B_BLUE}" text-anchor="middle" stroke="none">m</text>'),
    "volume": _svg(
        f'<path d="M36 8 L52 17 L36 26 L20 17 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<path d="M20 17 V33 L36 42 V26 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'
        f'<path d="M52 17 V33 L36 42 V26 Z" fill="none" stroke="{_B_BLUE}" stroke-width="2"/>'),
    "phase": _svg(
        f'<circle cx="36" cy="24" r="14" fill="none" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<path d="M36 24 V13 M36 24 l8 5" stroke="{_B_BLUE}" stroke-width="2.2"/>'
        f'<circle cx="36" cy="24" r="2.2" fill="{_B_BLUE}"/>'),
}


def attach(state: dict, icons: dict[str, str] = ICONS) -> dict:
    """Stamp ``_figure`` onto every matching process/store node in ``state``
    (in place). Recurses through place-graph branches and composite inner states.
    Returns ``state`` for convenient chaining."""
    if not isinstance(state, dict):
        return state
    for key, node in list(state.items()):
        if not isinstance(node, dict):
            continue
        t = node.get("_type")
        if t is not None:  # a process/step or a typed leaf store
            if key in icons:
                node["_figure"] = icons[key]
            inner = (node.get("config") or {}).get("state")
            if isinstance(inner, dict):
                attach(inner, icons)
        else:  # a plain place-graph branch → stamp if its name matches, then recurse
            if key in icons:
                node["_figure"] = icons[key]
            attach(node, icons)
    return state
