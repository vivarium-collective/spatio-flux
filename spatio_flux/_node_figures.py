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
        f'<defs><linearGradient id="g_fun" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{_FIELD_T}"/><stop offset="100%" stop-color="#eef2ff"/></linearGradient></defs>'
        f'<path d="M13 11 H59 L41 30 V39 L31 44 V30 Z" fill="url(#g_fun)" stroke="{_INK}" stroke-width="2.2"/>'),
    "analysis": _svg(
        f'<line x1="12" y1="40" x2="60" y2="40" stroke="{_INK}" stroke-width="1.8" opacity="0.6"/>'
        f'<g stroke-width="0">'
        f'<rect x="14" y="27" width="8" height="13" rx="2" fill="{_FIELD}"/>'
        f'<rect x="26" y="18" width="8" height="22" rx="2" fill="{_GREEN}"/>'
        f'<rect x="38" y="23" width="8" height="17" rx="2" fill="{_ENE}"/>'
        f'<rect x="50" y="12" width="8" height="28" rx="2" fill="{_CELL}"/></g>'
        f'<path d="M18 24 L30 15 L42 20 L54 9" stroke="{_INK}" stroke-width="2" opacity="0.55"/>'),

    # ── stores ─────────────────────────────────────────────────────────────
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
        else:  # a plain place-graph branch → recurse
            attach(node, icons)
    return state
