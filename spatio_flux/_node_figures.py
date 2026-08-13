"""Illustrative per-node figures for the Fig-1 composites.

Each entry is a small, original inline-SVG icon (no third-party assets) keyed by
the node's name as it appears in the fig-1a / 1b / 1c states. `attach(state)`
walks a composite state and stamps ``_figure`` onto each matching process / store
node; the loom renders it inside the card when Detail → Figures is on (Auto shows
it whenever a node carries one).
"""
from __future__ import annotations


def _svg(body: str, w: int = 68, h: int = 46) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{body}</svg>')


# palette
_DNA = "#2563eb"; _RNA = "#0891b2"; _PROT = "#7c3aed"; _MET = "#0d9488"
_ENE = "#f59e0b"; _CELL = "#db2777"; _FIELD = "#3b82f6"; _GREEN = "#16a34a"
_INK = "#334155"

ICONS: dict[str, str] = {
    # ── processes ──────────────────────────────────────────────────────────
    "gene_expression": _svg(
        f'<path d="M14 8 C30 16 20 30 36 38 M14 38 C30 30 20 16 36 8" fill="none" stroke="{_DNA}" stroke-width="2.4"/>'
        f'<line x1="18" y1="13" x2="32" y2="13" stroke="{_DNA}" stroke-width="1.6"/><line x1="20" y1="23" x2="30" y2="23" stroke="{_DNA}" stroke-width="1.6"/><line x1="18" y1="33" x2="32" y2="33" stroke="{_DNA}" stroke-width="1.6"/>'
        f'<path d="M42 23 h14 M52 19 l4 4 -4 4" fill="none" stroke="{_INK}" stroke-width="2"/>'),
    "transcription": _svg(
        f'<path d="M12 8 C28 16 18 30 34 38 M12 38 C28 30 18 16 34 8" fill="none" stroke="{_DNA}" stroke-width="2.2"/>'
        f'<path d="M40 12 q8 6 0 12 q-8 6 0 12" fill="none" stroke="{_RNA}" stroke-width="2.4"/>'),
    "translation": _svg(
        f'<path d="M8 30 q8 -8 16 0 t16 0" fill="none" stroke="{_RNA}" stroke-width="2.2"/>'
        f'<circle cx="46" cy="24" r="9" fill="#ede9fe" stroke="{_PROT}" stroke-width="2.2"/><circle cx="46" cy="24" r="3.5" fill="{_PROT}"/>'),
    "rna_degradation": _svg(
        f'<path d="M8 24 q6 -7 12 0 t12 0" fill="none" stroke="{_RNA}" stroke-width="2.2"/>'
        f'<path d="M40 14 l12 20 M40 34 l12 -20" stroke="{_INK}" stroke-width="2" fill="none"/><circle cx="38" cy="14" r="3" fill="none" stroke="{_INK}" stroke-width="1.6"/><circle cx="38" cy="34" r="3" fill="none" stroke="{_INK}" stroke-width="1.6"/>'),
    "metabolism": _svg(
        f'<circle cx="20" cy="14" r="4" fill="{_MET}"/><circle cx="14" cy="30" r="4" fill="{_MET}"/><circle cx="30" cy="34" r="4" fill="{_MET}"/><circle cx="40" cy="18" r="4" fill="{_MET}"/><circle cx="52" cy="30" r="4" fill="{_MET}"/>'
        f'<path d="M20 14 L14 30 L30 34 L40 18 Z M40 18 L52 30" fill="none" stroke="{_MET}" stroke-width="1.6"/>'),
    "morphogen_gradient": _svg(
        f'<defs><radialGradient id="mg"><stop offset="0%" stop-color="{_FIELD}"/><stop offset="100%" stop-color="#dbeafe"/></radialGradient></defs>'
        f'<rect x="6" y="6" width="56" height="34" rx="5" fill="url(#mg)"/><circle cx="22" cy="23" r="8" fill="none" stroke="{_FIELD}" stroke-width="1.6" opacity="0.7"/>'),
    "diffusion": _svg(
        f'<circle cx="16" cy="23" r="5" fill="{_FIELD}"/>'
        f'<circle cx="30" cy="16" r="2.4" fill="{_FIELD}"/><circle cx="34" cy="30" r="2.4" fill="{_FIELD}"/><circle cx="44" cy="20" r="2" fill="{_FIELD}"/><circle cx="48" cy="32" r="1.7" fill="{_FIELD}"/><circle cx="56" cy="24" r="1.4" fill="{_FIELD}"/>'),
    "multicellular_interactions": _svg(
        f'<circle cx="18" cy="16" r="6" fill="none" stroke="{_CELL}" stroke-width="2.2"/><circle cx="42" cy="14" r="6" fill="none" stroke="{_CELL}" stroke-width="2.2"/><circle cx="30" cy="34" r="6" fill="none" stroke="{_CELL}" stroke-width="2.2"/>'
        f'<path d="M18 16 L42 14 M18 16 L30 34 M42 14 L30 34" stroke="{_CELL}" stroke-width="1.6"/>'),
    "abm": _svg(
        f'<circle cx="18" cy="16" r="5.5" fill="none" stroke="{_CELL}" stroke-width="2.2"/><circle cx="42" cy="14" r="5.5" fill="none" stroke="{_CELL}" stroke-width="2.2"/><circle cx="30" cy="34" r="5.5" fill="none" stroke="{_CELL}" stroke-width="2.2"/>'
        f'<path d="M18 16 L42 14 M18 16 L30 34 M42 14 L30 34" stroke="{_CELL}" stroke-width="1.6"/>'),
    "structural_packing": _svg(
        f'<circle cx="24" cy="18" r="7" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/><circle cx="40" cy="20" r="6" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/><circle cx="30" cy="32" r="6.5" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/><circle cx="44" cy="32" r="5" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/>'),
    "growth": _svg(
        f'<path d="M14 36 v-16 M14 20 l-4 4 M14 20 l4 4" stroke="{_GREEN}" stroke-width="2.2" fill="none"/>'
        f'<path d="M30 36 v-24 M30 12 l-5 5 M30 12 l5 5" stroke="{_GREEN}" stroke-width="2.2" fill="none"/>'
        f'<path d="M46 36 v-16 M46 20 l-4 4 M46 20 l4 4" stroke="{_GREEN}" stroke-width="2.2" fill="none"/>'),
    "division": _svg(
        f'<circle cx="22" cy="23" r="10" fill="none" stroke="{_CELL}" stroke-width="2.2"/><circle cx="44" cy="23" r="10" fill="none" stroke="{_CELL}" stroke-width="2.2"/>'
        f'<line x1="33" y1="12" x2="33" y2="34" stroke="{_CELL}" stroke-width="1.6" stroke-dasharray="3 3"/>'),
    "preprocess": _svg(
        f'<path d="M12 10 h44 l-16 16 v14 l-12 -6 v-8 Z" fill="#eef2ff" stroke="{_INK}" stroke-width="2"/>'),
    "analysis": _svg(
        f'<rect x="12" y="26" width="7" height="12" fill="{_FIELD}"/><rect x="24" y="18" width="7" height="20" fill="{_GREEN}"/><rect x="36" y="22" width="7" height="16" fill="{_ENE}"/><rect x="48" y="14" width="7" height="24" fill="{_CELL}"/>'),

    # ── stores ─────────────────────────────────────────────────────────────
    "DNA": _svg(
        f'<path d="M22 8 C40 16 26 30 44 38 M22 38 C40 30 26 16 44 8" fill="none" stroke="{_DNA}" stroke-width="2.4"/>'
        f'<line x1="27" y1="14" x2="39" y2="14" stroke="{_DNA}" stroke-width="1.6"/><line x1="28" y1="23" x2="38" y2="23" stroke="{_DNA}" stroke-width="1.6"/><line x1="27" y1="32" x2="39" y2="32" stroke="{_DNA}" stroke-width="1.6"/>', w=68),
    "mRNA": _svg(f'<path d="M10 30 q8 -14 16 0 t16 0 t14 0" fill="none" stroke="{_RNA}" stroke-width="2.6"/>'),
    "protein": _svg(
        f'<path d="M18 24 q-4 -10 8 -12 q12 -2 14 8 q2 10 -8 12 q-12 2 -14 -8 Z" fill="#ede9fe" stroke="{_PROT}" stroke-width="2.2"/><circle cx="30" cy="24" r="3" fill="{_PROT}"/>'),
    "metabolites": _svg(
        f'<path d="M34 8 l14 8 v14 l-14 8 -14 -8 v-14 Z" fill="none" stroke="{_MET}" stroke-width="2.2"/><circle cx="34" cy="23" r="3.4" fill="{_MET}"/>'),
    "energy": _svg(f'<path d="M36 6 L20 26 h12 l-6 16 18 -22 h-12 Z" fill="{_ENE}" stroke="{_ENE}" stroke-width="1.5" stroke-linejoin="round"/>'),
    "reg_signals": _svg(
        f'<path d="M10 34 h10 l4 -20 6 32 5 -22 4 14 h13" fill="none" stroke="{_ENE}" stroke-width="2.2"/>'),
    "fields": _svg(
        f'<defs><linearGradient id="fg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="{_FIELD}"/><stop offset="100%" stop-color="#dbeafe"/></linearGradient></defs>'
        f'<rect x="8" y="8" width="52" height="30" rx="3" fill="url(#fg)"/><g stroke="#fff" stroke-width="1" opacity="0.6"><line x1="8" y1="18" x2="60" y2="18"/><line x1="8" y1="28" x2="60" y2="28"/><line x1="25" y1="8" x2="25" y2="38"/><line x1="43" y1="8" x2="43" y2="38"/></g>'),
    "cell_population": _svg(
        f'<circle cx="18" cy="18" r="6" fill="#fce7f3" stroke="{_CELL}" stroke-width="2"/><circle cx="36" cy="15" r="6" fill="#fce7f3" stroke="{_CELL}" stroke-width="2"/><circle cx="27" cy="31" r="6" fill="#fce7f3" stroke="{_CELL}" stroke-width="2"/><circle cx="45" cy="30" r="6" fill="#fce7f3" stroke="{_CELL}" stroke-width="2"/>'),
    "mass": _svg(
        f'<path d="M18 14 h32 l-4 24 h-24 Z" fill="none" stroke="{_INK}" stroke-width="2.2"/><line x1="24" y1="14" x2="24" y2="8" stroke="{_INK}" stroke-width="2.2"/><line x1="44" y1="14" x2="44" y2="8" stroke="{_INK}" stroke-width="2.2"/>'),
    "volume": _svg(
        f'<path d="M20 16 l14 -8 14 8 -14 8 Z" fill="#eef2ff" stroke="{_INK}" stroke-width="1.8"/><path d="M20 16 v14 l14 8 v-14 Z" fill="#e0e7ff" stroke="{_INK}" stroke-width="1.8"/><path d="M48 16 v14 l-14 8 v-14 Z" fill="#c7d2fe" stroke="{_INK}" stroke-width="1.8"/>'),
    "phase": _svg(
        f'<circle cx="34" cy="23" r="14" fill="none" stroke="{_INK}" stroke-width="2.2" stroke-dasharray="4 3"/><path d="M34 23 v-9 M34 23 l7 4" stroke="{_INK}" stroke-width="2.2" fill="none"/>'),
    "structure": _svg(
        f'<circle cx="24" cy="18" r="6" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/><circle cx="40" cy="20" r="5.5" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/><circle cx="31" cy="32" r="6" fill="#ede9fe" stroke="{_PROT}" stroke-width="1.8"/>'),
    "raw_data": _svg(
        f'<rect x="12" y="8" width="44" height="30" rx="3" fill="#f8fafc" stroke="{_INK}" stroke-width="1.8"/><g fill="{_FIELD}"><circle cx="20" cy="16" r="1.6"/><circle cx="28" cy="16" r="1.6"/><circle cx="36" cy="16" r="1.6"/><circle cx="20" cy="23" r="1.6"/><circle cx="28" cy="23" r="1.6"/><circle cx="44" cy="23" r="1.6"/><circle cx="20" cy="30" r="1.6"/><circle cx="36" cy="30" r="1.6"/><circle cx="44" cy="30" r="1.6"/></g>'),
    "results": _svg(
        f'<rect x="14" y="6" width="40" height="34" rx="3" fill="#f8fafc" stroke="{_INK}" stroke-width="1.8"/>'
        f'<path d="M20 30 l6 -8 5 5 8 -12" fill="none" stroke="{_FIELD}" stroke-width="2"/><circle cx="45" cy="16" r="4" fill="none" stroke="{_GREEN}" stroke-width="2"/>'),
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
