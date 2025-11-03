from pathlib import Path
from copy import deepcopy

from bigraph_viz import plot_bigraph
from vivarium.vivarium import VivariumTypes
from process_bigraph import register_types as register_process_types
from spatio_flux.processes import PROCESS_DOCS
from spatio_flux import register_types

# ------------- helpers -----------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def type_label(schema_piece: dict) -> str:
    """
    Produce a readable, reasonably unique label for a port/store type.
    Handles:
      - {'_type': 'float'} or 'map'/'array' with nested _value/_shape
      - nested dicts with '_value' holding inner type
    Falls back to 'unknown' when schematic info is too thin.
    """
    if not isinstance(schema_piece, dict):
        return "unknown"

    # direct type
    if "_type" in schema_piece and isinstance(schema_piece["_type"], str):
        t = schema_piece["_type"]

        # Expand containers with more context
        if t in ("map", "array", "struct"):
            inner = schema_piece.get("_value", {})
            inner_t = type_label(inner)
            shape = schema_piece.get("_shape")
            if t == "array" and shape is not None:
                return f"{t}<{inner_t}>[{shape}]"
            return f"{t}<{inner_t}>"
        return t

    # nested value
    if "_value" in schema_piece:
        return type_label(schema_piece["_value"])

    return "unknown"

def add_process_node(state: dict, proc_name: str):
    if proc_name not in state:
        state[proc_name] = {}
    state[proc_name].update({
        "_type": "process",
        "_inputs": {},
        "_outputs": {},
        "inputs": {},
        "outputs": {},
    })


# ------------- main pipeline -----------------

def build_per_process_figs(core, outdir: Path, show_types: bool = True):
    for proc_name in sorted(PROCESS_DOCS.keys()):
        get_doc = PROCESS_DOCS[proc_name]
        try:
            doc = get_doc(core=core)
            schema, state = core.generate({}, doc)
        except Exception as e:
            print(f"[skip] {proc_name}: failed to generate ({e})")
            continue

        # reshape state for bigraph_viz
        state = deepcopy(state)
        add_process_node(state, proc_name)

        inputs = schema[proc_name]["_inputs"]
        outputs = schema[proc_name]["_outputs"]

        # annotate schemas into the node (for disconnected view)
        for k, s in inputs.items():
            state[proc_name]["_inputs"][k] = s
        for k, s in outputs.items():
            state[proc_name]["_outputs"][k] = s

        # disconnected
        plot_bigraph(
            state=state,
            core=core,
            out_dir=str(outdir),
            filename=f"{proc_name}_disconnected",
            dpi="300",
            collapse_redundant_processes=True,
            show_types=show_types,
        )

        # connect each port to a store with the *same name* (your current pattern)
        for k in inputs:
            state[proc_name]["inputs"][k] = [k]
        for k in outputs:
            state[proc_name]["outputs"][k] = [k]

        # connected
        plot_bigraph(
            state=state,
            core=core,
            out_dir=str(outdir),
            filename=f"{proc_name}_connected",
            dpi="300",
            collapse_redundant_processes=True,
            show_types=show_types,
        )


def main():
    outdir = ensure_dir("out/metacomposite")

    # establish core
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    print("Registered Processes:")
    for proc_name in sorted(PROCESS_DOCS.keys()):
        print(f"- {proc_name}")

    # 1) keep your per-process disconnected/connected figs
    build_per_process_figs(core, outdir, show_types=True)


if __name__ == "__main__":
    main()
