import math
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, UnidentifiedImageError

from bigraph_viz import plot_bigraph
from vivarium.vivarium import VivariumTypes
from process_bigraph import register_types as register_process_types
from spatio_flux.processes import PROCESS_DOCS
from spatio_flux import register_types


# ---------- small helpers ----------
def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def add_process_node(state, name):
    state.setdefault(name, {}).update({"_type":"process","_inputs":{},"_outputs":{},"inputs":{},"outputs":{}})

def _grid_shape(n_items, max_cols):
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols) if n_items > 0 else 0
    return rows, cols

def _load_rgba(path):
    try:
        with Image.open(path) as im:
            return im.convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        print(f"⚠ Skipping unreadable image: {path} ({e.__class__.__name__})")
        return None

def _collect_images(outdir, suffix):
    outdir = Path(outdir)
    return sorted(p for p in outdir.glob(f"*_{suffix}.png") if not p.name.startswith("overview"))


# ---------- per-process figures ----------
def build_per_process_figs(core, outdir, show_types=True, per_fig_dpi=300):
    for name in sorted(PROCESS_DOCS.keys()):
        try:
            doc = PROCESS_DOCS[name](core=core)
            schema, state = core.generate({}, doc)
        except Exception as e:
            print(f"[skip] {name}: {e}"); continue

        state = deepcopy(state)
        add_process_node(state, name)

        ins = schema[name]["_inputs"]
        outs = schema[name]["_outputs"]

        for k, v in ins.items():  state[name]["_inputs"][k]  = v
        for k, v in outs.items(): state[name]["_outputs"][k] = v

        # (a) disconnected
        plot_bigraph(
            state=state, core=core, out_dir=str(outdir),
            filename=f"{name}_disconnected", dpi=str(per_fig_dpi),
            collapse_redundant_processes=True, show_types=show_types
        )

        # (b) connected (wire ports to same-named stores)
        for k in ins:  state[name]["inputs"][k]  = [k]
        for k in outs: state[name]["outputs"][k] = [k]
        plot_bigraph(
            state=state, core=core, out_dir=str(outdir),
            filename=f"{name}_connected", dpi=str(per_fig_dpi),
            collapse_redundant_processes=True, show_types=show_types
        )


# ---------- pairwise composite figures ----------
def build_pairwise_composite_figs(core, outdir, show_types=True, per_fig_dpi=300):

    process_names = sorted(PROCESS_DOCS.keys())
    compiled = {}
    for name in process_names:
        try:
            doc = PROCESS_DOCS[name](core=core)
            schema, state = core.generate({}, doc)
        except Exception as e:
            print(f"[skip] {name}: {e}"); continue

        state = deepcopy(state)
        add_process_node(state, name)

        # add the ports
        ins = schema[name]["_inputs"]
        outs = schema[name]["_outputs"]
        for k, v in ins.items():  state[name]["_inputs"][k]  = v
        for k, v in outs.items(): state[name]["_outputs"][k] = v

        compiled[name] = (schema, state)

        # # wire ports to same-named stores
        # for k in ins:  state[name]["inputs"][k]  = [k]
        # for k in outs: state[name]["outputs"][k] = [k]

    # --- Identify processes with matching ports ---
    pairs = []
    for i_idx, i in enumerate(process_names):
        for j in process_names[i_idx + 1:]:
            schema_i, state_i = compiled[i]
            schema_j, state_j = compiled[j]

            i_inputs = state_i[i]["_inputs"]
            # TODO -- we need to get the 'map[concentration]' to come through instead of 'map[float]'

            # breakpoint()
            pass


        # plot_bigraph(
        #     state=state, core=core, out_dir=str(outdir),
        #     filename=f"{name}_connected", dpi=str(per_fig_dpi),
        #     collapse_redundant_processes=True, show_types=show_types
        # )


# ---------- overview assembly (matplotlib) ----------
def assemble_overview_abc(
    outdir,
    cols=3,
    figsize=(16, 30),      # taller canvas for three panels
    fig_dpi=200,
    save_dpi=300,
    title_fs=22,
    save_name="overview_abc.png"
):
    outdir = Path(outdir)
    disc_paths = _collect_images(outdir, "disconnected")
    conn_paths = _collect_images(outdir, "connected")
    comp_paths = _collect_images(outdir, "composite")

    fig = plt.figure(figsize=figsize, dpi=fig_dpi)
    gs_root = GridSpec(3, 1, height_ratios=(1, 1, 1), hspace=0.08, figure=fig)

    # (a) Disconnected
    a_rows, a_cols = _grid_shape(len(disc_paths), max_cols=cols)
    if len(disc_paths) > 0 and a_rows > 0:
        gs_a = GridSpecFromSubplotSpec(a_rows, a_cols, subplot_spec=gs_root[0], wspace=0.02, hspace=0.04)
        axes_a = [fig.add_subplot(gs_a[i // a_cols, i % a_cols]) for i in range(a_rows * a_cols)]
        imgs_a = [im for p in disc_paths for im in [_load_rgba(p)] if im is not None]
        for i, ax in enumerate(axes_a):
            ax.axis("off")
            if i < len(imgs_a):
                ax.imshow(imgs_a[i], interpolation="none")
    else:
        ax = fig.add_subplot(gs_root[0]); ax.axis("off")
        ax.text(0.5, 0.5, "No disconnected figures found", ha="center", va="center", fontsize=12)
    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.006, "(a) Disconnected processes", fontsize=title_fs, weight="bold", va="bottom", ha="left")

    # (b) Connected (self-wired)
    b_rows, b_cols = _grid_shape(len(conn_paths), max_cols=cols)
    if len(conn_paths) > 0 and b_rows > 0:
        gs_b = GridSpecFromSubplotSpec(b_rows, b_cols, subplot_spec=gs_root[1], wspace=0.02, hspace=0.04)
        axes_b = [fig.add_subplot(gs_b[i // b_cols, i % b_cols]) for i in range(b_rows * b_cols)]
        imgs_b = [im for p in conn_paths for im in [_load_rgba(p)] if im is not None]
        for i, ax in enumerate(axes_b):
            ax.axis("off")
            if i < len(imgs_b):
                ax.imshow(imgs_b[i], interpolation="none")
    else:
        ax = fig.add_subplot(gs_root[1]); ax.axis("off")
        ax.text(0.5, 0.5, "No connected figures found", ha="center", va="center", fontsize=12)
    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.006, "(b) Connected processes", fontsize=title_fs, weight="bold", va="bottom", ha="left")

    # (c) Composites
    c_rows, c_cols = _grid_shape(len(comp_paths), max_cols=cols)
    if len(comp_paths) > 0 and c_rows > 0:
        gs_c = GridSpecFromSubplotSpec(c_rows, c_cols, subplot_spec=gs_root[2], wspace=0.02, hspace=0.04)
        axes_c = [fig.add_subplot(gs_c[i // c_cols, i % c_cols]) for i in range(c_rows * c_cols)]
        imgs_c = [im for p in comp_paths for im in [_load_rgba(p)] if im is not None]
        for i, ax in enumerate(axes_c):
            ax.axis("off")
            if i < len(imgs_c):
                ax.imshow(imgs_c[i], interpolation="none")
    else:
        ax = fig.add_subplot(gs_root[2]); ax.axis("off")
        ax.text(0.5, 0.5, "No composite figures found", ha="center", va="center", fontsize=12)
    pos_c = gs_root[2].get_position(fig)
    fig.text(pos_c.x0, pos_c.y1 + 0.006, "(c) Composites (pairwise-compatible)", fontsize=title_fs, weight="bold", va="bottom", ha="left")

    out_path = outdir / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=save_dpi)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[overview] saved {out_path}")
    print(f"[overview] saved {out_path.with_suffix('.pdf')}")


# ---------- driver ----------
def main():
    outdir = ensure_dir("out/metacomposite")

    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    print("Registered Processes:")
    for n in sorted(PROCESS_DOCS.keys()):
        print(f"- {n}")

    # 1) build per-process figs at high dpi
    build_per_process_figs(core, outdir, show_types=True, per_fig_dpi=300)

    # 2) build pairwise composite figs
    build_pairwise_composite_figs(core, outdir, show_types=True, per_fig_dpi=300)

    # 3) assemble overview (a, b, c)
    assemble_overview_abc(outdir, cols=3, figsize=(16, 30), fig_dpi=200, save_dpi=300, title_fs=22)

if __name__ == "__main__":
    main()
