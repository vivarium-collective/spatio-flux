import os
import json
import pprint
import shutil
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import numpy as np
from bigraph_viz import plot_bigraph
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.emitter import emitter_from_wires
from spatio_flux.plots.colors import build_plot_settings
from urllib.parse import quote as url_quote


def _url_href(path: str) -> str:
    # URL-encode filenames for href/src attributes (spaces, #, etc.)
    # Keep it relative (no leading slash)
    return url_quote(path)


def build_path(base_path, mol_id, i=None, j=None):
    """
    Constructs a path list for a molecule, optionally appending indices.

    Parameters:
        base_path (list of str): The base path prefix (e.g., ["..", "fields"]).
        mol_id (str): The molecule ID to insert in the path.
        i (int, optional): First index to append, if provided.
        j (int, optional): Second index to append, if provided.

    Returns:
        list: The full path as a list of path elements.
    """
    full_path = base_path + [mol_id]
    if i is not None:
        full_path.append(i)
    if j is not None:
        full_path.append(j)
    return full_path


def initialize_fields(n_bins, initial_min_max=None):
    initial_min_max = initial_min_max or {}
    fields = {}
    for field, minmax in initial_min_max.items():
        fields[field] = np.random.uniform(low=minmax[0], high=minmax[1], size=n_bins)
    return fields


def get_standard_emitter(state_keys):
    OPTIONAL_KEYS = {'fields', 'particles'}
    # Always include 'global_time', include optional keys if present
    included_keys = ['global_time'] + [key for key in OPTIONAL_KEYS if key in state_keys]
    emitter_spec = {key: [key] for key in included_keys}
    return emitter_from_wires(emitter_spec)


def run_composite_document(
        document, core=None, name=None, time=None, outdir="out", show_types=False, show_values=False):
    """
    Instantiates and runs a Composite simulation.

    Args:
        document (dict): Composition document with initial state and optional schema.
        time (float): Simulation duration.
        core (VivariumTypes): Core schema registration object.
        name (str): Output name prefix.
        outdir (str): Output directory.

    Returns:
        dict: Simulation results emitted during the run.
    """
    time = time or 60
    os.makedirs(outdir, exist_ok=True)

    core = core or allocate_core()

    if name is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'spatio_flux_{date}'

    # Ensure proper structure for Vivarium Composite
    document = {'state': document} if 'state' not in document else document
    if 'emitter' not in document['state']:
        state_keys = list(document['state'].keys())
        document['state']['emitter'] = get_standard_emitter(state_keys=state_keys)

    print(f"🧩 Making composite {name}...")

    sim = Composite(document, core=core)

    # Save composition JSON
    sim.save(filename=f"{name}.json", outdir=outdir)

    # Save representation string (human-readable schema summary)
    representation = core.render(sim.schema)
    rep_file = os.path.join(outdir, f"{name}_schema.json")
    with open(rep_file, "w") as f:
        json.dump(representation, f, indent=2)
    print(f"💾 Saved schema representation → {rep_file}")

    # Save the underlying state as JSON (machine-readable)
    state_file = os.path.join(outdir, f"{name}_state.json")
    try:
        serialized = core.serialize(sim.schema, sim.state)
        with open(state_file, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"💾 Saved state JSON → {state_file}")
    except Exception as e:
        print(f"⚠ Could not save state JSON: {e}")

    # Visualize initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.schema.items() if k not in ['global_time', 'emitter']}

    # only include one particle for visualization purposes
    if 'particles' in plot_state and plot_state['particles']:
        first_particle_key = next(iter(plot_state['particles']))
        plot_state['particles'] = {
            first_particle_key: plot_state['particles'][first_particle_key]
        }

    # get particles for coloring
    particle_ids = []
    if 'particles' in plot_state and plot_state['particles']:
        particle_ids = list(plot_state['particles'].keys())

    n_bins = ()
    if 'fields' in plot_state:
        for value in plot_state['fields'].values():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    n_bins =  value.shape  # (n, m)

    # spatio-flux-specific plot settings
    plot_settings = build_plot_settings(
        particle_ids=particle_ids,
        n_bins=n_bins
    )
    plot_settings.update(dict(
        dpi='300',
        show_values=show_values,
        show_types=show_types,
        collapse_redundant_processes={
            'exclude': [  # dont collapse these
                ('particle_movement',),
                ('particle_division',),
                ('enforce_boundaries',),
                ('glucose eater',),
                ('acetate eater',),
                ('newtonian_particles',),
                ('particle_division',),
            ]
        },
        value_char_limit=20,
        type_char_limit=40,
    ))

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir=outdir,
        filename=f"{name}_viz",
        **plot_settings
    )

    print(f"⏱ Simulating {name} for {time}s...")
    sim.run(time)
    results = gather_emitter_results(sim)
    print(f"✅ Simulation complete: {name}")

    return results[('emitter',)]

def prepare_output_dir(output_dir):
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"🧹 Clearing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

def _safe_id(raw: str) -> str:
    """Stable, HTML-safe anchor id."""
    s = str(raw)
    return "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in s)


def _load_state_data_from_json(json_path: Path) -> dict:
    """
    Load JSON and return a 'state-like' dict with top-level 'emitter' and 'global_time' removed.

    Works for:
      - composition docs containing {"state": {...}}
      - plain state dicts (wrap-as-state behavior)
    """
    with open(json_path, "r") as jf:
        full_data = json.load(jf)

    if not isinstance(full_data, dict):
        return {}

    if "state" not in full_data or not isinstance(full_data.get("state"), dict):
        full_data = {"state": full_data}

    state = full_data.get("state", {})
    return {k: v for k, v in state.items() if k not in ("emitter", "global_time")}


def _render_json_api_viewer_block(test: str, state_data: dict) -> str:
    """
    Render an interactive JSON browser:
      - Left pane: top-level keys (plus search results)
      - Right pane: selected value
      - Arrays: shown as compact preview; 2D arrays shown as a table (cropped)
    """
    safe = _safe_id(test)
    blob = json.dumps(state_data, ensure_ascii=False)

    # NOTE: blob is inserted into a <script type="application/json"> tag.
    # It's valid JSON; we do not html-escape it so JSON.parse works reliably.
    return f"""
<div class="json-viewer" data-test="{html_escape(safe)}">
  <div class="json-toolbar">
    <input class="json-search" placeholder="Search paths (e.g. fields.glucose, particles.p_12.position)" />
    <button type="button" class="json-reset">Top-level</button>
    <span class="json-status"></span>
  </div>

  <div class="json-layout">
    <div class="json-nav"></div>
    <div class="json-main">
      <div class="json-path"></div>
      <div class="json-value"></div>
    </div>
  </div>

  <script type="application/json" id="json-data-{html_escape(safe)}">
{blob}
  </script>
</div>
""".strip()


def _json_viewer_css_lines() -> list[str]:
    return [
        # existing base style can stay; these extend it
        ".json-viewer { border: 1px solid #ddd; background: #fff; border-radius: 10px; padding: 10px; margin: 10px 0; }",
        ".json-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }",
        ".json-toolbar input { flex: 1; padding: 6px 10px; border: 1px solid #ccc; border-radius: 8px; }",
        ".json-toolbar button { padding: 6px 10px; border: 1px solid #ccc; border-radius: 8px; background: #f5f5f5; cursor: pointer; }",
        ".json-toolbar button:hover { background: #eee; }",
        ".json-status { font-size: 12px; color: #555; }",
        ".json-layout { display: grid; grid-template-columns: 340px 1fr; gap: 10px; height: 460px; }",
        ".json-nav { overflow: auto; border-right: 1px solid #eee; padding-right: 8px; }",
        ".json-main { overflow: auto; padding-left: 6px; }",
        ".json-item { padding: 5px 8px; border-radius: 8px; cursor: pointer; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.25; }",
        ".json-item:hover { background: #f3f3f3; }",
        ".json-item.active { background: #e9eefc; }",
        ".json-path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; margin-bottom: 8px; color: #333; }",
        ".json-value pre { background: #f8f8f8; border: 1px solid #eee; border-radius: 10px; padding: 10px; overflow: auto; }",
        ".json-pill { display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:12px; margin-right:6px; background:#fafafa; color:#333; }",
        ".json-table { border-collapse: collapse; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; margin-top: 8px; }",
        ".json-table td, .json-table th { border: 1px solid #ddd; padding: 3px 6px; }",
        ".json-note { margin-top: 8px; font-size: 12px; color: #666; }",
    ]


def _json_viewer_js() -> str:
    # Plain JS: no dependencies; works in a local report.html.
    return r"""
<script>
(function(){
  function isPlainObject(x){ return x && typeof x === "object" && !Array.isArray(x); }

  function walk(obj, fn, path){
    path = path || [];
    fn(obj, path);
    if (Array.isArray(obj)){
      for (let i=0;i<obj.length;i++) walk(obj[i], fn, path.concat([String(i)]));
    } else if (isPlainObject(obj)){
      for (const k of Object.keys(obj)) walk(obj[k], fn, path.concat([k]));
    }
  }

  function getAtPath(root, path){
    let cur = root;
    for (const p of path){
      if (cur == null) return undefined;
      if (Array.isArray(cur)) cur = cur[Number(p)];
      else cur = cur[p];
    }
    return cur;
  }

  function prettyScalar(v){
    return JSON.stringify(v);
  }

  function renderValue(container, value){
    container.innerHTML = "";

    // Scalars
    if (value === null || typeof value !== "object"){
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(value, null, 2);
      container.appendChild(pre);
      return;
    }

    // Arrays
    if (Array.isArray(value)){
      const n = value.length;
      const header = document.createElement("div");
      header.innerHTML = `<span class="json-pill">array</span><span class="json-pill">len=${n}</span>`;
      container.appendChild(header);

      // 1D scalar array: show compact if long
      const allScalar = value.every(v => v === null || typeof v !== "object");
      if (allScalar){
        const maxInline = 200;
        const pre = document.createElement("pre");
        if (value.length <= maxInline){
          pre.textContent = JSON.stringify(value, null, 2);
        } else {
          const head = value.slice(0, 50);
          const tail = value.slice(-10);
          pre.textContent =
            JSON.stringify(head, null, 2) +
            `\n... (${value.length - 60} omitted) ...\n` +
            JSON.stringify(tail, null, 2);
        }
        container.appendChild(pre);
        return;
      }

      // 2D grid of scalars -> render as table (cropped)
      const is2D = value.length > 0 && value.every(row => Array.isArray(row));
      if (is2D){
        const widths = value.map(r => r.length);
        const sameWidth = widths.every(w => w === widths[0]);
        const scalarCells = value.flat().every(v => v === null || typeof v !== "object");
        if (sameWidth && scalarCells){
          const maxRows = 60, maxCols = 80;
          const rows = Math.min(value.length, maxRows);
          const cols = Math.min(widths[0], maxCols);

          const table = document.createElement("table");
          table.className = "json-table";
          const tbody = document.createElement("tbody");

          for (let i=0;i<rows;i++){
            const tr = document.createElement("tr");
            for (let j=0;j<cols;j++){
              const td = document.createElement("td");
              td.textContent = String(value[i][j]);
              tr.appendChild(td);
            }
            tbody.appendChild(tr);
          }
          table.appendChild(tbody);
          container.appendChild(table);

          if (value.length > maxRows || widths[0] > maxCols){
            const note = document.createElement("div");
            note.className = "json-note";
            note.textContent = `Showing ${rows}x${cols} (cropped). Full size: ${value.length}x${widths[0]}.`;
            container.appendChild(note);
          }
          return;
        }
      }

      // Fallback: pretty JSON
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(value, null, 2);
      container.appendChild(pre);
      return;
    }

    // Objects
    const keys = Object.keys(value);
    const header = document.createElement("div");
    header.innerHTML = `<span class="json-pill">object</span><span class="json-pill">keys=${keys.length}</span>`;
    container.appendChild(header);

    // For moderate objects, show a compact key list up front
    if (keys.length <= 60){
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(value, null, 2);
      container.appendChild(pre);
    } else {
      const pre = document.createElement("pre");
      // compact: show key preview
      const preview = {};
      for (let i=0;i<Math.min(keys.length, 50);i++){
        const k = keys[i];
        const v = value[k];
        preview[k] = (v === null || typeof v !== "object") ? v : (Array.isArray(v) ? `[array len=${v.length}]` : "[object]");
      }
      pre.textContent = JSON.stringify(preview, null, 2) + `\n... (${keys.length - 50} more keys) ...`;
      container.appendChild(pre);
    }
  }

  function buildNav(root, navEl, mainPathEl, mainValueEl, searchEl, statusEl){
    function renderList(paths){
      navEl.innerHTML = "";
      paths.forEach((p, idx) => {
        const item = document.createElement("div");
        item.className = "json-item";
        item.textContent = p.join(".");
        item.onclick = () => {
          navEl.querySelectorAll(".json-item").forEach(x => x.classList.remove("active"));
          item.classList.add("active");
          mainPathEl.textContent = p.join(".");
          const v = getAtPath(root, p);
          renderValue(mainValueEl, v);
        };
        navEl.appendChild(item);
        if (idx === 0) item.click();
      });
      statusEl.textContent = `${paths.length} paths`;
    }

    const top = Object.keys(root || {}).map(k => [k]);
    renderList(top);

    searchEl.addEventListener("input", () => {
      const q = (searchEl.value || "").trim().toLowerCase();
      if (!q){
        renderList(top);
        return;
      }
      const hits = [];
      walk(root, (node, path) => {
        const s = path.join(".").toLowerCase();
        if (s.includes(q)) hits.push(path);
      });

      // de-dupe & cap
      const seen = new Set();
      const uniq = [];
      for (const p of hits){
        const key = p.join(".");
        if (!seen.has(key)){
          seen.add(key);
          uniq.push(p);
        }
        if (uniq.length >= 600) break;
      }
      renderList(uniq.length ? uniq : top);
    });
  }

  document.querySelectorAll(".json-viewer").forEach(viewer => {
    const test = viewer.dataset.test;
    const dataEl = document.getElementById("json-data-" + test);
    if (!dataEl) return;

    let root = null;
    try {
      root = JSON.parse(dataEl.textContent);
    } catch (e) {
      const mainValueEl = viewer.querySelector(".json-value");
      mainValueEl.innerHTML = "<pre>Could not parse embedded JSON.</pre>";
      return;
    }

    const navEl = viewer.querySelector(".json-nav");
    const mainPathEl = viewer.querySelector(".json-path");
    const mainValueEl = viewer.querySelector(".json-value");
    const searchEl = viewer.querySelector(".json-search");
    const statusEl = viewer.querySelector(".json-status");
    const resetBtn = viewer.querySelector(".json-reset");

    buildNav(root, navEl, mainPathEl, mainValueEl, searchEl, statusEl);

    resetBtn.addEventListener("click", () => {
      searchEl.value = "";
      // rebuild nav to top-level
      buildNav(root, navEl, mainPathEl, mainValueEl, searchEl, statusEl);
    });
  });
})();
</script>
""".strip()


def generate_html_report(
    output_dir,
    simulations,
    descriptions,
    runtimes=None,
    total_sim_time=None
):
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"
    all_files = list(output_dir.glob("*"))

    html = [
        "<html><head><title>Simulation Results</title>",
        "<style>",
        "body { font-family: sans-serif; padding: 20px; background: #fcfcfc; color: #222; }",
        "h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }",
        "pre { background-color: #f8f8f8; padding: 8px; border: 1px solid #ddd; overflow-x: auto; }",
        "details { margin: 6px 0; padding-left: 1em; }",
        "summary { font-weight: 600; cursor: pointer; }",
        "code { background: #f1f1f1; padding: 2px 4px; border-radius: 4px; }",
        "nav ul { list-style: none; padding-left: 0; }",
        "nav ul li { margin: 5px 0; }",
        'a.download-btn { display: inline-block; margin: 8px 0; padding: 4px 8px; background: #eee; border: 1px solid #ccc; text-decoration: none; font-size: 0.9em; border-radius: 4px; }',
        *_json_viewer_css_lines(),
        "</style>",
        "</head><body>",
        "<h1>Simulation Results</h1>",
    ]

    # ------------------------------------------------------------------
    # Group files by simulation
    # ------------------------------------------------------------------
    test_files: dict[str, list[Path]] = {test: [] for test in simulations}
    others: list[Path] = []

    for file in all_files:
        if file.name == report_path.name:
            continue
        for test in test_files:
            if file.name.startswith(str(test)):
                test_files[test].append(file)
                break
        else:
            others.append(file)

    # Only show tests that actually have output files
    available_tests = [test for test, files in test_files.items() if files]

    # ------------------------------------------------------------------
    # Table of Contents
    # ------------------------------------------------------------------
    html.append("<nav><h2>Contents</h2><ul>")
    for test in available_tests:
        sid = _safe_id(test)
        html.append(f'<li><a href="#{html_escape(sid)}">{html_escape(str(test))}</a></li>')
    html.append("</ul></nav>")

    # ------------------------------------------------------------------
    # Per-simulation sections
    # ------------------------------------------------------------------
    for test in available_tests:
        files = test_files[test]
        files = sorted(files, key=lambda p: p.name)

        sid = _safe_id(test)
        html.append(f'<h2 id="{html_escape(sid)}">{html_escape(str(test))}</h2>')

        description = descriptions.get(test, "")
        if description:
            html.append(f"<p><em>{html_escape(description)}</em></p>")

        if runtimes and test in runtimes:
            html.append(f"<p><strong>Runtime:</strong> {runtimes[test]:.2f} seconds</p>")

        # Full downloadable JSON (prefer composition doc)
        download_json = (
                next((f for f in files if f.name == f"{test}.json"), None)
                or next((f for f in files if f.suffix == ".json"), None)
        )

        # JSON used for the viewer (prefer state JSON)
        viewer_json = (
                next((f for f in files if f.name == f"{test}_state.json"), None)
                or download_json
        )

        viz_file = next((f for f in files if f.name == f"{test}_viz.png"), None)
        pngs = [f for f in files if f.suffix == ".png" and f != viz_file]
        gifs = [f for f in files if f.suffix == ".gif"]

        # ---- JSON section ----
        if viewer_json:
            # Title: what the embedded viewer is based on (optional but honest)
            html.append(f"<h3>{html_escape(viewer_json.name)}</h3>")

            # Download: always the full json if present
            if download_json:
                html.append(
                    f'<a class="download-btn" href="{_url_href(download_json.name)}" target="_blank">'
                    f'View full JSON</a>'
                )

            # Viewer: show filtered state view
            try:
                state_data = _load_state_data_from_json(viewer_json)
                if state_data:
                    html.append(_render_json_api_viewer_block(test=str(test), state_data=state_data))
                else:
                    html.append('<p><em>No state-like content found in JSON.</em></p>')
            except Exception as e:
                html.append(f"<pre>Could not load JSON: {html_escape(str(e))}</pre>")

        # Bigraph visualization
        if viz_file:
            html.append(f"<h3>{html_escape(viz_file.name)}</h3>")
            html.append(
                f'<img src="{html_escape(viz_file.name)}" style="max-width:100%; height:auto; max-height:600px;">'
            )

        # PNG plots
        for f in pngs:
            html.append(f"<h3>{html_escape(f.name)}</h3>")
            html.append(f'<img src="{html_escape(f.name)}" style="max-width:100%"><hr>')

        # GIFs
        for f in gifs:
            html.append(f"<h3>{html_escape(f.name)}</h3>")
            html.append(f'<img src="{html_escape(f.name)}" style="max-width:100%"><hr>')

    # ------------------------------------------------------------------
    # Other files & total runtime
    # ------------------------------------------------------------------
    if others:
        html.append("<h2>Other Generated Files</h2>")
        for f in sorted(others, key=lambda p: p.name):
            html.append(f"<p>{html_escape(f.name)}</p>")

    if total_sim_time is not None:
        html.append(
            f"<h2>Total Simulation Time</h2><p><strong>{total_sim_time:.2f} seconds</strong></p>"
        )

    # Attach the JS once at the end
    html.append(_json_viewer_js())
    html.append("</body></html>")

    report_path.write_text("\n".join(html), encoding="utf-8")



def pf(obj):
    pp = pprint.PrettyPrinter(indent=4)
    return pp.pformat(obj)


def reversed_tuple(tu):
    return tuple(reversed(tu))


def inverse_tuple(tu):
    return tuple(-x for x in tu)
