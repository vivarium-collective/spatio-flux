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
    OPTIONAL_KEYS = {'fields', 'particles', 'lattice'}
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
    plot_settings = build_plot_settings(particle_ids=particle_ids)
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


def _html_link(url: str, label: str | None = None) -> str:
    """Safe HTML <a> link for external references."""
    label = label or url
    # escape label; url is used as an attribute so escape too
    return f'<a href="{html_escape(url)}" target="_blank" rel="noopener noreferrer">{html_escape(label)}</a>'


def _spatio_flux_process_families_table_html() -> str:
    """HTML rendering of the Spatio–Flux process family table (with color column)."""

    COLORS = {
        "metabolic": "#B34A44",          # dfba_process
        "transport": "#D6C35F",          # diffusion
        "movement": "#8EC09A",           # newtonian_particles_process (movement family)
        "coupling": "#CBDD8A",           # exchange_adapter
        "structural": "#A9DDE3",         # particle_graph_rewrite
    }

    def swatch(hex_color: str, label: str) -> str:
        return (
            f'<span class="sf-swatch" style="background:{html_escape(hex_color)}" '
            f'title="{html_escape(label)}"></span>'
            f'<span class="sf-swatch-label">{html_escape(label)}</span>'
        )

    def pills(items: list[str]) -> str:
        return " ".join(f'<code class="sf-pill">{html_escape(x)}</code>' for x in items)

    rows = [
        {
            "family": "Metabolic processes",
            "color": swatch(COLORS["metabolic"], "Metabolism"),
            "processes": pills(["DynamicFBA", "MonodKinetics", "SpatialDFBA"]),
            "role": (
                "<strong>Turns nutrients into growth.</strong> Computes uptake, secretion, and biomass production "
                "either at individual sites or across spatial grids by operating on substrate fields and biomass variables."
            ),
        },
        {
            "family": "Field transport",
            "color": swatch(COLORS["transport"], "Transport"),
            "processes": pills(["DiffusionAdvection"]),
            "role": (
                "<strong>Makes space matter.</strong> Updates dissolved species fields via diffusion and advection, "
                "so local metabolic activity can influence distant regions over time."
            ),
        },
        {
            "family": "Particle movement",
            "color": swatch(COLORS["movement"], "Movement"),
            "processes": pills(["BrownianMovement", "PymunkParticleMovement"]),
            "role": (
                "<strong>Moves agents through continuous space.</strong> Brownian motion provides stochastic movement; "
                "Newtonian motion adds mass, velocity, inertia, friction, and elastic interactions."
            ),
        },
        {
            "family": "Particle–field coupling",
            "color": swatch(COLORS["coupling"], "Coupling"),
            "processes": pills(["ParticleExchange"]),
            "role": (
                "<strong>Bridges discrete and continuous.</strong> Mediates bidirectional exchange between particle-local "
                "state and spatial fields, syncing internal particle chemistry with nearby lattice values."
            ),
        },
        {
            "family": "Structural and boundary processes",
            "color": swatch(COLORS["structural"], "Rewrite"),
            "processes": pills(["ParticleDivision", "ManageBoundaries"]),
            "role": (
                "<strong>Changes the population.</strong> Rewrites the particle store by creating, removing, or relocating "
                "particles in response to conditions like growth thresholds or boundary crossings."
            ),
        },
    ]

    body_rows = []
    for r in rows:
        body_rows.append(f"""
<tr>
  <td class="sf-family">
    <div class="sf-family-title">{r['family']}</div>
  </td>
  <td class="sf-color">{r['color']}</td>
  <td class="sf-procs">{r['processes']}</td>
  <td class="sf-role">{r['role']}</td>
</tr>
""".strip())

    return f"""
<section class="sf-table-wrap">
  <h3>Spatio–Flux process families</h3>

  <p class="sf-lede">
    Think of these as <strong>lego bricks for multiscale simulation</strong>.
    Each family does one thing well—metabolism, transport, motion, coupling, or structural change—
    and complex behaviors emerge by composing families rather than extending any single process.
  </p>

  <table class="sf-table sf-table-blog">
    <thead>
      <tr>
        <th>Family</th>
        <th>Color</th>
        <th>Processes</th>
        <th>Role</th>
      </tr>
    </thead>
    <tbody>
      {"".join(body_rows)}
    </tbody>
  </table>
</section>
""".strip()


def _how_to_read_bigraph_html() -> str:
    """A basics-only, blog-like guide to reading the bigraph image (no table repetition)."""
    return r"""
<details class="note">
  <summary>How to read the bigraph visualization</summary>

  <p>
    Each diagram is a <strong>map of a composite simulation</strong>: what state exists, what processes run,
    and how data flows between them.
  </p>

  <div class="sf-how-grid">
    <div class="sf-how-card">
      <h4>1) Nodes</h4>
      <p>
        <strong>Circles</strong> are <em>state</em> (variables or structured stores).
        <strong>Boxes</strong> are <em>processes</em> (update rules that run on a schedule).
      </p>
    </div>

    <div class="sf-how-card">
      <h4>2) Edges</h4>
      <p>
        Edges show <strong>read/write dependency</strong>: a process reads state to compute updates,
        and writes deltas back into state.
      </p>
    </div>

    <div class="sf-how-card">
      <h4>3) Hierarchy</h4>
      <p>
        Big nodes often contain nested nodes. That nesting reflects the <strong>hierarchical state tree</strong>
        (e.g., collections, sub-stores, or typed substructures).
      </p>
    </div>

    <div class="sf-how-card">
      <h4>4) A quick way to scan</h4>
      <p>
        Start by locating the <strong>main stores</strong> (large circles), then follow edges into the
        <strong>processes</strong> that touch them. The “story” is the loop: state → process → state.
      </p>
    </div>
  </div>

  <p class="sf-how-tip">
    <strong>Tip:</strong> If a diagram feels busy, focus on one store (fields or particles), then trace only the
    processes connected to it. The report sections below let you compare structure (bigraph),
    serialized state (JSON viewer), and behavior (plots/GIFs) side-by-side.
  </p>
</details>
""".strip()


def _vivarium2_ecosystem_html() -> str:
    items = [
        {
            "name": "bigraph-schema",
            "url": "https://github.com/vivarium-collective/bigraph-schema",
            "desc": (
                "Defines a compositional type system and hierarchical data structures using JSON-based schemas. "
                "Provides the type engine, schema compilation, state validation, and type-specific update operators "
                "used in process–bigraph delta semantics."
            ),
        },
        {
            "name": "process-bigraph",
            "url": "https://github.com/vivarium-collective/process-bigraph",
            "desc": (
                "The dynamic core. Defines typed Process and Composite abstractions, event scheduling, global time "
                "management, and orchestration logic for executing process–bigraph documents."
            ),
        },
        {
            "name": "bigraph-viz",
            "url": "https://github.com/vivarium-collective/bigraph-viz",
            "desc": (
                "Parses process–bigraph documents and renders their structure as inspectable graphs."
            ),
        },
        {
            "name": "spatio-flux",
            "url": "https://github.com/vivarium-collective/spatio-flux",
            "desc": (
                "A domain-specific application repository implementing concrete process families for metabolism, "
                "spatial fields, particle dynamics, and particle–field coupling, with an extended test suite of "
                "executable compositions."
            ),
        },
    ]

    lis = []
    for it in items:
        name_link = _html_link(it["url"], it["name"])
        lis.append(
            f"<li><strong><code>{name_link}</code></strong> — {html_escape(it['desc'])}</li>"
        )

    return f"""
<section class="sf-ecosystem">
  <h3>Vivarium 2.0 ecosystem</h3>

  <p>
    Spatio–Flux is a reference application in the <strong>Vivarium 2.0</strong> software suite:
    an open-source ecosystem for building, executing, and visualizing Process–Bigraph compositions.
  </p>

  <ul>
    {"".join(lis)}
  </ul>
</section>
""".strip()


def _spatio_flux_intro_html(total_sim_time=None, outdir: str | None = None) -> str:
    """
    Engaging intro for the report: invites exploration, explains what clicking reveals,
    includes process families table + Vivarium 2.0 ecosystem + references.
    """
    github_url = "https://github.com/vivarium-collective/spatio-flux"
    paper_url = "https://arxiv.org/abs/2512.23754"

    parts = []
    parts.append('<section class="intro-card" id="about">')

    parts.append("<h2>Explore the Spatio–Flux test suite</h2>")
    parts.append(
        "<p>"
        "This page is a catalog of executable compositions built with the "
        "Process Bigraph protocol. Each entry below is a self-contained simulation that "
        "demonstrates how distinct modeling concerns—metabolism, spatial transport, particle dynamics, and "
        "structural change—can be composed through explicit interfaces and shared state."
        "</p>"
    )
    parts.append(
        "<p>"
        "Rather than presenting biological conclusions, the goal of this test suite is to make "
        "model structure visible and inspectable. Clicking into a simulation lets you see "
        "<em>how</em> it is built: which processes are present, how they are wired to state, and how "
        "different process families combine to form more complex behaviors."
        "</p>"
    )

    parts.append("""
<div class="callout-grid">
  <div class="callout">
    <h4>Bigraph structure</h4>
    <p>Inspect a rendered process–bigraph showing state variables, processes, and their read/write dependencies.</p>
  </div>
  <div class="callout">
    <h4>Interactive state</h4>
    <p>Browse serialized state trees to explore spatial fields, particles, and intermediate variables.</p>
  </div>
  <div class="callout">
    <h4>Dynamics & behavior</h4>
    <p>View plots and animations illustrating spatial gradients, particle motion, growth, and division.</p>
  </div>
  <div class="callout">
    <h4>Composition documents</h4>
    <p>Download the machine-readable process–bigraph documents that define each simulation.</p>
  </div>
</div>
""".strip())

    parts.append('<p class="hint">Scroll down to explore individual compositions, or use the Contents section to jump to a simulation.</p>')

    parts.append(_spatio_flux_process_families_table_html())
    parts.append(_vivarium2_ecosystem_html())

    parts.append("<h3>References</h3>")
    parts.append("<ul>")
    parts.append(f"<li>{_html_link(github_url, 'Spatio–Flux GitHub repository')}</li>")
    parts.append(f"<li>{_html_link(paper_url, 'Process Bigraphs paper (arXiv:2512.23754)')}</li>")
    parts.append("</ul>")

    meta_bits = []
    if total_sim_time is not None:
        meta_bits.append(
            f"<span class='meta-pill'><strong>Total sim time:</strong> {total_sim_time:.2f}s</span>"
        )
    meta_bits.append(
        f"<span class='meta-pill'><strong>Generated:</strong> "
        f"{html_escape(datetime.now().isoformat(timespec='seconds'))}</span>"
    )
    parts.append("<div class='meta-row'>" + " ".join(meta_bits) + "</div>")

    parts.append("</section>")
    return "\n".join(parts)


def _json_viewer_js() -> str:
    # Plain JS: no dependencies; works in a local report.html.
    # Includes:
    #   - JSON viewer behavior (existing)
    #   - floating "Back to Contents" button behavior (new; no-op if button missing)
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

  // --------------------------
  // JSON viewers
  // --------------------------
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
      buildNav(root, navEl, mainPathEl, mainValueEl, searchEl, statusEl);
    });
  });

  // --------------------------
  // Floating "Back to Contents" button
  // --------------------------
  const backBtn = document.querySelector(".back-to-toc");
  if (backBtn){
    function onScroll(){
      backBtn.classList.toggle("show", window.scrollY > 600);
    }
    window.addEventListener("scroll", onScroll, {passive:true});
    onScroll();
  }
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
        "<html><head><title>Spatio–Flux Test Suite Report</title>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<style>",
        # Base page
        "html { scroll-behavior: smooth; }",
        "body { font-family: sans-serif; padding: 20px; background: #fcfcfc; color: #222; }",
        "h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }",
        "h3 { margin-top: 1.2em; }",
        "pre { background-color: #f8f8f8; padding: 8px; border: 1px solid #ddd; overflow-x: auto; }",
        "details { margin: 8px 0; }",
        "summary { font-weight: 600; cursor: pointer; }",
        "code { background: #f1f1f1; padding: 2px 4px; border-radius: 4px; }",
        "nav ul { list-style: none; padding-left: 0; }",
        "nav ul li { margin: 5px 0; }",
        "nav { margin: 0; }",
        'a.download-btn { display: inline-block; margin: 8px 0; padding: 4px 8px; background: #eee; border: 1px solid #ccc; text-decoration: none; font-size: 0.9em; border-radius: 4px; }',

        # Better anchor positioning when jumping
        ":target { scroll-margin-top: 16px; }",

        # Layout: sticky sidebar TOC
        ".layout { display: grid; grid-template-columns: 320px 1fr; gap: 18px; align-items: start; }",
        ".sidebar { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow: auto; padding-right: 10px; }",
        ".content { min-width: 0; }",
        ".sidebar-card { background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 12px 12px; }",
        ".sidebar-card h2 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 6px; }",
        ".sidebar-card a { text-decoration: none; color: #222; }",
        ".sidebar-card a:hover { text-decoration: underline; }",
        "@media (max-width: 980px){ .layout { grid-template-columns: 1fr; } .sidebar { position: static; max-height: none; padding-right: 0; } }",

        # Floating back-to-contents button (shows after scroll)
        ".back-to-toc { position: fixed; right: 18px; bottom: 18px; padding: 10px 12px; border-radius: 999px; border: 1px solid #ddd; background: #fff; box-shadow: 0 6px 18px rgba(0,0,0,0.08); text-decoration: none; font-size: 13px; color: #222; display: none; z-index: 9999; }",
        ".back-to-toc:hover { background: #f7f7f7; }",
        ".back-to-toc.show { display: inline-block; }",

        # Intro styling
        ".intro-card { background: #ffffff; border-left: 4px solid #e0e0e0; padding: 12px 16px; margin: 12px 0 18px 0; border-radius: 10px; }",
        ".intro-card h2 { margin-top: 0; }",
        ".intro-card ul { margin: 8px 0 0 18px; }",
        ".meta-row { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }",
        ".meta-pill { display:inline-block; padding: 3px 10px; border: 1px solid #e2e2e2; border-radius: 999px; background: #fafafa; font-size: 12px; color: #333; }",
        ".hint { margin-top: 10px; color: #333; font-size: 13px; }",

        # Callout grid
        ".callout-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 18px 0; }",
        ".callout { background: #ffffff; border: 1px solid #eee; border-radius: 10px; padding: 10px 12px; }",
        ".callout h4 { margin: 0 0 6px 0; }",
        ".callout p { margin: 0; color: #333; font-size: 13px; line-height: 1.35; }",

        # Notes
        ".note { color: #444; background: #fff; border-left: 4px solid #ddd; padding: 10px 12px; border-radius: 10px; }",

        # Table styling
        ".sf-table-wrap { margin-top: 12px; }",
        ".sf-table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e6e6e6; border-radius: 10px; overflow: hidden; }",
        ".sf-table th, .sf-table td { border-bottom: 1px solid #eee; padding: 10px 12px; vertical-align: top; }",
        ".sf-table th { text-align: left; background: #fafafa; font-weight: 700; }",
        ".sf-table tr:last-child td { border-bottom: none; }",
        ".sf-ecosystem { margin-top: 12px; }",
        ".sf-ecosystem ul { margin: 8px 0 0 18px; }",

        # Blog-like table enhancements
        ".sf-lede { margin: 8px 0 12px 0; color: #333; }",
        ".sf-table-blog td { font-size: 13px; line-height: 1.35; }",
        ".sf-family-title { font-weight: 700; }",
        ".sf-procs { white-space: normal; }",
        ".sf-pill { display: inline-block; margin: 0 6px 6px 0; padding: 2px 8px; border-radius: 999px; border: 1px solid #e2e2e2; background: #f8f8f8; font-size: 12px; }",

        # Color swatches
        ".sf-color { width: 120px; }",
        ".sf-swatch { display:inline-block; width: 18px; height: 18px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.12); vertical-align: middle; margin-right: 8px; }",
        ".sf-swatch-label { font-size: 12px; color: #444; vertical-align: middle; }",

        # How-to section cards
        ".sf-how-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 10px 0 10px 0; }",
        ".sf-how-card { background:#fff; border:1px solid #eee; border-radius:10px; padding:10px 12px; }",
        ".sf-how-card h4 { margin: 0 0 6px 0; }",
        ".sf-how-card p { margin: 0; font-size: 13px; line-height: 1.35; color:#333; }",
        ".sf-how-tip { margin-top: 8px; font-size: 13px; color:#333; }",

        # Header
        ".hero { margin: 0 0 12px 0; }",
        ".hero h1 { margin: 0; }",
        ".hero p { margin: 6px 0 0 0; color: #333; }",

        *_json_viewer_css_lines(),
        "</style>",
        "</head><body>",
        # Top anchor for "back to contents"
        '<a id="top"></a>',
    ]

    # Title
    html.append("""
<div class="hero">
  <h1>Spatio–Flux Test Suite Report</h1>
  <p>
    Browse a suite of runnable composites, inspect their structure, and explore the emitted state and dynamics.
  </p>
</div>
""".strip())

    # ------------------------------------------------------------------
    # Group files by simulation (needed for Contents)
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

    available_tests = [test for test, files in test_files.items() if files]

    # ------------------------------------------------------------------
    # Build TOC HTML once (we'll put it in the sticky sidebar)
    # ------------------------------------------------------------------
    toc_bits = []
    toc_bits.append('<nav id="toc" class="sidebar-card">')
    toc_bits.append("<h2>Contents</h2><ul>")
    toc_bits.append('<li><a href="#about">About / Overview</a></li>')
    toc_bits.append('<li><a href="#how-to">How to read the bigraph visualization</a></li>')
    for test in available_tests:
        sid = _safe_id(test)
        toc_bits.append(f'<li><a href="#{html_escape(sid)}">{html_escape(str(test))}</a></li>')
    toc_bits.append("</ul></nav>")
    toc_html = "\n".join(toc_bits)

    # ------------------------------------------------------------------
    # Layout wrapper (sidebar + main content)
    # ------------------------------------------------------------------
    html.append('<div class="layout">')
    html.append('<aside class="sidebar">')
    html.append(toc_html)
    html.append("</aside>")
    html.append('<main class="content">')

    # ------------------------------------------------------------------
    # About / Overview (collapsible)
    # ------------------------------------------------------------------
    intro_html = _spatio_flux_intro_html(
        total_sim_time=total_sim_time,
        outdir=str(output_dir)
    )
    if 'id="about"' not in intro_html:
        intro_html = f'<div id="about"></div>\n{intro_html}'

    html.append(f"""
<details class="note" open>
  <summary>About / Overview</summary>
  {intro_html}
</details>
""".strip())

    # ------------------------------------------------------------------
    # How-to-read block
    # ------------------------------------------------------------------
    how_block = _how_to_read_bigraph_html()
    if 'id="how-to"' not in how_block:
        how_block = f'<div id="how-to"></div>\n{how_block}'
    html.append(how_block)

    # ------------------------------------------------------------------
    # Per-simulation sections
    # ------------------------------------------------------------------
    for test in available_tests:
        files = sorted(test_files[test], key=lambda p: p.name)
        sid = _safe_id(test)

        html.append(f'<h2 id="{html_escape(sid)}">{html_escape(str(test))}</h2>')

        description = descriptions.get(test, "")
        if description:
            html.append(f"<p><em>{html_escape(description)}</em></p>")

        if runtimes and test in runtimes:
            html.append(f"<p><strong>Runtime:</strong> {runtimes[test]:.2f} seconds</p>")

        download_json = (
            next((f for f in files if f.name == f"{test}.json"), None)
            or next((f for f in files if f.suffix == ".json"), None)
        )

        viewer_json = (
            next((f for f in files if f.name == f"{test}_state.json"), None)
            or download_json
        )

        viz_file = next((f for f in files if f.name == f"{test}_viz.png"), None)
        pngs = [f for f in files if f.suffix == ".png" and f != viz_file]
        gifs = [f for f in files if f.suffix == ".gif"]

        # JSON section
        if viewer_json:
            html.append(f"<h3>{html_escape(viewer_json.name)}</h3>")

            if download_json:
                html.append(
                    f'<a class="download-btn" href="{_url_href(download_json.name)}" target="_blank">'
                    f'View full JSON</a>'
                )

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

    # Other files & total runtime
    if others:
        html.append("<h2>Other Generated Files</h2>")
        for f in sorted(others, key=lambda p: p.name):
            html.append(f"<p>{html_escape(f.name)}</p>")

    if total_sim_time is not None:
        html.append(
            f"<h2>Total Simulation Time</h2><p><strong>{total_sim_time:.2f} seconds</strong></p>"
        )

    # Close layout wrapper
    html.append("</main></div>")

    # Floating back-to-contents button
    html.append('<a class="back-to-toc" href="#toc" title="Back to contents">↑ Contents</a>')

    # JS (viewer + back-to-toc)
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