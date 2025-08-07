import json
import pprint
import shutil
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import numpy as np
from bigraph_viz import plot_bigraph
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires
from vivarium.vivarium import VivariumTypes


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


def run_composite_document(document, core=None, name=None, time=None):
    """
    Instantiates and runs a Composite simulation.

    Args:
        document (dict): Composition document with initial state and optional schema.
        time (float): Simulation duration.
        core (VivariumTypes): Core schema registration object.
        name (str): Output name prefix.

    Returns:
        dict: Simulation results emitted during the run.
    """
    time = time or 60
    if core is None:
        from spatio_flux import register_types
        core = VivariumTypes()
        core = register_types(core)
    if name is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'spatio_flux_{date}'

    document = {'state': document} if 'state' not in document else document
    if 'emitter' not in document['state']:
        state_keys = list(document['state'].keys())
        document['state']['emitter'] = get_standard_emitter(state_keys=state_keys)

    print(f'Making composite {name}...')
    sim = Composite(document, core=core)

    # Save composition JSON
    sim.save(filename=f'{name}.json', outdir='out')

    # Save visualization of the initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.composition.items() if k not in ['global_time', 'emitter']}

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir='out',
        filename=f'{name}_viz',
        dpi='300',
        collapse_redundant_processes=True
    )

    print(f'Simulating {name}...')
    sim.run(time)
    results = gather_emitter_results(sim)
    return results[('emitter',)]


def prepare_output_dir(output_dir):
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"🧹 Clearing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def generate_html_report(
        output_dir,
        simulations,
        descriptions,
        runtimes=None,
        total_sim_time=None
):
    output_dir = Path(output_dir)
    report_path = output_dir / 'report.html'
    all_files = list(output_dir.glob('*'))

    html = [
        '<html><head><title>Simulation Results</title>',
        '<style>',
        'body { font-family: sans-serif; padding: 20px; background: #fcfcfc; color: #222; }',
        'h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }',
        'pre { background-color: #f8f8f8; padding: 8px; border: 1px solid #ddd; overflow-x: auto; }',
        'details { margin: 6px 0; padding-left: 1em; }',
        'summary { font-weight: 600; cursor: pointer; }',
        'code { background: #f1f1f1; padding: 2px 4px; border-radius: 4px; }',
        'nav ul { list-style: none; padding-left: 0; }',
        'nav ul li { margin: 5px 0; }',
        'a.download-btn { display: inline-block; margin: 8px 0; padding: 4px 8px; background: #eee; border: 1px solid #ccc; text-decoration: none; font-size: 0.9em; border-radius: 4px; }',
        '</style>',
        '</head><body>',
        '<h1>Simulation Results</h1>'
    ]

    # Table of Contents
    html.append('<nav><h2>Contents</h2><ul>')
    for test in simulations:
        html.append(f'<li><a href="#{test}">{test}</a></li>')
    html.append('</ul></nav>')

    test_files = {test: [] for test in simulations}
    others = []

    for file in all_files:
        if file.name == report_path.name:
            continue
        for test in test_files:
            if file.name.startswith(test):
                test_files[test].append(file)
                break
        else:
            others.append(file)

    def json_to_html(obj):
        if isinstance(obj, dict):
            return ''.join(
                f"<details><summary>{html_escape(str(k))}</summary>{json_to_html(v)}</details>"
                for k, v in obj.items()
            )
        elif isinstance(obj, list):
            return ''.join(
                f"<details><summary>[{i}]</summary>{json_to_html(v)}</details>"
                for i, v in enumerate(obj)
            )
        else:
            return f"<code>{html_escape(json.dumps(obj))}</code>"

    for test, files in test_files.items():
        if not files:
            continue
        html.append(f'<h2 id="{test}">{test}</h2>')

        # Optional description
        description = descriptions.get(test, '')
        if description:
            html.append(f'<p><em>{description}</em></p>')

        # Runtime
        if runtimes and test in runtimes:
            html.append(f'<p><strong>Runtime:</strong> {runtimes[test]:.2f} seconds</p>')

        # Sort files
        json_file = next((f for f in files if f.suffix == '.json'), None)
        viz_file = next((f for f in files if f.name == f"{test}_viz.png"), None)
        pngs = sorted(f for f in files if f.suffix == '.png' and f != viz_file)
        gifs = sorted(f for f in files if f.suffix == '.gif')

        # JSON viewer
        if json_file:
            html.append(f'<h3>{json_file.name}</h3>')
            html.append(f'<a class="download-btn" href="{json_file.name}" target="_blank">View full JSON</a>')
            try:
                with open(json_file, 'r') as jf:
                    full_data = json.load(jf)
                # Filter out 'emitter' and 'global_time' top-level keys
                state_data = {
                    key: value for key, value in full_data.get('state', {}).items()
                    if key not in ['emitter', 'global_time']
                }

                if state_data:
                    html.append(json_to_html(state_data))
                else:
                    html.append('<p><em>No "state" key found.</em></p>')
            except Exception as e:
                html.append(f'<pre>Could not load JSON: {e}</pre>')

        # Bigraph visualization
        if viz_file:
            html.append(f'<h3>{viz_file.name}</h3>')
            html.append(f'<img src="{viz_file.name}" style="max-width:100%; height:auto; max-height:600px;">')
            # html.append(f'<img src="{viz_file.name}" style="max-width:100%"><hr>')

        # PNG plots
        for f in pngs:
            html.append(f'<h3>{f.name}</h3><img src="{f.name}" style="max-width:100%"><hr>')

        # GIFs
        for f in gifs:
            html.append(f'<h3>{f.name}</h3><img src="{f.name}" style="max-width:100%"><hr>')

    if others:
        html.append('<h2>Other Generated Files</h2>')
        for f in sorted(others):
            html.append(f'<p>{f.name}</p>')

    if total_sim_time:
        html.append(f'<h2>Total Simulation Time</h2><p><strong>{total_sim_time:.2f} seconds</strong></p>')

    html.append('</body></html>')

    with open(report_path, 'w') as f:
        f.write('\n'.join(html))


def pf(obj):
    pp = pprint.PrettyPrinter(indent=4)
    return pp.pformat(obj)


def reversed_tuple(tu):
    return tuple(reversed(tu))


def inverse_tuple(tu):
    return tuple(-x for x in tu)
