import os
import io
import base64
import random
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.colors import hsv_to_rgb
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as mcolors



def _evenly_spaced_indices(n_items, n_pick):
    """Choose n_pick indices spread across [0, n_items-1]."""
    n_pick = max(1, min(n_pick, n_items))
    return [int(round(k*(n_items-1)/(n_pick-1))) for k in range(n_pick)] if n_pick > 1 else [0]

def _global_minmax_per_field(sorted_results, field_names):
    """Compute global vmin/vmax per field across all time points."""
    mm = {}
    for f in field_names:
        frames = sorted_results['fields'][f]
        flat = np.concatenate([np.asarray(fr).ravel() for fr in frames])
        mm[f] = (float(np.min(flat)), float(np.max(flat)))
    return mm

def _ensure_dir_and_path(out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename) if out_dir else filename

def _coerce_axes_grid(axes, n_rows, n_cols):
    """Return axes[r][c] for both 1D/2D cases."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.array([axes if isinstance(axes, (list, np.ndarray)) else [axes]])
    if n_cols == 1:
        return np.array([[ax] for ax in (axes if isinstance(axes, (list, np.ndarray)) else [axes])])
    return axes

def _draw_particles(ax, particles, mass_scaling, xmax, ymax, color='b', min_mass=0.01):
    """Overlay particles on ax (reuses your logic)."""
    for p in particles.values():
        x, y = p['position']
        mass = max(p.get('mass', min_mass), min_mass)
        if 0 <= x <= xmax and 0 <= y <= ymax:
            ax.scatter(x, y, s=mass * mass_scaling, color=color)


def field_for_imshow(field_2d):
    """Assumes field is stored as (ny, nx) indexed [y, x]."""
    return np.asarray(field_2d)

def sort_results(results):
    if ('emitter',) in results:
        results = results[('emitter',)]
    if results[0] is None:
        return

    sorted_results = {'fields': {
        key: [] for key in results[0].get('fields', {}).keys()
    }, 'time': []}

    for results in results:
        time = results['global_time']
        fields = results.get('fields', {})
        sorted_results['time'].append(time)
        for key, value in fields.items():
            sorted_results['fields'][key].append(value)
    return sorted_results


def plot_model_grid(model_grid, out_dir='out', filename='model_grid.png',
                    title=None, show_border_coords=True):

    if not model_grid or not isinstance(model_grid, (list, tuple)):
        raise ValueError("model_grid must be a non-empty 2D list/tuple.")

    n_rows = len(model_grid)
    n_cols = len(model_grid[0]) if n_rows else 0
    if any(len(row) != n_cols for row in model_grid):
        raise ValueError("All rows must have the same number of columns.")

    # Add coordinate headers if requested
    if show_border_coords:
        table_data = [[""] + [str(c) for c in range(n_cols)]]
        for r in range(n_rows):
            table_data.append([str(r)] + list(model_grid[r]))
    else:
        table_data = [list(row) for row in model_grid]

    # Create figure sized tightly around content
    fig_w = max(1.5, len(table_data[0]) * 0.6)
    fig_h = max(1.5, (len(table_data)+1) * 0.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis('off')

    # Draw table with full borders
    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        edges='closed'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Uniform borders
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor('black')

    # Tight layout: remove extra space above/below
    if title:
        ax.set_title(title, fontsize=10, pad=2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return filepath

def place_legend_outside_right(
    ax,
    fig,
    *,
    width_fraction=0.28,   # fraction of figure reserved for legend
    loc="center left",
    anchor=(1.02, 0.5),
    fontsize=9,
    frameon=False,
    **legend_kwargs,
):
    """
    Place legend to the right of the axes with guaranteed no overlap.

    Parameters
    ----------
    ax : matplotlib Axes
    fig : matplotlib Figure
    width_fraction : float
        Fraction of figure width reserved for legend (0.25–0.35 is typical).
    loc : str
        Legend location relative to bbox_to_anchor.
    anchor : tuple
        (x, y) anchor in axes coordinates.
    """
    # Shrink plot area to make room
    fig.subplots_adjust(right=1.0 - width_fraction)

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        fontsize=fontsize,
        frameon=frameon,
        **legend_kwargs,
    )
    return legend



def plot_time_series(
        results,
        field_names=None,
        coordinates=None,
        out_dir=None,
        filename='time_series.png',
        display=False,
        log_scale=False,
        normalize=False,

        # sizing / appearance
        figsize=(12, 6),
        dpi=None,
        title=None,
        legend=True,
        legend_kwargs=None,

        # axis units and labeling
        time_units=None,          # e.g. "s", "min", "h"
        time_scale=1.0,           # multiply times by this (e.g. 1/60 for min if input is sec)
        y_label_base="Value",     # e.g. "Concentration"
        field_units=None,         # dict: field -> units string, e.g. {"glucose":"mM"}
        normalized_label="normalized",

        # per-field styling
        field_colors=None,        # dict: field -> matplotlib color
        field_styles=None,        # dict: field -> dict of kwargs passed to ax.plot
        linewidth=2.0,

        # color reservation / auto assignment
        reserved_colors=None,     # iterable of colors to avoid reusing for non-semantic fields
        auto_color_cycle=None,    # optional explicit list of colors to cycle through

        # legend placement (outside right)
        legend_outside=True,
        legend_width_fraction=0.30,  # fraction of figure width reserved for legend
        legend_anchor=(1.02, 0.5),   # (x,y) in axes coords
):
    """
    Plots time series for specified fields and coordinates from the results.

    Features:
      - figsize/dpi control
      - optional title (default None)
      - time units + optional scaling
      - per-field units
      - normalization labeling
      - per-field colors/styles
      - reserved semantic colors won't be reused for other fields
      - legend placed outside to the right (no overlap)

    Expects `sort_results(results)` to exist and return:
      {
        'time': [...],
        'fields': { field_name: [scalar_or_array_per_time, ...], ... }
      }
    """
    # Defaults
    field_names = field_names or ['glucose', 'acetate', 'dissolved biomass']
    field_units = field_units or {}
    field_colors = field_colors or {}
    field_styles = field_styles or {}
    legend_kwargs = legend_kwargs or {}

    sorted_results = sort_results(results)
    times = sorted_results['time']

    if time_scale != 1.0:
        times = [t * time_scale for t in times]

    # ---- Build an explicit color for every field (semantic or auto-assigned),
    #      avoiding reserved colors for non-semantic fields.
    reserved_colors = reserved_colors or []
    reserved_rgba = {mcolors.to_rgba(c) for c in reserved_colors}

    if auto_color_cycle is None:
        candidate_cycle = list(plt.get_cmap("tab20").colors)
    else:
        candidate_cycle = list(auto_color_cycle)

    # Filter cycle colors that would collide with reserved colors
    candidate_cycle = [c for c in candidate_cycle if mcolors.to_rgba(c) not in reserved_rgba]

    assigned_colors = dict(field_colors)
    cycle_idx = 0
    for fname in field_names:
        if fname in assigned_colors:
            continue
        if cycle_idx < len(candidate_cycle):
            assigned_colors[fname] = candidate_cycle[cycle_idx]
            cycle_idx += 1
        # else: no color assigned; matplotlib will pick (rare). Keeping silent on purpose.

    # ---- Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _series_label(fname: str, coord=None) -> str:
        unit = field_units.get(fname)
        base = fname if coord is None else f"{fname} @ {coord}"
        if unit:
            base = f"{base} ({unit})"
        if normalize:
            base = f"{base} [{normalized_label}]"
        return base

    for field_name in field_names:
        if field_name not in sorted_results['fields']:
            print(f"Field '{field_name}' not found in results['fields']")
            continue

        field_data = sorted_results['fields'][field_name]

        style = dict(linewidth=linewidth)
        if field_name in assigned_colors:
            style["color"] = assigned_colors[field_name]
        style.update(field_styles.get(field_name, {}))

        # Strict guard (optional): prevent non-semantic fields from using reserved colors
        if "color" in style and reserved_rgba:
            c_rgba = mcolors.to_rgba(style["color"])
            if (c_rgba in reserved_rgba) and (field_name not in field_colors):
                raise ValueError(
                    f"Field '{field_name}' is using a reserved color {style['color']} "
                    f"but is not in field_colors. Add it to field_colors or choose another color."
                )

        if coordinates is None:
            data = field_data
            if normalize:
                initial = data[0] if data[0] != 0 else 1e-12
                data = [v / initial for v in data]
            ax.plot(times, data, label=_series_label(field_name), **style)
        else:
            for coord in coordinates:
                x, y = coord
                try:
                    series = [field_data[t][y, x] for t in range(len(times))]
                    if normalize:
                        initial = series[0] if series[0] != 0 else 1e-12
                        series = [v / initial for v in series]
                    ax.plot(times, series, label=_series_label(field_name, coord), **style)
                except Exception as e:
                    print(f"Error plotting {field_name} at {coord}: {e}")

    # Axis scaling
    if log_scale:
        ax.set_yscale('log')

    # Labels
    ax.set_xlabel(f"Time ({time_units})" if time_units else "Time")

    unique_units = {field_units.get(f) for f in field_names if field_units.get(f)}
    units_suffix = f" ({list(unique_units)[0]})" if len(unique_units) == 1 else ""
    norm_suffix = f" ({normalized_label})" if normalize else ""
    scale_suffix = " (log)" if log_scale else ""
    ax.set_ylabel(f"{y_label_base}{units_suffix}{norm_suffix}{scale_suffix}")

    # Title (default: none)
    if title:
        ax.set_title(title)

    # Legend: outside right, fixed space so it never overlaps
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside:
                # Reserve space on the right for the legend
                fig.subplots_adjust(right=1.0 - legend_width_fraction)

                # Pull some common legend style defaults
                fontsize = legend_kwargs.pop("fontsize", 9)
                frameon = legend_kwargs.pop("frameon", False)

                # Remove conflicting keys if user provided them
                legend_kwargs = dict(legend_kwargs)  # copy defensively
                legend_kwargs.pop("loc", None)
                legend_kwargs.pop("bbox_to_anchor", None)
                legend_kwargs.pop("borderaxespad", None)

                leg = ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=legend_anchor,
                    borderaxespad=0.0,
                    fontsize=fontsize,
                    frameon=frameon,
                    **legend_kwargs,
                )

                # Make text left-aligned inside the legend box (nice for long labels)
                try:
                    leg._legend_box.align = "left"
                except Exception:
                    pass
            else:
                ax.legend(**legend_kwargs)

    # Save
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, bbox_inches="tight", dpi=dpi)
    if display:
        plt.show()
    plt.close(fig)
    return filepath





def plot_snapshots(
        results,
        field_names,
        times,
):
    """
    Plots snapshots of specified fields at specified times from the results dictionary.

    :param results: Dictionary containing the results with keys 'time' and 'fields'.
                    Example: {'time': [0.1, 0.2, 0.3 ...], 'fields': {'glucose': [nparray_time0, nparray_time1], ...}}
    :param field_names: List of field names to plot.
                        Example: ['glucose', 'other chemical']
    :param times: List of times at which to plot the snapshots.
                  Example: [0.1, 0.3]
    """
    sorted_results = sort_results(results)
    time_indices = [sorted_results['time'].index(t) for t in times]

    num_rows = len(times)
    num_cols = len(field_names)

    # xmax, ymax = bounds
    # extent = [0, xmax, 0, ymax]

    # Compute global min and max for each field
    global_min_max = {}
    for field_name in field_names:
        field_data_all_times = np.concatenate(
            [sorted_results['fields'][field_name][t].flatten() for t in range(len(sorted_results['time']))])
        global_min_max[field_name] = (np.min(field_data_all_times), np.max(field_data_all_times))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    for row, time_index in enumerate(time_indices):
        for col, field_name in enumerate(field_names):
            if field_name in sorted_results['fields']:
                field_data = sorted_results['fields'][field_name][time_index]
                ax = axes[row, col] if num_rows > 1 and num_cols > 1 else (axes[row] if num_rows > 1 else axes[col])
                im = ax.imshow(field_data, cmap='viridis', aspect='auto', origin='lower',
                               # extent=extent,
                               vmin=global_min_max[field_name][0], vmax=global_min_max[field_name][1])
                ax.set_title(f'{field_name} at t={sorted_results["time"][time_index]}')
                fig.colorbar(im, ax=ax)
            else:
                print(f"Field '{field_name}' not found in results['fields']")

    plt.tight_layout()
    plt.show()


def plot_species_distributions_to_gif(
        results,
        out_dir=None,
        filename='species_distribution.gif',
        title='',
        skip_frames=1,
        species_to_show=None,
):
    # Sort the results as before
    sorted_results = sort_results(results)

    # Full list of species available
    all_species = list(sorted_results['fields'].keys())

    # Determine which species to show
    if species_to_show is None:
        species_names = all_species
    else:
        # Keep only valid species
        species_names = [s for s in species_to_show if s in all_species]

    n_species = len(species_names)
    if n_species == 0:
        raise ValueError("No valid species selected to show.")

    times = sorted_results['time']
    n_times = len(times)

    # Compute global min and max for each selected species
    global_min_max = {
        species: (
            np.min(np.concatenate([
                val.flatten() if isinstance(val, np.ndarray) else np.array([val])
                for val in sorted_results['fields'][species][:n_times]
            ])),
            np.max(np.concatenate([
                val.flatten() if isinstance(val, np.ndarray) else np.array([val])
                for val in sorted_results['fields'][species][:n_times]
            ]))
        )
        for species in species_names
    }

    images = []

    for i in range(0, n_times, skip_frames):
        fig, axs = plt.subplots(1, n_species, figsize=(5 * n_species, 4))

        if n_species == 1:
            axs = [axs]

        for j, species in enumerate(species_names):
            field_val = sorted_results['fields'][species][i]

            if not isinstance(field_val, np.ndarray) or field_val.ndim != 2:
                continue

            ax = axs[j]
            vmin, vmax = global_min_max[species]

            img = ax.imshow(field_val, interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(f'{species} at t = {times[i]:.2f}')
            plt.colorbar(img, ax=ax)

        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.tight_layout(pad=0.2)

        # Save the current figure to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    # Create the output directory if not exists
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
    else:
        filepath = filename

    # Save GIF
    imageio.mimsave(filepath, images, duration=0.5, loop=0)

    # Show inline in Jupyter
    with open(filepath, 'rb') as file:
        data = file.read()
        data_url = 'data:image/gif;base64,' + base64.b64encode(data).decode()
    display(HTML(f'<img src="{data_url}" alt="{title}" style="max-width:100%;"/>'))


def plot_particles_snapshot(ax, particles, mass_scaling=1.0, xmax=1.0, ymax=1.0, color='b', min_mass=0.01):
    """
    Plot particles on the given matplotlib axis.
    """
    for particle in particles.values():
        x, y = particle['position']
        mass = max(particle['mass'], min_mass)

        if 0 <= x <= xmax and 0 <= y <= ymax:
            ax.scatter(x, y, s=mass * mass_scaling, color=color)


def plot_species_distributions_with_particles_to_gif(
        results,
        out_dir=None,
        filename='species_distribution_with_particles.gif',
        title='',
        skip_frames=1,
        bounds=(1.0, 1.0),
        mass_scaling=10.0
):
    """Create a GIF showing spatial fields and particles over time."""

    # Sort and extract data
    sorted_results = sort_results(results)
    species_names = list(sorted_results['fields'].keys())
    n_species = len(species_names)
    times = sorted_results['time']
    n_times = len(times)
    xmax, ymax = bounds
    extent = [0, xmax, 0, ymax]

    # Compute global min/max per species for consistent color scaling
    global_min_max = {
        species: (
            np.min(np.concatenate([sorted_results['fields'][species][i].flatten() for i in range(n_times)])),
            np.max(np.concatenate([sorted_results['fields'][species][i].flatten() for i in range(n_times)]))
        )
        for species in species_names
    }

    # Handle emitter results if nested
    if ('emitter',) in results:
        results = results[('emitter',)]

    images = []

    for i in range(0, n_times, skip_frames):
        fields = results[i].get('fields', {})
        particles = results[i]['particles']

        fig, axs = plt.subplots(
            1, max(1, n_species),
            figsize=(5 * max(1, n_species), 5)
        )

        if n_species == 0:
            # Ensure axs is always iterable
            if not isinstance(axs, np.ndarray):
                axs = [axs]

            ax = axs[0]
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_aspect('equal')
            ax.set_title(f'Particles at t = {times[i]:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            plot_particles_snapshot(ax,
                           particles,
                           mass_scaling=mass_scaling,
                           xmax=xmax, ymax=ymax,
                           color='b', min_mass=0.01)
        else:
            # Ensure axs is a list of axes
            if n_species == 1:
                axs = [axs]

            for j, species in enumerate(species_names):
                ax = axs[j]
                field = field_for_imshow(fields[species])
                vmin, vmax = global_min_max[species]

                im = ax.imshow(field, interpolation='nearest', cmap='viridis',
                               vmin=vmin, vmax=vmax, extent=extent, origin='lower')
                ax.set_title(f'{species} at t = {times[i]:.2f}')
                plt.colorbar(im, ax=ax)

                plot_particles_snapshot(ax,
                               particles,
                               mass_scaling=mass_scaling,
                               xmax=xmax, ymax=ymax,
                               color='b', min_mass=0.01)

        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=0.1)

        # Save to buffer and append to GIF frames
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    # Output filepath
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
    else:
        filepath = filename

    # print(f'Saving GIF to {filepath}')
    imageio.mimsave(filepath, images, duration=0.5, loop=0)

    # Inline display for Jupyter
    with open(filepath, 'rb') as file:
        data = file.read()
        data_url = 'data:image/gif;base64,' + base64.b64encode(data).decode()
    display(HTML(f'<img src="{data_url}" alt="{title}" style="max-width:100%;"/>'))


def plot_particles(
        # total_time,
        env_size,
        history,
        out_dir=None,
        filename='multi_species_diffusion.gif',
        fps=20,
        mass_scaling=10.0,
):
    """
    Plot particle movements and save the animation as a GIF.

    Parameters:
    - total_time: Total time (frames) of the simulation.
    - env_size: Tuple indicating the xlim and ylim of the environment, as ((xmin, xmax), (ymin, ymax)).
    - history: History of particles at each time step.
    - filename: Filename for the output GIF.
    - fps: Frames per second for the output GIF.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(*env_size[0])
    ax.set_ylim(*env_size[1])
    ax.set_aspect('equal')
    n_frames = len(history)

    images = []
    for frame in range(n_frames):  # Include initial position
        ax.clear()
        ax.set_title(f'time {frame}')
        ax.set_xlim(*env_size[0])
        ax.set_ylim(*env_size[1])
        ax.set_aspect('equal')

        particles = history[frame]

        for particle_id, particle in particles.items():
            ax.scatter(particle['position'][0], particle['position'][1],
                       s=particle['mass']*mass_scaling, color='b')

        # Save the current figure to a temporary buffer
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', dpi=120)
        except:
            continue
            # import ipdb; ipdb.set_trace()
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()

    plt.close(fig)

    # Create the output directory if not exists
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
    else:
        filepath = filename

    # Create and save the GIF with loop=0 for infinite loop
    # print(f'saving {filepath}')
    imageio.mimsave(filepath, images, fps=fps, loop=0)

    # Optionally display the GIF in a Jupyter notebook
    with open(filepath, 'rb') as file:
        data = file.read()
        data_url = 'data:image/gif;base64,' + base64.b64encode(data).decode()
    display(HTML(f'<img src="{data_url}" alt="Particle Diffusion" style="max-width:100%;"/>'))


def plot_particles_mass(results, out_dir=None, filename='particles_mass_plot.png', display=False, max_legend=10):
    """
    Plot mass trajectories of individual particles over time and optionally save/display the plot.

    Parameters:
    - results: list of dicts, each with keys: 'global_time', 'particles' (dict of particle_id -> dict with 'mass')
    - out_dir: directory to save the plot (optional)
    - filename: name of the output plot file (default: 'particles_mass_plot.png')
    - display: if True, shows the plot inline (default: False)
    - max_legend: maximum number of particle IDs to show in the legend (default: 10)
    """
    # Aggregate particle mass data
    particle_traces = {}  # pid -> list of (time, mass)

    for entry in results:
        time = entry['global_time']
        particles = entry.get('particles', {})

        for pid, pdata in particles.items():
            if pid not in particle_traces:
                particle_traces[pid] = []
            particle_traces[pid].append((time, pdata['mass']))

    # Sort particle IDs for consistency
    sorted_pids = sorted(particle_traces.keys())
    total_particles = len(sorted_pids)

    # Plot
    plt.figure(figsize=(10, 6))
    for idx, pid in enumerate(sorted_pids):
        times, masses = zip(*particle_traces[pid])
        label = pid if idx < max_legend else None  # Only label first max_legend particles
        plt.plot(times, masses, label=label)

    plt.xlabel("Time")
    plt.ylabel("Mass")

    title = "Particle Mass Over Time"
    plt.title(title)

    if total_particles > max_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                   title=f"First {max_legend} particle ids")
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()

    # Save the plot to a file
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath)

    if display:
        plt.show()


def plot_snapshots_grid(
    results,
    *,
    field_names=None,
    n_snapshots=4,
    bounds=None,
    cmap='viridis',
    out_dir=None,
    filename='snapshots.png',
    suptitle=None,
    mass_scaling=10.0,
    row_label_pad=0.3,
):
    """Plot spatial fields and particles (if present) consistently in world coordinates."""
    if bounds is None:
        raise ValueError("bounds=(xmax, ymax) required.")
    xmax, ymax = bounds
    extent = [0, xmax, 0, ymax]

    # --- Normalize results structure ---
    if isinstance(results, dict):
        if ('emitter',) in results:
            data = results[('emitter',)]
        elif 'emitter' in results:
            data = results['emitter']
        else:
            vals = [v for v in results.values() if isinstance(v, list)]
            data = vals[0] if vals else results
    else:
        data = results
    if not isinstance(data, list) or not data:
        raise ValueError("results must be a non-empty list of simulation steps")

    # --- Extract fields & times ---
    times = [step['global_time'] for step in data]
    field_keys = list((data[0].get('fields') or {}).keys())
    fields = {f: [step.get('fields', {}).get(f) for step in data] for f in field_keys}

    # --- Choose fields ---
    if field_names is None:
        field_names = field_keys
    else:
        field_names = [f for f in field_names if f in field_keys]

    # >>> Early exit: no fields to plot
    if not field_names:
        return None

    n_rows = len(field_names)

    n_times = len(times)
    n_snapshots = min(n_snapshots, n_times)
    col_indices = np.linspace(0, n_times - 1, n_snapshots, dtype=int)
    col_times = [times[i] for i in col_indices]
    n_cols = len(col_indices)

    # --- Compute vmin/vmax per field ---
    vminmax = {}
    for f in field_names:
        arrs = [np.asarray(x) for x in fields[f] if x is not None]
        if not arrs:
            vminmax[f] = (0.0, 1.0)
            continue
        flat = np.concatenate([np.ravel(a) for a in arrs])
        vminmax[f] = (float(np.min(flat)), float(np.max(flat)))

    # --- Setup figure ---
    fig_w = max(3.5 * n_cols, 6)
    fig_h = max(3.0 * n_rows, 3)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.02, hspace=0.05
    )

    # --- Titles across top ---
    for j, t in enumerate(col_times):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(f"t = {t:.2f}", pad=6)
        ax.remove()

    # --- Plot each field row ---
    for r, field in enumerate(field_names):
        vmin, vmax = vminmax[field]
        images = []
        for c, ti in enumerate(col_indices):
            ax = fig.add_subplot(gs[r, c])

            arr = field_for_imshow(fields[field][ti])
            im = ax.imshow(
                arr, cmap=cmap, vmin=vmin, vmax=vmax,
                extent=extent, origin='lower', interpolation='nearest', aspect='equal'
            )
            images.append(im)

            # Overlay particles if present
            step = data[ti]
            if 'particles' in step and step['particles']:
                for p in step['particles'].values():
                    x, y = p['position']
                    if 0 <= x <= xmax and 0 <= y <= ymax:
                        size = max(p.get('mass', 0.01), 0.01) * mass_scaling
                        ax.scatter(x, y, s=size, color=p.get('color', 'b'))

            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, xmax])
            ax.set_yticks([0, ymax])
            ax.tick_params(axis='both', which='both', length=2, labelsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
            if r > 0:
                ax.set_title('')

        # Colorbar for this row
        cax = fig.add_subplot(gs[r, -1])
        cb = fig.colorbar(images[-1], cax=cax)
        cb.ax.tick_params(length=2, labelsize=8)

        # Row label
        first_ax = fig.axes[r * (n_cols + 1)]
        pos = first_ax.get_position(fig)
        x = pos.x0 - (row_label_pad / fig.get_size_inches()[0])
        y = (pos.y0 + pos.y1) / 2
        fig.text(x, y, field, ha='right', va='center', fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.98)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
    else:
        path = filename

    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path



# --------- phylogeny coloring ---------
def _mother_id(agent_id: str):
    return agent_id[:-2] if len(agent_id) >= 2 and agent_id[-2] == '_' and agent_id[-1] in ('0', '1') else None

def _mutate_hsv(h, s, v, rng, dh=0.05, ds=0.03, dv=0.03):
    h = (h + rng.uniform(-dh, dh)) % 1.0
    s = min(1.0, max(0.0, s + rng.uniform(-ds, ds)))
    v = min(1.0, max(0.0, v + rng.uniform(-dv, dv)))
    return h, s, v

def build_phylogeny_colors(frames, agents_key='agents', seed=None, base_s=0.70, base_v=0.95, dh=0.05, ds=0.03, dv=0.03):
    rng, hsv_map = random.Random(seed), {}
    for step in frames:
        for aid in (step.get(agents_key) or {}).keys():
            if aid in hsv_map: continue
            mom = _mother_id(aid)
            if mom in hsv_map:
                hsv_map[aid] = _mutate_hsv(*hsv_map[mom], rng, dh, ds, dv)
            else:
                hsv_map[aid] = (rng.random(), base_s, base_v)
    return {aid: tuple(hsv_to_rgb(hsv)) for aid, hsv in hsv_map.items()}


# --------- combined fields + agents GIF ---------

def fields_and_agents_to_gif(
    data,
    config,
    *,
    agents_key='agents',
    fields_key='fields',
    filename='simulation_with_fields.gif',
    out_dir='out',
    skip_frames=1,
    title='',
    figure_size_inches=(6, 6),
    dpi=90,
    show_time_title=False,
    # coloring:
    color_by_phylogeny=False,
    color_seed=None,
    base_s=0.70, base_v=0.95,
    mutate_dh=0.05, mutate_ds=0.03, mutate_dv=0.03,
    default_rgb=(0.2, 0.6, 0.9),
    uniform_color=(0.2, 0.6, 0.9),  # set None to disable uniforming
):
    """
    Merge of:
      - plot_species_distributions_with_particles_to_gif (fields done right)
      - pymunk_simulation_to_gif       (particles/agents config & coloring)

    Assumes `data` is an iterable of frames, where each `frame` is a dict like:
        {
            'time': <float> (optional),
            'fields': {
                'species_1': 2D np.array,
                'species_2': 2D np.array,
                ...
            },
            'agents': {
                agent_id_1: {'position': (x, y), 'radius': r, ...},
                agent_id_2: {...},
                ...
            }
        }

    `config` must contain:
        config['bounds'] = (xmax, ymax)
    """

    # Make list of frames & downsample
    if isinstance(data, (list, tuple)):
        frames = data[::max(1, int(skip_frames))]
    else:
        frames = list(data)[::max(1, int(skip_frames))]

    if not frames:
        raise ValueError("No frames to render.")

    xmax, ymax = config['bounds']
    extent = [0, xmax, 0, ymax]

    # --- Collect species & global min/max over all frames (for consistent colorbars) ---
    first_fields = None
    for step in frames:
        if fields_key in step and step[fields_key]:
            first_fields = step[fields_key]
            break

    if first_fields is None:
        species_names = []
    else:
        species_names = list(first_fields.keys())

    global_min_max = {}
    for species in species_names:
        vals = []
        for step in frames:
            fields = step.get(fields_key, {})
            if species in fields:
                arr = np.asarray(fields[species])
                vals.append(arr.ravel())
        if vals:
            cat = np.concatenate(vals)
            global_min_max[species] = (np.nanmin(cat), np.nanmax(cat))
        else:
            global_min_max[species] = (0.0, 1.0)

    # --- Color policy (mirrors pymunk_simulation_to_gif idea) ---
    if color_by_phylogeny:
        # Assumes you have a helper similar to your existing one.
        rgb_colors = build_phylogeny_colors(
            frames, agents_key=agents_key, seed=color_seed,
            base_s=base_s, base_v=base_v,
            dh=mutate_dh, ds=mutate_ds, dv=mutate_dv,
        )

        def _color(agent_id):
            return rgb_colors.get(agent_id, default_rgb)
    else:
        def _color(agent_id):
            return uniform_color if uniform_color is not None else default_rgb

    # --- Helper to draw agents/particles on an axis ---
    def draw_agents_on_axes(ax, agents):
        """
        agents can be:
            - dict: {agent_id: agent_dict}
            - list: [agent_dict, ...]
        agent_dict is expected to have:
            - 'position': (x, y)
            - 'radius': r
            - optional 'id'
        """
        if isinstance(agents, dict):
            iterable = agents.items()
        else:
            iterable = enumerate(agents)

        for key, agent in iterable:
            aid = agent.get('id', key)
            x, y = agent.get('position', (None, None))
            if x is None or y is None:
                continue
            r = agent.get('radius', 1.0)

            rgb = _color(aid)
            circle = plt.Circle((x, y), r, color=rgb, alpha=0.9)
            ax.add_patch(circle)

    # --- Render each frame with fields + agents ---
    images = []

    for idx, step in enumerate(frames):
        fields = step.get(fields_key, {})
        agents = step.get(agents_key, {})

        n_species = len(species_names)

        # Create subplots: 1 axis if no species; otherwise 1 per species
        fig, axs = plt.subplots(
            1, max(1, n_species),
            figsize=figure_size_inches,
            dpi=dpi,
        )

        # Normalize axs to list
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.flatten()

        if n_species == 0:
            ax = axs[0]
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_aspect('equal')
            if show_time_title:
                t = step.get('time', idx)
                ax.set_title(f'Particles at t = {t:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            draw_agents_on_axes(ax, agents)

        else:
            for j, species in enumerate(species_names):
                ax = axs[j]
                ax.set_xlim(0, xmax)
                ax.set_ylim(0, ymax)
                ax.set_aspect('equal')

                if species in fields:
                    # Match orientation behavior from your original fields function
                    raw_field = np.asarray(fields[species])
                    field_img = field_for_imshow(raw_field)

                    vmin, vmax = global_min_max[species]
                    im = ax.imshow(
                        field_img,
                        interpolation='nearest',
                        cmap='viridis',
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        origin='lower',
                    )

                    # --- colorbar same height as subplot ---
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                if show_time_title:
                    t = step.get('time', idx)
                    ax.set_title(f'{species} at t = {t:.2f}')
                else:
                    ax.set_title(species)

                draw_agents_on_axes(ax, agents)

        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=0.1)

        # Save to buffer and append to GIF frames
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    # --- Save GIF ---
    os.makedirs(out_dir, exist_ok=True)
    if not filename.lower().endswith('.gif'):
        filename = filename + '.gif'
    filepath = os.path.join(out_dir, filename)

    imageio.mimsave(filepath, images, duration=0.5, loop=0)

    # Inline display for Jupyter
    with open(filepath, 'rb') as f:
        data_bytes = f.read()
    data_url = 'data:image/gif;base64,' + base64.b64encode(data_bytes).decode()
    display(HTML(f'<img src="{data_url}" alt="{title}" style="max-width:100%;"/>'))

    return filepath

