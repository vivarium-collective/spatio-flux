from __future__ import annotations

import os
import io
import base64
import random
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.colors import hsv_to_rgb
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Wedge
import matplotlib.cm as cm


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

def plot_particles_mass_with_submasses(
    results,
    *,
    out_dir=None,
    filename='particles_mass_with_submasses.png',
    display=False,
    max_particle_legend=10,
    particles_key='particles',
    time_key='global_time',
    mass_key='mass',
    submasses_key='sub_masses',

    # NEW: total-mass coloring
    color_total_by_particle: bool = True,
    particle_cmap: str = "tab20",
    total_mass_color='black',   # used only if color_total_by_particle=False
    total_mass_lw=2.5,

    # submass styling
    submass_color_map=None,
    submass_cmap='tab20',
    submass_lw=1.2,
    submass_alpha=0.85,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as mpatches

    # --- aggregate ---
    particle_traces = {}
    particle_subtraces = {}
    times_seen = set()
    all_submass_labels = set()

    for entry in results:
        t = entry.get(time_key)
        if t is None:
            continue
        times_seen.add(t)
        particles = entry.get(particles_key, {}) or {}

        for pid, pdata in particles.items():
            particle_traces.setdefault(pid, {})
            particle_subtraces.setdefault(pid, {})

            if mass_key in pdata:
                particle_traces[pid][t] = float(pdata[mass_key])

            sm = pdata.get(submasses_key, {})
            if isinstance(sm, dict):
                for label, val in sm.items():
                    label = str(label)
                    all_submass_labels.add(label)
                    particle_subtraces[pid].setdefault(label, {})
                    particle_subtraces[pid][label][t] = float(val)

    if not particle_traces:
        raise ValueError("No particle mass data found.")

    times = sorted(times_seen)
    pids = sorted(particle_traces.keys())

    # --- submass colors (consistent by label) ---
    if submass_color_map is None:
        submass_color_map = {}
    else:
        submass_color_map = dict(submass_color_map)

    if all_submass_labels:
        sm_cmap = cm.get_cmap(submass_cmap, max(len(all_submass_labels), 1))
        for i, label in enumerate(sorted(all_submass_labels)):
            if label not in submass_color_map:
                submass_color_map[label] = sm_cmap(i)

    # --- particle colors (for total mass) ---
    if color_total_by_particle:
        p_cmap = cm.get_cmap(particle_cmap, max(len(pids), 1))
        particle_color_map = {pid: p_cmap(i) for i, pid in enumerate(pids)}
    else:
        particle_color_map = {pid: total_mass_color for pid in pids}

    # --- plot ---
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for idx, pid in enumerate(pids):
        total_series = [particle_traces[pid].get(t, np.nan) for t in times]
        label = pid if idx < max_particle_legend else None

        ax.plot(
            times,
            total_series,
            color=particle_color_map[pid],
            linewidth=total_mass_lw,
            alpha=0.9,
            label=label,
        )

        for label_sm, series_dict in particle_subtraces.get(pid, {}).items():
            sm_series = [series_dict.get(t, 0.0) for t in times]
            ax.plot(
                times,
                sm_series,
                color=submass_color_map.get(label_sm, "gray"),
                linewidth=submass_lw,
                alpha=submass_alpha,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Mass")
    ax.set_title("Particle Total Mass (by particle) and Submasses (by label)")

    # --- legends (robust two-legend pattern) ---
    particle_leg = None
    if pids and max_particle_legend > 0:
        particle_leg = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize='small',
            title=("Particles (total mass)" if len(pids) <= max_particle_legend
                   else f"Particles (first {max_particle_legend})"),
            frameon=False,
        )
        ax.add_artist(particle_leg)

    if all_submass_labels:
        submass_handles = [
            mpatches.Patch(color=submass_color_map[lbl], label=lbl)
            for lbl in sorted(all_submass_labels)
        ]
        ax.legend(
            handles=submass_handles,
            title="Submasses",
            bbox_to_anchor=(1.02, 0.0),
            loc='lower left',
            fontsize='small',
            frameon=False,
        )

    plt.tight_layout()

    # --- save/display ---
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches="tight", pad_inches=0.5)
    else:
        filepath = None

    if display:
        plt.show()
    else:
        plt.close()

    return filepath


def plot_snapshots_grid(
    results,
    *,
    field_names=None,
    n_snapshots=4,
    bounds=None,
    cmap="viridis",
    out_dir=None,
    filename="snapshots.png",
    suptitle=None,

    # particle rendering
    particles_key="particles",
    particles_row: str = "overlay",   # "overlay" | "separate" | "none"
    overlay_particles: bool = True,   # legacy-friendly switch; ignored if particles_row != "overlay"
    particle_radius_key: str = "radius",
    particle_mass_key: str = "mass",
    radius_fallback_from_mass: bool = True,
    mass_to_radius_scale: float = 1.0,   # if falling back: r = sqrt(mass) * scale (rough)
    particle_alpha: float = 0.9,
    particle_edgecolor=None,
    particle_linewidth: float = 0.0,

    # sub-mass pie rendering
    show_particle_submasses: bool = False,
    submasses_key: str = "sub_masses",
    submass_color_map=None,              # dict[label] -> color; if None, auto-assign from submass_cmap
    submass_cmap: str = "tab20",
    submass_min_fraction: float = 0.0,   # drop tiny slices (e.g. 0.01)
    submass_draw_legend: bool = False,
    submass_legend_fontsize: int = 8,

    # spacing / layout control
    figsize=None,
    col_width=2.4,
    row_height=2.4,
    cbar_width=0.06,
    wspace=0.02,
    hspace=0.06,
    left=0.06, right=0.98, top=0.95, bottom=0.10,

    # row labels
    row_label_pad=0.01,
    row_label_fontsize=11,

    # time labels
    show_time_labels=True,
    time_units=None,
    time_scale=1.0,
    time_label_fmt="t = {t:.0f}",
    time_label_fontsize=10,
    time_label_pad=0.02,
):
    """
    Plot spatial field snapshots with consistent world coordinates, and optionally particles.

    Particle modes:
      - particles_row="overlay": particles drawn on top of each field panel
      - particles_row="separate": add an extra last row with particles only
      - particles_row="none": do not draw particles

    If show_particle_submasses=True, particles with a dict at p[submasses_key]
    are drawn as pie charts whose wedges correspond to per-label masses.

    Legend placement:
      - If particles_row=="separate": legend is placed in the empty last-column cell
        of the particles row (aligned with that row).
      - If particles_row=="overlay": legend is placed OUTSIDE the grid to the right,
        so it cannot overlap field colorbars.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle, Wedge
    from matplotlib import cm
    from pathlib import Path

    if bounds is None:
        raise ValueError("bounds=(xmax, ymax) required.")
    xmax, ymax = bounds
    extent = [0, xmax, 0, ymax]

    # -------------------------
    # Normalize results structure
    # -------------------------
    if isinstance(results, dict):
        if ("emitter",) in results:
            data = results[("emitter",)]
        elif "emitter" in results:
            data = results["emitter"]
        else:
            vals = [v for v in results.values() if isinstance(v, list)]
            data = vals[0] if vals else results
    else:
        data = results

    if not isinstance(data, list) or not data:
        raise ValueError("results must be a non-empty list of simulation steps")

    # -------------------------
    # Extract times + fields
    # -------------------------
    times_raw = [step.get("global_time", np.nan) for step in data]
    n_times = len(times_raw)

    field_keys = list((data[0].get("fields") or {}).keys())
    fields = {f: [step.get("fields", {}).get(f) for step in data] for f in field_keys}

    if field_names is None:
        field_names = field_keys
    else:
        field_names = [f for f in field_names if f in field_keys]

    if not field_names and particles_row != "separate":
        return None

    # -------------------------
    # Snapshot indices / labels
    # -------------------------
    n_snapshots = min(int(n_snapshots), n_times)
    col_indices = np.linspace(0, n_times - 1, n_snapshots, dtype=int)
    col_times = [(times_raw[i] * time_scale) for i in col_indices]
    n_cols = len(col_indices)

    # -------------------------
    # vmin/vmax per field
    # -------------------------
    vminmax = {}
    for f in field_names:
        arrs = [np.asarray(x) for x in fields[f] if x is not None]
        if not arrs:
            vminmax[f] = (0.0, 1.0)
            continue
        flat = np.concatenate([np.ravel(a) for a in arrs])
        vminmax[f] = (float(np.nanmin(flat)), float(np.nanmax(flat)))

    # -------------------------
    # Layout: rows
    # -------------------------
    field_rows = len(field_names)
    add_particles_row = (particles_row == "separate")
    n_rows = field_rows + (1 if add_particles_row else 0)

    if figsize is None:
        fig_w = max(col_width * n_cols, 4.5)
        fig_h = max(row_height * n_rows, 3.5)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols + 1,  # + colorbar column (only used for field rows)
        figure=fig,
        width_ratios=[1.0] * n_cols + [cbar_width],
        wspace=wspace,
        hspace=hspace,
    )
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    # -------------------------
    # Helpers
    # -------------------------
    def field_for_imshow(a):
        return np.asarray(a)

    def iter_particles(step):
        particles = step.get(particles_key) or {}
        if isinstance(particles, dict):
            return particles.values()
        if isinstance(particles, list):
            return particles
        return []

    def particle_radius(p):
        r = p.get(particle_radius_key, None)
        if r is not None:
            try:
                return float(r)
            except Exception:
                pass

        if radius_fallback_from_mass:
            m = p.get(particle_mass_key, None)
            if m is None:
                return None
            try:
                m = max(float(m), 0.0)
            except Exception:
                return None
            return (m ** 0.5) * float(mass_to_radius_scale)
        return None

    # -------------------------
    # Build consistent submass label -> color mapping
    # -------------------------
    def collect_submass_labels():
        if not show_particle_submasses:
            return []
        labels = set()
        for ti in col_indices:
            step = data[ti]
            for p in iter_particles(step):
                sm = p.get(submasses_key, None)
                if isinstance(sm, dict):
                    labels.update(str(k) for k in sm.keys())
        return sorted(labels)

    submass_labels = collect_submass_labels()

    if submass_color_map is None:
        submass_color_map = {}
    else:
        submass_color_map = dict(submass_color_map)

    if show_particle_submasses and submass_labels:
        cmap_obj = cm.get_cmap(submass_cmap, max(len(submass_labels), 1))
        for i, lab in enumerate(submass_labels):
            if lab not in submass_color_map:
                submass_color_map[lab] = cmap_obj(i)

    legend_handles = []
    if show_particle_submasses and submass_draw_legend and submass_labels:
        import matplotlib.patches as mpatches
        for lab in submass_labels:
            legend_handles.append(mpatches.Patch(color=submass_color_map[lab], label=lab))

    def draw_particle_pie(ax, x, y, r, sub_masses: dict) -> bool:
        items = []
        total = 0.0
        for k, v in sub_masses.items():
            try:
                val = float(v)
            except Exception:
                continue
            if val <= 0:
                continue
            items.append((str(k), val))
            total += val

        if total <= 0 or not items:
            return False

        if submass_min_fraction > 0:
            items = [(k, v) for (k, v) in items if (v / total) >= submass_min_fraction]
            total = sum(v for _, v in items)
            if total <= 0:
                return False

        items.sort(key=lambda kv: kv[0])

        start_angle = 0.0
        for label, val in items:
            frac = val / total
            sweep = 360.0 * frac
            color = submass_color_map.get(label, "gray")
            ax.add_patch(
                Wedge(
                    (x, y),
                    r,
                    start_angle,
                    start_angle + sweep,
                    facecolor=color,
                    edgecolor=particle_edgecolor,
                    linewidth=particle_linewidth,
                    alpha=particle_alpha,
                )
            )
            start_angle += sweep

        if particle_edgecolor is not None and particle_linewidth > 0:
            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    fill=False,
                    edgecolor=particle_edgecolor,
                    linewidth=particle_linewidth,
                    alpha=particle_alpha,
                )
            )
        return True

    def draw_particles(ax, step):
        for p in iter_particles(step):
            pos = p.get("position", None)
            if not pos or len(pos) != 2:
                continue
            x, y = pos
            if x is None or y is None:
                continue
            if not (0 <= x <= xmax and 0 <= y <= ymax):
                continue

            r = particle_radius(p)
            if r is None or r <= 0:
                continue

            if show_particle_submasses:
                sm = p.get(submasses_key, None)
                if isinstance(sm, dict) and sm:
                    if draw_particle_pie(ax, x, y, r, sm):
                        continue

            color = p.get("color", "b")
            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    facecolor=color,
                    edgecolor=particle_edgecolor,
                    linewidth=particle_linewidth,
                    alpha=particle_alpha,
                )
            )

    def style_axes(ax):
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def add_time_label(ax, t):
        label = time_label_fmt.format(t=t)
        if time_units:
            label = f"{label} {time_units}"
        ax.text(
            0.5,
            -time_label_pad,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=time_label_fontsize,
        )

    def add_row_label(ax, text):
        pos = ax.get_position(fig)
        fig.text(
            pos.x0 - row_label_pad,
            (pos.y0 + pos.y1) / 2,
            text,
            ha="right",
            va="center",
            fontsize=row_label_fontsize,
        )

    # -------------------------
    # Draw field rows
    # -------------------------
    for r, field in enumerate(field_names):
        vmin, vmax = vminmax[field]
        last_im = None
        first_ax_in_row = None

        for c, ti in enumerate(col_indices):
            ax = fig.add_subplot(gs[r, c])
            if first_ax_in_row is None:
                first_ax_in_row = ax

            arr_raw = fields[field][ti]
            arr = field_for_imshow(arr_raw) if arr_raw is not None else np.zeros((2, 2))

            last_im = ax.imshow(
                arr,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )

            if particles_row == "overlay" and overlay_particles:
                draw_particles(ax, data[ti])

            style_axes(ax)

            if show_time_labels and (r == n_rows - 1) and not add_particles_row:
                add_time_label(ax, col_times[c])

        if last_im is not None:
            cax = fig.add_subplot(gs[r, -1])
            cb = fig.colorbar(last_im, cax=cax)
            cb.ax.tick_params(length=2, labelsize=8)

        if first_ax_in_row is not None:
            add_row_label(first_ax_in_row, field)

    # -------------------------
    # Optional particles-only row
    # -------------------------
    legend_ax = None  # we'll store the rightmost cell for legend if separate
    if add_particles_row:
        pr = n_rows - 1
        first_ax = None
        for c, ti in enumerate(col_indices):
            ax = fig.add_subplot(gs[pr, c])
            if first_ax is None:
                first_ax = ax

            ax.axis("off")
            style_axes(ax)
            draw_particles(ax, data[ti])

            if show_time_labels:
                add_time_label(ax, col_times[c])

        legend_ax = fig.add_subplot(gs[pr, -1])
        legend_ax.axis("off")

        if first_ax is not None:
            add_row_label(first_ax, "particles")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # -------------------------
    # Legend placement rules
    # -------------------------
    if show_particle_submasses and submass_draw_legend and legend_handles:
        if add_particles_row and legend_ax is not None:
            # Aligned with particles row: use that empty cell
            legend_ax.legend(
                handles=legend_handles,
                loc="center left",
                frameon=False,
                fontsize=submass_legend_fontsize,
                borderaxespad=0.0,
                handlelength=1.2,
                labelspacing=0.3,
            )
        else:
            # Overlay mode: put legend to the RIGHT of the colorbar column (outside the grid)
            fig.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),   # outside; won't overlap field colorbars
                frameon=False,
                fontsize=submass_legend_fontsize,
                borderaxespad=0.0,
                handlelength=1.2,
                labelspacing=0.3,
            )

    # -------------------------
    # Save
    # -------------------------
    out_path = Path(out_dir) / filename if out_dir else Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)





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

    # coloring (agent base color)
    color_by_phylogeny=False,
    color_seed=None,
    base_s=0.70, base_v=0.95,
    mutate_dh=0.05, mutate_ds=0.03, mutate_dv=0.03,
    default_rgb=(0.2, 0.6, 0.9),
    uniform_color=(0.2, 0.6, 0.9),  # set None to disable uniforming

    # NEW: submass pies
    show_agent_submasses: bool = False,
    submasses_key: str = "sub_masses",
    submass_color_map=None,          # dict[label] -> matplotlib color
    submass_cmap: str = "tab20",     # used to auto-assign missing labels
    submass_min_fraction: float = 0.0,  # drop tiny wedges
    submass_alpha: float = 0.95,
    submass_edgecolor=None,
    submass_linewidth: float = 0.0,
    draw_submass_outline: bool = False,  # draw circle outline around pie
    submass_outline_color="k",
    submass_outline_lw: float = 0.4,

    # optional legend (rendered into each frame, so off by default)
    draw_submass_legend: bool = False,
    submass_legend_loc: str = "upper right",
    submass_legend_fontsize: int = 8,
):
    """
    Render fields + agents to an animated GIF, with optional sub-mass pie rendering.

    Frame format:
        {
            'time': <float> (optional),
            'fields': { name: 2D array, ... },
            'agents': { agent_id: {'position': (x,y), 'radius': r, 'sub_masses': {...}, ...}, ... }
        }

    config must contain:
        config['bounds'] = (xmax, ymax)

    Submass pies:
      - If show_agent_submasses=True and agent[submasses_key] is a dict,
        the agent is drawn as a pie chart (Wedges) within its radius.
      - Colors for submass labels are consistent across frames (within this GIF),
        or can be enforced by passing submass_color_map.
    """
    import os
    import io
    import base64
    import numpy as np
    import imageio
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from IPython.display import display, HTML

    # ---------- your existing helper expected ----------
    # field_for_imshow should exist in your module; keep behavior consistent.
    # If not, uncomment a simple default:
    # def field_for_imshow(a): return np.asarray(a)

    # Make list of frames & downsample
    if isinstance(data, (list, tuple)):
        frames = data[::max(1, int(skip_frames))]
    else:
        frames = list(data)[::max(1, int(skip_frames))]

    if not frames:
        raise ValueError("No frames to render.")

    xmax, ymax = config['bounds']
    extent = [0, xmax, 0, ymax]

    # --- Collect species & global min/max over all frames (consistent colorbars) ---
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
            global_min_max[species] = (float(np.nanmin(cat)), float(np.nanmax(cat)))
        else:
            global_min_max[species] = (0.0, 1.0)

    # --- Color policy (mirrors pymunk_simulation_to_gif idea) ---
    if color_by_phylogeny:
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

    # -------------------------
    # NEW: build consistent submass label -> color mapping across ALL frames used
    # -------------------------
    def iter_agents(frame_agents):
        if isinstance(frame_agents, dict):
            return frame_agents.items()
        return enumerate(frame_agents)

    def collect_submass_labels():
        if not show_agent_submasses:
            return []
        labels = set()
        for step in frames:
            agents = step.get(agents_key, {})
            for key, agent in iter_agents(agents):
                sm = agent.get(submasses_key, None)
                if isinstance(sm, dict):
                    labels.update(str(k) for k in sm.keys())
        return sorted(labels)

    submass_labels = collect_submass_labels()

    if submass_color_map is None:
        submass_color_map = {}
    else:
        submass_color_map = dict(submass_color_map)

    if show_agent_submasses and submass_labels:
        cmap_obj = cm.get_cmap(submass_cmap, max(len(submass_labels), 1))
        for i, lab in enumerate(submass_labels):
            if lab not in submass_color_map:
                submass_color_map[lab] = cmap_obj(i)

    legend_handles = []
    if show_agent_submasses and draw_submass_legend and submass_labels:
        import matplotlib.patches as mpatches
        for lab in submass_labels:
            legend_handles.append(mpatches.Patch(color=submass_color_map[lab], label=lab))

    def draw_agent_pie(ax, x, y, r, sub_masses: dict) -> bool:
        items = []
        total = 0.0
        for k, v in sub_masses.items():
            try:
                val = float(v)
            except Exception:
                continue
            if val <= 0:
                continue
            items.append((str(k), val))
            total += val

        if total <= 0 or not items:
            return False

        if submass_min_fraction > 0:
            items = [(k, v) for (k, v) in items if (v / total) >= submass_min_fraction]
            total = sum(v for _, v in items)
            if total <= 0:
                return False

        items.sort(key=lambda kv: kv[0])  # stable ordering for consistent wedge arrangement

        start = 0.0
        for label, val in items:
            frac = val / total
            sweep = 360.0 * frac
            color = submass_color_map.get(label, "gray")
            ax.add_patch(
                Wedge(
                    (x, y),
                    r,
                    start,
                    start + sweep,
                    facecolor=color,
                    edgecolor=submass_edgecolor,
                    linewidth=submass_linewidth,
                    alpha=submass_alpha,
                )
            )
            start += sweep

        if draw_submass_outline:
            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    fill=False,
                    edgecolor=submass_outline_color,
                    linewidth=submass_outline_lw,
                    alpha=min(1.0, submass_alpha + 0.05),
                )
            )

        return True

    # --- Helper to draw agents/particles on an axis ---
    def draw_agents_on_axes(ax, agents):
        """
        agents can be dict: {agent_id: agent_dict} or list: [agent_dict, ...]
        agent_dict expected:
            'position': (x, y)
            'radius': r
            optional 'id'
            optional submasses dict at submasses_key
        """
        for key, agent in iter_agents(agents):
            aid = agent.get('id', key)
            x, y = agent.get('position', (None, None))
            if x is None or y is None:
                continue
            r = float(agent.get('radius', 1.0))

            if show_agent_submasses:
                sm = agent.get(submasses_key, None)
                if isinstance(sm, dict) and sm:
                    if draw_agent_pie(ax, x, y, r, sm):
                        continue  # done

            # fallback: solid circle using base agent color
            rgb = _color(aid)
            circle = plt.Circle((x, y), r, color=rgb, alpha=0.9)
            ax.add_patch(circle)

    # --- Render each frame with fields + agents ---
    images = []

    for idx, step in enumerate(frames):
        fields = step.get(fields_key, {})
        agents = step.get(agents_key, {})

        n_species = len(species_names)

        fig, axs = plt.subplots(
            1, max(1, n_species),
            figsize=figure_size_inches,
            dpi=dpi,
        )

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.flatten()

        if n_species == 0:
            ax = axs[0]
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if show_time_title:
                t = step.get('time', idx)
                ax.set_title(f'Particles at t = {t:.2f}')

            draw_agents_on_axes(ax, agents)

            # legend (per-frame; can be costly/visually busy)
            if show_agent_submasses and draw_submass_legend and legend_handles:
                ax.legend(handles=legend_handles, loc=submass_legend_loc,
                          fontsize=submass_legend_fontsize, frameon=False)

        else:
            for j, species in enumerate(species_names):
                ax = axs[j]
                ax.set_xlim(0, xmax)
                ax.set_ylim(0, ymax)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                if species in fields:
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

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                if show_time_title:
                    t = step.get('time', idx)
                    ax.set_title(f'{species} at t = {t:.2f}')
                else:
                    ax.set_title(species)

                draw_agents_on_axes(ax, agents)

                # If you want ONE legend per frame, put it only on first axis:
                if show_agent_submasses and draw_submass_legend and legend_handles and j == 0:
                    ax.legend(handles=legend_handles, loc=submass_legend_loc,
                              fontsize=submass_legend_fontsize, frameon=False)

        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=0.1)

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
