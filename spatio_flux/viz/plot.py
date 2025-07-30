import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
import base64

from IPython.core.display import HTML
from IPython.core.display_functions import display
from IPython.display import display, HTML
from imageio import v2 as imageio
from matplotlib import pyplot as plt


def sort_results(results):
    if ('emitter',) in results:
        results = results[('emitter',)]
    if results[0] is None:
        return
    sorted_results = {'fields': {
        key: [] for key in results[0]['fields'].keys()
    }, 'time': []}

    for results in results:
        time = results['global_time']
        fields = results['fields']
        sorted_results['time'].append(time)
        for key, value in fields.items():
            sorted_results['fields'][key].append(value)
    return sorted_results


def plot_time_series(
        results,
        field_names=None,
        coordinates=None,
        out_dir=None,
        filename='time_series.png',
        display=False,
):
    """
    Plots time series for specified fields and coordinates from the results dictionary.

    :param results: Dictionary containing the results with keys 'time' and 'fields'.
                    Example: {'time': [0.1, 0.2, 0.3 ...], 'fields': {'glucose': [nparray_time0, nparray_time1], ...}}
    :param field_names: List of field names to plot.
                        Example: ['glucose', 'other chemical']
    :param coordinates: List of coordinates (indices) to plot.
                        Example: [(0, 0), (1, 2)]
    """
    # coordinates = coordinates or [(0, 0)]
    field_names = field_names or ['glucose', 'acetate', 'biomass']
    sorted_results = sort_results(results)
    times = sorted_results['time']

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for field_name in field_names:
        if field_name in sorted_results['fields']:
            field_data = sorted_results['fields'][field_name]
            if coordinates is None:
                ax.plot(times, field_data, label=field_name)
            else:
                for coord in coordinates:
                    x, y = coord
                    time_series = [field_data[t][x, y] for t in range(len(times))]
                    ax.plot(times, time_series, label=f'{field_name} at {coord}')
                    # plot log scale on y axis
                    # ax.set_yscale('log')
        else:
            print(f"Field '{field_name}' not found in results['fields']")

    # Adding plot labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Plot')
    ax.legend()

    # Save the plot to a file
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
        # print(f'saving {filepath}')
        plt.savefig(filepath)
    else:
        filepath = filename
    if display:
        # Display the plot
        plt.show()


def plot_snapshots(
        results,
        field_names,
        times
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
                im = ax.imshow(field_data, cmap='viridis', aspect='auto',
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
        skip_frames=1
):
    # Sort the results as before
    sorted_results = sort_results(results)
    species_names = [key for key in sorted_results['fields'].keys()]
    n_species = len(species_names)
    times = sorted_results['time']
    n_times = len(times)

    # Compute global min and max for each species
    global_min_max = {
        species: (np.min(np.concatenate([sorted_results['fields'][species][i].flatten() for i in range(n_times)])),
                  np.max(np.concatenate([sorted_results['fields'][species][i].flatten() for i in range(n_times)])))
        for species in species_names}

    images = []
    for i in range(0, n_times, skip_frames):
        fig, axs = plt.subplots(1, n_species, figsize=(5 * n_species, 4))
        if n_species == 1:
            axs = [axs]

        for j, species in enumerate(species_names):
            ax = axs[j]
            vmin, vmax = global_min_max[species]
            img = ax.imshow(sorted_results['fields'][species][i], interpolation='nearest', vmin=vmin, vmax=vmax)
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

    # Create and save the GIF with loop=0 for infinite loop
    # print(f'saving {filepath}')
    imageio.mimsave(filepath, images, duration=0.5, loop=0)

    # Optionally display the GIF in a Jupyter notebook
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
        fields = results[i]['fields']
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
                field = np.fliplr(np.rot90(fields[species], k=3))
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


def plot_particles_mass(results, out_dir=None, filename='particles_mass_plot.png',
                        display=False,
                        max_legend=10):
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
        label = pid if idx < max_legend else None
        plt.plot(times, masses, label=label)

    plt.xlabel("Time")
    plt.ylabel("Mass")

    title = "Particle Mass Over Time"
    if total_particles > max_legend:
        title += f" (showing {max_legend} of {total_particles})"
    plt.title(title)

    if total_particles > max_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title=f"First {max_legend} particles")
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
