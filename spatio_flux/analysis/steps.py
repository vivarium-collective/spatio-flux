"""Analysis-flush steps: thin file-writing wrappers over ``spatio_flux.plots.plot``.

Faithful reproduction is guaranteed by reusing the exact plotting code the old
test-suite ``plot_func``s used — each wrapper just maps a config dict to the
primitive's kwargs and returns the path(s) written. Registered into the flush
harness by name (see FLUSH_SPEC in scripts/scaffold_studies.py).

Signature of every step: ``(trajectory, state, out_dir, config) -> list[str]``.

Structure artifacts (``<slug>_viz.png`` / ``_schema.json`` / ``_state.json``) are
NOT produced here — the study runner's ``run_composite_document`` writes them.
"""
import os

from spatio_flux.analysis.flush import register_step
from spatio_flux.plots import plot as P
from spatio_flux.plots.multibody_plots import pymunk_simulation_to_gif


def _png(name):
    return name if name.endswith(".png") else f"{name}.png"


def _gif(name):
    return name if name.endswith(".gif") else f"{name}.gif"


@register_step("timeseries")
def timeseries_step(trajectory, state, out_dir, config):
    # community_dfba / dfba_kinetics_community write the filename verbatim (no .png).
    fname = config["filename"] if config.get("verbatim_filename") else _png(config["filename"])
    P.plot_time_series(
        trajectory,
        field_names=config.get("field_names"),
        coordinates=config.get("coordinates"),
        out_dir=out_dir, filename=fname,
        log_scale=config.get("log_scale", False),
        normalize=config.get("normalize", False),
        figsize=config.get("figsize", (12, 6)),
        title=config.get("title"),
        time_units=config.get("time_units", "min"),
        field_units=config.get("field_units"),
        field_colors=config.get("field_colors"),
        legend_kwargs=config.get("legend_kwargs"),
    )
    return [os.path.join(out_dir, fname)]


@register_step("species_dist_gif")
def species_dist_gif_step(trajectory, state, out_dir, config):
    fname = _gif(config["filename"])
    P.plot_species_distributions_to_gif(
        trajectory, out_dir=out_dir, filename=fname,
        species_to_show=config.get("species_to_show"))
    return [os.path.join(out_dir, fname)]


@register_step("species_dist_with_particles_gif")
def species_dist_with_particles_gif_step(trajectory, state, out_dir, config):
    fname = _gif(config["filename"])
    P.plot_species_distributions_with_particles_to_gif(
        trajectory, out_dir=out_dir, filename=fname,
        bounds=tuple(config.get("bounds", (1.0, 1.0))))
    return [os.path.join(out_dir, fname)]


@register_step("snapshots_grid")
def snapshots_grid_step(trajectory, state, out_dir, config):
    fname = _png(config["filename"])
    kwargs = {k: config[k] for k in (
        "field_names", "n_snapshots", "bounds", "particles_row", "time_units",
        "wspace", "hspace", "col_width", "row_height", "cbar_width",
        "show_particle_submasses", "submass_draw_legend", "submass_color_map",
    ) if k in config}
    P.plot_snapshots_grid(trajectory, out_dir=out_dir, filename=fname, **kwargs)
    return [os.path.join(out_dir, fname)]


@register_step("particles_mass")
def particles_mass_step(trajectory, state, out_dir, config):
    fname = _png(config["filename"])
    P.plot_particles_mass(trajectory, out_dir=out_dir, filename=fname)
    return [os.path.join(out_dir, fname)]


@register_step("particles_mass_submasses")
def particles_mass_submasses_step(trajectory, state, out_dir, config):
    fname = _png(config["filename"])
    P.plot_particles_mass_with_submasses(trajectory, out_dir=out_dir, filename=fname)
    return [os.path.join(out_dir, fname)]


@register_step("particle_traces")
def particle_traces_step(trajectory, state, out_dir, config):
    fname = _png(config["filename"])
    history = [step["particles"] for step in trajectory if "particles" in step]
    P.plot_particle_traces(
        history=history, bounds=tuple(config["bounds"]), out_dir=out_dir, filename=fname,
        radius_scaling=config.get("radius_scaling", 1.0),
        min_brightness=config.get("min_brightness", 0.15),
        legend=config.get("legend", True),
        units=config.get("units"))
    return [os.path.join(out_dir, fname)]


@register_step("model_grid")
def model_grid_step(trajectory, state, out_dir, config):
    fname = _png(config["filename"])
    model_grid = config["model_grid"]
    P.plot_model_grid(model_grid, out_dir=out_dir, filename=fname,
                      title=config.get("title", "model grid"),
                      show_border_coords=config.get("show_border_coords", True))
    return [os.path.join(out_dir, fname)]


@register_step("pymunk_gif")
def pymunk_gif_step(trajectory, state, out_dir, config):
    fname = _gif(config["filename"])
    pymunk_simulation_to_gif(
        trajectory, config=config["pymunk_config"], agents_key="particles",
        filename=fname, out_dir=out_dir)
    return [os.path.join(out_dir, fname)]


@register_step("fields_agents_gif")
def fields_agents_gif_step(trajectory, state, out_dir, config):
    fname = _gif(config["filename"])
    P.fields_and_agents_to_gif(
        data=trajectory, config=config["pymunk_config"],
        agents_key="particles", fields_key="fields",
        filename=fname, out_dir=out_dir,
        figure_size_inches=config.get("figure_size_inches", (10, 6)),
        show_agent_submasses=config.get("show_agent_submasses", False),
        submass_color_map=config.get("submass_color_map"))
    return [os.path.join(out_dir, fname)]
