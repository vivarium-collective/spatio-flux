"""Per-scenario analysis-flush specifications.

FLUSH_SPEC[slug] is the ordered list of ``{step, config}`` the post-run flush
runs to reproduce that scenario's figures. Configs are copied verbatim from the
old ``test_suite.py`` ``plot_func`` call sites. Values that depend on the run's
state (bounds, lattice size, pymunk config, model grid, dynamic field lists) are
``$``-placeholders the runner resolves from the composite state:

  $all_fields          -> list(state['fields'].keys())
  $community_species    -> biomass species ids parsed from state (community dFBA)
  $colors              -> STANDARD_FIELD_COLORS
  $bounds              -> resolved (xmax, ymax)
  $n_bins              -> resolved (nx, ny)
  $coordinates_corners  -> [(0, 0), (nx-1, ny-1)]
  $coordinates_comets   -> [(0, nx//2), (nx//2, nx//2), (nx-1, nx//2)]
  $pymunk_config        -> state['newtonian_particles']['config']
  $model_grid          -> state['spatial_dFBA']['config']['model_grid']
"""

# Copied from test_suite.py:64-76 (decoupled so test_suite.py can be retired).
STANDARD_FIELD_COLORS = {
    "glucose": "#1f77b4",
    "acetate": "#ff7f0e",
    "formate": "#9467bd",
    "ammonium": "#bcbd22",
    "biomass": "#2ca02c",
    "dfba_biomass": "#2ca02c",
    "ecoli core biomass": "#2ca02c",
    "ecoli_core_biomass": "#2ca02c",
    "kinetic_biomass": "#1b9e77",
    "monod_biomass": "#98df8a",
    "dissolved biomass": "#17becf",
}

_CONC_UNITS = {"glucose": "mM", "acetate": "mM", "formate": "mM",
               "ammonium": "mM", "biomass": "gDW", "kinetic_biomass": "gDW"}
_LEGEND = {"fontsize": 8, "loc": "best"}
_SUBMASS_COLORS = {"ecoli_1": "#1f77b4", "ecoli_2": "#d62728"}


def _scalar_ts(filename, title, field_names="$all_fields", figsize=(4.5, 3.5),
               field_units=None, **extra):
    cfg = {"filename": filename, "field_names": field_names, "title": title,
           "figsize": list(figsize), "time_units": "min",
           "field_units": field_units or {"glucose": "mM", "acetate": "mM",
                                          "biomass": "gDW", "kinetic_biomass": "gDW"},
           "field_colors": "$colors", "legend_kwargs": _LEGEND}
    cfg.update(extra)
    return {"step": "timeseries", "config": cfg}


def _particle_bundle(prefix, particles_row="separate"):
    """The plot_particles_sim bundle: gif + mass + snapshots + traces."""
    return [
        {"step": "species_dist_with_particles_gif",
         "config": {"filename": f"{prefix}_video", "bounds": "$bounds"}},
        {"step": "particles_mass", "config": {"filename": f"{prefix}_mass"}},
        {"step": "snapshots_grid",
         "config": {"filename": f"{prefix}_snapshots", "field_names": ["glucose", "acetate"],
                    "n_snapshots": 4, "bounds": "$bounds", "particles_row": particles_row,
                    "time_units": "min", "wspace": 0.1, "hspace": 0.1,
                    "col_width": 1.8, "row_height": 2.0}},
        {"step": "particle_traces",
         "config": {"filename": f"{prefix}_particles_traces", "bounds": "$bounds",
                    "radius_scaling": 0.1, "min_brightness": 0.1, "legend": False, "units": "µm"}},
    ]


def _nt_comets_bundle(prefix, n_snapshots, particles_row):
    """plot_newtonian_particle_comets: ts + mass + submasses + fields/agents gif + snapshots."""
    return [
        {"step": "timeseries",
         "config": {"filename": f"{prefix}_timeseries",
                    "field_names": ["glucose", "acetate", "dissolved biomass"],
                    "coordinates": "$coordinates_corners"}},
        {"step": "particles_mass", "config": {"filename": f"{prefix}_mass"}},
        {"step": "particles_mass_submasses", "config": {"filename": f"{prefix}_mass_submasses"}},
        {"step": "fields_agents_gif",
         "config": {"filename": f"{prefix}_video", "pymunk_config": "$pymunk_config",
                    "show_agent_submasses": True, "submass_color_map": _SUBMASS_COLORS,
                    "figure_size_inches": [10, 6]}},
        {"step": "snapshots_grid",
         "config": {"filename": f"{prefix}_snapshots",
                    "field_names": ["glucose", "acetate", "dissolved biomass"],
                    "n_snapshots": n_snapshots, "bounds": "$bounds",
                    "particles_row": particles_row, "time_units": "min",
                    "show_particle_submasses": True, "submass_draw_legend": True,
                    "submass_color_map": _SUBMASS_COLORS}},
    ]


FLUSH_SPEC = {
    # ---- metabolism scalar timeseries ----
    "monod_kinetics": [_scalar_ts("monod_kinetics", "Monod kinetics", figsize=(5, 4),
                                  field_units={"glucose": "mM", "acetate": "mM", "biomass": "gDW"})],
    "ecoli_core_dfba": [_scalar_ts("ecoli_core_dfba", "dFBA")],
    "ecoli_dfba": [_scalar_ts("ecoli_dfba", "dFBA")],
    "yeast_dfba": [_scalar_ts("yeast_dfba", "dFBA")],
    "community_dfba": [_scalar_ts("community_dfba", "hybrid community",
                                  field_names="$community_species",
                                  log_scale=True, normalize=True, field_units={})],
    "dfba_kinetics_community": [_scalar_ts(
        "dfba_kinetics_community", None,
        field_names=["glucose", "acetate", "dfba biomass", "kinetic biomass"],
        field_units={"glucose": "mM", "acetate": "mM", "dfba_biomass": "gDW"})],
    # ---- spatial ----
    "spatial_many_dfba": [
        {"step": "timeseries", "config": {"filename": "spatial_many_dfba_timeseries",
                                          "coordinates": [[0, 0], [1, 1], [1, 3]]}},
        {"step": "species_dist_gif", "config": {"filename": "spatial_many_dfba_video"}},
    ],
    "spatial_dfba_process": [
        {"step": "timeseries", "config": {"filename": "spatial_dfba_process_timeseries",
                                          "coordinates": [[0, 0], [1, 1], [2, 2]]}},
        {"step": "model_grid", "config": {"filename": "spatial_dfba_process_model_grid",
                                          "model_grid": "$model_grid", "title": "model grid",
                                          "show_border_coords": True}},
        {"step": "species_dist_gif", "config": {
            "filename": "spatial_dfba_process_video",
            "species_to_show": ["glucose", "acetate", "ammonium", "formate",
                                "glutamate", "dissolved biomass"]}},
    ],
    "diffusion_process": [
        {"step": "species_dist_gif", "config": {"filename": "diffusion_process_video"}}],
    # ---- brownian particles ----
    "brownian_particles": _particle_bundle("brownian_particles"),
    "br_particles_kinetics": _particle_bundle("br_particles_kinetics"),
    "br_particles_dfba": _particle_bundle("br_particles_dfba"),
    # ---- COMETS ----
    "comets_diffusion": [
        {"step": "timeseries", "config": {"filename": "comets_diffusion_timeseries",
                                          "coordinates": "$coordinates_comets"}},
        {"step": "snapshots_grid", "config": {
            "filename": "comets_diffusion_snapshots",
            "field_names": ["glucose", "acetate", "dissolved biomass"], "n_snapshots": 5,
            "bounds": "$bounds", "time_units": "min", "wspace": 0.1, "hspace": 0.1,
            "col_width": 1.8, "row_height": 2.0}},
        {"step": "species_dist_gif", "config": {"filename": "comets_diffusion_video"}},
    ],
    "comets_br_particles_kinetics": [
        {"step": "timeseries", "config": {"filename": "comets_br_particles_kinetics_timeseries",
                                          "coordinates": "$coordinates_corners"}},
        {"step": "particles_mass", "config": {"filename": "comets_br_particles_kinetics_mass"}},
        {"step": "snapshots_grid", "config": {
            "filename": "comets_br_particles_kinetics_snapshots",
            "field_names": ["glucose", "acetate"], "n_snapshots": 5, "bounds": "$bounds",
            "particles_row": "separate"}},
        {"step": "species_dist_with_particles_gif", "config": {
            "filename": "comets_br_particles_kinetics_video", "bounds": "$bounds"}},
    ],
    "comets_br_particles_dfba": [
        {"step": "timeseries", "config": {"filename": "comets_br_particles_dfba_timeseries",
                                          "field_names": ["glucose", "acetate", "dissolved biomass"],
                                          "coordinates": "$coordinates_corners"}},
        {"step": "particles_mass", "config": {"filename": "comets_br_particles_dfba_mass"}},
        {"step": "snapshots_grid", "config": {
            "filename": "comets_br_particles_dfba_snapshots",
            "field_names": ["glucose", "acetate"], "n_snapshots": 4, "bounds": "$bounds",
            "particles_row": "separate"}},
        {"step": "species_dist_with_particles_gif", "config": {
            "filename": "comets_br_particles_dfba_video", "bounds": "$bounds"}},
    ],
    # ---- pymunk ----
    "newtonian_particles": [
        {"step": "particles_mass", "config": {"filename": "newtonian_particles_mass"}},
        {"step": "pymunk_gif", "config": {"filename": "newtonian_particles_video",
                                          "pymunk_config": "$pymunk_config"}},
        {"step": "particle_traces", "config": {"filename": "newtonian_particles_particles_traces",
                                               "bounds": "$bounds", "radius_scaling": 0.1,
                                               "min_brightness": 0.1}},
    ],
    "comets_nt_particles_dfba": _nt_comets_bundle("comets_nt_particles_dfba", 5, "overlay"),
    # ---- reference demos ----
    "spatioflux_reference_demo": _nt_comets_bundle("spatioflux_reference_demo", 8, "separate"),
    "reference_demo_x2y2": _nt_comets_bundle("reference_demo_x2y2", 8, "separate"),
}


def _final_name(step, config):
    name = config["filename"]
    if step in ("species_dist_gif", "species_dist_with_particles_gif",
                "pymunk_gif", "fields_agents_gif"):
        return name if name.endswith(".gif") else f"{name}.gif"
    return name if name.endswith(".png") else f"{name}.png"


def expected_files(slug):
    """Figure files the flush writes + the structure artifacts the runner writes."""
    files = [_final_name(s["step"], s["config"]) for s in FLUSH_SPEC[slug]]
    files += [f"{slug}_viz.png", f"{slug}_schema.json", f"{slug}_state.json"]
    return files
