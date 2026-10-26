"""Interactive Plotly visualizations for the discovery-engine studies.

Reads a run's zarr and writes a self-contained HTML (``include_plotlyjs=True``)
so it renders offline in the published read-only workbench (inlined as an iframe
``srcdoc``). Three figure kinds: an animated field-heatmap movie (time slider),
a particle-movie scatter animation, and a sweep phase portrait.
"""
from __future__ import annotations

import os

import numpy as np


def _write(fig, out_html: str) -> str:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return out_html


def field_heatmap_movie(ds, fields: list[str], out_html: str, title: str = "") -> str:
    """Animated heatmap(s) of 2D field(s) over time with a play/slider control."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fields = [f for f in fields if f in ds.data_vars and ds[f].ndim == 3]
    if not fields:
        return _write(go.Figure(), out_html)
    T = ds[fields[0]].shape[0]
    cols = len(fields)
    fig = make_subplots(rows=1, cols=cols, subplot_titles=fields)

    def frame_traces(t):
        traces = []
        for i, f in enumerate(fields):
            arr = np.asarray(ds[f].values[t], dtype=float)
            traces.append(go.Heatmap(z=arr, coloraxis="coloraxis", name=f,
                                     xaxis=f"x{i+1}" if i else "x", yaxis=f"y{i+1}" if i else "y"))
        return traces

    for tr in frame_traces(0):
        fig.add_trace(tr, row=1, col=fields.index(tr.name) + 1)
    frames = [go.Frame(name=str(t), data=frame_traces(t)) for t in range(T)]
    fig.frames = frames
    fig.update_layout(
        title=title, coloraxis={"colorscale": "Viridis"},
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "▶ play", "method": "animate",
             "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
            {"label": "⏸ pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}]}],
        sliders=[{"steps": [{"label": str(t), "method": "animate",
                             "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for t in range(T)]}])
    return _write(fig, out_html)


def particle_movie(ds, out_html: str, color_by: str = "mass", title: str = "", bounds=None) -> str:
    """Scatter animation of particle positions over time; size=mass, color=strain|mass."""
    import plotly.graph_objects as go

    if "particle_x" not in ds.data_vars:
        return _write(go.Figure(), out_html)
    x = np.asarray(ds["particle_x"].values, dtype=float)
    y = np.asarray(ds["particle_y"].values, dtype=float)
    mass = np.asarray(ds["particle_mass"].values, dtype=float)
    strain = ds["particle_strain"].values if "particle_strain" in ds.data_vars else None
    T = x.shape[0]

    def frame(t):
        m = np.isfinite(x[t]) & np.isfinite(y[t])
        size = 6 + 20 * np.nan_to_num(mass[t][m]) / (np.nanmax(mass) or 1.0)
        if color_by == "strain" and strain is not None:
            col = [str(s) for s in strain[t][m]]
            palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
            uniq = sorted({c for c in col if c})
            cmap = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
            marker = {"size": size, "color": [cmap.get(c, "#999999") for c in col], "line": {"width": 0.5}}
        else:
            marker = {"size": size, "color": np.nan_to_num(mass[t][m]), "colorscale": "Viridis",
                      "colorbar": {"title": "mass"}, "line": {"width": 0.5}}
        return go.Scatter(x=x[t][m], y=y[t][m], mode="markers", marker=marker)

    fig = go.Figure(data=[frame(0)],
                    frames=[go.Frame(name=str(t), data=[frame(t)]) for t in range(T)])
    _mx = float(np.nanmax(np.concatenate([x, y]))) if np.isfinite(np.concatenate([x, y])).any() else 1.0
    rng = bounds or [0, _mx if _mx > 0 else 1.0]
    fig.update_layout(
        title=title, xaxis={"range": rng, "title": "x"}, yaxis={"range": rng, "title": "y", "scaleanchor": "x"},
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "▶ play", "method": "animate",
             "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]}]}],
        sliders=[{"steps": [{"label": str(t), "method": "animate",
                             "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for t in range(T)]}])
    return _write(fig, out_html)


def phase_portrait(xs, ys, out_html: str, xlabel="", ylabel="", title="", annotate_cliff=False) -> str:
    """A metric-vs-axis line (phase diagram); optionally annotate the sharpest drop."""
    import plotly.graph_objects as go

    xs, ys = list(map(float, xs)), list(map(float, ys))
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line={"width": 3}))
    if annotate_cliff and len(ys) > 1:
        diffs = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
        i = int(np.argmax(diffs))
        fig.add_vrect(x0=xs[i], x1=xs[i + 1], fillcolor="red", opacity=0.12, line_width=0,
                      annotation_text="transition", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return _write(fig, out_html)
