import os, math, random
import numpy as np
from math import pi
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb

# ----------------- LineWidthData -----------------
class LineWidthData(Line2D):
    """A Line2D whose linewidth is specified in data units (world units)."""
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1.0)
        super().__init__(*args, **kwargs)
        self._lw_data = float(_lw_data)
    def _get_lw(self):
        if self.axes is None:
            return 1.0
        (_, y0), (_, y1) = self.axes.transData.transform([(0, 0), (0, 1)])
        px_per_data = abs(y1 - y0)
        ppd = 72.0 / self.axes.figure.dpi  # points per pixel
        return max(0.1, px_per_data * self._lw_data * ppd)
    def _set_lw(self, lw): self._lw_data = float(lw)
    _linewidth = property(_get_lw, _set_lw)

# --------- tiny utils ---------
def _ensure_gif_filename(path):
    root, ext = os.path.splitext(path)
    return f"{path}.gif" if ext == "" else path

def _finite(*vals): return all(math.isfinite(v) for v in vals)
def _norm_angle(theta): return (theta + pi) % (2*pi) - pi

def _pixels_per_data_y(ax):
    (_, y0), (_, y1) = ax.transData.transform([(0, 0), (0, 1)])
    return abs(y1 - y0)

def _bbox_outside(px_bbox, img_bbox, pad_px):
    ix0, iy0, ix1, iy1 = img_bbox
    x0, y0, x1, y1 = px_bbox
    return (x1 < ix0 - pad_px or x0 > ix1 + pad_px or
            y1 < iy0 - pad_px or y0 > iy1 + pad_px)

def _infer_plot_type(o):
    t = o.get('type')
    if t in ('circle', 'segment'): return t
    if 'radius' in o and 'length' in o: return 'segment'
    if 'radius' in o: return 'circle'
    return None

def _is_plot_entity(o):
    if not isinstance(o, dict): return False
    loc = o.get('location')
    return isinstance(loc, (tuple, list)) and len(loc) == 2 and _infer_plot_type(o) is not None

def _is_entity_map(v): return isinstance(v, dict) and all(_is_plot_entity(x) for x in v.values())

def merge_plot_layers(data, merged_key='agents'):
    out = []
    for step in data:
        step_out = dict(step)
        base = dict(step_out.get(merged_key, {}))
        existing = set(base.keys())
        for k, v in step.items():
            if k == merged_key or not _is_entity_map(v): continue
            for ent_id, ent in v.items():
                out_id = ent_id if ent_id not in existing else f"{k}:{ent_id}"
                if ent.get('type') not in ('circle', 'segment'):
                    t = _infer_plot_type(ent)
                    if t is not None: ent = {**ent, 'type': t}
                base[out_id] = ent
                existing.add(out_id)
            step_out.pop(k, None)
        step_out[merged_key] = base
        out.append(step_out)
    return out

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

# ================= CORE RENDERER =================
class GifRenderer:
    def __init__(
        self, bounds, barriers, figure_size_inches, dpi, show_time_title,
        world_pad, max_line_px
    ):
        x_max, y_max = bounds
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.show_time_title = show_time_title
        self.max_line_px = int(max_line_px)

        # fig/ax/canvas
        self.fig = plt.figure(figsize=figure_size_inches, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, self.x_max)
        self.ax.set_ylim(0, self.y_max)
        self.ax.set_axis_off()
        self.ax.set_autoscale_on(False)

        # establish pixel geometry
        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        self.img_bbox = (0, 0, w - 1, h - 1)
        self.ypu = _pixels_per_data_y(self.ax)
        self.pad_px = int(round(world_pad * self.ypu))
        self.title_obj = self.ax.set_title("") if show_time_title else None

        # draw barriers once
        self._draw_barriers(barriers)

        # pools (grow-on-demand)
        self.circle_pool = []
        self.segment_pool = []

        # background for blitting
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def close(self): plt.close(self.fig)

    def _draw_barriers(self, barriers, color='gray'):
        dpi_fig = self.fig.dpi
        for b in barriers or []:
            (sx, sy), (ex, ey) = b['start'], b['end']
            if not _finite(sx, sy, ex, ey): continue
            t_world = float(b.get('thickness', 1.0))
            lw_px = max(1, min(int(round(t_world * self.ypu)), self.max_line_px))
            lw_pt = lw_px * 72.0 / dpi_fig
            self.ax.add_line(Line2D([sx, ex], [sy, ey], linewidth=lw_pt, color=color, antialiased=False))

    # ------- pools -------
    def _need_circle(self, idx):
        while len(self.circle_pool) <= idx:
            c = Circle((0, 0), 1.0, fill=True, antialiased=False, visible=False)
            self.ax.add_patch(c)
            self.circle_pool.append(c)
        return self.circle_pool[idx]

    def _need_segment(self, idx):
        while len(self.segment_pool) <= idx:
            ln = LineWidthData([0, 0], [0, 0], linewidth=1.0, solid_capstyle='round', antialiased=False, visible=False)
            self.ax.add_line(ln)
            self.segment_pool.append(ln)
        return self.segment_pool[idx]

    # ------- per-frame update -------
    def draw_frame(self, step, agents_key, color_fn, max_radius_px):
        c_vis = s_vis = 0
        layer = step.get(agents_key) or {}

        # circles
        for aid, o in layer.items():
            if o.get('type') != 'circle': continue
            cx, cy = o['location']; r = float(o['radius'])
            if not _finite(cx, cy, r) or r <= 0: continue
            px0, py0 = self.ax.transData.transform((cx - r, cy - r))
            px1, py1 = self.ax.transData.transform((cx + r, cy + r))
            if _bbox_outside((int(px0), int(py0), int(px1), int(py1)), self.img_bbox, self.pad_px): continue
            art = self._need_circle(c_vis)
            art.center = (cx, cy); art.set_radius(r)
            rgb = color_fn(aid); art.set_facecolor(rgb); art.set_edgecolor(rgb)
            if not art.get_visible(): art.set_visible(True)
            c_vis += 1

        # hide extra circles
        for i in range(c_vis, len(self.circle_pool)):
            if self.circle_pool[i].get_visible(): self.circle_pool[i].set_visible(False)

        # segments (shortened by exactly 2*radius)
        for aid, o in layer.items():
            if o.get('type') != 'segment': continue
            L = float(o['length'])
            r = float(o['radius'])
            ang = _norm_angle(float(o['angle']))
            cx, cy = o['location']
            if not _finite(cx, cy, L, r, ang) or L <= 0 or r <= 0: continue

            half = 0.5 * L
            length_offset = max(half - r, 0.0)  # trims 1*r off each end
            dx = math.cos(ang) * length_offset; dy = math.sin(ang) * length_offset
            x0, y0 = cx - dx, cy - dy; x1, y1 = cx + dx, cy + dy

            lw_nom_px = int(round(2.0 * r * self.ypu))
            lw_px = max(1, min(lw_nom_px, self.max_line_px))
            lw_data = lw_px / max(self.ypu, 1e-9)

            p0 = self.ax.transData.transform((x0, y0))
            p1 = self.ax.transData.transform((x1, y1))
            pad = lw_px * 0.5
            xmin, xmax = int(min(p0[0], p1[0]) - pad), int(max(p0[0], p1[0]) + pad)
            ymin, ymax = int(min(p0[1], p1[1]) - pad), int(max(p0[1], p1[1]) + pad)
            if _bbox_outside((xmin, ymin, xmax, ymax), self.img_bbox, self.pad_px): continue

            art = self._need_segment(s_vis)
            art.set_xdata([x0, x1]); art.set_ydata([y0, y1])
            art.set_linewidth(lw_data); art.set_color(color_fn(aid))
            if not art.get_visible(): art.set_visible(True)
            s_vis += 1

        # hide extra segments
        for i in range(s_vis, len(self.segment_pool)):
            if self.segment_pool[i].get_visible(): self.segment_pool[i].set_visible(False)

        # title
        if self.show_time_title and self.title_obj is not None:
            self.title_obj.set_text(f"t={step.get('time', 0):.1f}")

        # blit
        self.canvas.restore_region(self.background)
        if self.show_time_title and self.title_obj is not None: self.ax.draw_artist(self.title_obj)
        for art in self.circle_pool:
            if art.get_visible(): self.ax.draw_artist(art)
        for art in self.segment_pool:
            if art.get_visible(): self.ax.draw_artist(art)
        self.canvas.blit(self.ax.bbox)

        # read back
        w, h = self.fig.canvas.get_width_height()
        rgba = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        return Image.fromarray(rgba[:, :, :3].copy())

# ================= PUBLIC API =================
def simulation_to_gif(
    data,
    config,
    agents_key='agents',
    filename='simulation.gif',
    out_dir='out',
    skip_frames=1,
    figure_size_inches=(6, 6),
    dpi=90,
    show_time_title=False,
    # culling/clamping controls:
    world_pad=50.0,       # extra world-units beyond env_size to still draw
    max_line_px=40,       # max segment diameter (2*radius) in pixels
    max_radius_px=40,     # kept for API-compat; used in culling path if needed
    # coloring:
    color_by_phylogeny=False,   # << default to uniform color
    color_seed=None,
    base_s=0.70, base_v=0.95,
    mutate_dh=0.05, mutate_ds=0.03, mutate_dv=0.03,
    default_rgb=(0.2, 0.6, 0.9),
    uniform_color=(0.2, 0.6, 0.9),  # <set None to disable uniforming
):
    """
    Efficient Matplotlib renderer:
      - merges plot-capable layers
      - blitting + grow-on-demand artist pools (no prescan)
      - segments shortened by 2*radius
      - uniform adjustable color by default; optional phylogeny coloring
    """
    # merge layers & downsample frames
    if isinstance(data, (list, tuple)) and data:
        data = merge_plot_layers(data, merged_key=agents_key)
    frames = list(data)[::max(1, int(skip_frames))]
    if not frames: raise ValueError("No frames to render.")

    # paths
    os.makedirs(out_dir, exist_ok=True)
    filename = _ensure_gif_filename(filename)
    out_path = filename if os.path.dirname(filename) else os.path.join(out_dir, filename)
    out_path = os.path.abspath(os.path.expanduser(out_path))

    bounds = config['bounds']
    barriers = config.get('barriers', [])

    # color policy
    if color_by_phylogeny:
        rgb_colors = build_phylogeny_colors(
            frames, agents_key=agents_key, seed=color_seed,
            base_s=base_s, base_v=base_v, dh=mutate_dh, ds=mutate_ds, dv=mutate_dv
        )
        def _color(aid):
            # ignore uniform_color when phylogeny is on
            return rgb_colors.get(aid, default_rgb)
    else:
        # no phylogeny: use uniform if provided, otherwise fallback default
        def _color(aid):
            return uniform_color if uniform_color is not None else default_rgb

    # render
    renderer = GifRenderer(bounds, barriers, figure_size_inches, dpi, show_time_title, world_pad, max_line_px)
    try:
        pil_frames = [renderer.draw_frame(step, agents_key, _color, max_radius_px) for step in frames]
    finally:
        renderer.close()

    # save
    if not pil_frames: raise ValueError("No frames to save.")
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=100, loop=0, optimize=False, disposal=2
    )
    print(f"GIF saved to {out_path}")
    return out_path
