"""Figures for 2D orientation maps (EBSD maps).

Each function takes plain arrays and returns a ``matplotlib.figure.Figure``
without showing it, so the caller decides where it goes. Axes are in
micrometres when ``step_um`` is given; the y axis points down, as in an image.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

__all__ = [
    "euler_map_figure", "ipf_map_figure", "gb_pixel_map_figure",
    "gb_disorientation_figure",
]

_EULER_ANGLE_NAMES = ("phi1", "Phi", "phi2")
_EULER_LABELS = ("phi1 (deg)", "Phi (deg)", "phi2 (deg)")


def _extent(shape, step_um):
    ny, nx = shape
    s = float(step_um)
    return (-0.5 * s, (nx - 0.5) * s, (ny - 0.5) * s, -0.5 * s)


def _axis_labels(ax, step_um):
    unit = "um" if step_um != 1.0 else "pixels"
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")


def _boundary_overlay(ax, lfi, step_um, color):
    from upxo.viz.ebsdviz import _gb_mask
    mask = _gb_mask(lfi)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = [*color, 1.0]
    ax.imshow(rgba, interpolation="nearest", extent=_extent(lfi.shape, step_um))


def _polygon_fill_overlay(ax, polygons_by_gid, color_by_gid, step_um=1.0):
    """Render each grain as a filled polygon (holes handled correctly via
    a compound matplotlib Path, same technique descartes/geopandas use)
    instead of a per-pixel raster -- the "Smoothed" display counterpart
    to imshow(). color_by_gid supplies an RGB(A) tuple per gid; a gid
    with no polygon or no color is simply skipped (left unfilled)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    s = float(step_um)
    for gid, poly in polygons_by_gid.items():
        color = color_by_gid.get(gid)
        if color is None:
            continue
        parts = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
        for part in parts:
            verts, codes = [], []
            for ring in (part.exterior, *part.interiors):
                pts = np.asarray(ring.coords) * s
                if len(pts) < 2:
                    continue
                verts.extend(pts.tolist())
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY])
            if verts:
                ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                                       edgecolor='none', linewidth=0))
    bounds = [poly.bounds for poly in polygons_by_gid.values()]
    if bounds:
        xmin = min(b[0] for b in bounds) * s
        xmax = max(b[2] for b in bounds) * s
        ymin = min(b[1] for b in bounds) * s
        ymax = max(b[3] for b in bounds) * s
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)   # y points down, matches _extent's convention
        ax.set_aspect('equal')


def _grain_mean_scalar(grains, values):
    """{gid: mean(values[grains == gid])} via scipy.ndimage.mean -- one
    representative value per grain for the Smoothed display, which fills
    a whole polygon with a single color and so cannot show intra-grain
    pixel variation the way the raster display can."""
    from scipy import ndimage
    gids = [int(g) for g in np.unique(grains) if g > 0]
    if not gids:
        return {}
    means = np.atleast_1d(ndimage.mean(values, labels=grains, index=gids))
    return dict(zip(gids, means))


def _grain_first_pixel_rgb(grains, rgb_raster):
    """{gid: rgb_raster[one of that grain's pixels]} -- exact (not an
    approximation) when rgb_raster is already constant per grain, as
    build_ipf_rgb's output is (built from a per-grain-averaged
    quaternion), so sampling any single pixel of a grain gives its one
    true color."""
    gids = [int(g) for g in np.unique(grains) if g > 0]
    out = {}
    for gid in gids:
        idx = np.argwhere(grains == gid)
        if idx.size:
            r, c = idx[0]
            out[gid] = tuple(rgb_raster[r, c])
    return out


def euler_map_figure(lfi, euler_rad, angle_index, cmap="hsv", step_um=1.0,
                     show_boundaries=True, boundary_color=(0, 0, 0),
                     figsize=(6.5, 6.5), polygons=None, grains=None,
                     pixel_stride=1):
    """One Bunge Euler angle (phi1, Phi or phi2) as a single pixel map --
    or, if `polygons` is given, a Smoothed display: each grain filled
    with its own mean angle (see module docstring's Pixellated/Smoothed
    note -- a filled polygon has one color, so intra-grain pixel
    variation isn't shown in Smoothed mode the way it is in Pixellated).

    angle_index : 0 (phi1), 1 (Phi), or 2 (phi2).
    polygons, grains : output of ebsd_smoothing.smooth_technique_a/b.
    When given, pixel_stride is ignored (Smoothed mode doesn't rasterize
    at all, so there is nothing to subsample).
    pixel_stride : int >= 1 -- Pixellated mode only. Show every Nth pixel
    along each axis (nearest-neighbour subsampling, same physical extent)
    for a faster, lighter plot of a large map.
    """
    if angle_index not in (0, 1, 2):
        raise ValueError("angle_index must be 0 (phi1), 1 (Phi) or 2 (phi2).")
    euler_deg = np.degrees(euler_rad)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    label = _EULER_LABELS[angle_index]

    if polygons is not None:
        values = _grain_mean_scalar(grains, euler_deg[:, :, angle_index])
        vmin, vmax = (min(values.values()), max(values.values())) if values else (0.0, 1.0)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        colors = {gid: mappable.to_rgba(v) for gid, v in values.items()}
        _polygon_fill_overlay(ax, polygons, colors, step_um=step_um)
        fig.colorbar(mappable, ax=ax, label="degrees", fraction=0.046, pad=0.04)
    else:
        stride = max(1, int(pixel_stride))
        lfi_s = lfi[::stride, ::stride]
        data = euler_deg[::stride, ::stride, angle_index].copy()
        data[lfi_s <= 0] = np.nan
        extent = _extent(lfi_s.shape, step_um * stride)
        image = ax.imshow(data, cmap=cmap, interpolation="nearest", extent=extent)
        fig.colorbar(image, ax=ax, label="degrees", fraction=0.046, pad=0.04)
        if show_boundaries:
            _boundary_overlay(ax, lfi_s, step_um * stride, boundary_color)

    ax.set_title(f"Bunge {label}")
    _axis_labels(ax, step_um)
    fig.tight_layout()
    return fig


def ipf_map_figure(lfi, quat, sample_direction, direction_name, step_um=1.0,
                   show_boundaries=False, boundary_color=(0, 0, 0), figsize=(11, 5.5),
                   polygons=None, smooth_grains=None, pixel_stride=1):
    """Pixel IPF map for one sample direction, with the IPF colour key --
    or, if `polygons` is given, a Smoothed display (see euler_map_figure's
    docstring for the same Pixellated/Smoothed convention). IPF colour is
    already one value per grain either way (built_ipf_rgb is driven by a
    per-grain-averaged quaternion), so Smoothed mode loses nothing extra
    here -- only the boundary shape changes, not the colouring itself.
    """
    from upxo.viz.ebsdviz import build_ipf_rgb
    from upxo.xtalphy.crystal_orientation import grain_avg_quats
    from upxo.pxtal.twinned_simple_3d.viz_3d import plot_ipf_triangle_key

    gids, q_mean = grain_avg_quats(lfi, quat)
    rgb = build_ipf_rgb(lfi.astype(np.int64), {int(g): q for g, q in zip(gids, q_mean)},
                        sample_direction=sample_direction)
    fig = Figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[70, 30])
    ax = fig.add_subplot(grid[0, 0])

    if polygons is not None:
        colors = _grain_first_pixel_rgb(smooth_grains, rgb)
        _polygon_fill_overlay(ax, polygons, colors, step_um=step_um)
    else:
        stride = max(1, int(pixel_stride))
        lfi_s = lfi[::stride, ::stride]
        rgb_s = rgb[::stride, ::stride]
        extent = _extent(lfi_s.shape, step_um * stride)
        ax.imshow(rgb_s, interpolation="nearest", extent=extent)
        if show_boundaries:
            _boundary_overlay(ax, lfi_s, step_um * stride, boundary_color)

    ax.set_title(f"IPF map, sample direction {direction_name}")
    _axis_labels(ax, step_um)
    key_ax = fig.add_subplot(grid[0, 1])
    plot_ipf_triangle_key(ax=key_ax)
    key_ax.set_title("IPF key")
    fig.tight_layout()
    return fig


def gb_pixel_map_figure(lfi, cmap="nipy_spectral", step_um=1.0, boundary_color=(0, 0, 0),
                        figsize=(8, 6.5), polygons=None, smooth_grains=None, pixel_stride=1):
    """Grain-boundary pixels over the grain map -- or, if `polygons` is
    given, each grain filled (smoothed boundary) and coloured by its own
    label the same way the raster imshow(..., cmap=...) already colours
    by raw grain ID (see euler_map_figure's docstring for the same
    Pixellated/Smoothed convention)."""
    from upxo.viz.ebsdviz import _gb_mask

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if polygons is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        gids = sorted(polygons)
        mappable = ScalarMappable(norm=Normalize(vmin=min(gids), vmax=max(gids)), cmap=cmap) \
            if gids else None
        colors = {gid: mappable.to_rgba(gid) for gid in gids} if mappable else {}
        _polygon_fill_overlay(ax, polygons, colors, step_um=step_um)
        n_gb = len(gids)
        ax.set_title(f"Grain-boundary pixels (smoothed, {n_gb} grains)")
    else:
        stride = max(1, int(pixel_stride))
        lfi_s = lfi[::stride, ::stride]
        grains_f = lfi_s.astype(float)
        grains_f[grains_f <= 0] = np.nan
        extent = _extent(lfi_s.shape, step_um * stride)
        ax.imshow(grains_f, cmap=cmap, interpolation="nearest", extent=extent)
        _boundary_overlay(ax, lfi_s, step_um * stride, boundary_color)
        n_gb = int(_gb_mask(lfi_s).sum())
        ax.set_title(f"Grain-boundary pixels ({n_gb} px, {100 * n_gb / lfi_s.size:.1f}% of the map)")

    _axis_labels(ax, step_um)
    fig.tight_layout()
    return fig


COLORBAR_LOCATIONS = ("right", "left", "top", "bottom")
_KIND_STYLE = {   # same line width for every kind; only the dash pattern differs
    "twin_twin": ("--", "twin - twin"),
    "grain_grain": (":", "grain - grain"),
    "twin_grain": ("-", "twin - grain"),
}


def gb_disorientation_figure(result, cmap="viridis", step_um=1.0, vmin=None, vmax=None,
                             linewidth=1.4, show_edge_lines=True, only_in_range=False,
                             twin_ids=None, style_by_type=False,
                             colorbar_location="right", colorbar_label=None,
                             colorbar_label_size=10, colorbar_tick_size=9,
                             colorbar_length=1.0, colorbar_pad=0.04, figsize=(8, 6.5)):
    """Smoothed grain-boundary lines coloured by disorientation angle.

    ``result`` is a :class:`upxo.xtalphy.gb_disorientation_map.GBDisorientation`.
    Grains are not coloured. Lines on the map edge, which separate a single
    grain from nothing, are drawn thin and grey (or left out).

    only_in_range : draw only the lines whose angle lies in ``[vmin, vmax]``;
        otherwise angles outside the range take the colormap's end colours.
    style_by_type : with ``twin_ids`` (labels of the input map that are twins),
        draw twin-twin boundaries dashed, grain-grain dotted and twin-grain
        continuous, all at the same ``linewidth``, with a legend.
    colorbar_location : ``right``, ``left``, ``top`` or ``bottom``. The label
        and ticks follow the side; ``colorbar_length`` (0-1) and
        ``colorbar_pad`` set its extent and gap.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from upxo.xtalphy.gb_disorientation_map import (
        classify_boundaries, GRAIN_GRAIN, TWIN_GRAIN, TWIN_TWIN)
    if colorbar_location not in COLORBAR_LOCATIONS:
        raise ValueError(f"colorbar_location must be one of {COLORBAR_LOCATIONS}")

    # constrained_layout (not tight_layout) reserves room for the title,
    # axis labels/ticks, the colour bar and the legend together, whichever
    # side each lands on -- tight_layout() does not know about a colour bar
    # placed via ``location=`` and let it overlap the axis labels/title on
    # that side.
    fig = Figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    s = float(step_um)
    shared = ~np.isnan(result.angles)
    if show_edge_lines:
        edge = [seg * s for seg, keep in zip(result.segments, shared) if not keep]
        if edge:
            ax.add_collection(LineCollection(edge, colors="0.7", linewidths=0.6))

    angles = result.angles
    low = float(np.nanmin(angles)) if vmin is None and shared.any() else vmin
    high = float(np.nanmax(angles)) if vmax is None and shared.any() else vmax
    draw = shared.copy()
    if only_in_range and shared.any():
        draw &= (angles >= low) & (angles <= high)

    kinds = None
    if style_by_type and twin_ids is not None:
        kinds = classify_boundaries(result, twin_ids)

    mappable = None
    if draw.any():
        norm = Normalize(vmin=low, vmax=high)
        if kinds is None:
            groups = [("all", "-", draw)]
        else:
            groups = [("twin_twin", _KIND_STYLE["twin_twin"][0], draw & (kinds == TWIN_TWIN)),
                      ("grain_grain", _KIND_STYLE["grain_grain"][0], draw & (kinds == GRAIN_GRAIN)),
                      ("twin_grain", _KIND_STYLE["twin_grain"][0], draw & (kinds == TWIN_GRAIN))]
        for _name, style, mask in groups:
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            collection = LineCollection([result.segments[i] * s for i in idx], cmap=cmap,
                                        norm=norm, linewidths=linewidth, linestyles=style)
            collection.set_array(angles[idx])
            ax.add_collection(collection)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        horizontal = colorbar_location in ("top", "bottom")
        bar = fig.colorbar(mappable, ax=ax, location=colorbar_location,
                           orientation="horizontal" if horizontal else "vertical",
                           fraction=0.046, pad=colorbar_pad,
                           shrink=float(colorbar_length), aspect=30 if horizontal else 20)
        bar.set_label(colorbar_label or "Disorientation angle (deg)", fontsize=colorbar_label_size)
        bar.ax.tick_params(labelsize=colorbar_tick_size)
        if kinds is not None:
            handles = [Line2D([], [], color="0.2", linestyle=style, linewidth=linewidth, label=label)
                       for style, label in (_KIND_STYLE["twin_twin"], _KIND_STYLE["grain_grain"],
                                            _KIND_STYLE["twin_grain"])]
            # A figure-level legend below everything, not an axes-anchored
            # one: "outside lower center" is constrained_layout's own
            # mechanism for reserving a band for it, so it stacks cleanly
            # under a bottom colour bar instead of guessing a fixed offset
            # that can drift off the figure, and it never competes with the
            # (figure-level) suptitle the way an "above axes" placement would.
            fig.legend(handles=handles, loc="outside lower center",
                      ncol=3, frameon=False, fontsize=colorbar_tick_size)

    ny, nx = result.shape
    ax.set_xlim(-0.5 * s, (nx - 0.5) * s)
    ax.set_ylim((ny - 0.5) * s, -0.5 * s)
    ax.set_aspect("equal")
    shown = int(draw.sum())
    # A figure-level suptitle, not an axes title: keeps the title out of the
    # axes' own top margin, which is where a top colour bar or an "above"
    # legend may also need room.
    fig.suptitle(f"Grain-boundary disorientation ({result.n_grains} grains, "
                f"{shown} of {int(shared.sum())} boundary segments shown)")
    _axis_labels(ax, step_um)
    return fig
