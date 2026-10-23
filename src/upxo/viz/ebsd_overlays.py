"""Grain-boundary and junction overlays for pixel-indexed grain maps.

The functions draw onto an existing matplotlib axes whose data coordinates are
pixel indices (the pixel at row i, column j is centred at x=j, y=i), as the
grain maps of ``upxo.viz.ebsdviz`` are. ``data_per_pixel`` rescales for an axes
in other units.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection

__all__ = ["junction_points", "polygon_boundary_lines", "overlay_grain_boundaries",
           "overlay_junctions", "JUNCTION_MARKERS"]

JUNCTION_MARKERS = {"circle": "o", "square": "s", "triangle": "^", "diamond": "D",
                    "plus": "P", "cross": "X", "star": "*"}


def junction_points(lfi):
    """Triple and quadruple junctions of a label map.

    A junction is an interior pixel corner (shared by four pixels) around which
    three, or four, different labels meet. Returns ``(triple_xy, quad_xy)``,
    each an ``(n, 2)`` array of (x, y) in pixel coordinates (the corner between
    pixels i, i+1 and j, j+1 is at x=j+0.5, y=i+0.5).
    """
    lfi = np.asarray(lfi)
    if lfi.ndim != 2 or min(lfi.shape) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    around = np.stack([lfi[:-1, :-1], lfi[:-1, 1:], lfi[1:, :-1], lfi[1:, 1:]])
    around = np.sort(around, axis=0)
    n_distinct = 1 + (np.diff(around, axis=0) != 0).sum(axis=0)
    out = []
    for mask in (n_distinct == 3, n_distinct >= 4):
        i, j = np.nonzero(mask)
        out.append(np.column_stack([j + 0.5, i + 0.5]))
    return out[0], out[1]


def polygon_boundary_lines(polygons):
    """Every ring (outer and holes) of a ``{gid: shapely polygon}`` mapping as
    an (n, 2) coordinate array."""
    lines = []
    for poly in polygons.values():
        parts = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
        for part in parts:
            for ring in (part.exterior, *part.interiors):
                xy = np.asarray(ring.coords, dtype=float)
                if len(xy) >= 2:
                    lines.append(xy)
    return lines


def overlay_grain_boundaries(ax, lfi, polygons=None, color='black', linewidth=1.0,
                             data_per_pixel=1.0, zorder=4):
    """Draw grain boundaries on ``ax``.

    With ``polygons`` (a smoothed structure) the smoothed boundary lines are
    drawn at ``linewidth``; without them every pixel edge between two different
    labels is drawn in ``color`` (a one-pixel-wide raster, so ``linewidth`` does
    not apply). Returns ``'smoothed'`` or ``'pixel'``.
    """
    dpp = float(data_per_pixel)
    if polygons:
        lines = [xy * dpp for xy in polygon_boundary_lines(polygons)]
        ax.add_collection(LineCollection(lines, colors=color, linewidths=linewidth, zorder=zorder))
        return 'smoothed'
    from matplotlib.colors import to_rgb
    from upxo.viz.ebsdviz import _gb_mask
    lfi = np.asarray(lfi)
    mask = _gb_mask(lfi)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = [*to_rgb(color), 1.0]
    ny, nx = lfi.shape
    extent = (-0.5 * dpp, (nx - 0.5) * dpp, (ny - 0.5) * dpp, -0.5 * dpp)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(rgba, interpolation='nearest', extent=extent, zorder=zorder)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return 'pixel'


def overlay_junctions(ax, lfi, triple=True, quadruple=True, marker='circle', size=12.0,
                      edgecolor='black', facecolor='black', linewidth=1.0,
                      data_per_pixel=1.0, zorder=5):
    """Mark the triple and/or quadruple junctions of ``lfi`` on ``ax``.

    ``marker`` is a key of :data:`JUNCTION_MARKERS` or a matplotlib marker
    code; ``size`` is the marker area in points squared. Returns the number of
    junctions drawn."""
    t_xy, q_xy = junction_points(lfi)
    pts = [xy for xy, on in ((t_xy, triple), (q_xy, quadruple)) if on and len(xy)]
    if not pts:
        return 0
    xy = np.vstack(pts) * float(data_per_pixel)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=JUNCTION_MARKERS.get(marker, marker),
               edgecolors=edgecolor, facecolors=facecolor, linewidths=linewidth, zorder=zorder)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return len(xy)
