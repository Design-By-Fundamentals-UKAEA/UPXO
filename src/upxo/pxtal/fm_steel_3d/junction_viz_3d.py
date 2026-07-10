"""
junction_viz_3d.py

PyVista visualization of junction lines (triple-line edges) and junction
points (vertices where >=3 domains meet) for FM steel 3-D microstructures
at any hierarchy level.

Triple-line edges and junction points are rendered as 3D cylinders and
spheres respectively, with full per-level customization of colours,
sizes, and visibility.

Public API
----------
view_junction_lines(fm, levels_config, ...)
    Highly customizable junction visualization with per-level control.

Example
-------
from upxo.pxtal.fm_steel_3d.junction_viz_3d import view_junction_lines

view_junction_lines(
    fm_sb,
    levels_config={
        'pag': {
            'show': True,
            'line_color': '#9B59B6',
            'line_radius': 0.3,
            'show_jps': True,
            'jp_color': '#9B59B6',
            'jp_radius': 0.5,
        },
        'block': {
            'show': True,
            'line_color': '#27AE60',
            'line_radius': 0.2,
            'show_jps': False,
        },
    },
    background_color='white',
    grain_background_cmap='viridis',
)
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np
import pyvista as pv
from typing import Dict, Optional, Tuple, Any

from upxo.pxtal.fm_steel_3d.geom_metrics_3d import (
    junction_line_segments,
    junction_point_coords,
)
from upxo.pxtal.fm_steel_3d.feature_props_3d import _LGI_BUILDERS


# Default per-level configuration
_DEFAULT_LEVELS_CONFIG: Dict[str, Dict[str, Any]] = {
    'grain': {
        'show': False,
        'line_color': '#95A5A6',
        'line_radius': 0.5,
        'show_jps': False,
        'jp_color': '#95A5A6',
        'jp_radius': 0.8,
    },
    'pag': {
        'show': True,
        'line_color': '#9B59B6',
        'line_radius': 0.35,
        'show_jps': True,
        'jp_color': '#9B59B6',
        'jp_radius': 0.6,
    },
    'packet': {
        'show': True,
        'line_color': '#E67E22',
        'line_radius': 0.28,
        'show_jps': False,
        'jp_color': '#E67E22',
        'jp_radius': 0.5,
    },
    'block': {
        'show': True,
        'line_color': '#27AE60',
        'line_radius': 0.20,
        'show_jps': False,
        'jp_color': '#27AE60',
        'jp_radius': 0.4,
    },
    'subblock': {
        'show': True,
        'line_color': '#E74C3C',
        'line_radius': 0.12,
        'show_jps': False,
        'jp_color': '#E74C3C',
        'jp_radius': 0.25,
    },
}


def _segments_to_polydata(starts: np.ndarray, ends: np.ndarray) -> pv.PolyData:
    """Convert parallel (n, 3) start/end arrays to a PyVista PolyData line mesh."""
    n = len(starts)
    if n == 0:
        return pv.PolyData()

    points = np.empty((2 * n, 3), dtype=np.float64)
    points[0::2] = starts
    points[1::2] = ends

    lines = np.empty(3 * n, dtype=np.intp)
    lines[0::3] = 2
    lines[1::3] = np.arange(0, 2 * n, 2)
    lines[2::3] = np.arange(1, 2 * n, 2)

    pd = pv.PolyData()
    pd.points = points
    pd.lines  = lines
    return pd


def _points_to_spheres(coords: np.ndarray, radius: float) -> pv.PolyData:
    """Convert (n, 3) coordinate array to a PolyData of sphere glyphs."""
    if len(coords) == 0:
        return pv.PolyData()
    pd = pv.PolyData(coords)
    return pd.glyph(scale=radius * 2.0, geom=pv.Sphere(radius=1.0, phi_resolution=8,
                                                        theta_resolution=8))


def _add_grain_background(pl: pv.Plotter, fm, opacity: float,
                           cmap: str = 'Greys') -> None:
    """Add a semi-transparent grain-structure voxel mesh to the plotter."""
    NX, NY, NZ = fm.lgi.shape
    dx = float(fm.voxel_size)
    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.spacing    = (dx, dx, dx)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data['GrainID'] = fm.lgi.flatten(order='F').astype(float)
    bg = grid.threshold(0.5, scalars='GrainID').extract_surface()
    pl.add_mesh(bg, cmap=cmap, opacity=opacity,
                show_scalar_bar=False, show_edges=False)


def view_junction_lines(
    fm,
    levels_config: Optional[Dict[str, Dict[str, Any]]] = None,
    show_grain_background: bool = True,
    grain_background_cmap: str = 'Greys',
    grain_background_opacity: float = 0.15,
    background_color: str = 'white',
    window_size: Tuple[int, int] = (1100, 900),
    title: Optional[str] = None,
) -> None:
    """
    Display junction lines and points for all FM steel hierarchy levels
    with full per-level customization.

    Triple-line edges are rendered as 3-D cylinders; junction points as
    spheres (optional per level).  All parameters are customizable per level
    via the `levels_config` dictionary.

    Parameters
    ----------
    fm : FMSteel3D* state object
        Any state along the pipeline (FMSteel3DBase through FMSteel3DWithSubBlocks).
        Must expose .lgi, .voxel_size, and the attributes for requested levels.
    levels_config : dict or None
        Per-level configuration.  Keys are level names ('grain', 'pag', 'packet',
        'block', 'subblock').  Values are dicts with keys:
            'show' (bool)              — whether to display this level
            'line_color' (str)         — hex colour for junction lines
            'line_radius' (float)      — tube radius for junction lines (in voxel_size units)
            'show_jps' (bool)          — whether to show junction points
            'jp_color' (str)           — hex colour for junction point spheres
            'jp_radius' (float)        — sphere radius for junction points
        If None or missing levels, uses default configuration from _DEFAULT_LEVELS_CONFIG.
    show_grain_background : bool
        If True, show the grain structure as a semi-transparent voxel mesh.
    grain_background_cmap : str
        Matplotlib colormap name for the grain background (e.g. 'Greys', 'viridis').
    grain_background_opacity : float
        Opacity of the grain background (0 invisible, 1 opaque).
    background_color : str
        Plotter background colour ('white', 'black', etc.).
    window_size : (int, int)
        PyVista plotter window size in pixels.
    title : str or None
        Window title; auto-generated if None.

    Notes
    -----
    * For large grids (>150³) the total segment/point count can exceed 10⁶.
      Rendering is efficient but may be slow on low-end GPUs.
    * Junction points (spheres) have noticeably higher GPU cost than lines
      (cylinders); disable them where not needed.
    * Grain background is semi-transparent so junction geometry remains visible.
    """
    # Merge user config with defaults
    cfg = {k: dict(_DEFAULT_LEVELS_CONFIG[k]) for k in _DEFAULT_LEVELS_CONFIG}
    if levels_config:
        for level, user_opts in levels_config.items():
            if level in cfg:
                cfg[level].update(user_opts)
            else:
                cfg[level] = user_opts

    dx = float(fm.voxel_size)
    auto_title = 'Junction lines and points — all levels'
    pl = pv.Plotter(
        window_size=list(window_size),
        title=title if title is not None else auto_title,
    )

    print('\nBuilding junction line and point meshes...')

    if show_grain_background:
        _add_grain_background(pl, fm, grain_background_opacity, grain_background_cmap)

    legend_entries: list = []
    n_total_segs = 0
    n_total_jps = 0

    for level in ['grain', 'pag', 'packet', 'block', 'subblock']:
        level_cfg = cfg.get(level, {})
        if not level_cfg.get('show', False):
            continue

        if level not in _LGI_BUILDERS:
            continue

        try:
            lgi, _, _ = _LGI_BUILDERS[level](fm)
        except (AttributeError, TypeError, KeyError):
            continue

        # --- Junction lines ---
        starts, ends = junction_line_segments(lgi, dx)
        n_segs = len(starts)

        if n_segs > 0:
            n_total_segs += n_segs
            line_color = level_cfg.get('line_color', '#FFFFFF')
            line_radius = level_cfg.get('line_radius', 0.2) * dx
            pd_lines = _segments_to_polydata(starts, ends)
            pd_tubes = pd_lines.tube(radius=line_radius, n_sides=6)
            pl.add_mesh(pd_tubes, color=line_color, smooth_shading=True,
                        show_scalar_bar=False)
            legend_entries.append([f'{level} lines ({n_segs:,})', line_color])
            print(f'  {level} lines: {n_segs:,} segments')

        # --- Junction points (optional) ---
        if level_cfg.get('show_jps', False):
            coords = junction_point_coords(lgi, dx)
            n_jps = len(coords)

            if n_jps > 0:
                n_total_jps += n_jps
                jp_color = level_cfg.get('jp_color', '#FFFFFF')
                jp_radius = level_cfg.get('jp_radius', 0.3) * dx
                pd_spheres = _points_to_spheres(coords, jp_radius)
                pl.add_mesh(pd_spheres, color=jp_color, smooth_shading=True,
                            show_scalar_bar=False)
                legend_entries.append([f'{level} points ({n_jps:,})', jp_color])
                print(f'  {level} points: {n_jps:,} vertices')

    if legend_entries:
        pl.add_legend(
            legend_entries,
            bcolor='white',
            border=True,
            size=(0.30, 0.06 * len(legend_entries) + 0.04),
        )

    pl.set_background(background_color)
    pl.show_axes()
    print(f'  Total: {n_total_segs:,} line segments, {n_total_jps:,} junction points')
    pl.show()
