"""
gsmesh2d.py — 2D grain-structure FE meshing orchestrator.

Dispatches to conformal or non-conformal meshing backends.
Non-conformal support is a placeholder for future implementation.

Public API
----------
mesh_gs          : Orchestrate meshing; returns a unified result dict.
visualize_gs_mesh: Visualize a mesh_gs result dict.

Shared helpers
--------------
_flatten_cells   : Expand MultiPolygons → individual Polygons (shared by all backends).
"""

from __future__ import annotations

import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_cells(cells: dict) -> tuple[dict, dict]:
    """
    Expand any MultiPolygon values into individual Polygon entries.

    Parameters
    ----------
    cells : {gid: Shapely Polygon or MultiPolygon}

    Returns
    -------
    flat_cells : {flat_id: Polygon}  — contiguous integer keys starting at 1.
    gid_map    : {flat_id: original_gid}
    """
    from shapely.geometry import Polygon, MultiPolygon
    flat_cells: dict = {}
    gid_map: dict = {}
    flat_id = 1
    for gid, geom in cells.items():
        if isinstance(geom, Polygon):
            parts = [geom]
        elif isinstance(geom, MultiPolygon):
            parts = list(geom.geoms)
        else:
            parts = []
        for part in parts:
            if part is not None and part.area > 0:
                flat_cells[flat_id] = part
                gid_map[flat_id] = gid
                flat_id += 1
    return flat_cells, gid_map


# ─────────────────────────────────────────────────────────────────────────────
# Backend: conformal (gmsh via confMesh2dGMSH)
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_conformal(
    cells: dict,
    mesh_size_gb: float,
    mesh_size_bulk: float,
    mesh_order: int,
    mesh_algo: int,
    recombine_to_quads: bool,
    dist_min: float,
    dist_max: float,
    out_dir: str | None,
    basename: str,
    formats: list | None,
    verbose: bool,
) -> dict:
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH

    flat_cells, gid_map = _flatten_cells(cells)
    if verbose:
        print(f'  [gsmesh2d] {len(flat_cells)} flat polygons from {len(cells)} grains')

    m = confMesh2dGMSH()
    t0 = time.perf_counter()
    m.femesh_gmsh(
        flat_cells, gid_map,
        mesh_size_gb=mesh_size_gb,
        mesh_size_bulk=mesh_size_bulk,
        mesh_algo=mesh_algo,
        mesh_order=mesh_order,
        recombine_to_quads=recombine_to_quads,
        out_dir=out_dir,
        basename=basename,
        formats=formats,
    )
    elapsed = time.perf_counter() - t0

    pts, _, triangles, quads = m.get_mesh_geometry()
    valid_mask = ~np.isnan(pts[:, 0])
    n_nodes = int(valid_mask.sum())
    n_tri   = len(triangles) if triangles is not None else 0
    n_quad  = len(quads)     if quads     is not None else 0

    return {
        'method':     'conformal',
        'mesher':     m,
        'flat_cells': flat_cells,
        'gid_map':    gid_map,
        'n_nodes':    n_nodes,
        'n_tri':      n_tri,
        'n_quad':     n_quad,
        'elapsed':    elapsed,
        'exported':   list(getattr(m, '_exported', [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def mesh_gs(
    cells: dict,
    method: str = 'conformal',
    mesh_size_gb: float = 0.75,
    mesh_size_bulk: float = 4.5,
    mesh_order: int = 1,
    mesh_algo: int = 8,
    recombine_to_quads: bool = True,
    dist_min: float = 0.5,
    dist_max: float = 5.0,
    out_dir: str | None = None,
    basename: str = 'gs_mesh',
    formats: list | None = None,
    verbose: bool = False,
) -> dict:
    """
    Orchestrate FE meshing for a dict of Shapely grain polygons.

    Parameters
    ----------
    cells            : {gid: Shapely Polygon or MultiPolygon}
    method           : 'conformal' (default).  'non_conformal' reserved for future use.
    mesh_size_gb     : Target element size on grain boundaries.
    mesh_size_bulk   : Target element size in grain interiors.
    mesh_order       : Element order (1 = linear, 2 = quadratic).
    mesh_algo        : Gmsh algorithm ID (6=Frontal, 8=Frontal-Delaunay quads).
    recombine_to_quads: Recombine triangles into quads after generation.
    dist_min         : Distance field DistMin (passed through; currently stored in result).
    dist_max         : Distance field DistMax (passed through; currently stored in result).
    out_dir          : Directory for exported files.  No export when None.
    basename         : Filename stem (extension appended per format).
    formats          : List of format extensions, e.g. ``['msh', 'inp', 'vtk']``.
    verbose          : Print progress messages.

    Returns
    -------
    dict
        method      str — meshing method used.
        mesher      confMesh2dGMSH instance with populated nodes / elConn.
        flat_cells  {flat_id: Polygon}
        gid_map     {flat_id: original_gid}
        n_nodes     int
        n_tri       int
        n_quad      int
        elapsed     float — total wall-clock seconds.
        exported    list[str] — paths of files written.
    """
    if method == 'conformal':
        return _mesh_conformal(
            cells,
            mesh_size_gb=mesh_size_gb,
            mesh_size_bulk=mesh_size_bulk,
            mesh_order=mesh_order,
            mesh_algo=mesh_algo,
            recombine_to_quads=recombine_to_quads,
            dist_min=dist_min,
            dist_max=dist_max,
            out_dir=out_dir,
            basename=basename,
            formats=formats,
            verbose=verbose,
        )
    elif method == 'non_conformal':
        raise NotImplementedError('Non-conformal meshing orchestration: coming soon.')
    else:
        raise ValueError(f'Unknown meshing method: {method!r}')


def visualize_gs_mesh(
    result: dict,
    figsize: tuple = (12, 9),
    dpi: int = 150,
    tri_edgecolor: str = 'steelblue',
    tri_linewidth: float = 0.2,
    tri_alpha: float = 0.7,
    quad_edgecolor: str = 'darkorange',
    quad_linewidth: float = 0.2,
    quad_alpha: float = 0.7,
    show_axis: bool = True,
    show_stats: bool = True,
    lw_gb: float = 0.6,
    color_gb: str = 'k',
) -> tuple:
    """
    Visualize a mesh_gs result dict using see_femesh + grain boundary overlay.

    Parameters
    ----------
    result     : dict returned by mesh_gs.
    figsize    : Figure size in inches.
    dpi        : Figure DPI.
    tri_*      : Styling for triangle elements.
    quad_*     : Styling for quad elements.
    show_axis  : Show axis ticks.
    show_stats : Show element count statistics in the figure.
    lw_gb      : Linewidth for grain boundary overlay.
    color_gb   : Colour for grain boundary overlay.

    Returns
    -------
    (fig, ax)
    """
    from upxo.viz.meshviz import see_femesh
    from matplotlib.collections import LineCollection
    import numpy as np

    m = result['mesher']
    pts, _, triangles, quads = m.get_mesh_geometry()

    title = (
        f'Conformal FE mesh — '
        f'{result["n_tri"]:,} tri + {result["n_quad"]:,} quad, '
        f'{result["n_nodes"]:,} nodes'
    )

    fig, ax = see_femesh(
        points_2d=pts,
        lines=None,
        triangles=triangles,
        quads=quads,
        figsize=figsize,
        dpi=dpi,
        tri_edgecolor=tri_edgecolor,
        tri_linewidth=tri_linewidth,
        tri_alpha=tri_alpha,
        quad_edgecolor=quad_edgecolor,
        quad_linewidth=quad_linewidth,
        quad_alpha=quad_alpha,
        show_axis=show_axis,
        show_stats=show_stats,
        title=title,
    )

    segments = [np.column_stack(poly.exterior.xy) for poly in result['flat_cells'].values()]
    ax.add_collection(LineCollection(segments, colors=color_gb, linewidths=lw_gb, zorder=2))

    return fig, ax
