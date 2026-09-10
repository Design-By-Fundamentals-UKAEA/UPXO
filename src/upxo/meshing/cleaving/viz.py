"""
viz.py
======
Optional pyvista-based visualisation helpers for the cleaving module.
Never required to run the core algorithm -- import-guarded, matching the
convention used elsewhere in ``upxo.meshing`` (independently; this module
does not import anything from ``confMesh3d``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from upxo.meshing.cleaving.lattice import BCCLattice
from upxo.meshing.cleaving.cleave import CleaveResult

try:
    import pyvista as pv
except ImportError:
    pv = None

_VTK_TETRA = 10


def _require_pyvista():
    if pv is None:
        raise ImportError(
            'pyvista is required for upxo.meshing.cleaving.viz '
            '(pip install pyvista)')


def _to_unstructured_grid(lattice: BCCLattice):
    _require_pyvista()
    n_tets = len(lattice.tets)
    cells = np.hstack(
        [np.full((n_tets, 1), 4, dtype=np.int64), lattice.tets]).ravel()
    cell_types = np.full(n_tets, _VTK_TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, lattice.vertices)


def visualize_lattice(
        lattice: BCCLattice,
        labels: Optional[np.ndarray] = None,
        show_tets: bool = True,
        point_size: float = 8.0,
        show: bool = True,
        screenshot: Optional[str] = None,
):
    """
    Render the BCC lattice: points coloured by sampled label (if given,
    else by primal/dual vertex kind), with the tet wireframe optionally
    overlaid.

    Intended for small synthetic test cases during development -- a real
    microstructure lattice has far too many tets to usefully render as a
    full wireframe.

    ``screenshot``, if given, saves a PNG to that path (works headless via
    off_screen rendering) -- the "render this test case, eyeball it against
    the last known-good picture" habit for catching stencil regressions.
    """
    _require_pyvista()
    pvp = pv.Plotter(off_screen=screenshot is not None and not show)
    pvp.set_background('white')

    cloud = pv.PolyData(lattice.vertices)
    if labels is not None:
        # Colour by rank, not raw value: the wall/padding sentinel label is
        # a huge outlier (e.g. 32767) that would otherwise dominate the
        # colour scale and make real grain labels indistinguishable.
        unique_vals, rank = np.unique(labels, return_inverse=True)
        cloud['scalars'] = rank
        title = 'Sampled label (rank)'
        legend = '  '.join(f'{r}={v}' for r, v in enumerate(unique_vals.tolist()))
        pvp.add_text(f'rank -> label:  {legend}', position='lower_left', font_size=8)
    else:
        cloud['scalars'] = lattice.vertex_kind
        title = 'Vertex kind (0=primal, 1=dual)'
    pvp.add_mesh(cloud, scalars='scalars', cmap='tab20', point_size=point_size,
                 render_points_as_spheres=True,
                 scalar_bar_args={'title': title})

    if show_tets and len(lattice.tets):
        grid = _to_unstructured_grid(lattice)
        pvp.add_mesh(grid, style='wireframe', color='gray', opacity=0.5,
                     line_width=1)

    pvp.add_title(
        f'BCC lattice  vertices={len(lattice.vertices)}  '
        f'tets={len(lattice.tets)}', font_size=10)
    pvp.add_axes()
    if show:
        pvp.show(screenshot=screenshot)
    elif screenshot:
        pvp.show(screenshot=screenshot, auto_close=True)
    return pvp


def _to_result_grid(result: CleaveResult):
    _require_pyvista()
    n_tets = len(result.tets)
    cells = np.hstack(
        [np.full((n_tets, 1), 4, dtype=np.int64), result.tets]).ravel()
    cell_types = np.full(n_tets, _VTK_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, result.node_coords)
    unique_vals, rank = np.unique(result.tet_labels, return_inverse=True)
    grid.cell_data['label_rank'] = rank
    return grid, unique_vals


def visualize_cleaved_mesh(
        result: CleaveResult,
        clip_normal: Optional[str] = 'x',
        exclude_labels: Optional[list] = None,
        show_edges: bool = True,
        show: bool = True,
        screenshot: Optional[str] = None,
):
    """
    Render the cleaved conformal tet mesh, coloured by grain label. A solid
    3D tet mesh is opaque, so by default this clips away half the domain
    (``clip_normal``, one of 'x'/'y'/'z', or None to show the exterior
    whole) so the interior grain-boundary surface is actually visible.

    ``exclude_labels``, e.g. ``[wall_label]``, drops tets with those labels
    before clipping -- otherwise the (usually much larger) padding shell
    dominates the clip plane and hides the actual grain structure.
    """
    _require_pyvista()
    grid, unique_vals = _to_result_grid(result)

    if exclude_labels:
        keep = ~np.isin(result.tet_labels, exclude_labels)
        grid = grid.extract_cells(np.nonzero(keep)[0])

    pvp = pv.Plotter(off_screen=screenshot is not None and not show)
    pvp.set_background('white')

    mesh = grid.clip(normal=clip_normal) if clip_normal else grid
    pvp.add_mesh(mesh, scalars='label_rank', cmap='tab20', show_edges=show_edges,
                 scalar_bar_args={'title': 'grain label (rank)'})

    legend = '  '.join(f'{r}={v}' for r, v in enumerate(unique_vals.tolist()))
    pvp.add_text(f'rank -> label:  {legend}', position='lower_left', font_size=8)
    pvp.add_title(
        f'Cleaved mesh  nodes={len(result.node_coords)}  tets={len(result.tets)}  '
        f'deferred={len(result.deferred_tet_indices)}', font_size=10)
    pvp.add_axes()
    if show:
        pvp.show(screenshot=screenshot)
    elif screenshot:
        pvp.show(screenshot=screenshot, auto_close=True)
    return pvp
