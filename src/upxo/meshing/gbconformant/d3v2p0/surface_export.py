"""PyVista views of surface complexes and tetrahedral meshes.

One place builds the ``PolyData`` / ``UnstructuredGrid`` objects that plots and
file writers need, so every consumer carries the same cell arrays.
"""
from pathlib import Path

import numpy as np


def surface_polydata(surface, *, exterior=None, rve_face=None):
    """Triangles of an interface or closed-RVE object as ``PolyData``.

    Cell arrays: ``grain_a`` and ``grain_b`` (from ``grain_pairs``),
    ``is_rve_cap`` (from ``exterior``) and ``rve_face`` when available. An
    object that already is a ``PolyData`` is returned unchanged.
    """
    if hasattr(surface, 'cell_data') and hasattr(surface, 'points'):
        return surface
    import pyvista as pv

    triangles = np.asarray(surface.triangles)
    cells = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    mesh = pv.PolyData(np.asarray(surface.points, dtype=float), cells)
    pairs = getattr(surface, 'grain_pairs', None)
    if pairs is not None:
        mesh.cell_data['grain_a'] = np.asarray(pairs)[:, 0]
        mesh.cell_data['grain_b'] = np.asarray(pairs)[:, 1]
    caps = exterior if exterior is not None else getattr(surface, 'exterior', None)
    if caps is not None:
        mesh.cell_data['is_rve_cap'] = np.asarray(caps).astype(np.uint8)
    face = rve_face if rve_face is not None else getattr(surface, 'rve_face', None)
    if face is not None:
        mesh.cell_data['rve_face'] = np.asarray(face)
    return mesh


def save_surface(path, surface, **kwargs):
    """Write :func:`surface_polydata` of ``surface`` to ``path`` (.vtp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    surface_polydata(surface, **kwargs).save(path)
    return path


def tet_grid(tets, *, quality_name='minSICN'):
    """Tetrahedra of a verified mesh as an ``UnstructuredGrid``.

    Cell arrays: ``grain_id`` and, when the mesh has one, the quality values
    under ``quality_name``.
    """
    import pyvista as pv

    tetrahedra = np.asarray(tets.tetrahedra, dtype=np.int64)
    cells = np.column_stack((np.full(len(tetrahedra), 4, dtype=np.int64), tetrahedra)).ravel()
    types = np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, np.asarray(tets.points, dtype=float))
    grid.cell_data['grain_id'] = np.asarray(tets.grain_ids)
    quality = getattr(tets, 'quality', None)
    if quality is not None:
        grid.cell_data[quality_name] = np.asarray(quality, dtype=float)
    return grid
