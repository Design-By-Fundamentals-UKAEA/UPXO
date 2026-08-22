"""
tetgen_mesh.py
===============
Conformal multi-grain tet mesh generation via TetGen -- an alternative to
volume_mesh.py's Gmsh discrete-entity backend.

Strategy
--------
Unlike volume_mesh.py (per-grain surface-loop + addVolume, one grain at a
time), TetGen tetrahedralizes the WHOLE domain in a single call from one
global piecewise-linear complex (PLC):

1. Every triangle in the ConformalSurfaceComplex (interior *and* cap) is
   passed once as a PLC facet -- interior facets become internal
   constrained faces TetGen must honor exactly, so grain boundaries land
   on tet faces.
2. One region marker point per grain is registered via
   TetGen.add_region(id, point_in_region) so TetGen can tag which tets
   belong to which grain. The marker point is derived from the ORIGINAL
   VOXEL DATA (lgi), not the mesh: the physical-space center of the voxel
   labelled `gid` nearest that grain's voxel-index centroid -- this
   sidesteps any point-in-polyhedron computation against the mesh itself.
3. tg.tetrahedralize(..., regionattrib=True) is called; tg.attributes
   gives the per-tet region marker (grain id) directly, read alongside
   tg.grid (a PyVista UnstructuredGrid) for node/tet connectivity.

API notes (confirmed by introspection + a minimal two-region smoke test
against tetgen 0.8.2): TetGen.add_region takes positional (id,
point_in_region, max_vol=0.0) -- not keyword marker=/point=. tg.attributes
has shape (n_tets, 1) and is aligned with tg.grid.cells_dict[10] (VTK_TETRA)
in tet order; tg.grid.cell_data does NOT carry the region markers.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from upxo.meshing.confMesh3d.config import TetGenMeshConfig
from upxo.meshing.confMesh3d.complex_builder import (
    ConformalSurfaceComplex, TetGenVolumeResult)

try:
    import tetgen
except ImportError:
    tetgen = None

_VTK_TETRA = 10


# ---------------------------------------------------------------------------
def _grain_interior_points(
        lgi: np.ndarray, voxel_size: float, grain_ids,
) -> Dict[int, np.ndarray]:
    """
    One interior point per grain, in physical (voxel_size-scaled)
    coordinates, for TetGen region markers.

    Uses the voxel nearest each grain's voxel-index centroid (rather than
    a point-in-polyhedron test against the mesh) -- robust and cheap,
    reusing the same lgi/voxel_size inputs validation.py already takes.
    Coordinate convention matches surface_nets.py: voxel index i maps to
    physical coordinate i * voxel_size.
    """
    points: Dict[int, np.ndarray] = {}
    for gid in grain_ids:
        idx = np.argwhere(lgi == gid)
        if idx.size == 0:
            continue
        centroid = idx.mean(axis=0)
        nearest = idx[np.argmin(np.sum((idx - centroid) ** 2, axis=1))]
        points[int(gid)] = nearest.astype(np.float64) * voxel_size
    return points


# ---------------------------------------------------------------------------
def generate_conformal_tet_mesh_tetgen(
        complex_   : ConformalSurfaceComplex,
        lgi        : np.ndarray,
        voxel_size : float,
        config     : Optional[TetGenMeshConfig] = None,
        verbose    : bool = True,
) -> TetGenVolumeResult:
    """
    Generate a conformal multi-grain tet mesh via TetGen, as an
    alternative to volume_mesh.generate_conformal_tet_mesh() (gmsh).

    Parameters
    ----------
    complex_   : ConformalSurfaceComplex (optionally post surface_remesh)
    lgi        : ndarray (nx,ny,nz) -- original labelled voxel image,
                 used only to place one interior region-marker point per
                 grain (see _grain_interior_points).
    voxel_size : float -- physical voxel edge length
    config     : TetGenMeshConfig or None
    verbose    : bool

    Returns
    -------
    TetGenVolumeResult
    """
    if tetgen is None:
        raise RuntimeError('tetgen is not installed.')
    cfg = config or TetGenMeshConfig()
    t0 = time.perf_counter()

    grain_ids = sorted(
        int(gid) for gid in complex_.grain_to_triangle_ids
        if int(gid) != complex_.wall_label)

    if verbose:
        print(f'[TetGenMesh] Building global PLC  {len(complex_.vertices)} vertices  '
              f'{len(complex_.triangles)} triangles  {len(grain_ids)} grains')

    tg = tetgen.TetGen(complex_.vertices, complex_.triangles)

    interior_points = _grain_interior_points(lgi, voxel_size, grain_ids)
    missing = [gid for gid in grain_ids if gid not in interior_points]
    if missing and verbose:
        print(f'  WARNING: no voxels found for {len(missing)} grain(s) in lgi -- '
              f'they will not receive a region marker: {missing[:10]}'
              f'{"..." if len(missing) > 10 else ""}')

    for gid in grain_ids:
        pt = interior_points.get(gid)
        if pt is None:
            continue
        tg.add_region(gid, pt.tolist())

    if verbose:
        print(f'  Tetrahedralizing (quality={cfg.quality}  '
              f'minratio={cfg.minratio}  mindihedral={cfg.mindihedral})...',
              end='', flush=True)
    t1 = time.perf_counter()
    tg.tetrahedralize(
        plc=True, quality=cfg.quality, regionattrib=True,
        minratio=cfg.minratio, mindihedral=cfg.mindihedral,
        verbose=cfg.verbose,
    )
    print(f' done ({time.perf_counter()-t1:.1f}s)')

    grid = tg.grid
    node_coords = np.asarray(grid.points, dtype=np.float64)
    tets_all = grid.cells_dict.get(_VTK_TETRA)
    if tets_all is None:
        raise RuntimeError('TetGen produced no tetrahedral elements.')
    tets_all = np.asarray(tets_all, dtype=np.int64)

    attrib = getattr(tg, 'attributes', None)
    markers = None
    if attrib is not None:
        attrib = np.asarray(attrib, dtype=np.float64).reshape(-1)
        if len(attrib) == len(tets_all):
            markers = attrib.round().astype(np.int64)

    tets_by_grain: Dict[int, np.ndarray] = {}
    if markers is not None:
        for gid in grain_ids:
            mask = markers == gid
            if mask.any():
                tets_by_grain[gid] = tets_all[mask]
    else:
        if verbose:
            print('  WARNING: region attributes unavailable/mismatched -- '
                  'returning ungrouped mesh under key -1.')
        tets_by_grain[-1] = tets_all

    n_tets = int(sum(len(t) for t in tets_by_grain.values()))
    if verbose:
        print(f'[TetGenMesh] Complete ({time.perf_counter()-t0:.1f}s)  '
              f'nodes={len(node_coords)}  tets={n_tets}  '
              f'grains_tagged={len(tets_by_grain)}')

    return TetGenVolumeResult(
        node_coords   = node_coords,
        tets_by_grain = tets_by_grain,
        n_nodes       = len(node_coords),
        n_tetrahedra  = n_tets,
    )
