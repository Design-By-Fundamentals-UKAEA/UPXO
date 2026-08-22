"""
complex_builder.py
==================
Build a ConformalSurfaceComplex from a SurfaceNetsResult.

A ConformalSurfaceComplex holds a single shared vertex table plus per-triangle
adjacency metadata.  This is the canonical input to the gmsh discrete-entity
tet-meshing pipeline (adapted from kali.tet.surface_complex).

Interior surfaces (grain A -- grain B) are stored once and referenced by both
grains with opposite orientations.  RVE cap surfaces (grain -- wall) are stored
once and referenced only by their owning grain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh as tm

from upxo.meshing.confMesh3d.config import ComplexConfig
from upxo.meshing.confMesh3d.surface_nets import SurfaceNetsResult


# ---------------------------------------------------------------------------
# Data structures (adapted from kali.tet.surface_complex)
# ---------------------------------------------------------------------------

@dataclass
class ConformalSurfaceComplex:
    """
    Canonical shared-node surface complex for multi-grain tet meshing.

    vertices                : (V, 3) float64 — global node coordinates
    triangles               : (T, 3) int64   — 0-based vertex indices
    triangle_adjacent_grains: list of (gA, gB) or (gA, wall_label)
    triangle_source_types   : list of 'interior' or 'cap'
    triangle_source_ids     : list of str keys  e.g. 'int_(1,2)' or 'cap_1_x0'
    grain_to_triangle_ids   : {gid: [triangle indices belonging to grain]}
    """
    vertices                 : np.ndarray
    triangles                : np.ndarray
    triangle_adjacent_grains : List[Tuple[int, ...]]
    triangle_source_types    : List[str]
    triangle_source_ids      : List[str]
    grain_to_triangle_ids    : Dict[int, List[int]] = field(default_factory=dict)
    wall_label               : int = 32767
    voxel_size               : float = 1.0


@dataclass
class GmshVolumeResult:
    """Tags returned after building the gmsh model."""
    surface_tags_by_source_id : Dict[str, int]
    volume_tags_by_grain_id   : Dict[int, int]
    n_nodes                   : int = 0
    n_tetrahedra              : int = 0
    tet_meshing_attempted     : bool = False


@dataclass
class TetGenVolumeResult:
    """
    Result of TetGen tet-mesh generation.

    Unlike GmshVolumeResult, this is fully self-contained (plain arrays,
    no dependency on a live external-engine model) since TetGen returns
    its output as arrays rather than a queryable in-memory model.

    node_coords   : (N, 3) float64 -- global node coordinates
    tets_by_grain : {gid: (T, 4) int64} -- 0-based node indices into
                    node_coords, per grain
    """
    node_coords   : np.ndarray
    tets_by_grain : Dict[int, np.ndarray]
    n_nodes       : int = 0
    n_tetrahedra  : int = 0


# ---------------------------------------------------------------------------
# Vertex welding helper
# ---------------------------------------------------------------------------

class _VertexWelder:
    """
    Tolerance-based vertex deduplication using a rounded-grid hash.
    Vertices within *tol* of each other share one global index.
    """
    def __init__(self, tol: float = 1e-7):
        self._tol   = float(tol)
        self._scale = 1.0 / max(tol, 1e-15)
        self._table : Dict[Tuple[int,int,int], int] = {}
        self._coords: List[List[float]] = []

    def add_vertex(self, xyz: np.ndarray) -> int:
        key = (int(round(xyz[0] * self._scale)),
               int(round(xyz[1] * self._scale)),
               int(round(xyz[2] * self._scale)))
        if key not in self._table:
            self._table[key] = len(self._coords)
            self._coords.append(xyz.tolist())
        return self._table[key]

    def add_vertices(self, verts: np.ndarray) -> np.ndarray:
        """Add an (N,3) array; return global index array."""
        return np.array([self.add_vertex(v) for v in verts], dtype=np.int64)

    @property
    def global_vertices(self) -> np.ndarray:
        return np.array(self._coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_conformal_surface_complex(
        result  : SurfaceNetsResult,
        config  : Optional[ComplexConfig] = None,
        verbose : bool = True,
) -> ConformalSurfaceComplex:
    """
    Build a ConformalSurfaceComplex from the SurfaceNets3D output.

    Steps
    -----
    1. Triangulate the PyVista mesh (SurfaceNets3D output may contain quads).
    2. Separate cells into interior (grain-grain) and cap (grain-wall).
    3. For each unique grain-pair (A,B): extract the triangles, weld shared
       vertices into the global table, record source_id and adjacency.
    4. For each grain's cap faces: same process.
    5. Build grain_to_triangle_ids mapping.
    """
    cfg = config or ComplexConfig()
    t0  = time.perf_counter()

    # triangulate() converts any quads → triangles (SurfaceNets3D can output quads).
    # Do NOT call .clean() — it removes degenerate faces but does NOT update
    # BoundaryLabels cell data in sync, causing index-out-of-bounds in face lookup.
    # Our _VertexWelder handles vertex deduplication instead.
    mesh  = result.full_mesh.triangulate()
    bl    = mesh['BoundaryLabels'].astype(np.int32)   # (N_cells, 2)
    pts   = mesh.points                               # (V, 3)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]          # (T, 3)  after triangulate

    wall  = result.wall_label
    welder = _VertexWelder(tol=cfg.vertex_weld_tol)

    # --- Group cell indices by boundary-label pair --------------------------
    # Normalise pair so lower label always first: (min, max)
    pair_to_cells: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for ci, (a, b) in enumerate(bl.tolist()):
        pair_to_cells[(min(a,b), max(a,b))].append(ci)

    all_triangles     : List[List[int]]       = []
    all_adj_grains    : List[Tuple[int,...]]  = []
    all_source_types  : List[str]             = []
    all_source_ids    : List[str]             = []
    grain_to_tri_ids  : Dict[int, List[int]]  = defaultdict(list)

    n_pairs   = len(pair_to_cells)
    n_cells   = mesh.n_cells
    if verbose:
        print(f'[ComplexBuilder] {n_pairs} unique boundary pairs  '
              f'{n_cells:,} cells')
        print(f'  Building shared vertex table and triangle list...')

    _report_every = max(1, n_pairs // 20)   # progress every 5%
    _pair_done    = 0
    _t_loop       = time.perf_counter()

    for (la, lb), cell_indices in pair_to_cells.items():
        _pair_done += 1
        if verbose and (_pair_done % _report_every == 0 or _pair_done == n_pairs):
            _pct = 100 * _pair_done / n_pairs
            _ela = time.perf_counter() - _t_loop
            _rate = _pair_done / max(_ela, 1e-9)
            _eta  = (n_pairs - _pair_done) / max(_rate, 1e-9)
            print(f'  [{_pair_done:>6d}/{n_pairs}  {_pct:5.1f}%]  '
                  f'elapsed={_ela:.1f}s  ETA={_eta:.0f}s  '
                  f'verts_so_far={len(welder._coords):,}', flush=True)
        la, lb = int(la), int(lb)
        is_cap = (la == wall) or (lb == wall)

        if is_cap:
            grain = lb if la == wall else la
            src_type = 'cap'
            src_id   = f'cap_{grain}_w{wall}'
            adj      = (grain, wall)
        else:
            src_type = 'interior'
            src_id   = f'int_({la},{lb})'
            adj      = (la, lb)

        # Extract and weld triangles for this pair
        for ci in cell_indices:
            tri_local = faces[ci]               # 3 local vertex indices
            tri_xyz   = pts[tri_local]          # (3, 3)
            global_idx = welder.add_vertices(tri_xyz)
            t_id = len(all_triangles)
            all_triangles.append(global_idx.tolist())
            all_adj_grains.append(adj)
            all_source_types.append(src_type)
            all_source_ids.append(src_id)

            # Map triangle to owning grains
            if is_cap:
                grain_to_tri_ids[grain].append(t_id)
            else:
                grain_to_tri_ids[la].append(t_id)
                grain_to_tri_ids[lb].append(t_id)

    complex_ = ConformalSurfaceComplex(
        vertices                 = welder.global_vertices,
        triangles                = np.array(all_triangles, dtype=np.int64),
        triangle_adjacent_grains = all_adj_grains,
        triangle_source_types    = all_source_types,
        triangle_source_ids      = all_source_ids,
        grain_to_triangle_ids    = dict(grain_to_tri_ids),
        wall_label               = wall,
        voxel_size               = result.voxel_size,
    )

    if verbose:
        print(f'  Global vertices: {len(complex_.vertices)}  '
              f'Triangles: {len(complex_.triangles)}  '
              f'Grains: {len(complex_.grain_to_triangle_ids)}')
        print(f'[ComplexBuilder] Done ({time.perf_counter()-t0:.1f}s)')

    return complex_


# ---------------------------------------------------------------------------
# PyVista visualisation
# ---------------------------------------------------------------------------

def visualize_complex(
        complex_ : ConformalSurfaceComplex,
        mode     : str = 'by_type',    # 'by_type' | 'by_grain' | 'grain'
        grain_id : Optional[int] = None,
        show     : bool = True,
) -> pv.Plotter:
    """
    Visualise the ConformalSurfaceComplex.

    mode='by_type'  — colour interior (blue) vs cap (gray) faces
    mode='by_grain' — colour each grain differently
    mode='grain'    — show one grain's surface only (requires grain_id)
    """
    verts = complex_.vertices
    tris  = complex_.triangles
    pvp   = pv.Plotter()
    pvp.set_background('white')

    # Build PyVista PolyData from the complex
    faces_pv = np.hstack([np.full((len(tris),1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(verts, faces_pv)

    if mode == 'by_type':
        is_cap = np.array([1 if t == 'cap' else 0
                           for t in complex_.triangle_source_types], dtype=np.uint8)
        poly.cell_data['type'] = is_cap
        pvp.add_mesh(poly, scalars='type', cmap=['steelblue','lightyellow'],
                     show_edges=True)
        pvp.add_title('ConformalSurfaceComplex — blue=interior, yellow=cap', font_size=10)

    elif mode == 'by_grain':
        grain_label = np.zeros(len(tris), dtype=np.int32)
        for gid, tri_ids in complex_.grain_to_triangle_ids.items():
            grain_label[tri_ids] = gid
        poly.cell_data['grain'] = grain_label
        pvp.add_mesh(poly, scalars='grain', cmap='tab20', show_edges=False)
        pvp.add_title('ConformalSurfaceComplex — coloured by grain', font_size=10)

    elif mode == 'grain':
        if grain_id is None:
            raise ValueError('grain_id required for mode="grain"')
        tri_ids = complex_.grain_to_triangle_ids.get(int(grain_id), [])
        if not tri_ids:
            print(f'  WARNING: no triangles for grain {grain_id}')
        else:
            sub_tris = tris[np.array(tri_ids)]
            sub_faces = np.hstack([np.full((len(sub_tris),1), 3, dtype=np.int64),
                                   sub_tris]).ravel()
            sub_poly = pv.PolyData(verts, sub_faces)
            types = [complex_.triangle_source_types[i] for i in tri_ids]
            sub_poly.cell_data['is_cap'] = np.array(
                [1 if t == 'cap' else 0 for t in types], dtype=np.uint8)
            pvp.add_mesh(sub_poly, scalars='is_cap', cmap=['steelblue','lightyellow'],
                         show_edges=True)
            pvp.add_title(f'Grain {grain_id} — blue=interior, yellow=cap', font_size=10)

    pvp.add_axes()
    if show:
        pvp.show()
    return pvp
