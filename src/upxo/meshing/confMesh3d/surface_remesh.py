"""
surface_remesh.py
==================
Optional Gmsh-driven surface quality-optimization stage.

Sits between complex_builder.build_conformal_surface_complex() and
validation.validate_surface_complex() / the tet-meshing stage. Disabled by
default (SurfaceRemeshConfig.enabled=False) -- pipelines that do not call
remesh_conformal_surface(), or call it with the default config, reproduce
today's behaviour exactly.

v1 scope
--------
Boundary-pinned Laplacian relaxation of each grain-boundary/cap patch's
INTERIOR vertices only. A vertex is "boundary" if it is shared by >= 2
patches (triple lines, RVE edges) -- those vertices are never moved, so
conformality across patches is preserved by construction. This does not
re-parametrize or regenerate patches from scratch; that is a harder,
deferred v2 (see plan notes).

Why not gmsh.model.mesh.optimize() directly
--------------------------------------------
Empirically, gmsh's built-in smoothers ("Laplace2D", "Relocate2D", ...)
are no-ops on bare discrete entities added via addDiscreteEntity +
addElementsByType with no curve/parametrization info: gmsh has no notion
of which nodes are "free" vs "on the boundary" without classified
topology, and classifySurfaces() detects boundaries from local dihedral
angle -- not from our combinatorial (grain-pair) boundary definition. So
the position update is computed directly (uniform one-ring Laplacian,
identical math to what a boundary-aware "Laplace2D" would do); gmsh is
used to build each patch as a discrete entity (mirroring volume_mesh.py's
idiom) and, genuinely, as the accept/reject quality gate via
gmsh.model.mesh.getElementQualities('minSICN') -- a patch's relaxed
positions are kept only if they do not reduce minimum triangle quality.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from upxo.meshing.confMesh3d.config import SurfaceRemeshConfig
from upxo.meshing.confMesh3d.complex_builder import ConformalSurfaceComplex

try:
    import gmsh
except ImportError:
    gmsh = None


# ---------------------------------------------------------------------------
def _find_boundary_vertices(complex_: ConformalSurfaceComplex) -> np.ndarray:
    """
    A vertex is a boundary (triple-line / RVE-edge) vertex if it belongs
    to triangles from >= 2 distinct source_ids (patches). These vertices
    are shared across patches and must never move.
    """
    n_verts = len(complex_.vertices)
    vertex_source_ids: List[Set[str]] = [set() for _ in range(n_verts)]
    for tri, src_id in zip(complex_.triangles, complex_.triangle_source_ids):
        for v in tri:
            vertex_source_ids[int(v)].add(src_id)
    return np.array([len(s) >= 2 for s in vertex_source_ids], dtype=bool)


def _build_patch_adjacency(tris_local: np.ndarray, n_local: int) -> List[Set[int]]:
    """One-ring vertex adjacency (by local index) for a patch's triangles."""
    adjacency: List[Set[int]] = [set() for _ in range(n_local)]
    for tri in tris_local:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def _laplacian_relax_patch(
        pts_local         : np.ndarray,
        tris_local        : np.ndarray,
        is_boundary_local : np.ndarray,
        iterations        : int,
) -> np.ndarray:
    """Boundary-pinned uniform Laplacian relaxation on a single patch."""
    adjacency = _build_patch_adjacency(tris_local, len(pts_local))
    pts = pts_local.copy()
    for _ in range(max(0, iterations)):
        new_pts = pts.copy()
        for vi, neighbors in enumerate(adjacency):
            if is_boundary_local[vi] or not neighbors:
                continue
            neigh_idx = np.fromiter(neighbors, dtype=np.int64)
            new_pts[vi] = pts[neigh_idx].mean(axis=0)
        pts = new_pts
    return pts


# ---------------------------------------------------------------------------
def remesh_conformal_surface(
        complex_ : ConformalSurfaceComplex,
        config   : Optional[SurfaceRemeshConfig] = None,
        verbose  : bool = True,
) -> ConformalSurfaceComplex:
    """
    Optimize triangle quality of every grain-boundary/cap patch in place.

    Interior (non-shared) vertices of each patch are relaxed toward their
    one-ring barycenter; boundary vertices (shared by >= 2 patches) are
    never moved. Each patch's relaxed result is accepted only if gmsh
    reports its minimum element quality (minSICN) did not decrease --
    otherwise the patch's original SurfaceNets triangulation is kept
    unchanged. Topology (triangles, adjacency, source ids) is never
    altered -- only vertex positions.

    Parameters
    ----------
    complex_ : ConformalSurfaceComplex
    config   : SurfaceRemeshConfig or None. If config.enabled is False
               (the default), this is a pass-through returning complex_
               unchanged.
    verbose  : bool

    Returns
    -------
    ConformalSurfaceComplex -- new instance with (possibly) updated
    vertex positions; triangles/adjacency/source ids are shared with the
    input (unchanged).
    """
    cfg = config or SurfaceRemeshConfig()
    if not cfg.enabled:
        if verbose:
            print('[SurfaceRemesh] Disabled (config.enabled=False) -- pass-through.')
        return complex_
    if gmsh is None:
        raise RuntimeError('gmsh is not installed.')

    t0 = time.perf_counter()
    is_boundary_global = _find_boundary_vertices(complex_)

    src_to_tri_ids: Dict[str, List[int]] = defaultdict(list)
    for ti, src_id in enumerate(complex_.triangle_source_ids):
        src_to_tri_ids[str(src_id)].append(ti)

    new_vertices = complex_.vertices.copy()
    n_updated = 0
    n_skipped = 0

    gmsh_was_initialized = bool(gmsh.isInitialized())
    if not gmsh_was_initialized:
        gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1 if cfg.verbose_gmsh else 0)
    gmsh.model.add('confMesh3d_surface_remesh')

    if verbose:
        print(f'[SurfaceRemesh] {len(src_to_tri_ids)} patches  '
              f'boundary vertices={int(is_boundary_global.sum())}/{len(new_vertices)}')

    for src_id, tri_ids in src_to_tri_ids.items():
        tris_global = complex_.triangles[np.array(tri_ids, dtype=np.int64)]
        unique_vidx, inv = np.unique(tris_global.ravel(), return_inverse=True)
        tris_local = inv.reshape(-1, 3)
        pts_local = complex_.vertices[unique_vidx]
        is_boundary_local = is_boundary_global[unique_vidx]

        if is_boundary_local.all() or len(tris_local) < 4:
            n_skipped += 1
            continue   # nothing free to relax, or too small to matter

        ent = None
        try:
            n_local = len(pts_local)
            ent = gmsh.model.addDiscreteEntity(2)
            node_tags = np.arange(1, n_local + 1, dtype=np.int64)
            gmsh.model.mesh.addNodes(2, ent, node_tags.tolist(), pts_local.ravel().tolist())
            el_tags = np.arange(1, len(tris_local) + 1, dtype=np.int64)
            conn = node_tags[tris_local].ravel().tolist()
            gmsh.model.mesh.addElementsByType(ent, 2, el_tags.tolist(), conn)

            q_before = gmsh.model.mesh.getElementQualities(el_tags.tolist(), 'minSICN')
            q_before_min = float(np.min(q_before)) if len(q_before) else 0.0

            relaxed_pts = _laplacian_relax_patch(
                pts_local, tris_local, is_boundary_local, cfg.optimize_iterations)

            for i, tag in enumerate(node_tags.tolist()):
                gmsh.model.mesh.setNode(int(tag), relaxed_pts[i].tolist(), [])

            q_after = gmsh.model.mesh.getElementQualities(el_tags.tolist(), 'minSICN')
            q_after_min = float(np.min(q_after)) if len(q_after) else 0.0

            if q_after_min >= q_before_min - 1e-9:
                new_vertices[unique_vidx] = relaxed_pts
                n_updated += 1
            else:
                n_skipped += 1   # relaxation made shape quality worse -- keep original

            gmsh.model.mesh.removeElements(2, ent)
            gmsh.model.removeEntities([(2, ent)], recursive=True)
        except Exception as exc:
            n_skipped += 1
            if verbose:
                print(f'  WARNING patch {src_id}: {exc} -- keeping original triangulation.')

    if not gmsh_was_initialized:
        gmsh.finalize()

    if verbose:
        print(f'[SurfaceRemesh] Updated {n_updated} patches, kept {n_skipped} unchanged '
              f'({time.perf_counter()-t0:.1f}s)')

    new_complex = copy.copy(complex_)
    new_complex.vertices = new_vertices
    return new_complex
