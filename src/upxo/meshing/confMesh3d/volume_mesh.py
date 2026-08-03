"""
volume_mesh.py
==============
Conformal multi-grain tet mesh generation via gmsh's discrete-entity kernel.
Adapted from kali.tet.gmsh_volume_construction.

Strategy
--------
1. Add all shared vertices once as a 0D discrete entity (global node table).
2. For each unique grain-boundary surface patch: add as a 2D discrete entity
   using addElementsByType (triangles only — no B-rep reconstruction).
3. For each grain: addSurfaceLoop([signed_surface_tags]) + addVolume([loop]).
   Interior surfaces appear with +tag for one grain and -tag for the other
   (orientation reversal so normals are consistently outward for each grain).
4. Add physical groups: one per grain (named 'grain_<gid>').
5. Synchronise geometry, generate(3), optionally optimize mesh quality.

Element types
-------------
C3D4  (element type 4 in gmsh) — 4-node linear tet      (default)
C3D10 (element type 11)        — 10-node quadratic tet   (user option)
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from upxo.meshing.confMesh3d.config import GmshMeshConfig
from upxo.meshing.confMesh3d.complex_builder import (
    ConformalSurfaceComplex, GmshVolumeResult)

try:
    import gmsh
except ImportError:
    gmsh = None

_ELEMENT_TYPE_MAP = {
    'C3D4':  4,    # linear tet (4 nodes)
    'C3D10': 11,   # quadratic tet (10 nodes)
}


# ---------------------------------------------------------------------------
# Helpers (adapted from kali.tet.gmsh_volume_construction)
# ---------------------------------------------------------------------------

def _group_triangles_by_source(
        complex_: ConformalSurfaceComplex,
) -> Dict[str, List[int]]:
    """Triangle indices grouped by source_id."""
    grouped: Dict[str, List[int]] = defaultdict(list)
    for tri_id, src_id in enumerate(complex_.triangle_source_ids):
        grouped[str(src_id)].append(int(tri_id))
    return dict(sorted(grouped.items()))


def _grain_to_source_ids(
        complex_: ConformalSurfaceComplex,
) -> Dict[int, List[str]]:
    """Ordered unique source_ids for each grain's surface shell."""
    result: Dict[int, List[str]] = {}
    for gid, tri_ids in sorted(complex_.grain_to_triangle_ids.items()):
        seen: set = set()
        src_list: List[str] = []
        for ti in tri_ids:
            s = str(complex_.triangle_source_ids[ti])
            if s not in seen:
                seen.add(s)
                src_list.append(s)
        result[int(gid)] = src_list
    return result


def _signed_surface_tag(
        *,
        complex_: ConformalSurfaceComplex,
        source_id: str,
        grain_id: int,
        surface_tag: int,
) -> int:
    """
    Return ±surface_tag for a grain's orientation of one surface patch.

    For interior surfaces (A, B) with A < B:
      grain A → +tag (outward normal as stored)
      grain B → -tag (reversed, so outward for B)
    For cap surfaces: always +tag (already outward for the grain).
    """
    for tri_src, adj, src_type in zip(
            complex_.triangle_source_ids,
            complex_.triangle_adjacent_grains,
            complex_.triangle_source_types,
    ):
        if str(tri_src) != str(source_id):
            continue
        if src_type == 'interior' and len(adj) == 2:
            a, b = int(adj[0]), int(adj[1])
            # Stored orientation: normals point from lower-label to higher-label
            # (SurfaceNets3D convention: BoundaryLabels = (lower, higher))
            if int(grain_id) == a:      # grain A = lower label → outward
                return +int(surface_tag)
            else:                       # grain B = higher label → inward stored
                return -int(surface_tag)
        return +int(surface_tag)        # cap: always outward
    return +int(surface_tag)


def _add_all_nodes(complex_: ConformalSurfaceComplex) -> np.ndarray:
    """Add all complex vertices to gmsh as a 0D discrete entity."""
    verts     = np.asarray(complex_.vertices, dtype=np.float64)
    pt_entity = gmsh.model.addDiscreteEntity(0)
    node_tags = np.arange(1, verts.shape[0] + 1, dtype=np.int64)
    gmsh.model.mesh.addNodes(
        0, int(pt_entity),
        node_tags.tolist(),
        verts.ravel().tolist(),
    )
    return node_tags


def _add_source_surface(
        *,
        complex_   : ConformalSurfaceComplex,
        source_id  : str,
        tri_ids    : List[int],
        node_tags  : np.ndarray,
) -> int:
    """Add one source surface as a gmsh 2D discrete entity (triangles)."""
    surf_tag = gmsh.model.addDiscreteEntity(2)
    gmsh.model.setEntityName(2, surf_tag, str(source_id))

    tris_local = np.asarray(
        complex_.triangles[np.array(tri_ids, dtype=np.int64)], dtype=np.int64)
    el_start = int(gmsh.model.mesh.getMaxElementTag()) + 1
    el_tags  = np.arange(el_start, el_start + len(tri_ids), dtype=np.int64)
    # Map 0-based vertex indices → 1-based gmsh node tags
    tri_node_tags = node_tags[tris_local]   # (T, 3)

    gmsh.model.mesh.addElementsByType(
        int(surf_tag), 2,
        el_tags.tolist(),
        tri_node_tags.ravel().tolist(),
    )
    return int(surf_tag)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_conformal_tet_mesh(
        complex_  : ConformalSurfaceComplex,
        config    : Optional[GmshMeshConfig] = None,
        verbose   : bool = True,
) -> GmshVolumeResult:
    """
    Generate a conformal multi-grain tet mesh from the ConformalSurfaceComplex.

    Returns
    -------
    GmshVolumeResult  with gmsh entity tags and element counts.
    The gmsh model remains active (call gmsh.finalize() externally when done).
    """
    if gmsh is None:
        raise RuntimeError('gmsh is not installed.')

    cfg = config or GmshMeshConfig()
    vs  = complex_.voxel_size
    t0  = time.perf_counter()

    # ── Initialise gmsh ─────────────────────────────────────────────────────
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    gmsh.option.setNumber('General.Verbosity', 5 if verbose else 0)
    gmsh.option.setNumber('Mesh.Algorithm3D', cfg.algorithm_3d)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMin',
                          cfg.char_length_min_frac * vs)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax',
                          cfg.char_length_max_frac * vs)
    gmsh.model.add('confMesh3d')

    if verbose:
        n_grains = len([g for g in complex_.grain_to_triangle_ids
                        if g != complex_.wall_label])
        print(f'[VolumeMesh] Building discrete-entity gmsh model  '
              f'{n_grains} grains  {len(complex_.vertices)} vertices  '
              f'{len(complex_.triangles)} triangles')

    # ── Step 1: add all vertices as 0D entity ───────────────────────────────
    print('  Step 1/4  Adding global node table...', end='', flush=True)
    t1 = time.perf_counter()
    node_tags = _add_all_nodes(complex_)
    print(f' done ({time.perf_counter()-t1:.1f}s)  {len(node_tags)} nodes')

    # ── Step 2: add one discrete surface per unique source_id ──────────────
    print('  Step 2/4  Adding discrete surface entities...', end='', flush=True)
    t2 = time.perf_counter()
    src_to_tris = _group_triangles_by_source(complex_)
    surface_tags_by_source: Dict[str, int] = {}
    for src_id, tri_ids in src_to_tris.items():
        stag = _add_source_surface(
            complex_=complex_, source_id=src_id,
            tri_ids=tri_ids, node_tags=node_tags)
        surface_tags_by_source[src_id] = stag
    print(f' done ({time.perf_counter()-t2:.1f}s)  '
          f'{len(surface_tags_by_source)} surface entities')

    # ── Step 3: assemble grain volumes ─────────────────────────────────────
    print('  Step 3/4  Assembling grain volumes...', end='', flush=True)
    t3 = time.perf_counter()
    gid_to_src_ids = _grain_to_source_ids(complex_)
    volume_tags_by_grain: Dict[int, int] = {}

    for gid, src_ids in sorted(gid_to_src_ids.items()):
        gid = int(gid)
        if gid == complex_.wall_label:
            continue
        signed_tags = [
            _signed_surface_tag(
                complex_=complex_, source_id=sid,
                grain_id=gid,
                surface_tag=surface_tags_by_source[sid],
            )
            for sid in src_ids
            if sid in surface_tags_by_source
        ]
        if not signed_tags:
            continue
        try:
            loop_tag   = gmsh.model.geo.addSurfaceLoop(signed_tags)
            volume_tag = gmsh.model.geo.addVolume([loop_tag])
            volume_tags_by_grain[gid] = int(volume_tag)
        except Exception as exc:
            if verbose:
                print(f'\n  WARNING grain {gid}: {exc}')

    gmsh.model.geo.synchronize()
    print(f' done ({time.perf_counter()-t3:.1f}s)  '
          f'{len(volume_tags_by_grain)} volumes')

    # Physical groups (one per grain, named grain_<gid>)
    for gid, vtag in volume_tags_by_grain.items():
        pg = gmsh.model.addPhysicalGroup(3, [int(vtag)], tag=int(gid))
        gmsh.model.setPhysicalName(3, pg, f'grain_{gid}')

    # ── Step 4: generate tet mesh ───────────────────────────────────────────
    print(f'  Step 4/4  Generating tet mesh  '
          f'(alg={cfg.algorithm_3d}  '
          f'h=[{cfg.char_length_min_frac*vs:.3g}, '
          f'{cfg.char_length_max_frac*vs:.3g}] um)...', flush=True)
    t4 = time.perf_counter()
    gmsh.model.mesh.generate(3)
    if cfg.optimize:
        gmsh.model.mesh.optimize('Netgen')
    dt4 = time.perf_counter() - t4
    print(f'  Mesh generation done ({dt4:.1f}s)')

    # Element counts
    n_nodes   = int(len(gmsh.model.mesh.getNodes()[0]))
    elem_type = _ELEMENT_TYPE_MAP.get(cfg.element_type, 4)
    n_tets    = 0
    for etyp, etags, _ in zip(*gmsh.model.mesh.getElements(dim=3)):
        if int(etyp) == 4:   # always count C3D4
            n_tets += len(etags)

    if cfg.element_type == 'C3D10':
        gmsh.model.mesh.setOrder(2)
        if verbose:
            print('  Converted to quadratic (C3D10) elements.')

    if verbose:
        print(f'[VolumeMesh] Complete ({time.perf_counter()-t0:.1f}s)  '
              f'nodes={n_nodes}  tets={n_tets}')

    if cfg.show_gui:
        print('  Opening gmsh GUI — close window to continue.')
        gmsh.fltk.run()

    return GmshVolumeResult(
        surface_tags_by_source_id = surface_tags_by_source,
        volume_tags_by_grain_id   = volume_tags_by_grain,
        n_nodes                   = n_nodes,
        n_tetrahedra              = n_tets,
        tet_meshing_attempted     = True,
    )


# ---------------------------------------------------------------------------
# PyVista visualisation from live gmsh model
# ---------------------------------------------------------------------------

def visualize_tet_mesh(
        gmsh_result     : GmshVolumeResult,
        twin_role       : Optional[Dict[int,str]] = None,
        show            : bool = True,
) -> pv.Plotter:
    """
    Extract the tet mesh from the active gmsh model and visualise in PyVista.

    Colours elements by grain ID (or by role if twin_role is provided).
    Call BEFORE gmsh.finalize().
    """
    if gmsh is None:
        raise RuntimeError('gmsh is not available.')

    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)
    tag_to_idx  = {int(t): i for i, t in enumerate(node_tags)}

    # Extract tet elements and their grain IDs
    all_cells   : List[int] = []
    all_grain_id: List[int] = []

    for gid, vtag in gmsh_result.volume_tags_by_grain_id.items():
        etyp, etags, enodes = gmsh.model.mesh.getElements(dim=3, tag=int(vtag))
        for typ, tags, nodes in zip(etyp, etags, enodes):
            if int(typ) not in (4, 11):
                continue
            nodes_per = 4 if int(typ) == 4 else 10
            flat = np.array(nodes, dtype=np.int64).reshape(-1, nodes_per)
            # Use first 4 nodes for visualisation (works for both C3D4 and C3D10)
            for row in flat[:, :4].tolist():
                all_cells.extend([4] + [tag_to_idx[n] for n in row])
                all_grain_id.append(int(gid))

    if not all_cells:
        print('[VolumeMesh] No tet elements found in gmsh model.')
        return pv.Plotter()

    ug = pv.UnstructuredGrid(
        np.array(all_cells, dtype=np.int64),
        np.full(len(all_grain_id), pv.CellType.TETRA, dtype=np.uint8),
        node_coords,
    )
    ug.cell_data['grain_id'] = np.array(all_grain_id, dtype=np.int32)

    if twin_role:
        role_map = {'non_host': 0, 'host': 1, 'primary_twin': 2, 'secondary_twin': 3}
        ug.cell_data['role'] = np.array(
            [role_map.get(twin_role.get(g, 'non_host'), 0) for g in all_grain_id],
            dtype=np.uint8)

    pvp = pv.Plotter()
    pvp.set_background('white')
    scalar = 'role' if (twin_role and 'role' in ug.cell_data) else 'grain_id'
    cmap   = ['lightgray','steelblue','darkorange','crimson'] if scalar=='role' else 'tab20'
    pvp.add_mesh(ug.slice(normal='z'), scalars=scalar, cmap=cmap,
                 show_edges=True, edge_color='black', line_width=0.3,
                 label='Z-slice of tet mesh')
    pvp.add_title(f'Conformal tet mesh  nodes={len(node_coords)}  '
                  f'tets={len(all_grain_id)}', font_size=10)
    pvp.add_axes()
    if show:
        pvp.show()
    return pvp
