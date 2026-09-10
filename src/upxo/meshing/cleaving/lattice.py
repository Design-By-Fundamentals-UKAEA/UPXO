"""
lattice.py
==========
BCC (body-centered cubic) background lattice construction for the cleaving
tet-meshing module.

The BCC lattice is two interleaved point grids: "primal" vertices at integer
grid coordinates (i, j, k) and "dual" vertices at cell centres
(i+0.5, j+0.5, k+0.5), one per unit cell. Every tetrahedron in the lattice
has exactly one "long" edge (length = voxel_size, connecting two same-type
vertices -- primal-primal or dual-dual) and two "short" vertices of the
opposite type flanking it. Four tets fan around each long edge, one per
consecutive pair of the (up to four) opposite-type vertices surrounding it.

Reference: Bronson, Levine, Whitaker, "Lattice Cleaving: A Multimaterial
Tetrahedral Meshing Algorithm with Guarantees," IEEE TVCG 2013 -- "The BCC
lattice is composed of two grids of primal and dual vertices... Fanning out
from the dual vertex are 24 lattice tetrahedra, each of which spans two
lattice cells." and "The edges connecting both primal and dual vertices
differ in length than the edges connecting only primal or only dual
vertices. We refer to these edges as long and short, respectively."

Boundary handling: the lattice is built over the real (nx,ny,nz) domain
PLUS ``config.pad`` layers of extra primal cells on every side, so that
every long edge inside the real domain has full 4-vertex flanking support
(no half-starved boundary stencils). pad=1 is the minimum sufficient value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from upxo.meshing.cleaving.config import LatticeConfig

PRIMAL = 0
DUAL = 1


@dataclass
class BCCLattice:
    """
    BCC background lattice over a padded box enclosing an (nx,ny,nz) voxel
    domain.

    vertices    : (V, 3) float64 -- physical coords (axis-index * voxel_size)
    vertex_kind : (V,) int8      -- 0 = primal, 1 = dual
    tets        : (T, 4) int64   -- 0-based vertex indices, positive volume
    voxel_size  : float
    grid_shape  : (nx, ny, nz) -- extent of the REAL (unpadded) domain
    pad         : int          -- padding layers used on each side
    """
    vertices: np.ndarray
    vertex_kind: np.ndarray
    tets: np.ndarray
    voxel_size: float
    grid_shape: Tuple[int, int, int]
    pad: int


# ---------------------------------------------------------------------------
# Grid index helpers
# ---------------------------------------------------------------------------

def _primal_axis_grid(n: int, pad: int) -> Tuple[np.ndarray, int]:
    """Primal axis indices lo..(n-1+pad), inclusive. Returns (indices, lo)."""
    lo = -pad
    return np.arange(lo, n - 1 + pad + 1), lo


def _dual_axis_grid(n: int, pad: int) -> Tuple[np.ndarray, int]:
    """Dual axis indices lo..(n-2+pad), inclusive (one fewer than primal)."""
    lo = -pad
    return np.arange(lo, n - 2 + pad + 1), lo


def _lookup(id_array: np.ndarray,
            idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
            lo: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map raw axis indices (any-shape int arrays) to global vertex ids via
    id_array (a 3D lookup whose shape matches that grid's axis extents).

    Returns (ids, valid_mask); ids are meaningless wherever valid_mask is
    False (out-of-range axis indices, e.g. at the outermost padded edge).
    """
    shape = id_array.shape
    pos = [idx[d] - lo[d] for d in range(3)]
    valid = ((pos[0] >= 0) & (pos[0] < shape[0]) &
             (pos[1] >= 0) & (pos[1] < shape[1]) &
             (pos[2] >= 0) & (pos[2] < shape[2]))
    pos_c = [np.clip(pos[d], 0, shape[d] - 1) for d in range(3)]
    ids = id_array[pos_c[0], pos_c[1], pos_c[2]]
    return ids, valid


def _stack_tets(v1, m1, v2, m2, v3, m3, v4, m4) -> np.ndarray:
    """Assemble (T,4) tets from 4 vertex-id arrays, keeping fully-valid rows."""
    valid = m1 & m2 & m3 & m4
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.int64)
    return np.stack([v1[valid], v2[valid], v3[valid], v4[valid]],
                     axis=-1).astype(np.int64)


def _deduplicate_tets(tets: np.ndarray) -> np.ndarray:
    """
    Drop exact-duplicate tets (same 4 vertices, any order). Interior tets
    are generated twice -- once reachable via their primal-primal long
    edge, once via their dual-dual long edge -- so this is expected, not
    a symptom of a bug (see comment in build_bcc_lattice).
    """
    sorted_tets = np.sort(tets, axis=1)
    _, first_idx = np.unique(sorted_tets, axis=0, return_index=True)
    return tets[np.sort(first_idx)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_bcc_lattice(
        shape: Tuple[int, int, int],
        config: Optional[LatticeConfig] = None,
) -> BCCLattice:
    """
    Build the BCC background lattice over a domain padded around ``shape``.

    Parameters
    ----------
    shape  : (nx, ny, nz) -- extent of the real (unpadded) voxel domain that
             this lattice will later be labelled against (see labeling.py).
    config : LatticeConfig or None

    Returns
    -------
    BCCLattice
    """
    cfg = config or LatticeConfig()
    vs, pad = cfg.voxel_size, cfg.pad
    nx, ny, nz = shape

    px, plo_x = _primal_axis_grid(nx, pad)
    py, plo_y = _primal_axis_grid(ny, pad)
    pz, plo_z = _primal_axis_grid(nz, pad)
    Px, Py, Pz = len(px), len(py), len(pz)
    plo = (plo_x, plo_y, plo_z)

    dx, dlo_x = _dual_axis_grid(nx, pad)
    dy, dlo_y = _dual_axis_grid(ny, pad)
    dz, dlo_z = _dual_axis_grid(nz, pad)
    Dx, Dy, Dz = len(dx), len(dy), len(dz)
    dlo = (dlo_x, dlo_y, dlo_z)

    n_primal = Px * Py * Pz
    n_dual = Dx * Dy * Dz
    primal_id = np.arange(n_primal, dtype=np.int64).reshape(Px, Py, Pz)
    dual_id = (n_primal + np.arange(n_dual, dtype=np.int64)).reshape(Dx, Dy, Dz)

    # --- Vertex coordinates --------------------------------------------
    PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij')
    primal_coords = (np.stack([PX, PY, PZ], axis=-1).reshape(-1, 3)
                      * vs).astype(np.float64)

    DX, DY, DZ = np.meshgrid(dx + 0.5, dy + 0.5, dz + 0.5, indexing='ij')
    dual_coords = (np.stack([DX, DY, DZ], axis=-1).reshape(-1, 3)
                   * vs).astype(np.float64)

    vertices = np.vstack([primal_coords, dual_coords])
    vertex_kind = np.concatenate([
        np.full(n_primal, PRIMAL, dtype=np.int8),
        np.full(n_dual, DUAL, dtype=np.int8),
    ])

    all_tets = []

    # ------------------------------------------------------------------
    # Every tet has exactly two "long" edges -- the tet's pair of opposite
    # (skew, non-vertex-sharing) edges: one primal-primal, one dual-dual.
    # In the INTERIOR, fanning around dual-dual edges alone regenerates
    # every tet that primal-primal-edge fanning would also produce (each
    # tet is reachable from either of its two long edges) -- so generating
    # from only one family is enough there, and doing both would just
    # double-count in the interior.
    #
    # But near the lattice boundary the two families are NOT interchange-
    # able: a boundary dual vertex may be missing some of its 6 neighbour
    # cells (so dual-edge fanning alone starves it of tets it should have),
    # while the primal-primal edge that would reconstruct that same tet
    # from the OTHER side can still have full flanking support (the primal
    # grid boundary sits half a cell further out than the dual grid
    # boundary). Verified empirically: dual-edges-only under-covers a
    # small padded box by exactly 1/3 (216 vs. 324 tets for a (2,2,2)+pad1
    # case, each tet's volume individually correct at 1/12 -- so the
    # shortfall is missing boundary tets, not malformed ones).
    #
    # Fix: generate from BOTH families (3 primal-edge + 3 dual-edge
    # directions) and deduplicate by vertex set -- whichever family still
    # has full support at a given boundary location supplies the tet.
    # ------------------------------------------------------------------

    # -- Primal-primal X-edges: (i,j,k)-(i+1,j,k); duals vary in (j,k) ---
    I, J, K = np.meshgrid(px[:-1], py, pz, indexing='ij')
    P1, v1 = _lookup(primal_id, (I, J, K), plo)
    P2, v2 = _lookup(primal_id, (I + 1, J, K), plo)
    D1, w1 = _lookup(dual_id, (I, J - 1, K - 1), dlo)
    D2, w2 = _lookup(dual_id, (I, J, K - 1), dlo)
    D3, w3 = _lookup(dual_id, (I, J, K), dlo)
    D4, w4 = _lookup(dual_id, (I, J - 1, K), dlo)
    for Da, wa, Db, wb in ((D1, w1, D2, w2), (D2, w2, D3, w3),
                           (D3, w3, D4, w4), (D4, w4, D1, w1)):
        all_tets.append(_stack_tets(P1, v1, P2, v2, Da, wa, Db, wb))

    # -- Primal-primal Y-edges: (i,j,k)-(i,j+1,k); duals vary in (k,i) ---
    I, J, K = np.meshgrid(px, py[:-1], pz, indexing='ij')
    P1, v1 = _lookup(primal_id, (I, J, K), plo)
    P2, v2 = _lookup(primal_id, (I, J + 1, K), plo)
    D1, w1 = _lookup(dual_id, (I - 1, J, K - 1), dlo)
    D2, w2 = _lookup(dual_id, (I - 1, J, K), dlo)
    D3, w3 = _lookup(dual_id, (I, J, K), dlo)
    D4, w4 = _lookup(dual_id, (I, J, K - 1), dlo)
    for Da, wa, Db, wb in ((D1, w1, D2, w2), (D2, w2, D3, w3),
                           (D3, w3, D4, w4), (D4, w4, D1, w1)):
        all_tets.append(_stack_tets(P1, v1, P2, v2, Da, wa, Db, wb))

    # -- Primal-primal Z-edges: (i,j,k)-(i,j,k+1); duals vary in (i,j) ---
    I, J, K = np.meshgrid(px, py, pz[:-1], indexing='ij')
    P1, v1 = _lookup(primal_id, (I, J, K), plo)
    P2, v2 = _lookup(primal_id, (I, J, K + 1), plo)
    D1, w1 = _lookup(dual_id, (I - 1, J - 1, K), dlo)
    D2, w2 = _lookup(dual_id, (I, J - 1, K), dlo)
    D3, w3 = _lookup(dual_id, (I, J, K), dlo)
    D4, w4 = _lookup(dual_id, (I - 1, J, K), dlo)
    for Da, wa, Db, wb in ((D1, w1, D2, w2), (D2, w2, D3, w3),
                           (D3, w3, D4, w4), (D4, w4, D1, w1)):
        all_tets.append(_stack_tets(P1, v1, P2, v2, Da, wa, Db, wb))

    # -- Dual-dual X-edges: (i,j,k)-(i+1,j,k); shared primal face at x = i+1 --
    I, J, K = np.meshgrid(dx[:-1], dy, dz, indexing='ij')
    Q1, v1 = _lookup(dual_id, (I, J, K), dlo)
    Q2, v2 = _lookup(dual_id, (I + 1, J, K), dlo)
    R1, w1 = _lookup(primal_id, (I + 1, J, K), plo)
    R2, w2 = _lookup(primal_id, (I + 1, J + 1, K), plo)
    R3, w3 = _lookup(primal_id, (I + 1, J + 1, K + 1), plo)
    R4, w4 = _lookup(primal_id, (I + 1, J, K + 1), plo)
    for Ra, wa, Rb, wb in ((R1, w1, R2, w2), (R2, w2, R3, w3),
                           (R3, w3, R4, w4), (R4, w4, R1, w1)):
        all_tets.append(_stack_tets(Q1, v1, Q2, v2, Ra, wa, Rb, wb))

    # -- Dual-dual Y-edges: (i,j,k)-(i,j+1,k); shared primal face at y = j+1 --
    I, J, K = np.meshgrid(dx, dy[:-1], dz, indexing='ij')
    Q1, v1 = _lookup(dual_id, (I, J, K), dlo)
    Q2, v2 = _lookup(dual_id, (I, J + 1, K), dlo)
    R1, w1 = _lookup(primal_id, (I, J + 1, K), plo)
    R2, w2 = _lookup(primal_id, (I, J + 1, K + 1), plo)
    R3, w3 = _lookup(primal_id, (I + 1, J + 1, K + 1), plo)
    R4, w4 = _lookup(primal_id, (I + 1, J + 1, K), plo)
    for Ra, wa, Rb, wb in ((R1, w1, R2, w2), (R2, w2, R3, w3),
                           (R3, w3, R4, w4), (R4, w4, R1, w1)):
        all_tets.append(_stack_tets(Q1, v1, Q2, v2, Ra, wa, Rb, wb))

    # -- Dual-dual Z-edges: (i,j,k)-(i,j,k+1); shared primal face at z = k+1 --
    I, J, K = np.meshgrid(dx, dy, dz[:-1], indexing='ij')
    Q1, v1 = _lookup(dual_id, (I, J, K), dlo)
    Q2, v2 = _lookup(dual_id, (I, J, K + 1), dlo)
    R1, w1 = _lookup(primal_id, (I, J, K + 1), plo)
    R2, w2 = _lookup(primal_id, (I + 1, J, K + 1), plo)
    R3, w3 = _lookup(primal_id, (I + 1, J + 1, K + 1), plo)
    R4, w4 = _lookup(primal_id, (I, J + 1, K + 1), plo)
    for Ra, wa, Rb, wb in ((R1, w1, R2, w2), (R2, w2, R3, w3),
                           (R3, w3, R4, w4), (R4, w4, R1, w1)):
        all_tets.append(_stack_tets(Q1, v1, Q2, v2, Ra, wa, Rb, wb))

    tets = np.vstack([t for t in all_tets if len(t)])
    tets = _deduplicate_tets(tets)
    tets = _fix_tet_winding(vertices, tets)

    return BCCLattice(
        vertices=vertices, vertex_kind=vertex_kind, tets=tets,
        voxel_size=vs, grid_shape=shape, pad=pad,
    )


# ---------------------------------------------------------------------------
# Volume / winding
# ---------------------------------------------------------------------------

def signed_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of each tet: dot(cross(b-a,c-a), d-a) / 6."""
    a = vertices[tets[:, 0]]
    b = vertices[tets[:, 1]]
    c = vertices[tets[:, 2]]
    d = vertices[tets[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6.0


def _fix_tet_winding(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Swap the last two vertices of any tet with non-positive signed volume."""
    vol = signed_volumes(vertices, tets)
    bad = vol <= 0
    if np.any(bad):
        tets = tets.copy()
        tets[bad, 2], tets[bad, 3] = tets[bad, 3], tets[bad, 2]
    return tets
