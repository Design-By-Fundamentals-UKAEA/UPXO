"""
geom_metrics_3d.py

Fast numpy-only geometric metrics for 3-D labeled voxel arrays (LGI).

All functions accept an integer ndarray `lgi` of shape (NX, NY, NZ) where
each non-zero integer labels one closed-cell feature, and 0 is void/background.

Return convention
-----------------
All functions return 1-D ndarrays of length (max_label + 1) so that
result[label] holds the value for that label (label 0 = void, always 0).
This makes downstream indexing by label direct and branch-free.

Counting conventions
--------------------
  vol_nvox    : voxel count (trivial)
  surf_nfaces : number of voxel-face pairs where this feature borders a
                different label (including void=0, so domain-boundary
                exposure is included in surface area)
  jl_nedges   : number of voxel-edges (one edge = one unit-length segment)
                where ≥3 distinct labels meet among the 4 adjacent voxels
  jp_nverts   : number of voxel-vertices (one vertex shared by 8 voxels)
                where ≥3 distinct labels meet — each such vertex
                corresponds to a junction point on the real microstructure

Physical scaling
----------------
Caller multiplies by voxel_size^3, voxel_size^2, voxel_size, voxel_size^0
for vol, surf, jl, jp respectively.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def feature_volumes(lgi: np.ndarray) -> np.ndarray:
    """
    Per-feature voxel count.

    Returns
    -------
    ndarray, shape (max_label + 1,), dtype int64
        result[label] = voxel count for that label.
        result[0] = number of void voxels.
    """
    return np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1).astype(np.int64)


# ---------------------------------------------------------------------------
# Surface area (boundary face count)
# ---------------------------------------------------------------------------

def feature_surface_faces(lgi: np.ndarray) -> np.ndarray:
    """
    Per-feature boundary-face count including domain-boundary exposure.

    The LGI is padded by one void (0) layer on every face before counting so
    that voxels at the domain boundary accumulate faces against the padding,
    giving the correct total surface area even for truncated features.

    Returns
    -------
    ndarray, shape (max_label + 1,), dtype int64
    """
    max_label = int(lgi.max())
    padded = np.pad(lgi, pad_width=1, mode='constant', constant_values=0)
    all_a, all_b = [], []
    for axis in range(3):
        slc_lo = [slice(None)] * 3;  slc_lo[axis] = slice(None, -1)
        slc_hi = [slice(None)] * 3;  slc_hi[axis] = slice(1, None)
        a = padded[tuple(slc_lo)].ravel()
        b = padded[tuple(slc_hi)].ravel()
        diff = a != b
        all_a.append(a[diff])
        all_b.append(b[diff])
    a_cat = np.concatenate(all_a)
    b_cat = np.concatenate(all_b)
    return np.bincount(
        np.concatenate([a_cat, b_cat]),
        minlength=max_label + 1,
    ).astype(np.int64)


# ---------------------------------------------------------------------------
# Neighbour IDs
# ---------------------------------------------------------------------------

def feature_neighbours(lgi: np.ndarray) -> Dict[int, frozenset]:
    """
    Face-adjacent neighbour IDs for every non-void feature.

    Returns
    -------
    dict  {label: frozenset(neighbour_labels)}
        Void (0) is excluded from dict keys, but may appear as a value
        to signal that the feature touches the domain boundary.
        Callers that only want real-feature neighbours should filter
        ``{nb for nb in nbs if nb != 0}``.
    """
    all_pairs_a, all_pairs_b = [], []
    for axis in range(3):
        slc_lo = [slice(None)] * 3;  slc_lo[axis] = slice(None, -1)
        slc_hi = [slice(None)] * 3;  slc_hi[axis] = slice(1, None)
        a = lgi[tuple(slc_lo)].ravel()
        b = lgi[tuple(slc_hi)].ravel()
        diff = a != b
        all_pairs_a.append(a[diff])
        all_pairs_b.append(b[diff])

    a_arr = np.concatenate(all_pairs_a)
    b_arr = np.concatenate(all_pairs_b)

    # Unique unordered pairs (a <= b)
    lo = np.minimum(a_arr, b_arr)
    hi = np.maximum(a_arr, b_arr)
    unique_pairs = np.unique(np.column_stack([lo, hi]), axis=0)

    nb: Dict[int, set] = {}
    for lo_val, hi_val in unique_pairs.tolist():
        if lo_val != 0:
            nb.setdefault(lo_val, set()).add(hi_val)
        if hi_val != 0:
            nb.setdefault(hi_val, set()).add(lo_val)

    return {k: frozenset(v) for k, v in nb.items()}


# ---------------------------------------------------------------------------
# Junction line edges (triple-line segments)
# ---------------------------------------------------------------------------

def _get_edge_4_labels(
    lgi: np.ndarray, edge_axis: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the 4 voxel-label arrays adjacent to all edges parallel to
    `edge_axis`.  The 4 arrays have the same shape (derived by taking
    lo/hi slices along the two perpendicular axes).

    Edge axis 0 (x-parallel) → perpendicular axes 1, 2.
    Edge axis 1 (y-parallel) → perpendicular axes 0, 2.
    Edge axis 2 (z-parallel) → perpendicular axes 0, 1.
    """
    pa, pb = [a for a in range(3) if a != edge_axis]

    slc_ll = [slice(None)] * 3;  slc_ll[pa] = slice(None, -1);  slc_ll[pb] = slice(None, -1)
    slc_lh = [slice(None)] * 3;  slc_lh[pa] = slice(None, -1);  slc_lh[pb] = slice(1, None)
    slc_hl = [slice(None)] * 3;  slc_hl[pa] = slice(1, None);   slc_hl[pb] = slice(None, -1)
    slc_hh = [slice(None)] * 3;  slc_hh[pa] = slice(1, None);   slc_hh[pb] = slice(1, None)

    return (
        lgi[tuple(slc_ll)],
        lgi[tuple(slc_lh)],
        lgi[tuple(slc_hl)],
        lgi[tuple(slc_hh)],
    )


def _n_distinct_sorted(arr: np.ndarray) -> np.ndarray:
    """
    Given a sorted integer array of shape (..., K), return the per-position
    count of distinct values.  Uses consecutive-diff trick: O(N·K).
    """
    return 1 + (np.diff(arr, axis=-1) != 0).sum(axis=-1)


def _per_feature_counts_from_multi_label_positions(
    labels_at_positions: np.ndarray,
    max_label: int,
) -> np.ndarray:
    """
    Given an array of shape (n_positions, K) where each row contains
    the (sorted) labels at a multi-feature junction position, return
    a 1-D array counts[label] = number of positions where that label
    participates (each label counted once per position even if repeated).

    Uses vectorised diff-based de-duplication — no Python loops.
    """
    n = len(labels_at_positions)
    if n == 0:
        return np.zeros(max_label + 1, dtype=np.int64)

    K = labels_at_positions.shape[1]
    # Mark first occurrence of each distinct value per row
    is_first = np.concatenate([
        np.ones((n, 1), dtype=bool),
        np.diff(labels_at_positions, axis=1) != 0,
    ], axis=1)  # (n, K)
    row_idx, col_idx = np.where(is_first)
    unique_labels = labels_at_positions[row_idx, col_idx]
    return np.bincount(unique_labels, minlength=max_label + 1).astype(np.int64)


def feature_junction_line_edges(lgi: np.ndarray) -> np.ndarray:
    """
    Per-feature triple-line edge count.

    A triple-line edge (one voxel-edge length) has ≥3 distinct labels
    among its 4 adjacent voxels.  The count for a feature is the number
    of such edges where it is one of the participating labels.

    Three edge directions (x-, y-, z-parallel) are each enumerated.

    Returns
    -------
    ndarray, shape (max_label + 1,), dtype int64
    """
    max_label = int(lgi.max())
    total = np.zeros(max_label + 1, dtype=np.int64)

    for ea in range(3):
        l00, l01, l10, l11 = _get_edge_4_labels(lgi, ea)
        # Sort per position so _n_distinct_sorted works
        labels4 = np.sort(
            np.stack([l00.ravel(), l01.ravel(), l10.ravel(), l11.ravel()], axis=1),
            axis=1,
        )  # (n_edges, 4)
        n_dist = _n_distinct_sorted(labels4)
        triple_mask = n_dist >= 3
        if not triple_mask.any():
            continue
        total += _per_feature_counts_from_multi_label_positions(
            labels4[triple_mask], max_label
        )

    return total


# ---------------------------------------------------------------------------
# Junction point vertices
# ---------------------------------------------------------------------------

def feature_junction_points(lgi: np.ndarray) -> np.ndarray:
    """
    Per-feature junction-point vertex count.

    A junction-point vertex (shared by 8 voxels) has ≥3 distinct labels
    among those 8 voxels.  Only interior vertices (all 8 neighbours exist)
    are considered.  Each feature is counted once per vertex it participates in.

    Returns
    -------
    ndarray, shape (max_label + 1,), dtype int64
    """
    max_label = int(lgi.max())

    l000 = lgi[:-1, :-1, :-1];  l001 = lgi[:-1, :-1, 1:]
    l010 = lgi[:-1, 1:,  :-1];  l011 = lgi[:-1, 1:,  1:]
    l100 = lgi[1:,  :-1, :-1];  l101 = lgi[1:,  :-1, 1:]
    l110 = lgi[1:,  1:,  :-1];  l111 = lgi[1:,  1:,  1:]

    labels8 = np.sort(
        np.stack([
            l000.ravel(), l001.ravel(), l010.ravel(), l011.ravel(),
            l100.ravel(), l101.ravel(), l110.ravel(), l111.ravel(),
        ], axis=1),
        axis=1,
    )  # (n_interior_verts, 8)

    n_dist = _n_distinct_sorted(labels8)
    jp_mask = n_dist >= 3
    if not jp_mask.any():
        return np.zeros(max_label + 1, dtype=np.int64)

    return _per_feature_counts_from_multi_label_positions(
        labels8[jp_mask], max_label
    )


# ---------------------------------------------------------------------------
# Triple-line segment coordinate extraction
# ---------------------------------------------------------------------------

def junction_line_segments(
    lgi: np.ndarray,
    voxel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the 3-D vertex coordinates of every triple-line edge segment.

    A triple-line edge (one voxel-edge in length) has ≥3 distinct labels
    among its 4 adjacent voxels.  All 3 edge directions are enumerated.

    Coordinate convention
    ---------------------
    Corner vertex (0,0,0) of voxel (i,j,k) is at world point (i,j,k)*dx.
    Perpendicular axes  →  vertex at  (perp_boundary_idx + 1) * dx
    Edge axis           →  segment from  edge_vox_idx * dx
                                      to (edge_vox_idx + 1) * dx

    Parameters
    ----------
    lgi        : (NX, NY, NZ) integer ndarray  — feature labels, 0 = void
    voxel_size : physical side length of one voxel (isotropic)

    Returns
    -------
    starts : (n_segs, 3) float64  — start vertex of each segment
    ends   : (n_segs, 3) float64  — end   vertex of each segment
    """
    dx = float(voxel_size)
    all_starts: list = []
    all_ends:   list = []

    for ea in range(3):
        pa, pb = [a for a in range(3) if a != ea]

        # Build 4-label arrays for edges parallel to axis ea.
        # Perpendicular slices along pa (lo/hi) and pb (lo/hi):
        slc_ll = [slice(None)]*3; slc_ll[pa] = slice(None,-1); slc_ll[pb] = slice(None,-1)
        slc_lh = [slice(None)]*3; slc_lh[pa] = slice(None,-1); slc_lh[pb] = slice(1, None)
        slc_hl = [slice(None)]*3; slc_hl[pa] = slice(1, None); slc_hl[pb] = slice(None,-1)
        slc_hh = [slice(None)]*3; slc_hh[pa] = slice(1, None); slc_hh[pb] = slice(1, None)

        block = lgi[tuple(slc_ll)]   # reference slice — determines shape
        shape = block.shape
        if block.size == 0:
            continue

        labels4 = np.sort(np.stack([
            block.ravel(),
            lgi[tuple(slc_lh)].ravel(),
            lgi[tuple(slc_hl)].ravel(),
            lgi[tuple(slc_hh)].ravel(),
        ], axis=1), axis=1)  # (n_edges, 4)

        n_dist      = 1 + (np.diff(labels4, axis=-1) != 0).sum(axis=-1)
        triple_flat = n_dist >= 3
        if not triple_flat.any():
            continue

        flat_idx = np.where(triple_flat)[0]
        i0, i1, i2 = np.unravel_index(flat_idx, shape)
        # i0/i1/i2 are axis-0/1/2 indices in the sliced array.
        # Axis == ea  → voxel index  (spans [idx, idx+1]*dx along the edge)
        # Axis != ea  → inter-voxel boundary index (vertex sits at (idx+1)*dx)

        n = len(flat_idx)
        starts = np.empty((n, 3), dtype=np.float64)
        ends   = np.empty((n, 3), dtype=np.float64)

        for ax, idx_arr in enumerate((i0, i1, i2)):
            if ax == ea:
                starts[:, ax] = idx_arr * dx
                ends[:, ax]   = (idx_arr + 1) * dx
            else:
                starts[:, ax] = ends[:, ax] = (idx_arr + 1) * dx

        all_starts.append(starts)
        all_ends.append(ends)

    if not all_starts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    return np.vstack(all_starts), np.vstack(all_ends)


def junction_point_coords(
    lgi: np.ndarray,
    voxel_size: float = 1.0,
) -> np.ndarray:
    """
    Extract the 3-D world coordinates of all junction point vertices.

    A junction point vertex (shared by 8 voxels) has ≥3 distinct labels
    among those 8 voxels.  Only interior vertices (all 8 neighbours exist)
    are considered.

    Coordinate convention
    ---------------------
    An interior vertex at array position (i, j, k) (0-indexed) sits at
    the corner where 8 voxels meet:
    - voxels: (i,j,k), (i,j,k+1), (i,j+1,k), (i,j+1,k+1),
              (i+1,j,k), (i+1,j,k+1), (i+1,j+1,k), (i+1,j+1,k+1)
    - world coordinates: ((i+1)*dx, (j+1)*dx, (k+1)*dx)

    Parameters
    ----------
    lgi        : (NX, NY, NZ) integer ndarray  — feature labels
    voxel_size : physical side length of one voxel

    Returns
    -------
    ndarray of shape (n_jps, 3) float64
        World coordinates (x, y, z) of each junction point vertex.
        If no junction points exist, returns (0, 3) empty array.
    """
    dx = float(voxel_size)

    l000 = lgi[:-1, :-1, :-1];  l001 = lgi[:-1, :-1, 1:]
    l010 = lgi[:-1, 1:,  :-1];  l011 = lgi[:-1, 1:,  1:]
    l100 = lgi[1:,  :-1, :-1];  l101 = lgi[1:,  :-1, 1:]
    l110 = lgi[1:,  1:,  :-1];  l111 = lgi[1:,  1:,  1:]

    shape = l000.shape  # (NX-1, NY-1, NZ-1)
    labels8 = np.sort(
        np.stack([
            l000.ravel(), l001.ravel(), l010.ravel(), l011.ravel(),
            l100.ravel(), l101.ravel(), l110.ravel(), l111.ravel(),
        ], axis=1),
        axis=1,
    )  # (n_interior_verts, 8)

    n_dist = 1 + (np.diff(labels8, axis=-1) != 0).sum(axis=-1)
    jp_mask = n_dist >= 3

    if not jp_mask.any():
        return np.zeros((0, 3), dtype=np.float64)

    flat_idx = np.where(jp_mask)[0]
    i0, i1, i2 = np.unravel_index(flat_idx, shape)

    coords = np.empty((len(flat_idx), 3), dtype=np.float64)
    coords[:, 0] = (i0 + 1) * dx
    coords[:, 1] = (i1 + 1) * dx
    coords[:, 2] = (i2 + 1) * dx

    return coords


# ---------------------------------------------------------------------------
# All metrics in one pass
# ---------------------------------------------------------------------------

def all_metrics(lgi: np.ndarray) -> Dict[str, object]:
    """
    Compute all per-feature metrics in a single call.

    Returns
    -------
    dict with keys:
        'vol'    ndarray (max_label+1,) int64 — voxel counts
        'surf'   ndarray (max_label+1,) int64 — boundary face counts
        'jl'     ndarray (max_label+1,) int64 — triple-line edge counts
        'jp'     ndarray (max_label+1,) int64 — junction-point vertex counts
        'nb_ids' dict {label: frozenset(neighbour_labels)}
    """
    return {
        'vol':    feature_volumes(lgi),
        'surf':   feature_surface_faces(lgi),
        'jl':     feature_junction_line_edges(lgi),
        'jp':     feature_junction_points(lgi),
        'nb_ids': feature_neighbours(lgi),
    }
