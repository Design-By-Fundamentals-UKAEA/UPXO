"""
slice_metrics_2d.py

Fast numpy-only geometric metrics for a single 2-D labeled pixel slice
(e.g. one cross-section extracted from a 3-D LGI). Mirrors the vectorised
edge/vertex counting style of geom_metrics_3d.py, one dimension down.

All functions accept an integer ndarray ``slice2d`` of shape (H, W) where
each non-zero integer labels one grain, and 0 is void/background.

Return convention
------------------
All functions return 1-D ndarrays of length (max_label + 1) so that
result[label] holds the value for that label (label 0 = void, always 0).

Counting conventions
---------------------
  area_npix     : pixel count (trivial)
  perimeter_npix: number of pixel-edge pairs where this grain borders a
                  different label (both orientations)
  n_neighbours  : count of distinct edge-adjacent other labels
  n_tjp         : number of pixel-corners (shared by 4 pixels) where >= 3
                  distinct labels meet -- the 2-D analogue of a 3-D
                  junction-point vertex, i.e. a triple junction point (TJP)
                  in the plane of this slice. Only interior corners (all 4
                  neighbouring pixels exist) are considered, matching
                  geom_metrics_3d.feature_junction_points' convention.
  tjp_boundary_npix : number of grain-boundary pixel-edges whose BOTH
                  endpoints are themselves TJP corners -- i.e. boundary
                  segments that run directly between two triple junction
                  points, as opposed to grain-boundary length in general
                  (see perimeter_npix for that).

Physical scaling
-----------------
Caller multiplies by voxel_size^2, voxel_size, voxel_size^0 for area,
perimeter/tjp_boundary length, and n_neighbours/n_tjp respectively.
"""

from __future__ import annotations

import numpy as np
from typing import Dict


def label_areas(slice2d: np.ndarray) -> np.ndarray:
    """Per-label pixel count. result[0] = void pixel count."""
    return np.bincount(slice2d.ravel(), minlength=int(slice2d.max()) + 1).astype(np.int64)


def label_aspect_ratio_bbox(slice2d: np.ndarray) -> np.ndarray:
    """Per-label 2-D aspect ratio from each label's axis-aligned bounding
    box: max(extent) / min(extent) across the 2 in-plane pixel axes.
    Dimensionless (bounding-box extents scale identically on both axes for
    square pixels, so voxel_size cancels out) -- same convention as
    twinned_simple_3d's compute_aspect_ratio_bbox, one dimension down."""
    from scipy import ndimage
    max_label = int(slice2d.max())
    ar = np.zeros(max_label + 1, dtype=np.float64)
    objects = ndimage.find_objects(slice2d)
    for lbl in range(1, max_label + 1):
        sl = objects[lbl - 1] if 0 <= lbl - 1 < len(objects) else None
        if sl is None:
            continue
        extents = [s.stop - s.start for s in sl]
        ar[lbl] = float(max(extents) / min(extents))
    return ar


def _edge_diffs(slice2d: np.ndarray):
    """Return (a_h, b_h, mask_h), (a_v, b_v, mask_v) -- horizontally- and
    vertically-adjacent pixel-pairs and their differing-label masks."""
    a_h, b_h = slice2d[:, :-1], slice2d[:, 1:]      # horizontal neighbours (shape H, W-1)
    a_v, b_v = slice2d[:-1, :], slice2d[1:, :]       # vertical neighbours   (shape H-1, W)
    return (a_h, b_h, a_h != b_h), (a_v, b_v, a_v != b_v)


def label_perimeters(slice2d: np.ndarray) -> np.ndarray:
    """Per-label boundary-edge count (both pixel-edge orientations)."""
    max_label = int(slice2d.max())
    (a_h, b_h, m_h), (a_v, b_v, m_v) = _edge_diffs(slice2d)
    a_cat = np.concatenate([a_h[m_h].ravel(), a_v[m_v].ravel()])
    b_cat = np.concatenate([b_h[m_h].ravel(), b_v[m_v].ravel()])
    return np.bincount(np.concatenate([a_cat, b_cat]), minlength=max_label + 1).astype(np.int64)


def label_neighbours(slice2d: np.ndarray) -> Dict[int, frozenset]:
    """{label: frozenset(edge-adjacent neighbour labels)} -- void (0) excluded
    from keys but may appear as a value (domain-boundary exposure)."""
    (a_h, b_h, m_h), (a_v, b_v, m_v) = _edge_diffs(slice2d)
    a_cat = np.concatenate([a_h[m_h].ravel(), a_v[m_v].ravel()])
    b_cat = np.concatenate([b_h[m_h].ravel(), b_v[m_v].ravel()])
    lo, hi = np.minimum(a_cat, b_cat), np.maximum(a_cat, b_cat)
    unique_pairs = np.unique(np.column_stack([lo, hi]), axis=0)

    nb: Dict[int, set] = {}
    for lo_val, hi_val in unique_pairs.tolist():
        if lo_val != 0:
            nb.setdefault(lo_val, set()).add(hi_val)
        if hi_val != 0:
            nb.setdefault(hi_val, set()).add(lo_val)
    return {k: frozenset(v) for k, v in nb.items()}


def _n_distinct_sorted(arr: np.ndarray) -> np.ndarray:
    return 1 + (np.diff(arr, axis=-1) != 0).sum(axis=-1)


def _per_label_counts(labels_at_positions: np.ndarray, max_label: int) -> np.ndarray:
    n = len(labels_at_positions)
    if n == 0:
        return np.zeros(max_label + 1, dtype=np.int64)
    K = labels_at_positions.shape[1]
    is_first = np.concatenate([
        np.ones((n, 1), dtype=bool),
        np.diff(labels_at_positions, axis=1) != 0,
    ], axis=1)
    row_idx, col_idx = np.where(is_first)
    unique_labels = labels_at_positions[row_idx, col_idx]
    return np.bincount(unique_labels, minlength=max_label + 1).astype(np.int64)


def _tjp_corner_grid(slice2d: np.ndarray) -> np.ndarray:
    """Boolean grid, shape (H-1, W-1): corner (i, j) is True iff the 4
    pixels sharing it -- (i,j),(i,j+1),(i+1,j),(i+1,j+1) -- carry >= 3
    distinct labels. Only interior corners are ever considered (matches
    geom_metrics_3d.feature_junction_points)."""
    l00 = slice2d[:-1, :-1]; l01 = slice2d[:-1, 1:]
    l10 = slice2d[1:, :-1];  l11 = slice2d[1:, 1:]
    labels4 = np.sort(
        np.stack([l00.ravel(), l01.ravel(), l10.ravel(), l11.ravel()], axis=1), axis=1)
    n_dist = _n_distinct_sorted(labels4)
    return (n_dist >= 3).reshape(l00.shape)


def label_junction_points(slice2d: np.ndarray) -> np.ndarray:
    """Per-label triple-junction-point (TJP) count in this slice."""
    max_label = int(slice2d.max())
    l00 = slice2d[:-1, :-1]; l01 = slice2d[:-1, 1:]
    l10 = slice2d[1:, :-1];  l11 = slice2d[1:, 1:]
    labels4 = np.sort(
        np.stack([l00.ravel(), l01.ravel(), l10.ravel(), l11.ravel()], axis=1), axis=1)
    n_dist = _n_distinct_sorted(labels4)
    jp_mask = n_dist >= 3
    if not jp_mask.any():
        return np.zeros(max_label + 1, dtype=np.int64)
    return _per_label_counts(labels4[jp_mask], max_label)


def label_tjp_boundary_length(slice2d: np.ndarray) -> np.ndarray:
    """Per-label count of grain-boundary pixel-edges whose two endpoint
    corners are BOTH triple junction points -- i.e. boundary run directly
    between two TJPs, in pixel-length units."""
    max_label = int(slice2d.max())
    H, W = slice2d.shape
    jp = _tjp_corner_grid(slice2d)  # shape (H-1, W-1)

    (a_h, b_h, m_h), (a_v, b_v, m_v) = _edge_diffs(slice2d)

    # Vertical edges (between horizontally-adjacent pixels, shape (H, W-1)):
    # edge at pixel-row i, column-boundary j has endpoint corners
    # jp[i-1, j] (above) and jp[i, j] (below) -- only rows 1..H-2 have both.
    both_v = np.zeros((H, W - 1), dtype=bool)
    if H >= 3:
        both_v[1:H - 1, :] = jp[:-1, :] & jp[1:, :]
    v_hit = both_v & m_h

    # Horizontal edges (between vertically-adjacent pixels, shape (H-1, W)):
    # edge at pixel-col j, row-boundary i has endpoint corners
    # jp[i, j-1] (left) and jp[i, j] (right) -- only cols 1..W-2 have both.
    both_h = np.zeros((H - 1, W), dtype=bool)
    if W >= 3:
        both_h[:, 1:W - 1] = jp[:, :-1] & jp[:, 1:]
    h_hit = both_h & m_v

    a_cat = np.concatenate([a_h[v_hit].ravel(), a_v[h_hit].ravel()])
    b_cat = np.concatenate([b_h[v_hit].ravel(), b_v[h_hit].ravel()])
    return np.bincount(np.concatenate([a_cat, b_cat]), minlength=max_label + 1).astype(np.int64)


def compute_slice_metrics(slice2d: np.ndarray, voxel_size: float = 1.0) -> Dict[str, np.ndarray]:
    """Convenience wrapper: every per-label metric for this slice, physically
    scaled by voxel_size, plus derived circle-equivalent-diameter and
    circularity. Returns a dict of 1-D arrays, each index-aligned by label
    (index 0 = void, always 0/undefined)."""
    area_px = label_areas(slice2d)
    perim_px = label_perimeters(slice2d)
    n_tjp = label_junction_points(slice2d)
    tjp_len_px = label_tjp_boundary_length(slice2d)
    neighbours = label_neighbours(slice2d)
    aspect_ratio = label_aspect_ratio_bbox(slice2d)
    max_label = int(slice2d.max())
    n_neighbours = np.zeros(max_label + 1, dtype=np.int64)
    for lbl, nbs in neighbours.items():
        n_neighbours[lbl] = len({n for n in nbs if n != 0})

    area = area_px.astype(np.float64) * (voxel_size ** 2)
    perimeter = perim_px.astype(np.float64) * voxel_size
    tjp_boundary_length = tjp_len_px.astype(np.float64) * voxel_size
    with np.errstate(divide="ignore", invalid="ignore"):
        circle_eq_dia = np.where(area > 0, 2.0 * np.sqrt(area / np.pi), 0.0)
        circularity = np.where(perimeter > 0, 4.0 * np.pi * area / (perimeter ** 2), 0.0)

    return {
        "area": area,
        "perimeter": perimeter,
        "circle_eq_dia": circle_eq_dia,
        "circularity": circularity,
        "n_neighbours": n_neighbours.astype(np.float64),
        "n_tjp": n_tjp.astype(np.float64),
        "tjp_boundary_length": tjp_boundary_length,
        "aspect_ratio": aspect_ratio,
    }
