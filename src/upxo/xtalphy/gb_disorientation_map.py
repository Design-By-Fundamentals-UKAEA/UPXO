"""Disorientation along the grain boundaries of a 2D orientation map.

The pixel grain map is converted to smooth boundary lines (UPXO's
geometrification, then smoothing of the boundary segments), and every line
between two grains gets the disorientation angle of that pair, so a map can
be drawn with the lines coloured by angle while the grains stay uncoloured.

Two steps, so the slow one is done once and the picture can be redrawn freely:

1. :func:`compute_gb_disorientation` -- geometrify, smooth, and find the
   disorientation of every boundary pair. Reports progress through
   :mod:`upxo._sup.progress`.
2. :func:`upxo.viz.ebsd_maps.gb_disorientation_figure` -- draw the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from upxo._sup import progress

__all__ = ["GBDisorientation", "compute_gb_disorientation", "pair_disorientation_deg",
           "classify_boundaries", "segments_and_pairs_from_rings",
           "EDGE", "GRAIN_GRAIN", "TWIN_GRAIN", "TWIN_TWIN"]

# Boundary kinds returned by classify_boundaries.
EDGE, GRAIN_GRAIN, TWIN_GRAIN, TWIN_TWIN = 0, 1, 2, 3


@dataclass
class GBDisorientation:
    """Smoothed boundary lines with their disorientation.

    segments  list of (n, 2) arrays of ``(x, y)`` in pixel units; pixel
              centres are at integer coordinates (``x`` = column, ``y`` = row).
    angles    disorientation of each segment in degrees; NaN for a line that
              lies on the edge of the map and so separates one grain only.
    pairs     (n, 2) grain ids on either side (0 where there is no neighbour).
              Ids refer to ``grain_ids`` after the 4-connected relabelling.
    """
    segments: List[np.ndarray]
    angles: np.ndarray
    pairs: np.ndarray
    shape: tuple
    n_grains: int
    smooth_passes: int
    max_smooth_levels: tuple
    original_ids: np.ndarray = None   # label in the input map of each relabelled grain (index = id)


def classify_boundaries(result: "GBDisorientation", twin_ids) -> np.ndarray:
    """Kind of every boundary segment: ``EDGE`` (lies on the map edge),
    ``GRAIN_GRAIN``, ``TWIN_GRAIN`` or ``TWIN_TWIN``.

    ``twin_ids`` are labels of the *input* map that are twins; the relabelled
    pieces of a twin grain count as twins too.
    """
    twin_ids = np.fromiter((int(t) for t in twin_ids), dtype=np.int64)
    is_twin = np.zeros(len(result.original_ids), dtype=bool)
    is_twin[1:] = np.isin(result.original_ids[1:], twin_ids)
    kinds = np.full(len(result.pairs), EDGE, dtype=int)
    shared = result.pairs[:, 1] > 0
    first = is_twin[result.pairs[shared, 0]]
    second = is_twin[result.pairs[shared, 1]]
    kinds[shared] = np.where(first & second, TWIN_TWIN,
                             np.where(first | second, TWIN_GRAIN, GRAIN_GRAIN))
    return kinds


def segments_and_pairs_from_rings(rings):
    """(segments, pairs) from a ``{gid: ring}`` dict, where each ring has a
    ``.segments`` list of smoothed boundary-segment objects produced by
    ``polygonised_grain_structure.smooth_gbsegs`` -- an interior boundary's
    segment is the SAME Python object on both owning grains' rings, so
    grouping by ``id(segment)`` (object identity, not geometry) recovers
    exactly which grain(s) own each segment without any coordinate
    comparison. This is the numerically exact way to recover a shared
    grain boundary: unlike intersecting two independently-built polygon
    boundaries (which only coincide up to floating-point precision and so
    routinely fragments into scattered sub-pixel pieces, see
    ``ebsd_smoothing.disorientation_from_polygons``'s own docstring), this
    never compares coordinates at all.

    Returns
    -------
    segments : list of (n, 2) float arrays
    pairs : (n, 2) int array
        ``(gid_a, gid_b)`` for an interior boundary shared by two grains,
        ``(gid, 0)`` for one lying on the map edge (only one owner).
    """
    owners = {}
    for gid, ring in rings.items():
        for segment in ring.segments:
            owners.setdefault(id(segment), (segment, []))[1].append(int(gid))

    segments, pairs = [], []
    for segment, grain_ids in owners.values():
        grain_ids = sorted(set(grain_ids))
        if len(grain_ids) == 2:
            pair = (grain_ids[0], grain_ids[1])
        elif len(grain_ids) == 1:
            pair = (grain_ids[0], 0)
        else:
            continue
        segments.append(np.asarray(segment.get_node_coords(), dtype=float))
        pairs.append(pair)
    return segments, np.asarray(pairs, dtype=int).reshape(-1, 2)


def pair_disorientation_deg(R_all, index_a, index_b, chunk=1500):
    """Disorientation (degrees, cubic symmetry) between grain ``index_a[k]``
    and ``index_b[k]`` for every k, from rotation matrices ``R_all``.

    Works through the pairs in chunks of ``chunk`` to bound memory, and
    reports progress.
    """
    from upxo.xtalphy.crystal_orientation import get_cubic_ops_np
    S = get_cubic_ops_np()
    out = np.empty(len(index_a), dtype=float)
    starts = range(0, len(index_a), chunk)
    for start in progress.track(starts, message="Disorientation of boundary pairs"):
        sl = slice(start, start + chunk)
        RA = R_all[index_a[sl]]
        RB = R_all[index_b[sl]]
        SA = np.einsum('sij,pjk->psik', S, RA)
        SB = np.einsum('sij,pjk->psik', S, RB)
        dR = np.einsum('pmab,pncb->pmnac', SB, SA)
        trace = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        out[sl] = np.degrees(np.arccos(cos_angle).min(axis=(1, 2)))
    return out


def compute_gb_disorientation(lfi, quat, smooth_passes: int = 2,
                              max_smooth_levels=(3, 3)) -> GBDisorientation:
    """
    Parameters
    ----------
    lfi : (ny, nx) int
        Grain label map with every pixel labelled (> 0).
    quat : (ny, nx, 4)
        Per-pixel unit quaternions ``[w, x, y, z]``.
    smooth_passes : int
        Smoothing passes over the boundary segments (1 or 2).
    max_smooth_levels : sequence of int
        Smoothing level for each pass.
    """
    import cc3d
    from upxo.gsdataops.gid_ops import find_neighs2d
    from upxo.pxtal.geometrification import polygonised_grain_structure
    from upxo.xtalphy.crystal_orientation import grain_avg_quats, quat_to_R_batch

    lfi = np.asarray(lfi)
    quat = np.asarray(quat, dtype=np.float64)
    if lfi.ndim != 2 or quat.shape[:2] != lfi.shape:
        raise ValueError("lfi must be 2D and quat must be (ny, nx, 4) on the same grid.")
    if (lfi <= 0).any():
        raise ValueError("Every pixel needs a grain label; clean the map first.")
    passes = int(smooth_passes)
    if passes not in (1, 2) or len(max_smooth_levels) < passes:
        raise ValueError("smooth_passes must be 1 or 2, with one smoothing level per pass.")

    with progress.stage("Preparing grains", 0.0, 0.08):
        progress.report(0, 3, "Relabelling disconnected pieces")
        # One id per connected piece: the geometrification builds one polygon
        # per id, so a grain split in two on the map must become two grains.
        grains = cc3d.connected_components(lfi, connectivity=4).astype(np.int32)
        gids, q_mean = grain_avg_quats(grains, quat)
        first_pixel = np.unique(grains.ravel(), return_index=True)[1]
        original_ids = np.zeros(int(grains.max()) + 1, dtype=np.int64)
        original_ids[1:] = lfi.ravel()[first_pixel]
        progress.report(1, 3, "Averaging grain orientations")
        R_all = quat_to_R_batch(q_mean)
        index_of = np.zeros(int(gids.max()) + 1, dtype=int)
        index_of[gids] = np.arange(len(gids))
        progress.report(2, 3, "Finding neighbours")
        neigh = {int(g): [int(v) for v in vs] for g, vs in find_neighs2d(grains, conn=4).items()}

    with progress.stage("Geometrifying the grain map", 0.08, 0.60):
        progress.report(None, None, "Building grain polygons and boundary segments")
        geom = polygonised_grain_structure(grains, np.unique(grains), neigh)
        geom.pix_to_geom(verbose=False)

    with progress.stage("Smoothing boundaries", 0.60, 0.70):
        progress.report(None, None, "Smoothing boundary segments")
        geom.smooth_gbsegs(geom.GB, npasses=passes,
                           max_smooth_levels=list(max_smooth_levels[:passes]),
                           plot=False, name="gbmap")
    rings = geom.smoothed["gbmap"]["GB"]
    segments, pairs = segments_and_pairs_from_rings(rings)

    angles = np.full(len(pairs), np.nan)
    shared = np.flatnonzero(pairs[:, 1] > 0)
    with progress.stage("Disorientation", 0.70, 1.0):
        if shared.size:
            unique_pairs, inverse = np.unique(pairs[shared], axis=0, return_inverse=True)
            unique_angles = pair_disorientation_deg(
                R_all, index_of[unique_pairs[:, 0]], index_of[unique_pairs[:, 1]])
            angles[shared] = unique_angles[inverse.ravel()]
        progress.report(1, 1, "Done")

    return GBDisorientation(
        segments=segments, angles=angles, pairs=pairs, shape=tuple(lfi.shape),
        n_grains=int(len(gids)), smooth_passes=passes,
        max_smooth_levels=tuple(max_smooth_levels[:passes]), original_ids=original_ids)
