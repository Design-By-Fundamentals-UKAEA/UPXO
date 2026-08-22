"""
subsetting.py
==============
Cuboidal sub-volume extraction from a cleaned twinned 3D structure:
tile generation, cropping, and descriptive statistics for a selected
subset.

Ported from ``pxtal/twinned_simple_3d/gui/pages_subset.py`` (previously
only reachable through ``SubsetExtractionPage`` -- none of the logic
here has any GUI/Tkinter dependency).
"""

import numpy as np

_AXES = ('x', 'y', 'z')
# lgi_clean carries the raw MC array's native (nz, ny, nx) axis order --
# same convention used throughout the twinned_simple_3d pipeline (e.g.
# TwinnedSimple3DBase.lgi, assess_hosting_representativeness_2d's axis_map).
_AXIS_TO_SHAPE_IDX = {'x': 2, 'y': 1, 'z': 0}


class SubsetCleaner:
    """Minimal duck-typed stand-in for StructureCleaner3D, exposing just
    the 4 attributes downstream consumers (e.g. Abaqus export, 3D
    visualization) actually read: lgi_clean/twin_role_clean/
    twin_parent_of_clean/all_quats_clean.

    is_subset=True marks this as a produced crop rather than a genuine
    StructureCleaner3D, so callers can tell the two apart."""
    is_subset = True

    def __init__(self, lgi_clean, twin_role_clean, twin_parent_of_clean, all_quats_clean):
        self.lgi_clean = lgi_clean
        self.twin_role_clean = twin_role_clean
        self.twin_parent_of_clean = twin_parent_of_clean
        self.all_quats_clean = all_quats_clean


def _axis_intervals(n, start_pct, length_pct, stride_pct, overflow_policy):
    """Per-axis tile intervals as [(i, start_pct, end_pct), ...].

    n == 1: a single cuboid spanning [start_pct, start_pct+length_pct] --
    unless that would overflow 100%, in which case the FULL axis extent
    [0, 100] is used instead (a lone cuboid that doesn't fit becomes the
    whole RVE along that axis, rather than being subject to the
    clip/skip policy below).

    n > 1: evenly strided tiles starting at start_pct + i*stride_pct,
    each length_pct wide. A tile that would extend past 100% is either
    clipped to the boundary or dropped entirely, per overflow_policy
    ('clip' or 'skip'). A tile that starts at/past 100% is always
    dropped regardless of policy (nothing to clip).
    """
    if n <= 1:
        s, e = start_pct, start_pct + length_pct
        if e > 100.0:
            s, e = 0.0, 100.0
        return [(0, s, e)]

    out = []
    for i in range(n):
        s = start_pct + i * stride_pct
        if s >= 100.0:
            continue
        e = s + length_pct
        if e > 100.0:
            if overflow_policy == 'clip':
                e = 100.0
            else:
                continue
        if e <= s:
            continue
        out.append((i, s, e))
    return out


def generate_subset_tiles(lgi_shape, start_pct, n_cuboids, length_pct, stride_pct, overflow_policy):
    """
    Build the full set of cuboid subset tiles for a cleaned 3D structure.

    Parameters
    ----------
    lgi_shape : tuple (nz, ny, nx)
    start_pct, length_pct, stride_pct : dict {'x': float, 'y': float, 'z': float}
        Percentages of the ORIGINAL domain length along each axis.
    n_cuboids : dict {'x': int, 'y': int, 'z': int}
    overflow_policy : 'clip' or 'skip'

    Returns
    -------
    list of dict, each: {'index': (i, j, k),
                          'pct': {'x': (s, e), 'y': (s, e), 'z': (s, e)},
                          'vox': {'x': (v0, v1), 'y': (v0, v1), 'z': (v0, v1)}}
    'vox' bounds are array-index, end-exclusive, ready for direct slicing
    of an (nz, ny, nx)-shaped array.
    """
    axis_size = {a: lgi_shape[_AXIS_TO_SHAPE_IDX[a]] for a in _AXES}
    per_axis = {
        a: _axis_intervals(n_cuboids[a], start_pct[a], length_pct[a], stride_pct[a], overflow_policy)
        for a in _AXES
    }

    tiles = []
    for i, sx, ex in per_axis['x']:
        for j, sy, ey in per_axis['y']:
            for k, sz, ez in per_axis['z']:
                pct = {'x': (sx, ex), 'y': (sy, ey), 'z': (sz, ez)}
                vox = {}
                ok = True
                for a in _AXES:
                    s, e = pct[a]
                    axsz = axis_size[a]
                    v0 = int(round(s / 100.0 * axsz))
                    v1 = int(round(e / 100.0 * axsz))
                    v1 = max(v1, v0 + 1)
                    v1 = min(v1, axsz)
                    v0 = min(v0, axsz - 1)
                    if v1 <= v0:
                        ok = False
                        break
                    vox[a] = (v0, v1)
                if not ok:
                    continue
                tiles.append({'index': (i, j, k), 'pct': pct, 'vox': vox})
    return tiles


def crop_cleaner(source, tile):
    """Crop a StructureCleaner3D-like ``source`` (lgi_clean/
    twin_role_clean/twin_parent_of_clean/all_quats_clean) to ``tile``'s
    voxel bounds, returning a SubsetCleaner with only the grain ids
    present in the crop."""
    z0, z1 = tile['vox']['z']
    y0, y1 = tile['vox']['y']
    x0, x1 = tile['vox']['x']
    lgi = source.lgi_clean[z0:z1, y0:y1, x0:x1].copy()
    present = set(int(g) for g in np.unique(lgi) if g > 0)
    role = {g: v for g, v in source.twin_role_clean.items() if g in present}
    parent = {g: v for g, v in source.twin_parent_of_clean.items() if g in present}
    quats = {g: v for g, v in source.all_quats_clean.items() if g in present}
    return SubsetCleaner(lgi, role, parent, quats)


def compute_subset_stats(cropped, voxel_size, units):
    """Descriptive-statistics bundle for a cropped subset: constituent-
    region volume fractions, boundary/internal grain counts, and mean/std
    grain size (sphere-equiv. diameter), aspect ratio, and coordination
    number."""
    import cc3d
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase

    lgi = cropped.lgi_clean
    role = cropped.twin_role_clean
    total_vox = int(np.sum(lgi > 0))
    gids = sorted(int(g) for g in np.unique(lgi) if g > 0)

    # Constituent-region volume fractions
    role_keys = ('non_host', 'host', 'primary_twin', 'secondary_twin')
    role_vox = {r: 0 for r in role_keys}
    if lgi.size:
        counts = np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1)
        for gid in gids:
            r = role.get(gid, 'non_host')
            if r not in role_vox:
                r = 'non_host'
            role_vox[r] += int(counts[gid])
    vf = {r: (v / total_vox if total_vox > 0 else 0.0) for r, v in role_vox.items()}
    total_twin_vf = vf['primary_twin'] + vf['secondary_twin']

    # (1) total grain count
    n_grains = len(gids)

    # (2) grains touching the subset RVE's own outer boundary vs internal
    boundary_mask = np.zeros_like(lgi, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    boundary_gids = set(int(g) for g in np.unique(lgi[boundary_mask]) if g > 0)
    n_boundary = len(boundary_gids)
    n_internal = n_grains - n_boundary

    # (3)/(4) mean/std grain size (sphere-equiv. diameter) and aspect ratio
    base = TwinnedSimple3DBase(lgi, voxel_size, units)
    base.char_morphology(volnv=True, eqdia=True, sanv=False, force_compute=True)
    base.compute_aspect_ratio_bbox(force_compute=True)
    eqdia_vals = np.array(list(base.mprop.get('eqdia', {}).values()), dtype=float)
    ar_vals = np.array(list(base.mprop.get('aspect_ratio', {}).values()), dtype=float)

    # (5) mean/std coordination number -- unique face-adjacent neighbour
    # count per grain (a SET per grain, not an edge tally, so a pair
    # touching across multiple disjoint contact patches is still
    # counted as exactly one neighbour).
    neighbours: dict = {}
    if lgi.size:
        edges = cc3d.region_graph(lgi.astype(np.int32), connectivity=6)
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            if a > 0 and b > 0:
                neighbours.setdefault(a, set()).add(b)
                neighbours.setdefault(b, set()).add(a)
    coord_vals = np.array([len(neighbours.get(g, set())) for g in gids], dtype=float)

    def _mean_std(arr):
        if arr.size == 0:
            return (float('nan'), float('nan'))
        return (float(np.mean(arr)), float(np.std(arr)))

    return {
        'volume_fractions': vf,
        'total_twin_vf': total_twin_vf,
        'n_grains': n_grains,
        'n_boundary': n_boundary,
        'n_internal': n_internal,
        'eqdia_mean_std': _mean_std(eqdia_vals),
        'aspect_ratio_mean_std': _mean_std(ar_vals),
        'coord_mean_std': _mean_std(coord_vals),
        'units': units,
    }
