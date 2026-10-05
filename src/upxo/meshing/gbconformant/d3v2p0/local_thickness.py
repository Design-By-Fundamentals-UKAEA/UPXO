"""Local thickness of labelled voxel grains.

The local thickness of a voxel is the size of the largest ball that lies
inside its grain and contains the voxel, as an odd voxel count: 1 for a slab
or bar one or two voxels thick, 3 for three or four, and so on. It uses the
same convention as thin_freeze's inscribed thickness (2*r - 1, r the largest
distance in voxels from a grain voxel centre to the nearest voxel centre
outside the grain), but per voxel: a thick lobe joined by a thin neck has a
large value in the lobe and a small value in the neck. A grain cut by an RVE
face is treated as continuing beyond it (mirrored), so the box itself does
not make grains thin; thin_freeze counts the RVE exterior as outside instead.
"""
import numpy as np


def local_thickness(labels, mirror_width=8):
    """Per-voxel local thickness (voxels, odd integers >= 1), same shape as labels.

    mirror_width: voxels of mirrored grain added beyond each RVE face a grain
    touches (thicknesses above about 2 x mirror_width are not resolved there).
    """
    from scipy.ndimage import distance_transform_edt, find_objects
    a = np.asarray(labels)
    if a.ndim != 3 or not a.size or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError('Expected nonnegative 3D integer labels')
    out = np.ones(a.shape, dtype=np.int32)
    shifted = a.astype(np.int64) + 1                       # find_objects skips label 0
    for index, box in enumerate(find_objects(shifted), start=1):
        if box is None:
            continue
        own_box = shifted[box] == index
        widths = []
        for axis, sl in enumerate(box):
            low = mirror_width if sl.start == 0 else 1
            high = mirror_width if sl.stop == a.shape[axis] else 1
            widths.append((low, high))
        mask = np.pad(own_box, widths, mode='symmetric')
        # sides not on the RVE boundary get a plain outside layer instead
        for axis, (low, high) in enumerate(widths):
            if low == 1:
                mask[(slice(None),) * axis + (slice(0, 1),)] = False
            if high == 1:
                mask[(slice(None),) * axis + (slice(-1, None),)] = False
        r = distance_transform_edt(mask)                 # 1 on boundary voxels
        thickness = np.where(mask, 1, 0).astype(np.int32)
        k = 1
        while True:
            seeds = r > k                                  # centres of balls of radius k that fit
            if not seeds.any():
                break
            reach = distance_transform_edt(~seeds)          # distance to the nearest such centre
            covered = mask & (reach <= k)
            thickness[covered] = 2 * k + 1
            k += 1
        inner = tuple(slice(low, mask.shape[k] - high) for k, (low, high) in enumerate(widths))
        region = out[box]
        region[own_box] = thickness[inner][own_box]
    return out


def node_local_thickness(labels, lattice_points, grain_ids=None, neighbourhood=2):
    """Local grain thickness at each lattice node.

    For each grain at the node, take the largest local thickness among its
    voxels within ``neighbourhood`` voxels of the node; the node value is the
    smallest of these over the node's grains. A thick grain's own corners and
    edges therefore read thick (their nearby interior is thick), while a thin
    plate or neck reads thin (none of its nearby voxels is thick).

    lattice_points: (n, 3) integer voxel-corner indices (corner (i, j, k)
    touches voxels (i-1..i, j-1..j, k-1..k)). grain_ids: one set of grain IDs
    per node (the grains whose interfaces meet there); None uses the grains of
    the eight touching voxels. Returns (n,) int.
    """
    a = np.asarray(labels)
    if isinstance(neighbourhood, bool) or not isinstance(neighbourhood, (int, np.integer)) or neighbourhood < 1:
        raise ValueError('neighbourhood must be a positive integer')
    lt = local_thickness(a)
    p = np.asarray(lattice_points, dtype=int)
    n = len(p)
    shape = np.array(a.shape)
    near = np.arange(-neighbourhood, neighbourhood)            # voxels within the window around a corner
    offsets = np.array(np.meshgrid(near, near, near, indexing='ij')).reshape(3, -1).T
    touching = np.array(np.meshgrid([-1, 0], [-1, 0], [-1, 0], indexing='ij')).reshape(3, -1).T
    if grain_ids is None:
        grain_ids = []
        for row in p:
            v = row + touching
            v = v[np.all((v >= 0) & (v < shape), axis=1)]
            grain_ids.append(set(map(int, a[tuple(v.T)])))
    width = max(len(g) for g in grain_ids) if n else 1
    table = np.full((n, width), -1, dtype=np.int64)
    for i, g in enumerate(grain_ids):
        table[i, :len(g)] = sorted(g)
    best = np.zeros((n, width), dtype=np.int64)               # per node and grain: largest nearby thickness
    for off in offsets:
        v = p + off
        inside = np.all((v >= 0) & (v < shape), axis=1)
        rows = np.flatnonzero(inside)
        owner = a[tuple(v[inside].T)]
        value = lt[tuple(v[inside].T)]
        match = table[rows] == owner[:, None]
        best[rows] = np.where(match, np.maximum(best[rows], value[:, None]), best[rows])
    best = np.where((table >= 0) & (best > 0), best, np.iinfo(np.int32).max)
    out = best.min(axis=1)
    out[out == np.iinfo(np.int32).max] = 1
    return out
