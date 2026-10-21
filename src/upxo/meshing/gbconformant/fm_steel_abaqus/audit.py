"""Reports on how cleaning and meshing changed feature ownership.

``reallocation_report`` compares the raw label array with the cleaned labels
that were meshed. ``location_audit`` looks up the raw voxel under every
tetrahedron centroid and compares it with the identity the tetrahedron was
given; it stops the export when a grain's tetrahedra mostly lie over raw
features that are not part of that grain's merge history.
"""
import numpy as np

from .hierarchy import parent_tables


class IdentityError(RuntimeError):
    """A grain's tetrahedra do not lie over any of its own raw features."""


def reallocation_report(raw_labels, meshed_labels, identity, hierarchy):
    """Voxel ownership change between the raw labels and the meshed labels.

    ``meshed_labels`` is the cleaned label array in pipeline labels (before
    any upscaling), the same array the grain surfaces were extracted from.
    """
    raw = np.asarray(raw_labels)
    meshed = np.asarray(meshed_labels)
    if raw.shape != meshed.shape:
        raise ValueError('raw and meshed label arrays must have the same shape')
    lut = identity.lookup()
    if meshed.max() >= len(lut) or np.any(lut[meshed] < 0):
        raise ValueError('meshed labels contain a label with no resolved identity')
    resolved = lut[meshed]
    moved = resolved != raw
    n_moved = int(moved.sum())
    pag, pck, _, blk, _ = parent_tables(hierarchy)
    old, new = raw[moved], resolved[moved]
    lost = dict(zip(*[a.tolist() for a in np.unique(old, return_counts=True)]))
    gained = dict(zip(*[a.tolist() for a in np.unique(new, return_counts=True)]))
    per_feature = {int(label): {'raw_voxels': int(hierarchy.voxel_counts[label]),
                                'lost': int(lost.get(label, 0)),
                                'gained': int(gained.get(label, 0))}
                   for label in sorted(hierarchy.features)}
    report = {
        'voxels': int(raw.size),
        'voxels_moved': n_moved,
        'fraction_moved': n_moved / raw.size,
        'moved_across_pag': int(np.sum(pag[old] != pag[new])),
        'moved_across_packet': int(np.sum((pck[old] != pck[new]))) if pck.max() >= 0 else 0,
        'moved_across_block': int(np.sum((blk[old] != blk[new]))) if blk.max() >= 0 else 0,
        'absorbed_raw_labels': identity.report['lost_raw_labels'],
        'merged_groups': identity.report['merged_groups'],
        'per_feature': per_feature,
    }
    return report


def location_audit(points, tetrahedra, tet_grain, identity, hierarchy,
                   mesh_voxel_length=1.0, chunk=500_000):
    """Compare each tetrahedron's identity with the raw voxel under its centroid.

    ``mesh_voxel_length`` is the length of one raw voxel in mesh coordinates.
    Smoothing and cleaning legitimately move boundaries, so a mismatch
    fraction is reported per feature; only a grain whose volume lies mostly
    over raw features outside its own merge history raises ``IdentityError``.
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    g = np.asarray(tet_grain)
    lut = identity.lookup()
    if g.max() >= len(lut) or np.any(lut[g] < 0):
        raise IdentityError('a tetrahedron carries a grain with no resolved identity')
    shape = np.array(hierarchy.lgi.shape)
    assigned = lut[g]
    raw_under = np.empty(len(t), dtype=np.int64)
    volume = np.empty(len(t))
    for start in range(0, len(t), chunk):
        v = p[t[start:start + chunk]]
        volume[start:start + chunk] = np.abs(np.einsum(
            'ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))) / 6
        idx = np.clip(np.floor(v.mean(axis=1) / mesh_voxel_length).astype(np.int64), 0, shape - 1)
        raw_under[start:start + chunk] = hierarchy.lgi[idx[:, 0], idx[:, 1], idx[:, 2]]
    match = raw_under == assigned
    stride = int(max(raw_under.max(), assigned.max())) + 1
    keys, inverse = np.unique(assigned * stride + raw_under, return_inverse=True)
    share = np.bincount(inverse, weights=volume)
    pair_assigned, pair_raw = keys // stride, keys % stride
    group_of = {w: set(identity.members[m]) for m, w in identity.winner.items()}
    totals, matched, dominant = {}, {}, {}
    for a, r, s in zip(pair_assigned.tolist(), pair_raw.tolist(), share.tolist()):
        totals[a] = totals.get(a, 0.0) + s
        if r == a:
            matched[a] = s
        if s > dominant.get(a, (None, -1.0))[1]:
            dominant[a] = (r, s)
    violations = [{'label': a, 'dominant_raw_label': r, 'volume_share': s / totals[a]}
                  for a, (r, s) in dominant.items() if r not in group_of[a]]
    per_feature = {int(a): {'volume': totals[a],
                            'fraction_over_own_raw_label': matched.get(a, 0.0) / totals[a]}
                   for a in sorted(totals)}
    worst = sorted(per_feature.items(), key=lambda kv: kv[1]['fraction_over_own_raw_label'])[:10]
    report = {
        'tetrahedra': int(len(t)),
        'volume_over_own_raw_label': float(volume[match].sum() / volume.sum()),
        'worst_features': [{'label': k, **v} for k, v in worst],
        'violations': violations,
        'per_feature': per_feature,
    }
    if violations:
        raise IdentityError(
            f'{len(violations)} grain(s) lie mostly over raw features outside their '
            f'merge history (first: {violations[0]}).')
    return report
