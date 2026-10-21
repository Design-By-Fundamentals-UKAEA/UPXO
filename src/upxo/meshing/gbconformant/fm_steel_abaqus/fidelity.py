"""Per-feature fidelity of the mesh to the raw voxel structure.

For every feature of the chosen hierarchy level, the raw voxels and the resolved mesh are measured
the same way: volume, boundary area, number of neighbours and aspect ratio. A feature whose volume
shrank by more than a limit is flagged. Lengths are in output units.

The voxel boundary area is the area of the stepped voxel faces, which exceeds the area of a smooth
surface enclosing the same voxels; the ratio of the two areas is a measure of that, not a defect.
Neighbours are features that share a face (a voxel face, or a triangle of the mesh).
"""
from __future__ import annotations

import numpy as np

from .comparison import ComparisonSettings, _units, describe, tet_volumes

_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _check(hierarchy, tet_grain, tetrahedra):
    grains = np.asarray(tet_grain)
    if grains.shape != (len(tetrahedra),):
        raise ValueError("tet_grain needs one resolved label per tetrahedron.")
    return grains.astype(np.int64, copy=False)


# --- voxel side ----------------------------------------------------------------------------------------
def voxel_boundaries(labels):
    """``(area_faces, pairs)``: exposed voxel faces per label (box faces included) and neighbour pair counts.

    ``pairs`` maps ``(a, b)`` with ``a < b`` to the number of faces the two labels share. Label 0 is
    treated as outside every feature.
    """
    a = np.asarray(labels).astype(np.int64, copy=False)
    top = int(a.max()) + 1 if a.size else 1
    faces = np.zeros(top, dtype=np.int64)
    pair_parts = []
    for axis in range(3):
        below, above = [slice(None)] * 3, [slice(None)] * 3
        below[axis], above[axis] = slice(None, -1), slice(1, None)
        lo = a[tuple(below)].ravel()
        hi = a[tuple(above)].ravel()
        differ = lo != hi
        faces += np.bincount(lo[differ], minlength=top) + np.bincount(hi[differ], minlength=top)
        pair_parts.append(np.sort(np.column_stack((lo[differ], hi[differ])), axis=1))
        faces += np.bincount(np.take(a, 0, axis=axis).ravel(), minlength=top)       # the two box faces
        faces += np.bincount(np.take(a, -1, axis=axis).ravel(), minlength=top)
    pairs = np.vstack(pair_parts) if pair_parts else np.zeros((0, 2), dtype=np.int64)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return faces, {(int(x), int(y)): int(n) for (x, y), n in zip(unique, counts)}


def _moments(weights, coords, labels, top):
    """Weighted count, centroid and covariance of points grouped by label."""
    w = np.bincount(labels, weights=weights, minlength=top)
    sums = [np.bincount(labels, weights=weights * coords[:, i], minlength=top) for i in range(3)]
    prods = {(i, j): np.bincount(labels, weights=weights * coords[:, i] * coords[:, j], minlength=top)
             for i in range(3) for j in range(i, 3)}
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = np.stack([s / w for s in sums], axis=1)
        cov = np.zeros((top, 3, 3))
        for (i, j), value in prods.items():
            cov[:, i, j] = cov[:, j, i] = value / w - centre[:, i] * centre[:, j]
    return w, centre, cov


def _aspect(cov):
    """Largest over smallest principal semi-axis of covariance matrices (NaN where undefined)."""
    out = np.full(len(cov), np.nan)
    ok = np.isfinite(cov).all(axis=(1, 2))
    if ok.any():
        values = np.linalg.eigvalsh(cov[ok])
        low = np.maximum(values[:, 0], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.sqrt(values[:, 2] / low)
    return out


def voxel_aspect(labels, top):
    a = np.asarray(labels).astype(np.int64, copy=False)
    index = np.indices(a.shape).reshape(3, -1).T.astype(float)
    _, _, cov = _moments(np.ones(a.size), index, a.ravel(), top)
    cov = cov + np.eye(3) / 12.0                      # a voxel is a cube, not a point
    return _aspect(cov)


# --- mesh side -------------------------------------------------------------------------------------------
def mesh_boundaries(points, tetrahedra, tet_grain):
    """``(area, pairs)``: boundary area per label (box faces included) and shared area per neighbour pair.

    A triangle is a boundary of a feature when the tetrahedra on its two sides carry different labels,
    or when only one tetrahedron has it (a face of the box). Areas are in mesh coordinates.
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    grain = np.asarray(tet_grain, dtype=np.int64)
    top = int(grain.max()) + 1 if grain.size else 1
    tri = np.sort(t[:, _FACES].reshape(-1, 3), axis=1)
    owner = np.repeat(grain, 4)
    order = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    tri, owner = tri[order], owner[order]
    same_as_next = np.all(tri[1:] == tri[:-1], axis=1)
    if len(same_as_next) > 1 and np.any(same_as_next[1:] & same_as_next[:-1]):
        raise ValueError("A triangle belongs to three tetrahedra: the mesh is not conforming.")
    first_of_pair = np.flatnonzero(same_as_next)
    in_pair = np.zeros(len(tri), dtype=bool)
    in_pair[first_of_pair] = in_pair[first_of_pair + 1] = True
    single = np.flatnonzero(~in_pair)
    differ = first_of_pair[owner[first_of_pair] != owner[first_of_pair + 1]]

    def areas(rows):
        v = p[tri[rows]]
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    area = np.zeros(top)
    area += np.bincount(owner[single], weights=areas(single), minlength=top)
    a_diff = areas(differ)
    area += np.bincount(owner[differ], weights=a_diff, minlength=top)
    area += np.bincount(owner[differ + 1], weights=a_diff, minlength=top)
    pair_labels = np.sort(np.column_stack((owner[differ], owner[differ + 1])), axis=1)
    unique, inverse = np.unique(pair_labels, axis=0, return_inverse=True)
    shared = np.bincount(inverse.ravel(), weights=a_diff)
    return area, {(int(x), int(y)): float(s) for (x, y), s in zip(unique, shared)}


def mesh_aspect(points, tetrahedra, tet_grain, top):
    """Aspect ratio of each label from the second moments of its tetrahedra (volume-weighted)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    grain = np.asarray(tet_grain, dtype=np.int64)
    v = p[t]
    volume = tet_volumes(p, t)
    centroid = v.mean(axis=1)
    offsets = v - centroid[:, None, :]
    own = np.einsum("nki,nkj->nij", offsets, offsets) * (volume / 20.0)[:, None, None]   # about each centroid
    w, centre, cov = _moments(volume, centroid, grain, top)
    for i in range(3):
        for j in range(i, 3):
            extra = np.bincount(grain, weights=own[:, i, j], minlength=top)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov[:, i, j] += extra / w
                if i != j:
                    cov[:, j, i] = cov[:, i, j]
    return _aspect(cov)


# --- put together -----------------------------------------------------------------------------------------
def _neighbours(pairs):
    out = {}
    for a, b in pairs:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


COLUMNS = ("label", "volume_before", "volume_after", "volume_change", "area_before", "area_after",
           "area_ratio", "neighbours_before", "neighbours_after", "neighbours_lost",
           "neighbours_gained", "aspect_before", "aspect_after", "flag")


def grain_fidelity(hierarchy, points, tetrahedra, tet_grain, settings=None, *, shrink_limit=0.10):
    """Volume, boundary area, neighbours and aspect ratio of every feature, before and after.

    ``tet_grain`` is the resolved raw label of each tetrahedron. ``shrink_limit`` is the fraction of
    volume a feature may lose before it is flagged ``shrunk``; a feature with no tetrahedra is ``lost``.
    Returns ``{'units', 'columns', 'rows', 'stats', 'shrink_limit', 'flagged', 'lost'}``.
    """
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    scale = per_voxel / settings.mesh_voxel_length
    grain = _check(hierarchy, tet_grain, tetrahedra)
    lgi = np.asarray(hierarchy.lgi)
    labels = sorted(set(int(k) for k in hierarchy.features) | set(int(g) for g in np.unique(grain)))
    top = max(max(labels, default=0), int(lgi.max()) if lgi.size else 0, int(grain.max()) if grain.size else 0) + 1

    v_faces, v_pairs = voxel_boundaries(lgi)
    v_aspect = voxel_aspect(lgi, top)
    counts = np.bincount(lgi.ravel().astype(np.int64), minlength=top)
    m_area, m_pairs = mesh_boundaries(points, tetrahedra, grain)
    m_aspect = mesh_aspect(points, tetrahedra, grain, top)
    m_volume = np.bincount(grain, weights=tet_volumes(points, tetrahedra), minlength=top)
    present = np.bincount(grain, minlength=top) > 0
    before_n, after_n = _neighbours(v_pairs), _neighbours(m_pairs)

    def padded(array, label, default=np.nan):
        return float(array[label]) if label < len(array) else default

    rows = []
    for label in labels:
        vb = counts[label] * per_voxel ** 3 if label < len(counts) else 0.0
        ab = padded(v_faces, label, 0) * per_voxel ** 2
        has = bool(label < len(present) and present[label])
        va = float(m_volume[label]) * scale ** 3 if has else float("nan")
        aa = padded(m_area, label) * scale ** 2 if has else float("nan")
        nb, na = before_n.get(label, set()), after_n.get(label, set()) if has else set()
        change = (va / vb - 1.0) if (has and vb > 0) else float("nan")
        flag = "lost" if not has and vb > 0 else ("shrunk" if has and change < -float(shrink_limit) else "")
        rows.append([label, vb, va, change, ab, aa, (aa / ab) if (has and ab > 0) else float("nan"),
                     len(nb), len(na) if has else float("nan"), len(nb - na) if has else float("nan"),
                     len(na - nb) if has else float("nan"), padded(v_aspect, label),
                     padded(m_aspect, label) if has else float("nan"), flag])
    table = {name: [row[i] for row in rows] for i, name in enumerate(COLUMNS)}
    stats = {name: describe([v for v in table[name] if v == v]) for name in
             ("volume_change", "area_ratio", "neighbours_lost", "neighbours_gained")}
    stats["aspect_change"] = describe([a / b for a, b in zip(table["aspect_after"], table["aspect_before"])
                                       if a == a and b == b and b > 0])
    return {"units": units, "columns": list(COLUMNS), "rows": rows, "stats": stats,
            "shrink_limit": float(shrink_limit),
            "flagged": [r[0] for r in rows if r[-1] == "shrunk"], "lost": [r[0] for r in rows if r[-1] == "lost"]}
