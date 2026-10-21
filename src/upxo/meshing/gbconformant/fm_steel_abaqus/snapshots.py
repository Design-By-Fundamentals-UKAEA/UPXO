"""The grain structure measured the same way at several points of the pipeline.

A :class:`Snapshot` holds, for one representation of the structure, everything the before/after comparison needs:
the grains present (raw feature labels), their volume and aspect ratio when the representation encloses them, the
area shared by every pair of neighbouring grains with its misorientation, the junction lines and the volume of the box.
Five representations can be measured:

* ``voxels``: the voxel structure of the chosen hierarchy level,
* ``surface_raw``: the grain boundary surface as first extracted from the voxels, before smoothing,
* ``surface_smoothed``: the same surface after smoothing,
* ``surface_prepared``: the closed, remeshed surface handed to the tetrahedral mesher,
* ``tets``: the tetrahedral mesh.

The two open surfaces hold the interior grain boundaries only, so they have no grain volume. Areas are those of the
interior grain boundaries (the faces of the box are not counted), the same for every representation. Lengths are in
output units, as in :mod:`.comparison`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .comparison import AFTER_TITLE, BEFORE_TITLE, ComparisonSettings, _line_summary, _units, tet_volumes
from .fidelity import _aspect, mesh_aspect, mesh_boundaries, voxel_aspect, voxel_boundaries
from .junctions import SET_WIDTH, JunctionEdges, junction_lines, tet_junction_edges, voxel_junction_edges

VOXELS, SURFACE_RAW, SURFACE_SMOOTHED, SURFACE_PREPARED, TETS = (
    'voxels', 'surface_raw', 'surface_smoothed', 'surface_prepared', 'tets')
ORDER = (VOXELS, SURFACE_RAW, SURFACE_SMOOTHED, SURFACE_PREPARED, TETS)
TITLES = {VOXELS: BEFORE_TITLE, SURFACE_RAW: 'Surface before smoothing',
          SURFACE_SMOOTHED: 'Surface after smoothing', SURFACE_PREPARED: 'Remeshed surface before tet meshing',
          TETS: AFTER_TITLE}
HAS_VOLUME = {VOXELS: True, SURFACE_RAW: False, SURFACE_SMOOTHED: False, SURFACE_PREPARED: True, TETS: True}
_TRIANGLE_EDGES = ((0, 1), (1, 2), (2, 0))


def key_of_title(title: str) -> str:
    """The snapshot key of a title (or of a key)."""
    text = str(title)
    if text in ORDER:
        return text
    for key, name in TITLES.items():
        if name == text:
            return key
    raise ValueError(f"{title!r} is not one of {', '.join(TITLES.values())}.")


@dataclass
class Snapshot:
    """The structure at one point of the pipeline; arrays aligned with ``labels`` unless said otherwise."""

    key: str
    title: str
    units: str
    labels: np.ndarray
    pair_labels: np.ndarray
    """(P, 2) raw labels of every pair of grains that share a boundary, the lower label first."""
    pair_area: np.ndarray
    """(P,) area they share, in output units squared."""
    pair_angle: np.ndarray
    """(P,) misorientation in degrees; NaN where it cannot be given."""
    lines: dict
    rve_volume: float
    volume: Optional[np.ndarray] = None
    aspect: Optional[np.ndarray] = None
    note: str = ''
    problems: dict = field(default_factory=dict)

    @property
    def has_volume(self) -> bool:
        return self.volume is not None

    def _positions(self, labels):
        return np.searchsorted(self.labels, np.asarray(labels, dtype=np.int64))

    def neighbour_count(self) -> np.ndarray:
        """Number of neighbours of every grain (grains it shares a boundary with)."""
        both = self._positions(self.pair_labels.ravel())
        return np.bincount(both, minlength=len(self.labels))

    def grain_area(self) -> np.ndarray:
        """Interior boundary area of every grain."""
        both = self._positions(self.pair_labels.ravel())
        return np.bincount(both, weights=np.repeat(self.pair_area, 2), minlength=len(self.labels))

    def mean_misorientation(self) -> np.ndarray:
        """Area-weighted mean of the misorientations across the boundaries of every grain (NaN where unavailable)."""
        ok = np.isfinite(self.pair_angle)
        out = np.full(len(self.labels), np.nan)
        if not ok.any():
            return out
        both = self._positions(self.pair_labels[ok].ravel())
        weight = np.repeat(self.pair_area[ok], 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (np.bincount(both, weights=weight * np.repeat(self.pair_angle[ok], 2), minlength=len(self.labels))
                   / np.bincount(both, weights=weight, minlength=len(self.labels)))
        return out


# --- pair misorientation -------------------------------------------------------------------------------------
def pair_misorientation(hierarchy, pairs, symmetry='cubic') -> np.ndarray:
    """Misorientation in degrees of every pair of features (NaN for a symmetry other than cubic, or a missing feature)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    out = np.full(len(pairs), np.nan)
    if symmetry != 'cubic' or not len(pairs):
        return out
    from upxo.xtalphy.crystal_orientation import cubic_misorientation

    cache = hierarchy.__dict__.setdefault('_misorientation_cache', {})
    for i, (a, b) in enumerate(pairs):
        key = (int(a), int(b))
        if key not in cache:
            fa, fb = hierarchy.features.get(key[0]), hierarchy.features.get(key[1])
            cache[key] = np.nan if fa is None or fb is None else float(cubic_misorientation(fa.euler, fb.euler)[0])
        out[i] = cache[key]
    return out


def _pairs_from_dict(pairs):
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    keys = sorted(pairs)
    return np.array(keys, dtype=np.int64), np.array([pairs[k] for k in keys], dtype=float)


def _summary(lines, scale):
    out = _line_summary(lines, scale)
    return {'length': out['length'], 'per_grain': out['per_grain'], 'per_set': out['per_set'],
            'stats': out['lines'], 'per_grain_stats': out['per_grain_total']}


def _label_array(labels):
    return np.unique(np.asarray(list(labels), dtype=np.int64))


# --- voxels -----------------------------------------------------------------------------------------------------
def voxel_snapshot(hierarchy, settings: Optional[ComparisonSettings] = None) -> Snapshot:
    """The voxel structure of the hierarchy level (raw labels)."""
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    lgi = np.asarray(hierarchy.lgi)
    labels = _label_array(k for k, n in hierarchy.voxel_counts.items() if n > 0 and k > 0)
    top = int(lgi.max()) + 1 if lgi.size else 1
    _, pair_faces = voxel_boundaries(lgi)
    pair_labels, faces = _pairs_from_dict(pair_faces)
    counts = np.array([hierarchy.voxel_counts[int(k)] for k in labels], dtype=float)
    aspect = voxel_aspect(lgi, top)
    edges = voxel_junction_edges(lgi, settings.include_rve_faces)
    return Snapshot(
        VOXELS, TITLES[VOXELS], units, labels, pair_labels, faces * per_voxel ** 2,
        pair_misorientation(hierarchy, pair_labels, settings.symmetry), _summary(junction_lines(edges), per_voxel),
        float(counts.sum()) * per_voxel ** 3, volume=counts * per_voxel ** 3, aspect=aspect[labels],
        problems={'truncated_edges': int(edges.truncated)})


# --- tetrahedra ---------------------------------------------------------------------------------------------------
def tet_snapshot(hierarchy, points, tetrahedra, resolved, settings: Optional[ComparisonSettings] = None) -> Snapshot:
    """The tetrahedral mesh; ``resolved`` is the raw label of every tetrahedron."""
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    scale = per_voxel / settings.mesh_voxel_length
    grain = np.asarray(resolved, dtype=np.int64)
    top = int(grain.max()) + 1 if grain.size else 1
    volume = np.bincount(grain, weights=tet_volumes(points, tetrahedra), minlength=top)
    labels = np.flatnonzero(np.bincount(grain, minlength=top) > 0).astype(np.int64)
    _, pairs = mesh_boundaries(points, tetrahedra, grain)
    pair_labels, area = _pairs_from_dict(pairs)
    edges = tet_junction_edges(points, tetrahedra, grain, settings.include_rve_faces)
    return Snapshot(
        TETS, TITLES[TETS], units, labels, pair_labels, area * scale ** 2,
        pair_misorientation(hierarchy, pair_labels, settings.symmetry), _summary(junction_lines(edges), scale),
        float(volume.sum()) * scale ** 3, volume=volume[labels] * scale ** 3,
        aspect=mesh_aspect(points, tetrahedra, grain, top)[labels], problems={'truncated_edges': int(edges.truncated)})


# --- surfaces -----------------------------------------------------------------------------------------------------
def surface_junction_edges(points, triangles, region_a, region_b, include_rve_faces=False) -> JunctionEdges:
    """Junction edges of a triangulated grain boundary surface.

    ``region_a``/``region_b`` are the grains on the two sides of every triangle; a triangle on the box has the same
    grain on both sides and borders the exterior (region 0), which counts only with ``include_rve_faces``.
    """
    p = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    a = np.asarray(region_a, dtype=np.int64)
    b = np.asarray(region_b, dtype=np.int64)
    exterior = a == b
    keep = ~exterior | bool(include_rve_faces)
    tri, a, b = tri[keep], a[keep], np.where(exterior[keep], 0, b[keep])
    if not len(tri):
        return JunctionEdges(np.zeros((0, 2), dtype=np.int64), np.full((0, SET_WIDTH), -1, dtype=np.int64), np.zeros(0))
    n, stride = len(p), int(max(a.max(), b.max())) + 2
    keys, regions = [], []
    for i, j in _TRIANGLE_EDGES:
        key = np.minimum(tri[:, i], tri[:, j]) * n + np.maximum(tri[:, i], tri[:, j])
        keys += [key, key]
        regions += [a + 1, b + 1]
    pair = np.unique(np.concatenate(keys) * stride + np.concatenate(regions))
    edge, region = pair // stride, pair % stride - 1
    unique_edges, inverse = np.unique(edge, return_inverse=True)
    counts = np.bincount(inverse)
    first = np.concatenate(([0], np.flatnonzero(np.diff(edge)) + 1))
    rank = np.arange(len(edge)) - first[inverse]
    fill = (counts[inverse] >= 3) & (rank < SET_WIDTH)
    sets = np.full((len(unique_edges), SET_WIDTH), -1, dtype=np.int64)
    sets[inverse[fill], rank[fill]] = region[fill]
    chosen = counts >= 3
    if not chosen.any():
        return JunctionEdges(np.zeros((0, 2), dtype=np.int64), np.full((0, SET_WIDTH), -1, dtype=np.int64), np.zeros(0))
    ends = unique_edges[chosen]
    nodes = np.column_stack((ends // n, ends % n))
    return JunctionEdges(nodes, np.sort(sets[chosen], axis=1), np.linalg.norm(p[nodes[:, 0]] - p[nodes[:, 1]], axis=1),
                         truncated=int((counts[chosen] > SET_WIDTH).sum()))


def closed_volumes(points, triangles, tet_points, tetrahedra, tet_labels):
    """Volume and aspect ratio of every grain, from the surface triangles that bound it.

    Each triangle is a face of the tetrahedra on one or both of its sides; the tetrahedron says which way the
    triangle faces into its grain, whatever the winding of the stored triangle. ``points`` are the coordinates the
    triangles are measured in (the surface), ``tet_points`` those of the tetrahedra (whose nodes ``0..len(points)``
    are the surface nodes). Returns ``(labels, volume, aspect, problems)``; ``problems['unmatched']`` counts triangles
    that are not a face of the tetrahedra (the volumes are then incomplete).
    """
    from upxo.meshing.gbconformant.pipeline.mesh_regions import tet_faces

    p = np.asarray(points, dtype=float)
    tri = np.sort(np.asarray(triangles, dtype=np.int64), axis=1)
    raw = np.asarray(triangles, dtype=np.int64)
    t = np.asarray(tetrahedra, dtype=np.int64)
    q = np.asarray(tet_points, dtype=float)
    labels = np.asarray(tet_labels, dtype=np.int64)
    n = len(q)
    if n ** 3 >= 2 ** 62:
        raise ValueError('The mesh has too many nodes to match surface triangles to tetrahedra.')
    face_nodes, first, second, _ = tet_faces(t, n)
    key = lambda f: (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    face_key = key(face_nodes)
    where = np.searchsorted(face_key, key(tri))
    where = np.minimum(where, len(face_key) - 1)
    found = face_key[where] == key(tri)
    shifted = p - p.mean(axis=0)
    v = shifted[raw]
    triple = np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0
    cov_part = (np.einsum('ni,nj->nij', v[:, 0], v[:, 0]) + np.einsum('ni,nj->nij', v[:, 1], v[:, 1])
                + np.einsum('ni,nj->nij', v[:, 2], v[:, 2])
                + np.einsum('ni,nj->nij', v.sum(axis=1), v.sum(axis=1))) / 20.0
    centre = v.sum(axis=1) / 4.0
    owners, weights, rows = [], [], []
    for side in (first, second):
        tet = side[where]
        use = found & (tet >= 0)
        which = np.flatnonzero(use)
        tet = tet[use]
        corners = t[tet]
        in_face = (corners[:, :, None] == tri[which][:, None, :]).any(axis=2)
        fourth = corners[np.arange(len(which)), np.argmin(in_face, axis=1)]
        d = q[fourth] - q[raw[which, 0]]
        facing = np.einsum('ij,ij->i', np.cross(q[raw[which, 1]] - q[raw[which, 0]], q[raw[which, 2]] - q[raw[which, 0]]), d)
        outward = np.where(facing > 0, -1.0, 1.0)         # the stored winding points into the grain when facing > 0
        owners.append(labels[tet])
        weights.append(outward * triple[which])
        rows.append(which)
    owner, weight, row = np.concatenate(owners), np.concatenate(weights), np.concatenate(rows)
    top = int(labels.max()) + 1
    volume = np.bincount(owner, weights=weight, minlength=top)
    moment = np.stack([np.bincount(owner, weights=weight * centre[row, k], minlength=top) for k in range(3)], axis=1)
    cov = np.zeros((top, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = cov[:, j, i] = np.bincount(owner, weights=weight * cov_part[row, i, j], minlength=top)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = moment / volume[:, None]
        covariance = cov / volume[:, None, None] - np.einsum('ni,nj->nij', mean, mean)
    present = np.flatnonzero(np.bincount(labels, minlength=top) > 0)
    return present, volume[present], _aspect(covariance)[present], {'unmatched': int(np.sum(~found))}


def surface_snapshot(key, hierarchy, points, triangles, region_a, region_b, settings: Optional[ComparisonSettings] = None,
                     *, tets=None, note='') -> Snapshot:
    """A grain boundary surface; ``region_a``/``region_b`` are the raw labels on the two sides of every triangle.

    A triangle on the box has the same grain on both sides. With ``tets`` (the nodes, tetrahedra and raw label of each
    tetrahedron of a mesh built on this surface) the grain volume and aspect ratio are measured too; an open surface
    has interior boundaries only and no grain volume.
    """
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    scale = per_voxel / settings.mesh_voxel_length
    p = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    a = np.asarray(region_a, dtype=np.int64)
    b = np.asarray(region_b, dtype=np.int64)
    interior = a != b
    v = p[tri[interior]]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    pairs = np.sort(np.column_stack((a[interior], b[interior])), axis=1)
    unique, inverse = (np.unique(pairs, axis=0, return_inverse=True) if len(pairs)
                       else (np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)))
    shared = np.bincount(inverse.ravel(), weights=area, minlength=len(unique)) * scale ** 2
    edges = surface_junction_edges(p, tri, a, b, settings.include_rve_faces)
    lines = _summary(junction_lines(edges), scale)
    problems = {'truncated_edges': int(edges.truncated)}
    labels = np.unique(np.concatenate((a, b))) if len(tri) else np.zeros(0, dtype=np.int64)
    volume = aspect = None
    box = float(np.prod(np.asarray(hierarchy.lgi.shape, dtype=float))) * per_voxel ** 3
    if tets is not None and len(tri):
        found_labels, found_volume, found_aspect, found = closed_volumes(p, tri, *tets)
        problems.update(found)
        if not found['unmatched']:
            labels, volume, aspect = found_labels, found_volume * scale ** 3, found_aspect
            box = float(volume.sum())
    return Snapshot(key, TITLES[key], units, labels, unique.astype(np.int64), shared,
                    pair_misorientation(hierarchy, unique, settings.symmetry), lines, box, volume=volume, aspect=aspect,
                    note=note, problems=problems)
