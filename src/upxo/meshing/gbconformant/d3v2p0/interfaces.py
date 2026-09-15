"""Extract and Laplacian-smooth internal voxel interfaces without volume meshing."""
from dataclasses import dataclass, field
from numbers import Real
import numpy as np


def upscale_labels(labels, factor=1, spacing=1.):
    """Subdivide each voxel into factor**3 equal, identically labelled voxels.

    Return (refined_labels, refined_spacing). Factor must be an integer >= 1;
    integer-valued real numbers such as 2.0 are accepted. Factor 1 returns the
    original array without copying. Label IDs and physical grain geometry are
    preserved exactly before smoothing. Memory grows cubically with factor.
    """
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
    if labels.ndim != 3 or not labels.size or labels.dtype.kind not in 'iu':
        raise ValueError('labels must be a nonempty 3D integer array')
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must be finite and positive')
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, Real) or not np.isfinite(factor) or factor < 1 or factor != int(factor):
        raise ValueError('factor must be an integer >= 1')
    factor = int(factor)
    refined = labels
    if factor > 1:
        for axis in range(3):
            refined = np.repeat(refined, factor, axis=axis)
    return refined, spacing/factor


@dataclass
class Interfaces:
    points: np.ndarray
    triangles: np.ndarray
    grain_pairs: np.ndarray
    original_points: np.ndarray
    node_kind: np.ndarray
    fixed_axes: np.ndarray
    accepted_iterations: int
    junction_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    def junction_metrics(self):
        """Turning angles at movable degree-two line nodes; straight = 0 degrees."""
        e = np.vstack([self.junction_edges, self.junction_edges[:, ::-1]])
        e = e[self.node_kind[e[:, 0]] == 2]
        e = e[np.argsort(e[:, 0], kind='stable')]
        if not len(e):
            return {'movable_nodes': 0, 'junction_edges': len(self.junction_edges)}
        grouped = e.reshape(-1, 2, 2)
        nodes = grouped[:, 0, 0]
        neighbors = grouped[:, :, 1]
        def angles(points):
            v = points[neighbors]-points[nodes, None, :]
            cosine = np.einsum('ij,ij->i', v[:, 0], v[:, 1])/np.maximum(
                np.linalg.norm(v[:, 0], axis=1)*np.linalg.norm(v[:, 1], axis=1), 1e-30)
            return np.degrees(np.arccos(np.clip(-cosine, -1, 1)))
        before, after = angles(self.original_points), angles(self.points)
        return {'movable_nodes': len(nodes), 'junction_edges': len(self.junction_edges),
                'mean_turn_before_deg': float(before.mean()), 'mean_turn_after_deg': float(after.mean()),
                'turns_over_45_before': int(np.sum(before > 45)),
                'turns_over_45_after': int(np.sum(after > 45)),
                'maximum_line_displacement': float(np.linalg.norm(self.points[nodes]-self.original_points[nodes], axis=1).max())}


def smooth_interfaces(labels, spacing=1., iterations=20, relaxation=.35,
                      junction_iterations=None, junction_relaxation=.25,
                      junction_max_displacement=None, frozen_grain_ids=()):
    """Return shared internal interface triangles; no tetrahedra or RVE caps.

    Voxel axes are x/y/z. Surface nodes average surface neighbors and junction
    nodes average junction neighbors. Junction endpoints, branches and >=4-grain
    nodes stay fixed, as do coordinates on RVE planes. Local step suppression
    and backtracking prevent triangle collapse and per-step normal reversal.
    This surface preview does not certify self-intersection-free geometry or
    preserve individual grain volumes, and differs from volume-constrained CM01.
    Junction smoothing has an independent iteration count (None uses iterations),
    relaxation and optional maximum displacement in physical units from the
    extracted geometry. Only edges incident to >=3 distinct grains are junction
    edges; endpoint label sets alone do not identify a junction edge.
    frozen_grain_ids locks every shared node incident to the selected grains
    in all three axes, preserving their voxel facets through Laplacian updates.
    """
    labels = np.asarray(labels)
    frozen_grain_ids = tuple(frozen_grain_ids)
    if not set(frozen_grain_ids) <= set(np.unique(labels)):
        raise ValueError('Unknown frozen grain ID')
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if labels.ndim != 3 or not labels.size or labels.dtype.kind not in 'iu':
        raise ValueError('labels must be a nonempty 3D integer array')
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must be finite and positive')
    if not isinstance(iterations, int) or iterations < 0 or not 0 < relaxation <= 1:
        raise ValueError('Invalid smoothing parameters')
    if junction_iterations is None:
        junction_iterations = iterations
    if not isinstance(junction_iterations, int) or junction_iterations < 0 or not 0 < junction_relaxation <= 1:
        raise ValueError('Invalid junction smoothing parameters')
    if junction_max_displacement is not None and (not np.isfinite(junction_max_displacement) or junction_max_displacement < 0):
        raise ValueError('junction_max_displacement must be finite and nonnegative')
    shape = np.array(labels.shape)+1
    face_blocks, pair_blocks = [], []
    for axis in range(3):
        lo, hi = [slice(None)]*3, [slice(None)]*3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(lo)], labels[tuple(hi)]
        indices = np.argwhere(a != b)
        if not len(indices):
            continue
        pairs = np.column_stack([a[tuple(indices.T)], b[tuple(indices.T)]])
        indices[:, axis] += 1
        other = [i for i in range(3) if i != axis]
        u, v = np.eye(3, dtype=int)[other]
        corners = indices[:, None, :] + np.array([np.zeros(3, dtype=int), u, u+v, v])
        quads = np.ravel_multi_index(corners.transpose(2, 0, 1), shape)
        face_blocks.append(quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))
        pair_blocks.append(np.repeat(pairs, 2, axis=0))
    if not face_blocks:
        return Interfaces(np.empty((0, 3)), np.empty((0, 3), dtype=int),
                          np.empty((0, 2), dtype=labels.dtype), np.empty((0, 3)),
                          np.empty(0, dtype=np.uint8), np.empty((0, 3), dtype=bool), 0)
    used, inverse = np.unique(np.vstack(face_blocks), return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    pairs = np.vstack(pair_blocks)
    lattice = np.column_stack(np.unravel_index(used, shape))
    points = lattice*spacing
    original = points.copy()
    fixed = (lattice == 0) | (lattice == shape-1)
    # Shared nodes are locked for both sides of each protected interface.
    frozen_faces = np.any(np.isin(pairs, frozen_grain_ids), axis=1)
    fixed[np.unique(triangles[frozen_faces])] = True
    signatures = [set() for _ in points]
    for tri, pair in zip(triangles, pairs):
        for node in tri:
            signatures[node].update(pair.tolist())
    kind = np.array([min(len(s)-1, 3) for s in signatures], dtype=np.uint8)
    edges, edge_inverse = np.unique(np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    edge_grains = [set() for _ in edges]
    for edge_id, pair in zip(edge_inverse, np.repeat(pairs, 3, axis=0)):
        edge_grains[edge_id].update(pair.tolist())
    junction = edges[np.array([len(s) >= 3 for s in edge_grains])]
    degree = np.bincount(junction.ravel(), minlength=len(points))
    kind[(kind == 2) & (degree != 2)] = 3
    neighbors = []
    for k, e in [(1, edges), (2, junction)]:
        directed = np.vstack([e, e[:, ::-1]])
        neighbors.append(directed[kind[directed[:, 0]] == k])
    neighbors = np.vstack(neighbors)
    degree = np.bincount(neighbors[:, 0], minlength=len(points))
    def normals(p):
        tri = p[triangles]
        return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area_floor = .05*np.linalg.norm(normals(original), axis=1)
    accepted = 0
    for iteration in range(max(iterations, junction_iterations)):
        average = np.column_stack([np.bincount(neighbors[:, 0], weights=points[neighbors[:, 1], j], minlength=len(points)) for j in range(3)])
        rate = np.zeros(len(points))
        if iteration < iterations:
            rate[kind == 1] = relaxation
        if iteration < junction_iterations:
            rate[kind == 2] = junction_relaxation
        delta = rate[:, None]*(average/np.maximum(degree[:, None], 1)-points)
        delta[(kind == 3) | (degree == 0)] = 0
        delta[fixed] = 0
        if junction_max_displacement is not None:
            line = kind == 2
            offset = points[line]+delta[line]-original[line]
            offset *= np.minimum(1., junction_max_displacement/np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-30))
            delta[line] = original[line]+offset-points[line]
        current = normals(points)
        for _ in range(20):
            proposed = normals(points+delta)
            bad = (np.linalg.norm(proposed, axis=1) < area_floor) | (np.einsum('ij,ij->i', current, proposed) <= 0)
            if not np.any(bad):
                break
            delta[np.unique(triangles[bad])] = 0
        for _ in range(20):
            proposed = normals(points+delta)
            if np.all(np.linalg.norm(proposed, axis=1) >= area_floor) and np.all(np.einsum('ij,ij->i', current, proposed) > 0):
                break
            delta *= .5
        else:
            break
        if np.linalg.norm(delta, axis=1).max() < 1e-9*spacing.min():
            break
        points += delta
        accepted += 1
    return Interfaces(points, triangles, pairs, original, kind, fixed, accepted, junction)
