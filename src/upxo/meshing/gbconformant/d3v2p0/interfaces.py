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
    smoothing_report: dict = field(default_factory=dict)

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
                      junction_max_displacement=None, frozen_grain_ids=(),
                      iterations_by_pair=None, ramp_rings=2, thickness_cap_fraction=None,
                      minimum_wedge_angle=None, minimum_corner_angle=None):
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

    Differential smoothing (all optional; defaults reproduce the above):
    iterations_by_pair maps a grain pair (a, b) to the number of Laplacian
    iterations its interface nodes receive; a node takes the smallest count of
    the pairs it touches, and counts rise by at most an equal step per ring of
    neighbours over ramp_rings rings, so no step change leaves a crease.
    thickness_cap_fraction caps each node's distance from its voxel position
    at that fraction of half the smallest local thickness (local_thickness) of
    the grains it touches, in physical units: thin necks and plates move
    little, thick regions are unaffected. minimum_wedge_angle (degrees) stops
    a node for the remaining iterations when its move would bring two
    triangles sharing an edge (a grain wedge at a junction line, or a fold)
    below that opening; such openings are reported, not repaired.
    minimum_corner_angle (degrees) does the same for patch corners: the angle
    of one interface patch at a fixed junction point (the sum of its triangle
    angles there, i.e. the angle between the two junction lines bounding it).
    A move that would bring a corner below the minimum, or make an already
    smaller corner smaller, stops the nodes involved.
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
    if isinstance(ramp_rings, bool) or not isinstance(ramp_rings, int) or ramp_rings < 0:
        raise ValueError('ramp_rings must be a nonnegative integer')
    if thickness_cap_fraction is not None and (not np.isfinite(thickness_cap_fraction) or thickness_cap_fraction <= 0):
        raise ValueError('thickness_cap_fraction must be positive')
    if minimum_wedge_angle is not None and (not np.isfinite(minimum_wedge_angle) or not 0 < minimum_wedge_angle < 90):
        raise ValueError('minimum_wedge_angle must lie in (0, 90) degrees')
    if minimum_corner_angle is not None and (not np.isfinite(minimum_corner_angle) or not 0 < minimum_corner_angle < 90):
        raise ValueError('minimum_corner_angle must lie in (0, 90) degrees')
    if iterations_by_pair is not None:
        iterations_by_pair = {tuple(sorted(map(int, k))): v for k, v in dict(iterations_by_pair).items()}
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in iterations_by_pair.values()):
            raise ValueError('iterations_by_pair values must be nonnegative integers')
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
    # Differential schedule: per-node iteration caps and displacement caps.
    total = max(iterations, junction_iterations)
    node_cap = np.full(len(points), total, dtype=np.int64)
    if iterations_by_pair:
        keys = np.sort(pairs, axis=1)
        tri_cap = np.array([iterations_by_pair.get((int(a), int(b)), total) for a, b in keys], dtype=np.int64)
        np.minimum.at(node_cap, triangles.ravel(), np.repeat(tri_cap, 3))
        if ramp_rings and node_cap.min() < total:
            step = max(1, int(np.ceil((total - node_cap.min()) / (ramp_rings + 1))))
            for _ in range(ramp_rings):
                lowest = node_cap.copy()
                np.minimum.at(lowest, edges[:, 0], node_cap[edges[:, 1]])
                np.minimum.at(lowest, edges[:, 1], node_cap[edges[:, 0]])
                node_cap = np.minimum(node_cap, lowest + step)
    node_limit = None
    if thickness_cap_fraction is not None:
        from .local_thickness import node_local_thickness
        thickness = node_local_thickness(labels, lattice, grain_ids=signatures)
        node_limit = thickness_cap_fraction * .5 * thickness * float(spacing.min())
    stopped = np.zeros(len(points), bool)
    if minimum_wedge_angle is not None:
        from .facet_angles import small_facet_angles
    if minimum_corner_angle is not None:
        # triangle corners at fixed junction points, grouped by (point, patch)
        patch_id = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)[1].ravel()
        corner_tri, corner_k = np.nonzero(kind[triangles] == 3)
        corner_key = np.unique(np.column_stack((triangles[corner_tri, corner_k], patch_id[corner_tri])),
                               axis=0, return_inverse=True)[1].ravel()
        n_keys = int(corner_key.max()) + 1 if len(corner_key) else 0

        def corner_sums(p):
            x = p[triangles[corner_tri]]
            a = x[np.arange(len(corner_tri)), corner_k]
            b = x[np.arange(len(corner_tri)), (corner_k + 1) % 3]
            c = x[np.arange(len(corner_tri)), (corner_k + 2) % 3]
            u, v = b - a, c - a
            cos = np.einsum('ij,ij->i', u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
            return np.bincount(corner_key, weights=np.degrees(np.arccos(np.clip(cos, -1, 1))), minlength=n_keys)
    accepted = 0
    for iteration in range(total):
        average = np.column_stack([np.bincount(neighbors[:, 0], weights=points[neighbors[:, 1], j], minlength=len(points)) for j in range(3)])
        rate = np.zeros(len(points))
        if iteration < iterations:
            rate[kind == 1] = relaxation
        if iteration < junction_iterations:
            rate[kind == 2] = junction_relaxation
        delta = rate[:, None]*(average/np.maximum(degree[:, None], 1)-points)
        delta[(kind == 3) | (degree == 0)] = 0
        delta[fixed] = 0
        delta[(iteration >= node_cap) | stopped] = 0
        if node_limit is not None:
            offset = points+delta-original
            offset *= np.minimum(1., node_limit[:, None]/np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-30))
            delta = original+offset-points
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
        if minimum_wedge_angle is not None or minimum_corner_angle is not None:
            # Stop nodes whose move would close an opening (or a patch corner)
            # below the minimum; recheck orientation after each change of the step.
            corner_now = corner_sums(points) if minimum_corner_angle is not None else None
            for _ in range(50):
                culprits = np.empty(0, int)
                if minimum_wedge_angle is not None:
                    folds, _ = small_facet_angles(points+delta, triangles, minimum_wedge_angle)
                    if len(folds):
                        culprits = np.unique(triangles[np.unique(folds)])
                if minimum_corner_angle is not None and n_keys:
                    after = corner_sums(points+delta)
                    closing = (after < minimum_corner_angle) & (after < corner_now - 1e-9)
                    if closing.any():
                        rows = np.flatnonzero(closing[corner_key])
                        culprits = np.union1d(culprits, np.unique(triangles[corner_tri[rows]]))
                culprits = culprits[np.linalg.norm(delta[culprits], axis=1) > 0]
                proposed = normals(points+delta)
                bad = (np.linalg.norm(proposed, axis=1) < area_floor) | (np.einsum('ij,ij->i', current, proposed) <= 0)
                flips = np.unique(triangles[bad]) if np.any(bad) else np.empty(0, int)
                flips = flips[np.linalg.norm(delta[flips], axis=1) > 0]
                if not len(culprits) and not len(flips):
                    break
                stopped[culprits] = True
                delta[culprits] = 0
                delta[flips] = 0
        if np.linalg.norm(delta, axis=1).max() < 1e-9*spacing.min():
            break
        points += delta
        accepted += 1
    report = {}
    if iterations_by_pair or node_limit is not None or minimum_wedge_angle is not None or minimum_corner_angle is not None:
        moved = np.linalg.norm(points-original, axis=1)
        report = dict(iterations_by_pair=len(iterations_by_pair or {}), ramp_rings=int(ramp_rings),
                      nodes_with_reduced_iterations=int(np.sum(node_cap < total)),
                      thickness_cap_fraction=thickness_cap_fraction,
                      nodes_at_thickness_cap=int(np.sum(moved >= node_limit-1e-9)) if node_limit is not None else 0,
                      minimum_wedge_angle=minimum_wedge_angle, minimum_corner_angle=minimum_corner_angle,
                      wedge_stopped_nodes=int(stopped.sum()))
        if minimum_corner_angle is not None and n_keys:
            report['smallest_patch_corner'] = float(corner_sums(points).min())
    return Interfaces(points, triangles, pairs, original, kind, fixed, accepted, junction, smoothing_report=report)
