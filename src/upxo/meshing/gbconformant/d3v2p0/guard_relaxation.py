"""Re-smooth geometry-guard rollback regions under the guard's own checks.

The geometry guard pulls defective neighbourhoods back toward voxel
coordinates by a per-node weight. The step in weight at the edge of a rollback
region leaves creases. This optional stage applies further Laplacian steps to
rolled-back nodes and a margin of neighbouring nodes only, and keeps a node's
move only while every changed triangle stays free of intersections, facet
openings below minimum_facet_angle, orientation reversal or collapse relative
to the voxel triangles, and RVE-plane clearance. Fixed junction points, fixed
coordinates (RVE planes, frozen grains) and junction-line displacement limits
are kept exactly as in smoothing. A full intersection scan certifies the result.
"""
from dataclasses import replace
import numpy as np
from .surface_intersections import find_surface_intersections
from .facet_angles import small_facet_angles


def fold_counts(points, triangles, grain_pairs, thresholds=(150., 120.), exclude_grain_ids=()):
    """Number of same-interface neighbouring triangle pairs with an opening
    (180 = flat) below each threshold, optionally excluding some grains."""
    f = np.asarray(triangles)
    pairs = np.asarray(grain_pairs)
    e = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind='stable')
    two = np.flatnonzero(cnt == 2)
    start = np.r_[0, np.cumsum(cnt)][two]
    i, j = order[start] // 3, order[start + 1] // 3
    keep = np.all(pairs[i] == pairs[j], axis=1)
    if len(exclude_grain_ids):
        keep &= ~np.isin(pairs[i], list(exclude_grain_ids)).any(axis=1)
    i, j = i[keep], j[keep]
    x = np.asarray(points)[f]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    opening = 180 - np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', n[i], n[j])), 0, 1)))
    return {f'below_{t:g}': int(np.sum(opening < t)) for t in thresholds}


def check_nodes_outside(p, triangles, lower, upper):
    """Nodes of the given triangles outside the RVE clearance box."""
    nodes = np.unique(triangles)
    return nodes[np.any((p[nodes] < lower[nodes] - 1e-12) | (p[nodes] > upper[nodes] + 1e-12), axis=1)]


def relax_guarded_interfaces(surface, smoothed, dimensions, enabled=True, iterations=20,
                             relaxation=.35, junction_relaxation=.25, margin_rings=2,
                             minimum_facet_angle=1., boundary_clearance=.025,
                             junction_max_displacement=None, exclude_grain_ids=()):
    """Return (Interfaces, report) after relaxing the guard's rollback regions.

    surface is the geometry-guard output and smoothed its input; both share
    triangles, original_points, node_kind, fixed_axes and junction_edges.
    exclude_grain_ids only affects the fold counts in the report.
    """
    extent = np.asarray(dimensions, dtype=float)
    if extent.shape != (3,) or np.any(~np.isfinite(extent)) or np.any(extent <= 0):
        raise ValueError('Invalid dimensions')
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    if isinstance(margin_rings, bool) or not isinstance(margin_rings, (int, np.integer)) or margin_rings < 0:
        raise ValueError('margin_rings must be a nonnegative integer')
    if not 0 < relaxation <= 1 or not 0 < junction_relaxation <= 1:
        raise ValueError('relaxation values must lie in (0, 1]')
    if not np.isfinite(minimum_facet_angle) or not 0 <= minimum_facet_angle < 180:
        raise ValueError('minimum_facet_angle must lie in [0, 180)')
    if not np.isfinite(boundary_clearance) or boundary_clearance <= 0:
        raise ValueError('boundary_clearance must be positive')
    if junction_max_displacement is not None and (not np.isfinite(junction_max_displacement) or junction_max_displacement < 0):
        raise ValueError('junction_max_displacement must be finite and nonnegative')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    f = surface.triangles
    if not np.array_equal(f, smoothed.triangles) or not np.array_equal(surface.original_points, smoothed.original_points):
        raise ValueError('surface and smoothed must share connectivity and reference geometry')
    p = surface.points.copy()
    folds_before = fold_counts(p, f, surface.grain_pairs, exclude_grain_ids=exclude_grain_ids)
    if not enabled:
        return replace(surface, points=p), dict(enabled=False, folds_before=folds_before)
    original, kind, fixed = surface.original_points, surface.node_kind, surface.fixed_axes
    edges = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    rolled = np.linalg.norm(p - smoothed.points, axis=1) > 1e-12
    region = rolled.copy()
    for _ in range(margin_rings):
        region[np.unique(edges[region[edges].any(axis=1)])] = True
    movable = region & (kind != 3)
    # Laplacian neighbours as in smooth_interfaces: surface nodes use all edge
    # neighbours, junction-line nodes use junction-line neighbours only.
    junction = np.asarray(surface.junction_edges).reshape(-1, 2)
    neighbours = []
    for k, e in ((1, edges), (2, junction)):
        directed = np.vstack([e, e[:, ::-1]])
        neighbours.append(directed[kind[directed[:, 0]] == k])
    neighbours = np.vstack(neighbours)
    degree = np.bincount(neighbours[:, 0], minlength=len(p))
    movable &= degree > 0
    rate = np.where(kind == 2, junction_relaxation, relaxation)
    x0 = original[f]
    normals0 = np.cross(x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0])
    area0 = np.linalg.norm(normals0, axis=1)
    lower = np.minimum(original, boundary_clearance)
    upper = extent - np.minimum(extent - original, boundary_clearance)
    history = []
    for iteration in range(iterations):
        average = np.column_stack([np.bincount(neighbours[:, 0], weights=p[neighbours[:, 1], j],
                                               minlength=len(p)) for j in range(3)]) / np.maximum(degree, 1)[:, None]
        delta = np.where(movable[:, None], rate[:, None] * (average - p), 0.)
        delta[fixed] = 0.
        if junction_max_displacement is not None:
            line = kind == 2
            offset = p[line] + delta[line] - original[line]
            offset *= np.minimum(1., junction_max_displacement / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-30))
            delta[line] = original[line] + offset - p[line]
        moving = np.linalg.norm(delta, axis=1) > 1e-12
        if not moving.any():
            break
        previous = p.copy()
        p = p + delta
        reverted = 0
        moved = np.any(p != previous, axis=1)
        check = np.flatnonzero(moved[f].any(axis=1))
        while len(check):
            # Every new shape is checked once; after a rollback only triangles
            # touching rolled-back nodes have new shapes and are rechecked.
            xyz = p[f[check]]
            n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            bad_face = np.zeros(len(f), bool)
            bad_face[check] = (np.linalg.norm(n, axis=1) < .05 * area0[check]) |                 (np.einsum('ij,ij->i', n, normals0[check]) <= 0)
            hits = find_surface_intersections(p, f, triangle_ids=check)
            if len(hits):
                bad_face[np.unique(hits)] = True
            # facet openings: all triangles around the edges of the checked ones
            near = np.flatnonzero(np.isin(f, np.unique(f[check])).any(axis=1))
            folds, _ = small_facet_angles(p, f[near], minimum_facet_angle)
            if len(folds):
                bad_face[near[np.unique(folds)]] = True
            bad_node = np.zeros(len(p), bool)
            bad_node[np.unique(f[bad_face])] = True
            bad_node[check_nodes_outside(p, f[check], lower, upper)] = True
            bad_node &= moved
            if not bad_node.any():
                break                      # remaining bad faces pre-existed
            ring = bad_node.copy()
            ring[np.unique(edges[bad_node[edges].any(axis=1)])] = True
            ring &= moved
            p[ring] = previous[ring]
            moved &= ~ring
            reverted += int(ring.sum())
            check = np.flatnonzero(ring[f].any(axis=1))
        accepted = int(np.count_nonzero(np.any(p != previous, axis=1)))
        history.append(dict(iteration=iteration + 1, proposed=int(moving.sum()), accepted=accepted, reverted=reverted))
        if not accepted:
            break
    if len(find_surface_intersections(p, f)):
        raise RuntimeError('Final full intersection check failed after guard relaxation')
    remaining, _ = small_facet_angles(p, f, minimum_facet_angle)
    before_folds, _ = small_facet_angles(surface.points, f, minimum_facet_angle)
    if len(remaining) > len(before_folds):
        raise RuntimeError('Guard relaxation added facet openings below the minimum')
    folds_after = fold_counts(p, f, surface.grain_pairs, exclude_grain_ids=exclude_grain_ids)
    move = np.linalg.norm(p - surface.points, axis=1)
    return replace(surface, points=p), dict(
        enabled=True, verified=True, iterations=len(history), rolled_back_nodes=int(rolled.sum()),
        region_nodes=int(region.sum()), moved_nodes=int(np.count_nonzero(move > 1e-12)),
        maximum_move=float(move.max()), folds_before=folds_before, folds_after=folds_after,
        remaining_intersections=0, minimum_facet_angle=float(minimum_facet_angle),
        frozen_junctions_preserved=bool(np.array_equal(p[kind == 3], surface.points[kind == 3])),
        fixed_coordinates_preserved=bool(np.array_equal(p[fixed], surface.points[fixed])),
        history=history)
