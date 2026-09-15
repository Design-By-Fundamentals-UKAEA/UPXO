"""Read-only surface checks before attempting tetrahedral meshing."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def validate_tet_surfaces(surface, expected_grains=None, min_triangle_quality=.05,
                           check_intersections=False, minimum_facet_angle=.1):
    """Check a ClosedRVE; return JSON-compatible diagnostics and bad face IDs.

    Manifoldness is assessed per grain: a shared triple line is valid in the
    global complex. Vertex links catch point pinches missed by edge incidence.
    Optional intersection checks use floating-point geometric predicates.
    This is a preliminary check, not an exact intersection/clearance certification or
    a guarantee that Gmsh will produce valid tetrahedra. No geometry is modified.
    """
    if not 0 <= min_triangle_quality <= 1:
        raise ValueError('min_triangle_quality must lie between 0 and 1')
    p, f = np.asarray(surface.points), np.asarray(surface.triangles)
    pairs = np.asarray(surface.grain_pairs)
    ext, face = np.asarray(surface.exterior, dtype=bool), np.asarray(surface.rve_face)
    blockers, warnings, bad = [], [], set()
    report = dict(blockers=blockers, warnings=warnings, per_grain={},
                  unchecked=['triangle self/intersections and overlapping caps',
                             'minimum separation between nonadjacent surfaces',
                             'nested-shell containment and tetrahedron quality'])
    def finish():
        report['status'] = 'BLOCKED' if blockers else 'PRELIMINARY_CHECKS_PASSED'
        report['bad_triangle_ids'] = sorted(bad)
        report['tetrahedralisation_certified'] = False
        return report
    if (p.ndim != 2 or p.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3
            or not len(f) or pairs.shape != (len(f), 2)
            or ext.shape != (len(f),) or face.shape != (len(f),)):
        blockers.append('Invalid or empty surface array shapes'); return finish()
    if f.dtype.kind not in 'iu' or np.any(f < 0) or np.any(f >= len(p)) or not np.all(np.isfinite(p)):
        blockers.append('Invalid connectivity or nonfinite coordinates'); return finish()
    extent = np.asarray(surface.report.get('rve_dimensions', []), dtype=float)
    if extent.shape != (3,) or np.any(~np.isfinite(extent)) or np.any(extent <= 0):
        blockers.append('Missing/invalid RVE dimensions'); return finish()
    tol = max(float(extent.max()), 1.)*1e-9
    if np.any(p < -tol) or np.any(p > extent+tol):
        blockers.append('Nodes outside the RVE bounds')
    if pairs.dtype.kind not in 'iu' or not np.array_equal(ext, face >= 0) or np.any((face < -1) | (face > 5)):
        blockers.append('Invalid grain labels or RVE face metadata'); return finish()
    invalid = (ext & (pairs[:, 0] != pairs[:, 1])) | (~ext & (pairs[:, 0] == pairs[:, 1]))
    if np.any(invalid):
        blockers.append('Invalid internal/cap grain-pair ownership'); bad.update(map(int, np.flatnonzero(invalid)))
    grains = np.unique(pairs)
    if expected_grains is not None and set(map(int, grains)) != set(map(int, expected_grains)):
        blockers.append('Grain IDs differ from the input voxel dataset')
    _, inv, counts = np.unique(np.sort(f, axis=1), axis=0, return_inverse=True, return_counts=True)
    duplicate = np.flatnonzero(counts[inv] > 1)
    if len(duplicate):
        blockers.append(f'{len(duplicate)} triangles have duplicate connectivity'); bad.update(map(int, duplicate))
    used = np.unique(f)
    coincident = cKDTree(p[used]).query_pairs(tol, output_type='ndarray')
    report['near_coincident_node_pairs'] = len(coincident)
    if len(coincident):
        blockers.append(f'{len(coincident)} independent node pairs are coincident within tolerance')
    xyz = p[f]
    normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    area2 = np.linalg.norm(normals, axis=1)
    lengths2 = np.sum((xyz[:, [1, 2, 0]]-xyz)**2, axis=2).sum(axis=1)
    quality = np.divide(2*np.sqrt(3)*area2, lengths2, out=np.zeros(len(f)), where=lengths2 > 0)
    degenerate = np.flatnonzero(area2 <= tol**2)
    if len(degenerate):
        blockers.append(f'{len(degenerate)} degenerate triangles'); bad.update(map(int, degenerate))
    poor = np.flatnonzero(quality < min_triangle_quality)
    if len(poor):
        warnings.append(f'{len(poor)} triangles below quality threshold {min_triangle_quality}')
        bad.update(map(int, poor))
    report['triangle_quality'] = dict(minimum=float(quality.min()),
        percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
        poor_count=len(poor), threshold=float(min_triangle_quality),
        definition='4*sqrt(3)*area / sum(squared edge lengths); equilateral=1')
    def topology(tri):
        edges = tri[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        direction = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
        return unique, inverse, counts, direction
    volumes = []
    for gid in grains:
        ids = np.flatnonzero(np.any(pairs == gid, axis=1))
        tri = f[ids].copy()
        reverse = ~ext[ids] & (pairs[ids, 1] == gid)
        tri[reverse] = tri[reverse][:, [0, 2, 1]]
        edges, inverse, counts, direction = topology(tri)
        broken = (counts != 2) | (direction != 0)
        bad.update(map(int, ids[np.unique(np.flatnonzero(broken[inverse])//3)]))
        # A closed manifold vertex link is one connected cycle.
        links = {}
        for a, b, c in tri.tolist():
            links.setdefault(a, []).append((b, c))
            links.setdefault(b, []).append((c, a))
            links.setdefault(c, []).append((a, b))
        bad_vertices = []
        for node, link in links.items():
            neighbours = {}
            for a, b in link:
                neighbours.setdefault(a, []).append(b); neighbours.setdefault(b, []).append(a)
            visited, stack = set(), [next(iter(neighbours))]
            while stack:
                a = stack.pop()
                if a not in visited:
                    visited.add(a); stack.extend(b for b in neighbours[a] if b not in visited)
            if len(visited) != len(neighbours) or any(len(v) != 2 for v in neighbours.values()):
                bad_vertices.append(node)
        if bad_vertices:
            bad.update(map(int, ids[np.any(np.isin(tri, bad_vertices), axis=1)]))
        nodes, local = np.unique(tri, return_inverse=True)
        local = local.reshape(-1, 3)
        e = local[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
        components = connected_components(graph, directed=False, return_labels=False)
        xyz = p[tri]
        volume = float(np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()/6)
        volumes.append(volume)
        item = dict(open_edges=int(np.sum(counts == 1)), nonmanifold_edges=int(np.sum(counts > 2)),
                    inconsistent_oriented_edges=int(np.sum(direction != 0)),
                    nonmanifold_vertices=len(bad_vertices), connected_components=int(components), signed_volume=volume)
        report['per_grain'][str(int(gid))] = item
        if np.any(broken) or bad_vertices or volume <= tol**3:
            blockers.append(f'Grain {gid}: invalid boundary topology/orientation or nonpositive volume')
        if components > 1:
            warnings.append(f'Grain {gid}: {components} separate shells; containment needs checking')
    _, _, counts, direction = topology(f[ext])
    if not len(counts) or np.any(counts != 2) or np.any(direction != 0):
        blockers.append('Exterior shell is open, nonmanifold, or inconsistently oriented')
    report['rve_faces'] = {}
    for k in range(6):
        ids = np.flatnonzero(face == k)
        axis, side = divmod(k, 2)
        error = float(np.max(np.abs(p[f[ids], axis]-side*extent[axis]))) if len(ids) else None
        area = float(area2[ids].sum()/2)
        expected = float(np.prod(np.delete(extent, axis)))
        valid = (len(ids) > 0 and error <= tol and np.isclose(area, expected, rtol=1e-8)
                 and np.all(normals[ids, axis]*(1 if side else -1) > 0))
        report['rve_faces'][str(k)] = dict(valid=bool(valid), area=area, expected_area=expected, plane_error=error)
        if not valid:
            blockers.append(f'RVE face {k}: missing, nonplanar, wrong area, or inward triangles'); bad.update(map(int, ids))
    report['total_signed_volume'] = sum(volumes)
    report['expected_rve_volume'] = float(np.prod(extent))
    if not np.isclose(sum(volumes), np.prod(extent), rtol=1e-8):
        blockers.append('Total grain signed volume differs from RVE volume')
    report['coordinate_tolerance'] = tol
    report['grains'] = len(grains)
    from .facet_angles import small_facet_angles
    folds,angles=small_facet_angles(p,f,minimum_facet_angle)
    report['small_facet_angle_pairs']=folds.tolist()
    report['minimum_facet_angle_threshold']=float(minimum_facet_angle)
    report['smallest_flagged_facet_angle']=float(angles.min()) if len(angles) else None
    if len(folds):
        blockers.append(f'{len(folds)} facet pairs have opening angles below {minimum_facet_angle} degrees')
        bad.update(map(int,np.unique(folds)))
    report['intersection_check_performed'] = bool(check_intersections and not len(degenerate))
    if report['intersection_check_performed']:
        from .surface_intersections import find_surface_intersections
        crossings = find_surface_intersections(p, f, tolerance=tol)
        report['intersecting_triangle_pairs'] = crossings.tolist()
        report['unchecked'].remove('triangle self/intersections and overlapping caps')
        if len(crossings):
            blockers.append(f'{len(crossings)} surface triangle intersections (PLC conflicts)')
            bad.update(map(int, np.unique(crossings)))
    return finish()
