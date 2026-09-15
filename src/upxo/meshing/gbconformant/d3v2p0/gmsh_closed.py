"""Joint remeshing of labelled internal interfaces and planar RVE patches."""
from types import SimpleNamespace
import numpy as np
from .gmsh_interfaces import remesh_interfaces_gmsh
from .rve_caps import ClosedRVE


def remesh_closed_rve_gmsh(surface, mesh_size=.75, algorithm=6, verbose=False,
                         check_intersections=False, intersection_retries=2,
                         max_chart_triangles=None, minimum_facet_angle=.1,
                         frozen_grain_ids=()):
    """Regenerate every 2D patch in one Gmsh model, keeping shared curves.

    Input is a ClosedRVE with oriented internal triangles and outward caps.
    Grain/face keys keep perpendicular box faces separate. Cap interior nodes
    are free to remesh; common patch nodes (including point contacts) stay fixed.
    Nonmanifold grain edges are reported, not repaired.
    """
    if check_intersections:
        from .facet_angles import small_facet_angles
        folds,_=small_facet_angles(surface.points,surface.triangles,minimum_facet_angle)
        if len(folds):
            raise ValueError(f'Input closed surface has {len(folds)} openings below {minimum_facet_angle} degrees; repair its geometry before remeshing')
    keys = np.column_stack((surface.grain_pairs, surface.rve_face))
    _, patch = np.unique(keys, axis=0, return_inverse=True)
    memberships = np.unique(np.column_stack((surface.triangles.ravel(),
                                            np.repeat(patch, 3))), axis=0)
    counts = np.bincount(memberships[:, 0], minlength=len(surface.points))
    source = SimpleNamespace(points=surface.points, triangles=surface.triangles,
                             grain_pairs=surface.grain_pairs,
                             node_kind=np.where(counts > 1, 3, 1),
                             fixed_axes=np.zeros_like(surface.points, dtype=bool))
    result = remesh_interfaces_gmsh(source, mesh_size, verbose, algorithm,
                                    check_intersections=check_intersections,
                                    intersection_retries=intersection_retries,
                                    max_chart_triangles=max_chart_triangles,
                                    minimum_facet_angle=minimum_facet_angle,
                                    rve_dimensions=surface.report['rve_dimensions'],
                                    _patch_keys=keys, _oriented=True,
                                    frozen_grain_ids=frozen_grain_ids)
    pairs, face_ids = result.grain_pairs[:, :2], result.grain_pairs[:, 2]
    exterior = face_ids >= 0
    points, faces = result.points, result.triangles
    extent = np.asarray(surface.report['rve_dimensions'])
    tol = 1e-9*max(1., float(extent.max()))
    for face_id in range(6):
        selected = face_ids == face_id
        if not np.any(selected):
            raise RuntimeError(f'Missing RVE face {face_id}')
        axis, side = divmod(face_id, 2)
        p = points[faces[selected]]
        if np.any(np.abs(p[:, :, axis]-side*extent[axis]) > tol):
            raise RuntimeError('Remeshed cap left its RVE plane')
        normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        ids = np.flatnonzero(selected)[normal[:, axis]*(1 if side else -1) < 0]
        faces[ids] = faces[ids][:, [0, 2, 1]]
        if not np.isclose(np.linalg.norm(normal, axis=1).sum()/2,
                          np.prod(np.delete(extent, axis)), rtol=1e-8):
            raise RuntimeError('Remeshed caps do not cover an RVE face')
    def edge_counts(f):
        return np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                         axis=0, return_counts=True)[1]
    if np.any(edge_counts(faces[exterior]) != 2):
        raise RuntimeError('Remeshing opened the exterior shell')
    volumes, nonmanifold = {}, []
    for gid in np.unique(surface.grain_pairs):
        selected = np.any(pairs == gid, axis=1)
        if not np.any(selected):
            raise RuntimeError(f'Remeshing lost grain {gid}')
        counts = edge_counts(faces[selected])
        if np.any(counts % 2):
            raise RuntimeError(f'Remeshing opened grain {gid}')
        if np.any(counts != 2):
            nonmanifold.append(int(gid))
        p = points[faces[selected]]
        sign = np.where(~exterior[selected] & (pairs[selected, 1] == gid), -1., 1.)
        volume = float(np.sum(sign*np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])))/6)
        if volume <= 0:
            raise RuntimeError(f'Nonpositive volume for grain {gid}')
        volumes[str(int(gid))] = volume
    if not np.isclose(sum(volumes.values()), np.prod(extent), rtol=1e-8):
        raise RuntimeError('Remeshed grain volumes do not sum to the RVE volume')
    report = dict(result.report, grains=len(volumes), cap_triangles=int(exterior.sum()),
                  internal_triangles=int((~exterior).sum()), all_six_faces_closed=True,
                  grain_surface_edge_closure_verified=True, rve_faces_planar=True,
                  grains_with_nonmanifold_edges=nonmanifold, enclosed_grain_volumes=volumes,
                  rve_dimensions=extent.tolist(), rve_volume=float(np.prod(extent)))
    report['surface_patch_groups'] = report.pop('grain_pairs')
    report['grain_pairs'] = len(np.unique(pairs[~exterior], axis=0))
    on_box = np.any((np.abs(surface.points) <= tol) |
                    (np.abs(surface.points-extent) <= tol), axis=1)
    report['rve_trace_nodes_preserved'] = int(np.sum(on_box & (source.node_kind == 3)))
    return ClosedRVE(points, faces, pairs, exterior, face_ids, report)
