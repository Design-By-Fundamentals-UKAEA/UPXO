"""Improve surface slivers by guarded, ownership-preserving diagonal flips."""
from dataclasses import replace
import numpy as np
from .surface_intersections import find_surface_intersections


def triangle_quality(points, triangles):
    xyz = points[triangles]
    area2 = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)
    length2 = np.sum((xyz[:, [1, 2, 0]]-xyz)**2, axis=(1, 2))
    return np.divide(2*np.sqrt(3)*area2, length2, out=np.zeros(len(xyz)), where=length2 > 0)


def improve_surface_quality(surface, enabled=True, minimum_quality=.05, max_passes=10,
                            minimum_facet_angle=.1):
    """Flip interior diagonals without moving nodes or changing patch ownership.

    Every accepted pair improves its minimum triangle quality. Shared grain
    junction edges and RVE patch boundaries cannot flip. Independent proposals
    are checked against the whole complex; proposals involved in an intersection
    are rolled back. This guards changed triangles, not pre-existing defects in
    untouched triangles: run the full tetrahedral surface validation afterwards.
    Nonplanar flips can change local interface shape and individual grain volumes.
    """
    if not 0 <= minimum_quality <= 1:
        raise ValueError('minimum_quality must lie between zero and one')
    if not np.isfinite(minimum_facet_angle) or not 0 <= minimum_facet_angle < 180:
        raise ValueError('minimum_facet_angle must lie in [0, 180)')
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError('max_passes must be a nonnegative integer')
    p, f = surface.points, surface.triangles.copy()
    keys = np.column_stack((surface.grain_pairs, surface.rve_face))
    q = triangle_quality(p, f)
    before = dict(minimum=float(q.min()), poor_count=int(np.sum(q < minimum_quality)))
    history = []
    for iteration in range(max_passes if enabled else 0):
        poor = np.flatnonzero(q < minimum_quality)
        if not len(poor): break
        edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind='stable')
        offsets = np.r_[0, np.cumsum(counts)]
        edge_ids = inverse.reshape(-1, 3)
        # Structured search avoids a Python set containing millions of edges.
        edge_dtype = np.dtype([('a', f.dtype), ('b', f.dtype)])
        edge_view = np.ascontiguousarray(unique).view(edge_dtype).ravel()
        reserved, proposals = set(), []
        trial = f.copy()
        for face in poor[np.argsort(q[poor])]:
            if reserved.intersection(f[face]): continue
            best = None
            for edge_id in edge_ids[face]:
                if counts[edge_id] != 2: continue
                owners = order[offsets[edge_id]:offsets[edge_id+1]]//3
                other = int(owners[0] if owners[1] == face else owners[1])
                if not np.array_equal(keys[face], keys[other]) or reserved.intersection(f[other]): continue
                a, b = unique[edge_id]
                first = f[face]
                i = int(np.flatnonzero(first == a)[0])
                if first[(i+1) % 3] != b: a, b = b, a
                c = int(first[(first != a) & (first != b)][0])
                d = int(f[other][(f[other] != a) & (f[other] != b)][0])
                if c == d: continue
                diagonal = np.array([tuple(sorted((c, d)))], dtype=edge_dtype)
                pos = int(np.searchsorted(edge_view, diagonal)[0])
                if pos < len(unique) and np.array_equal(unique[pos], np.sort([c, d])): continue
                candidate = np.array([[c, a, d], [c, d, b]], dtype=f.dtype)
                quality = triangle_quality(p, candidate)
                old = min(q[face], q[other])
                if quality.min() <= old+max(1e-12, .001*old): continue
                xyz = p[f[[face, other]]]
                reference = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]).sum(axis=0)
                xyz = p[candidate]
                normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
                if np.any(normals@reference <= 0): continue
                if best is None or quality.min() > best[0]: best = (float(quality.min()), other, candidate)
            if best is not None:
                _, other, candidate = best
                proposals.append((int(face), other))
                reserved.update(f[[face, other]].ravel())
                trial[[face, other]] = candidate
        if not proposals: break
        active = np.ones(len(proposals), dtype=bool)
        proposal_faces = np.asarray(proposals)
        # Reverting one proposal can expose another conflict, so recheck until
        # the remaining changed triangles are clear against the final trial.
        while np.any(active):
            hits = find_surface_intersections(p, trial, triangle_ids=proposal_faces[active].ravel())
            from .facet_angles import small_facet_angles
            folds,_=small_facet_angles(p,trial,minimum_facet_angle)
            hits=np.vstack((hits,folds))
            rejected = active & np.any(np.isin(proposal_faces, hits), axis=1)
            if not np.any(rejected): break
            ids = proposal_faces[rejected].ravel()
            trial[ids] = f[ids]
            active[rejected] = False
        accepted = int(active.sum())
        history.append(dict(pass_number=iteration+1, proposed=len(proposals), accepted=accepted,
                            rejected_geometry=int((~active).sum())))
        if not accepted: break
        f = trial
        q = triangle_quality(p, f)
    report = dict(surface.report)
    if 'enclosed_grain_volumes' in report:
        grains = np.unique(surface.grain_pairs)
        xyz = p[f]
        volume = np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2]))/6
        totals = np.bincount(np.searchsorted(grains, surface.grain_pairs[:, 0]), weights=volume, minlength=len(grains))
        internal = ~surface.exterior
        totals -= np.bincount(np.searchsorted(grains, surface.grain_pairs[internal, 1]), weights=volume[internal], minlength=len(grains))
        if np.any(totals <= 0): raise RuntimeError('Quality flips produced a nonpositive grain volume')
        report['enclosed_grain_volumes'] = dict(zip(map(str, grains.tolist()), map(float, totals)))
    report['surface_quality_repair'] = dict(enabled=bool(enabled), threshold=float(minimum_quality),
        minimum_facet_angle=float(minimum_facet_angle),
        before=before, after=dict(minimum=float(q.min()), poor_count=int(np.sum(q < minimum_quality))),
        history=history, accepted_flips=sum(h['accepted'] for h in history), node_coordinates_unchanged=True,
        full_validation_required=True)
    return replace(surface, triangles=f, report=report)
