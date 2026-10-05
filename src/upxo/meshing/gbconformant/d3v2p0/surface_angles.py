"""Raise the smallest surface-triangle angles by guarded local operations.

Additive companion to surface_quality.improve_surface_quality (flips only).
Operations, all inside one labelled patch (grain pair and RVE face):

- diagonal flips (topology only, shape change bounded by max_deviation);
- node relocation: patch-interior nodes move on their 1-ring surface, nodes on
  a patch-boundary curve (junction line or RVE-face trace) slide along that
  curve; RVE-plane coordinates are kept exactly;
- short-edge collapse: a movable node merges into a neighbour.

A node or triangle that touches a frozen grain never changes. Corner nodes of
curves (three or more curve edges) never move. Every operation must raise the
smallest angle of the triangles it changes and keep each changed triangle's
orientation; changed triangles are checked against the whole complex for
intersections and for facet openings below minimum_facet_angle, and rejected
operations are rolled back. Run the full tetrahedral surface validation after.
"""
from dataclasses import replace
import numpy as np
from .surface_intersections import find_surface_intersections
from .facet_angles import small_facet_angles


def triangle_angles(points, triangles):
    """Interior angles (degrees), shape (n, 3); column k is the angle at vertex k."""
    x = np.asarray(points)[np.asarray(triangles)]
    out = np.empty(x.shape[:2])
    for k in range(3):
        u, v = x[:, (k + 1) % 3] - x[:, k], x[:, (k + 2) % 3] - x[:, k]
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        c = np.divide(np.einsum('ij,ij->i', u, v), nu * nv, out=np.ones(len(x)), where=(nu * nv) > 0)
        out[:, k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def _normals(x):
    return np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])


def _closest_on_segments(y, a, b):
    """Closest points from y (3,) to segments a-b, shape (m, 3)."""
    d = b - a
    l2 = np.einsum('ij,ij->i', d, d)
    t = np.divide(np.einsum('ij,ij->i', y - a, d), l2, out=np.zeros(len(a)), where=l2 > 0)
    return a + np.clip(t, 0, 1)[:, None] * d


def _closest_on_triangles(y, x):
    """Closest points from y (3,) to triangles x (m, 3, 3), shape (m, 3)."""
    n = _normals(x)
    nn = np.einsum('ij,ij->i', n, n)
    proj = y - np.divide(np.einsum('ij,ij->i', y - x[:, 0], n), nn, out=np.zeros(len(x)), where=nn > 0)[:, None] * n
    # barycentric inside test of the plane projection
    inside = nn > 0
    for k in range(3):
        e = x[:, (k + 1) % 3] - x[:, k]
        inside &= np.einsum('ij,ij->i', np.cross(e, proj - x[:, k]), n) >= 0
    best = np.where(inside[:, None], proj, np.nan)
    for k in range(3):
        c = _closest_on_segments(y, x[:, k], x[:, (k + 1) % 3])
        better = ~inside & (np.isnan(best[:, 0]) |
                            (np.linalg.norm(c - y, axis=1) < np.linalg.norm(best - y, axis=1)))
        best[better] = c[better]
    return best


def _distance_to_triangles(y, x):
    if not len(x):
        return np.inf
    return float(np.linalg.norm(_closest_on_triangles(y, x) - y, axis=1).min())


def _project_to_triangles(y, x):
    c = _closest_on_triangles(y, x)
    return c[int(np.argmin(np.linalg.norm(c - y, axis=1)))]


def improve_surface_angles(surface, enabled=True, minimum_angle=30., max_passes=10,
                           max_deviation=None, minimum_facet_angle=.1, frozen_grain_ids=(),
                           flips=True, relocation=True, collapses=True,
                           max_normal_change=60., wedge_limit=None):
    """Raise triangle angles toward minimum_angle (degrees) on a ClosedRVE.

    max_deviation bounds how far the surface may move at any changed point
    (physical units); None uses 0.1 x the median edge length. A triangle is
    worked on while its smallest angle is below minimum_angle; operations stop
    when no further local improvement passes all checks. The target is not
    guaranteed: geometric corners sharper than it (patch corners, wedges)
    cannot be removed without changing the geometry beyond max_deviation.
    wedge_limit (degrees) adds the no-sharpening rule at every edge an
    operation changes: an opening below the limit may not get sharper, one at
    or above it may not fall below it.
    """
    if not np.isfinite(minimum_angle) or not 0 < minimum_angle < 60:
        raise ValueError('minimum_angle must lie in (0, 60) degrees')
    if not np.isfinite(minimum_facet_angle) or not 0 <= minimum_facet_angle < 180:
        raise ValueError('minimum_facet_angle must lie in [0, 180)')
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError('max_passes must be a nonnegative integer')
    if wedge_limit is not None and (not np.isfinite(wedge_limit) or not 0 < wedge_limit < 180):
        raise ValueError('wedge_limit must lie in (0, 180)')
    if not np.isfinite(max_normal_change) or not 0 < max_normal_change < 90:
        raise ValueError('max_normal_change must lie in (0, 90) degrees')
    if not all(isinstance(v, (bool, np.bool_)) for v in (enabled, flips, relocation, collapses)):
        raise ValueError('Switches must be boolean')
    p = np.array(surface.points, dtype=float)
    f = np.array(surface.triangles)
    pairs = np.asarray(surface.grain_pairs)
    exterior = np.asarray(surface.exterior, dtype=bool)
    rve_face = np.asarray(surface.rve_face)
    extent = surface.report.get('rve_dimensions')   # needed only to move nodes on RVE planes
    extent = None if extent is None else np.asarray(extent, dtype=float)
    edges0 = np.linalg.norm(p[f[:, 1]] - p[f[:, 0]], axis=1)
    if max_deviation is None:
        max_deviation = .1 * float(np.median(edges0))
    if not np.isfinite(max_deviation) or max_deviation < 0:
        raise ValueError('max_deviation must be finite and nonnegative')
    cos_normal = np.cos(np.radians(max_normal_change))
    frozen = np.isin(pairs, np.asarray(list(frozen_grain_ids), dtype=pairs.dtype)).any(axis=1) \
        if len(frozen_grain_ids) else np.zeros(len(f), bool)
    _, patch = np.unique(np.column_stack((pairs, rve_face)), axis=0, return_inverse=True)
    patch = patch.ravel()
    alive = np.ones(len(f), bool)

    a0 = triangle_angles(p, f).min(axis=1)
    before = _angle_summary(a0, minimum_angle)
    history = []
    totals = dict(flips=0, relocations=0, collapses=0)
    for iteration in range(max_passes if enabled else 0):
        ang = np.full(len(f), 180.)
        ang[alive] = triangle_angles(p, f[alive]).min(axis=1)
        work = np.flatnonzero(alive & (ang < minimum_angle) & ~frozen)
        if not len(work):
            break
        topo = _Topology(p, f, alive, patch, frozen, exterior, rve_face)
        topo.pairs = pairs
        reserved = np.zeros(len(p), bool)
        ops = []
        for t in work[np.argsort(ang[work])]:
            if reserved[f[t]].any() or not alive[t]:
                continue
            best = None
            for op in _candidates(t, p, f, topo, ang, flips, relocation, collapses,
                                  max_deviation, cos_normal, extent):
                if wedge_limit is not None and not _keeps_openings(op, p, f, topo, wedge_limit):
                    continue
                if best is None or op['gain'] > best['gain']:
                    best = op
            if best is None:
                continue
            touched = np.unique(np.concatenate([f[best['rows']].ravel(), best['new'].ravel()]))
            if reserved[touched].any():
                continue
            reserved[touched] = True
            if 'node' in best:
                reserved[topo.neighbours(best['node'])] = True
            best['old_rows'] = f[best['rows']].copy()
            best['old_alive'] = alive[best['rows']].copy()
            best['old_point'] = p[best['node']].copy() if 'node' in best else None
            _apply(best, p, f, alive)
            ops.append(best)
        if not ops:
            break
        active = np.ones(len(ops), bool)
        while active.any():
            ids = np.flatnonzero(alive)
            local = np.full(len(f), -1)
            local[ids] = np.arange(len(ids))
            changed = np.concatenate([o['rows'][alive[o['rows']]] for o, a in zip(ops, active) if a])
            hits = find_surface_intersections(p, f[ids], triangle_ids=local[changed])
            folds, _ = small_facet_angles(p, f[ids], minimum_facet_angle)
            bad = np.zeros(len(f), bool)
            for h in (hits, folds):
                if len(h):
                    bad[ids[np.unique(h)]] = True
            rejected = np.array([a and bool(bad[o['rows']].any()) for o, a in zip(ops, active)])
            if not rejected.any():
                break
            for o in [o for o, r in zip(ops, rejected) if r][::-1]:
                _undo(o, p, f, alive)
            active &= ~rejected
        accepted = [o for o, a in zip(ops, active) if a]
        counts = {k: sum(o['kind'] == k for o in accepted) for k in ('flip', 'relocation', 'collapse')}
        totals['flips'] += counts['flip']; totals['relocations'] += counts['relocation']
        totals['collapses'] += counts['collapse']
        history.append(dict(pass_number=iteration + 1, worked=int(len(work)), proposed=len(ops),
                            accepted=len(accepted), rejected_geometry=len(ops) - len(accepted), **counts))
        if not accepted:
            break

    # compact: drop collapsed triangles and unused nodes
    keep = np.flatnonzero(alive)
    f, pairs, exterior, rve_face = f[keep], pairs[keep], exterior[keep], rve_face[keep]
    used = np.unique(f)
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    p, f = p[used], remap[f]
    after = _angle_summary(triangle_angles(p, f).min(axis=1), minimum_angle)

    report = dict(surface.report)
    if 'enclosed_grain_volumes' in report:
        grains = np.unique(pairs)
        xyz = p[f]
        volume = np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])) / 6
        tot = np.bincount(np.searchsorted(grains, pairs[:, 0]), weights=volume, minlength=len(grains))
        internal = ~exterior
        tot -= np.bincount(np.searchsorted(grains, pairs[internal, 1]), weights=volume[internal], minlength=len(grains))
        if np.any(tot <= 0):
            raise RuntimeError('Angle repair produced a nonpositive grain volume')
        report['enclosed_grain_volumes'] = dict(zip(map(str, grains.tolist()), map(float, tot)))
    report['surface_angle_repair'] = dict(
        enabled=bool(enabled), minimum_angle=float(minimum_angle), max_deviation=float(max_deviation),
        minimum_facet_angle=float(minimum_facet_angle), max_normal_change=float(max_normal_change),
        operations=dict(flips=bool(flips), relocation=bool(relocation), collapses=bool(collapses)),
        wedge_limit=None if wedge_limit is None else float(wedge_limit),
        frozen_grains=len(frozen_grain_ids), before=before, after=after, history=history, **totals,
        full_validation_required=True)
    return replace(surface, points=p, triangles=f, grain_pairs=pairs, exterior=exterior,
                   rve_face=rve_face, report=report)


def _angle_summary(a, target):
    return dict(minimum=float(a.min()), percentile_1=float(np.percentile(a, 1)),
                below_target=int(np.sum(a < target)), below_20=int(np.sum(a < 20)),
                below_10=int(np.sum(a < 10)), below_5=int(np.sum(a < 5)), triangles=int(len(a)))


class _Topology:
    """Per-pass adjacency of the alive triangles."""

    def __init__(self, p, f, alive, patch, frozen, exterior, rve_face):
        ids = np.flatnonzero(alive)
        self.f, self.patch, self.exterior, self.rve_face = f, patch, exterior, rve_face
        rows = np.repeat(ids, 3)
        nodes = f[ids].ravel()
        order = np.argsort(nodes, kind='stable')
        self.node_tri = rows[order]
        self.offsets = np.r_[0, np.cumsum(np.bincount(nodes, minlength=len(p)))]
        self.edge_tri = {}
        for t in ids:
            a, b, c = f[t]
            for u, v in ((a, b), (b, c), (c, a)):
                self.edge_tri.setdefault((min(u, v), max(u, v)), []).append(t)
        self.fixed = np.zeros(len(p), bool)
        self.fixed[f[frozen & alive].ravel()] = True
        # curve edges: not exactly two owners in one patch
        self.curve = {}
        for e, ts in self.edge_tri.items():
            if len(ts) != 2 or patch[ts[0]] != patch[ts[1]]:
                key = frozenset(int(patch[t]) for t in ts)
                for n in e:
                    self.curve.setdefault(n, []).append((e, key))

    def ring(self, n):
        return self.node_tri[self.offsets[n]:self.offsets[n + 1]]

    def neighbours(self, n):
        return np.setdiff1d(np.unique(self.f[self.ring(n)]), [n])

    def kind(self, n):
        """'interior', ('curve', n1, n2) or None (fixed)."""
        if self.fixed[n]:
            return None
        c = self.curve.get(n)
        if not c:
            return 'interior'
        if len(c) == 2 and c[0][1] == c[1][1]:
            (e1, _), (e2, _) = c
            return ('curve', e1[0] if e1[1] == n else e1[1], e2[0] if e2[1] == n else e2[1])
        return None

    def planes(self, n):
        """(axis, value) pairs of RVE planes the node lies on (from its cap triangles)."""
        r = self.ring(n)
        faces = np.unique(self.rve_face[r][self.exterior[r]])
        return [(int(fc) // 2, int(fc) % 2) for fc in faces]


def _snap(y, planes, extent):
    y = y.copy()
    for axis, side in planes:
        y[axis] = side * extent[axis]
    return y


def _orientation_ok(old_x, new_x, cos_normal):
    no, nn = _normals(old_x), _normals(new_x)
    lo, ln = np.linalg.norm(no, axis=1), np.linalg.norm(nn, axis=1)
    if np.any(ln <= 1e-14 * max(1., float(np.max(lo)))):
        return False
    return bool(np.all(np.einsum('ij,ij->i', no, nn) > cos_normal * lo * ln))


def _candidates(t, p, f, topo, ang, flips, relocation, collapses, max_dev, cos_normal, extent):
    tri = f[t]
    if flips:
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            ts = topo.edge_tri.get((min(a, b), max(a, b)), [])
            if len(ts) != 2 or topo.patch[ts[0]] != topo.patch[ts[1]]:
                continue
            o = ts[0] if ts[1] == t else ts[1]
            c = int(tri[(k + 2) % 3])
            d = int(f[o][(f[o] != a) & (f[o] != b)][0])
            if c == d or (min(c, d), max(c, d)) in topo.edge_tri:
                continue
            new = np.array([[c, a, d], [c, d, b]], dtype=f.dtype)
            old_min = min(ang[t], ang[o])
            new_min = triangle_angles(p, new).min()
            if new_min <= old_min + 1e-6:
                continue
            ref = _normals(p[f[[t, o]]]).sum(axis=0)
            nn = _normals(p[new])
            if np.any(nn @ ref <= cos_normal * np.linalg.norm(nn, axis=1) * np.linalg.norm(ref)):
                continue
            height = abs(np.dot(p[d] - p[c], np.cross(p[a] - p[c], p[b] - p[c]))) / max(np.linalg.norm(ref), 1e-300)
            if height > max_dev:
                continue
            yield dict(kind='flip', rows=np.array([t, o]), new=new, gain=new_min - old_min)
    if relocation:
        for n in tri:
            kind = topo.kind(n)
            if kind is None:
                continue
            r = topo.ring(n)
            old_x = p[f[r]]
            old_min = ang[r].min()
            planes = topo.planes(n)
            if planes and extent is None:
                continue
            if kind == 'interior':
                target = p[topo.neighbours(n)].mean(axis=0)
            else:
                _, n1, n2 = kind
                target = .5 * (p[n1] + p[n2])
            best = None
            for step in (1., .5, .25):
                y = p[n] + step * (target - p[n])
                if kind == 'interior':
                    y = _project_to_triangles(y, old_x)
                else:
                    segs = _closest_on_segments(y, np.array([p[n1], p[n2]]), np.array([p[n], p[n]]))
                    y = segs[int(np.argmin(np.linalg.norm(segs - y, axis=1)))]
                y = _snap(y, planes, extent)
                new_x = old_x.copy()
                new_x[f[r] == n] = y
                if not _orientation_ok(old_x, new_x, cos_normal):
                    continue
                new_min = triangle_angles(new_x.reshape(-1, 3), np.arange(3 * len(r)).reshape(-1, 3)).min()
                if new_min <= old_min + 1e-6:
                    continue
                if kind == 'interior':
                    dev = _distance_to_triangles(p[n], new_x)
                else:
                    dev = float(np.linalg.norm(_closest_on_segments(
                        p[n], np.array([p[n1], y]), np.array([y, p[n2]])) - p[n], axis=1).min())
                if dev > max_dev:
                    continue
                if best is None or new_min > best[0]:
                    best = (new_min, y)
            if best is not None:
                yield dict(kind='relocation', rows=r, new=f[r], node=int(n), point=best[1],
                           gain=best[0] - old_min)
    if collapses:
        x = p[tri]
        lengths = [np.linalg.norm(x[(k + 1) % 3] - x[k]) for k in range(3)]
        k = int(np.argmin(lengths))
        a, b = int(tri[k]), int(tri[(k + 1) % 3])
        for gone, stay in ((a, b), (b, a)):
            kind = topo.kind(gone)
            if kind is None or (kind != 'interior' and stay not in kind[1:]):
                continue
            shared = topo.edge_tri.get((min(a, b), max(a, b)), [])
            opposite = {int(v) for s in shared for v in f[s] if v not in (a, b)}
            common = set(topo.neighbours(gone).tolist()) & set(topo.neighbours(stay).tolist())
            if common != opposite:
                continue                                   # link condition
            r = topo.ring(gone)
            keep = np.array([s for s in r if s not in shared])
            if not len(keep):
                continue
            new = f[keep].copy()
            new[new == gone] = stay
            old_min = ang[np.union1d(r, topo.ring(stay))].min()
            new_min = triangle_angles(p, new).min()
            if new_min <= old_min + 1e-6:
                continue
            if not _orientation_ok(p[f[keep]], p[new], cos_normal):
                continue
            if _distance_to_triangles(p[gone], p[new]) > max_dev:
                continue
            # every grain's shell must stay a single closed fan around the kept node
            others = np.array([s_ for s_ in topo.ring(stay) if s_ not in shared], dtype=int)
            fan_rows = np.concatenate((keep, others))
            fan_tris = np.vstack((new, f[others])) if len(others) else new
            if not _grain_fans_ok(stay, fan_tris, topo.pairs[fan_rows], topo.exterior[fan_rows]):
                continue
            yield dict(kind='collapse', rows=r, new=new, keep=keep, gone=np.array(shared),
                       gain=new_min - old_min)


def _keeps_openings(op, p, f, topo, limit):
    """No-sharpening rule at every edge the operation changes."""
    from .facet_angles import edge_openings, sharpens
    rows = np.asarray(op['rows'])
    new = np.asarray(op['new'])
    def edges(tris):
        return {(min(a, b), max(a, b)) for t in tris for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    old_edges = sorted(edges(f[rows]))
    old_values = edge_openings(p, f, old_edges, topo.edge_tri)
    before_of = dict(zip(old_edges, old_values))
    before = old_values.min()
    removed = set(rows.tolist())
    f2 = np.vstack((f, new))
    temp = np.arange(len(f), len(f) + len(new))
    new_edges = edges(new)
    mapping = {}
    for e in new_edges:
        mapping[e] = [t for t in topo.edge_tri.get(e, []) if t not in removed]
    for tid, t in zip(temp, new):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            mapping[(min(a, b), max(a, b))].append(int(tid))
    p2 = p
    if op['kind'] == 'relocation':
        p2 = p.copy(); p2[op['node']] = op['point']
    new_list = sorted(new_edges)
    after = edge_openings(p2, f2, new_list, mapping)
    # each surviving edge against its own previous opening; new edges against the local minimum
    previous = np.array([before_of.get(e, before) for e in new_list])
    return not np.any(sharpens(previous, after, limit))


def _grain_fans_ok(node, tris, pairs, exterior):
    """Each grain's triangles around node form one closed fan (manifold vertex and edges)."""
    grains = set(pairs[:, 0].tolist()) | set(pairs[~exterior, 1].tolist())
    for g in grains:
        rows = np.flatnonzero((pairs[:, 0] == g) | (~exterior & (pairs[:, 1] == g)))
        spokes = {}
        for r in rows:
            for x in tris[r]:
                if x != node:
                    spokes.setdefault(int(x), []).append(int(r))
        if any(len(v) != 2 for v in spokes.values()):
            return False                                   # an edge used by one or more than two triangles
        seen, stack = {int(rows[0])}, [int(rows[0])]
        while stack:
            r = stack.pop()
            for x in tris[r]:
                if x != node:
                    for q in spokes[int(x)]:
                        if q not in seen:
                            seen.add(q); stack.append(q)
        if len(seen) != len(rows):
            return False                                   # the grain would touch itself at this node
    return True


def _apply(op, p, f, alive):
    if op['kind'] == 'flip':
        f[op['rows']] = op['new']
    elif op['kind'] == 'relocation':
        p[op['node']] = op['point']
    else:
        f[op['keep']] = op['new']
        alive[op['gone']] = False


def _undo(op, p, f, alive):
    f[op['rows']] = op['old_rows']
    alive[op['rows']] = op['old_alive']
    if op['old_point'] is not None:
        p[op['node']] = op['old_point']
