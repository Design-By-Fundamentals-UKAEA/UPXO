"""Phase 3 of tet improvement: face and edge swaps inside grains.

2-3 swap: two tets sharing a face (a, b, c) with apexes d and e become three
tets around the new edge d-e. 3-2 swap: three tets around an edge d-e (ring
a, b, c) become two tets sharing the face (a, b, c). Only tets of one grain
take part, the swapped face or edge is inside that grain (never on a grain
surface or RVE face), so grain surfaces, conformity and grain volumes are
unchanged. A swap is kept only when
- the worst normalised angle quality of the tets involved rises (smallest
  dihedral relative to ``target``, largest relative to ``max_angle``),
- the number of tets outside the limits (below ``target`` or above
  ``max_angle``) does not rise, and
- every new tet has positive volume and the new tets fill exactly the volume
  of the old ones (so they neither overlap nor invert).
"""
from dataclasses import replace
from collections import defaultdict
import numpy as np
from .tet_angles import dihedral_angles_xyz, dihedral_summary


def _signed_volume(x):
    return float(np.dot(x[1] - x[0], np.cross(x[2] - x[0], x[3] - x[0])) / 6)


def _orient(points, tet):
    tet = list(tet)
    if _signed_volume(points[tet]) < 0:
        tet[0], tet[1] = tet[1], tet[0]
    return tet


class _Mesh:
    """Tets with lazy node->tet lookup, supporting removal and addition."""

    def __init__(self, tetrahedra, grain_ids, n_points):
        self.t = [list(map(int, x)) for x in tetrahedra]
        self.g = list(map(int, grain_ids))
        self.alive = [True] * len(self.t)
        flat = np.asarray(tetrahedra).ravel()
        order = np.argsort(flat, kind='stable')
        self.order, self.offsets = order // 4, np.r_[0, np.cumsum(np.bincount(flat, minlength=n_points))]
        self.extra = defaultdict(list)

    def star(self, n):
        base = self.order[self.offsets[n]:self.offsets[n + 1]].tolist()
        return [k for k in base + self.extra.get(n, []) if self.alive[k]]

    def add(self, tet, grain):
        k = len(self.t)
        self.t.append(list(map(int, tet)))
        self.g.append(int(grain))
        self.alive.append(True)
        for n in tet:
            self.extra[int(n)].append(k)
        return k


def swap_tets(points, tetrahedra, grain_ids, surface_triangles, *, enabled=True, target=30., max_angle=150.,
              max_passes=5, progress=None):
    """Return (tetrahedra, grain_ids, report) after face and edge swaps.

    surface_triangles: (m, 3) node indices of every grain-surface and RVE-face
    triangle; their faces and edges are never swapped.
    """
    p = np.asarray(points, dtype=float)
    t0 = np.asarray(tetrahedra)
    if t0.ndim != 2 or t0.shape[1] != 4 or len(grain_ids) != len(t0):
        raise ValueError('Expected (m, 4) tetrahedra with one grain ID each')
    if not np.isfinite(target) or not 0 < target < 70.5:
        raise ValueError('target must lie in (0, 70.5) degrees')
    if not np.isfinite(max_angle) or not 70.5 < max_angle < 180:
        raise ValueError('max_angle must lie in (70.5, 180) degrees')
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError('max_passes must be a nonnegative integer')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    surf = np.sort(np.asarray(surface_triangles), axis=1)
    surface_faces = set(map(tuple, surf.tolist()))
    surface_edges = set(map(tuple, np.sort(surf[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).tolist()))
    span = 180. - max_angle

    def measure(tets):
        a = dihedral_angles_xyz(p[np.asarray(tets)])
        mn, mx = a.min(axis=1), a.max(axis=1)
        return mn, mx, np.minimum(mn / target, (180. - mx) / span)

    def acceptable(old, new):
        """old/new: lists of tets (oriented). Returns (ok, new quality min)."""
        vols = [_signed_volume(p[x]) for x in new]
        if min(vols) <= 0:
            return False, None
        old_vol = sum(_signed_volume(p[x]) for x in old)
        if not np.isclose(sum(vols), old_vol, rtol=1e-9, atol=1e-12 * max(1., abs(old_vol))):
            return False, None
        omn, _, oq = measure(old)
        nmn, _, nq = measure(new)
        if nq.min() <= oq.min() + 1e-9:
            return False, None
        if np.sum(nq < 1.) > np.sum(oq < 1.):                    # tets outside either limit
            return False, None
        return True, float(nq.min())

    mesh = _Mesh(t0, grain_ids, len(p))
    mn, mx, q = measure(t0)
    before = dict(minimum_dihedral=float(mn.min()), below_target=int(np.sum(mn < target)),
                  above_max_angle=int(np.sum(mx > max_angle)), below_10=int(np.sum(mn < 10)),
                  below_15=int(np.sum(mn < 15)), below_20=int(np.sum(mn < 20)))
    quality = list(q)
    history = []
    counts = dict(swaps_2_3=0, swaps_3_2=0)
    for pass_number in range(max_passes if enabled else 0):
        bad = [k for k in range(len(mesh.t)) if mesh.alive[k] and quality[k] < 1.]
        bad.sort(key=lambda k: quality[k])
        done = {'2-3': 0, '3-2': 0}
        for k in bad:
            if not mesh.alive[k] or quality[k] >= 1.:
                continue
            best = None
            tet, grain = mesh.t[k], mesh.g[k]
            # 2-3 across each face
            for i in range(4):
                face = [tet[j] for j in range(4) if j != i]
                if tuple(sorted(face)) in surface_faces:
                    continue
                d = tet[i]
                shared = set(mesh.star(face[0])) & set(mesh.star(face[1])) & set(mesh.star(face[2]))
                others = [o for o in shared if o != k]
                if len(others) != 1 or mesh.g[others[0]] != grain:
                    continue
                o = others[0]
                e = [n for n in mesh.t[o] if n not in face][0]
                new = [_orient(p, [face[0], face[1], d, e]), _orient(p, [face[1], face[2], d, e]),
                       _orient(p, [face[2], face[0], d, e])]
                ok, val = acceptable([tet, mesh.t[o]], new)
                if ok and (best is None or val > best[0]):
                    best = (val, '2-3', [k, o], new)
            # 3-2 around each edge
            for i in range(4):
                for j in range(i + 1, 4):
                    d, e = tet[i], tet[j]
                    if tuple(sorted((d, e))) in surface_edges:
                        continue
                    ring = sorted(set(mesh.star(d)) & set(mesh.star(e)))
                    if len(ring) != 3 or any(mesh.g[r] != grain for r in ring):
                        continue
                    apex = defaultdict(int)
                    for r in ring:
                        for n in mesh.t[r]:
                            if n not in (d, e):
                                apex[n] += 1
                    if len(apex) != 3 or any(c != 2 for c in apex.values()):
                        continue                               # ring not closed: edge on a boundary
                    a, b, c = list(apex)
                    if tuple(sorted((a, b, c))) in surface_faces:
                        continue
                    new = [_orient(p, [a, b, c, d]), _orient(p, [a, b, c, e])]
                    ok, val = acceptable([mesh.t[r] for r in ring], new)
                    if ok and (best is None or val > best[0]):
                        best = (val, '3-2', ring, new)
            if best is None:
                continue
            _, kind, old_ids, new = best
            for r in old_ids:
                mesh.alive[r] = False
            nmn, nmx, nq = measure(new)
            for x, qq in zip(new, nq):
                mesh.add(x, grain)
                quality.append(float(qq))
            done[kind] += 1
        counts['swaps_2_3'] += done['2-3']
        counts['swaps_3_2'] += done['3-2']
        alive = np.flatnonzero(mesh.alive)
        amn, amx, _ = measure(np.asarray(mesh.t)[alive])
        entry = dict(pass_number=pass_number + 1, swaps_2_3=done['2-3'], swaps_3_2=done['3-2'],
                     below_target=int(np.sum(amn < target)), below_15=int(np.sum(amn < 15)),
                     below_10=int(np.sum(amn < 10)), minimum_dihedral=float(amn.min()))
        history.append(entry)
        if progress is not None:
            progress(entry)
        if not done['2-3'] and not done['3-2']:
            break
    alive = np.flatnonzero(mesh.alive)
    tets = np.asarray(mesh.t, dtype=t0.dtype)[alive]
    grains = np.asarray(mesh.g, dtype=np.asarray(grain_ids).dtype)[alive]
    mn, mx, _ = measure(tets)
    after = dict(minimum_dihedral=float(mn.min()), below_target=int(np.sum(mn < target)),
                 above_max_angle=int(np.sum(mx > max_angle)), below_10=int(np.sum(mn < 10)),
                 below_15=int(np.sum(mn < 15)), below_20=int(np.sum(mn < 20)))
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  before=before, after=after, history=history, **counts)
    return tets, grains, report


def swap_grain_tetrahedra(tets, surface, *, enabled=True, target=30., max_angle=150., max_passes=5,
                          progress=None):
    """Swap faces and edges of a verified GrainTetrahedra result and re-verify it.

    surface: the ClosedRVE whose triangles are the grain surfaces (points
    numbered as the first tet nodes). Conformity (each grain's boundary faces
    equal its surface triangles), positive volumes, unchanged grain volumes,
    quality (Gmsh minSICN), dihedral statistics and readiness are rechecked.
    """
    from .tet_smoothing import minimum_sicn_gmsh, _volumes
    new_t, new_g, swaps = swap_tets(tets.points, tets.tetrahedra, tets.grain_ids, surface.triangles,
                                    enabled=enabled, target=target, max_angle=max_angle,
                                    max_passes=max_passes, progress=progress)
    report = dict(tets.report)
    report['tet_swaps'] = swaps
    if not enabled:
        return replace(tets, report=report)
    return _reverify(tets, tets.points, new_t, new_g, surface, report, 'swaps')


def _reverify(tets, points, new_t, new_g, surface, report, what):
    """Shared checks after a topology change inside grains (grain surfaces unchanged)."""
    from .tet_smoothing import minimum_sicn_gmsh, _volumes
    volumes = _volumes(points[new_t])
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError(f'{what} produced a nonpositive tetrahedron volume')
    # conformity: each grain's single-use tet faces must be exactly its surface triangles
    pairs = np.asarray(surface.grain_pairs)
    exterior = np.asarray(surface.exterior, bool)
    pattern = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    faces = np.sort(new_t[:, pattern].reshape(-1, 3), axis=1)
    key = np.column_stack((np.repeat(new_g, 4), faces))
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError(f'Nonmanifold tet faces after {what}')
    boundary = uniq[counts == 1]
    st = np.sort(np.asarray(surface.triangles), axis=1)
    expected = np.unique(np.vstack((np.column_stack((pairs[:, 0], st)),
                                    np.column_stack((pairs[~exterior, 1], st[~exterior])))), axis=0)
    if not np.array_equal(boundary, expected):
        raise RuntimeError(f'Tet boundary differs from the grain surfaces after {what}')
    quality = minimum_sicn_gmsh(points, new_t)
    if np.any(~np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError(f'Invalid tetrahedron quality after {what}')
    per_grain = {}
    for gid, entry in tets.report['per_grain'].items():
        selected = new_g == int(gid)
        volume = float(volumes[selected].sum())
        if not np.isclose(volume, entry['volume'], rtol=1e-8):
            raise RuntimeError(f'Volume of grain {gid} changed during {what}')
        per_grain[gid] = dict(entry, tetrahedra=int(selected.sum()), volume=volume,
                              minimum_quality=float(quality[selected].min()))
    dihedral = dihedral_summary(points, new_t)
    passed = bool(quality.min() >= report['quality_threshold'])
    if report.get('minimum_dihedral_threshold') is not None:
        passed = passed and dihedral['minimum'] >= report['minimum_dihedral_threshold']
    report.update(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET', ready=passed,
                  per_grain=per_grain, tetrahedra=int(len(new_t)), nodes=int(len(points)),
                  total_volume=float(volumes.sum()), minimum_quality=float(quality.min()),
                  quality_percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
                  below_quality_threshold=int(np.sum(quality < report['quality_threshold'])),
                  dihedral_angles=dihedral)
    return replace(tets, points=points, tetrahedra=new_t, grain_ids=new_g, quality=quality, report=report)


_DIRS = np.array([d for d in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T if np.any(d)], float)
_DIRS /= np.linalg.norm(_DIRS, axis=1)[:, None]


def insert_nodes(points, tetrahedra, grain_ids, surface_triangles, *, enabled=True, target=30., max_angle=150.,
                 cavity_rings=(1, 2, 3), max_passes=3, search_iterations=40, samples=256, seed=0, progress=None):
    """Return (points, tetrahedra, grain_ids, report) after node insertion.

    For each tet below target (worst first), the cavity is first the union of
    the stars of its interior nodes (those nodes are replaced by the new one),
    then the tets around each of its edges not on a grain surface (the new node
    splits that edge; this removes tets lying flat on a grain surface), then
    this grain's tets at each of its grain-surface edges with their neighbours
    (splits a wide grain wedge held by one tet), then
    grown from the tet over
    faces shared with tets of the same grain that are not grain-surface
    faces (cavity_rings: the ring counts tried). One new node is connected to
    every boundary face of the cavity and placed by a direct search that
    maximises the worst normalised angle quality of the new tets, starting
    from the best of ``samples`` random points in the cavity (seeded). The result
    is kept only when all new tets have positive volume, they fill exactly
    the cavity volume, the worst quality rises and the number of tets outside
    the limits (below target or above max_angle) does not rise. Grain surfaces are cavity boundary faces and stay.
    """
    p = [np.asarray(x, float) for x in np.asarray(points, float)]
    t0 = np.asarray(tetrahedra)
    if t0.ndim != 2 or t0.shape[1] != 4 or len(grain_ids) != len(t0):
        raise ValueError('Expected (m, 4) tetrahedra with one grain ID each')
    if not np.isfinite(target) or not 0 < target < 70.5:
        raise ValueError('target must lie in (0, 70.5) degrees')
    if not np.isfinite(max_angle) or not 70.5 < max_angle < 180:
        raise ValueError('max_angle must lie in (70.5, 180) degrees')
    for name, value, low in (('max_passes', max_passes, 0), ('search_iterations', search_iterations, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise ValueError(f'{name} must be an integer >= {low}')
    rings = tuple(cavity_rings)
    if not rings or any(isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 0 for r in rings):
        raise ValueError('cavity_rings must be nonnegative integers')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError('samples must be a positive integer')
    rng = np.random.default_rng(seed)
    surface_faces = set(map(tuple, np.sort(np.asarray(surface_triangles), axis=1).tolist()))
    on_surface = np.zeros(len(points), bool)
    on_surface[np.unique(np.asarray(surface_triangles))] = True
    st = np.sort(np.asarray(surface_triangles), axis=1)
    surface_edges = set(map(tuple, np.sort(st[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).tolist()))
    span = 180. - max_angle
    P = np.asarray(points, float)
    points_list = [P]
    n_points = len(P)

    def coords():
        return np.vstack(points_list) if len(points_list) > 1 else points_list[0]

    def measure_xyz(x):
        a = dihedral_angles_xyz(x)
        mn, mx = a.min(axis=-1), a.max(axis=-1)
        return mn, mx, np.minimum(mn / target, (180. - mx) / span)

    mesh = _Mesh(t0, grain_ids, len(P))
    face_owner = {}

    def faces_of(k):
        t = mesh.t[k]
        return [tuple(sorted(t[j] for j in range(4) if j != i)) for i in range(4)]

    def neighbour(k, face):
        shared = set(mesh.star(face[0])) & set(mesh.star(face[1])) & set(mesh.star(face[2]))
        others = [o for o in shared if o != k]
        return others[0] if len(others) == 1 else None

    X = coords()
    _, _, q0 = measure_xyz(X[t0])
    quality = list(q0)
    mn0 = measure_xyz(X[t0])[0]
    before = dict(minimum_dihedral=float(mn0.min()), below_target=int(np.sum(mn0 < target)),
                  below_15=int(np.sum(mn0 < 15)), below_10=int(np.sum(mn0 < 10)))
    history, inserted = [], 0
    for pass_number in range(max_passes if enabled else 0):
        X = coords()
        bad = sorted((k for k in range(len(mesh.t)) if mesh.alive[k] and quality[k] < 1.), key=lambda k: quality[k])
        done = 0
        for k in bad:
            if not mesh.alive[k] or quality[k] >= 1.:
                continue
            grain = mesh.g[k]
            best = None
            interior = [n for n in mesh.t[k] if n >= len(on_surface) or not on_surface[n]]
            tet = mesh.t[k]
            inner_edges = [(tet[i], tet[j]) for i in range(4) for j in range(i + 1, 4)
                           if tuple(sorted((tet[i], tet[j]))) not in surface_edges]
            wedge_edges = [(tet[i], tet[j]) for i in range(4) for j in range(i + 1, 4)
                           if tuple(sorted((tet[i], tet[j]))) in surface_edges]
            options = ((['nodes'] if interior else []) + [('edge',) + e for e in inner_edges]
                       + [('wedge',) + e for e in wedge_edges] + list(rings))
            for ring in options:
                cavity = {k}
                frontier = [] if ring == 'nodes' or isinstance(ring, tuple) else [k]
                if isinstance(ring, tuple) and ring[0] == 'edge':
                    # the tets around one interior edge: inserting splits that edge
                    cavity = set(mesh.star(ring[1])) & set(mesh.star(ring[2]))
                    if any(mesh.g[c] != grain for c in cavity):
                        continue
                if isinstance(ring, tuple) and ring[0] == 'wedge':
                    # this grain's tets at a surface edge plus their same-grain neighbours:
                    # the new node splits the grain's wedge at that edge between more tets
                    cavity = {c for c in set(mesh.star(ring[1])) & set(mesh.star(ring[2])) if mesh.g[c] == grain}
                    frontier = list(cavity)
                    ring = ('wedge', ring[1], ring[2], 1)
                if ring == 'nodes':
                    # the stars of the tet's interior nodes: those nodes are replaced
                    for n in interior:
                        cavity.update(mesh.star(n))
                    if any(mesh.g[c] != grain for c in cavity):
                        continue
                steps = ring if isinstance(ring, (int, np.integer)) else (ring[3] if ring[0] == 'wedge' else 0)
                for _ in range(steps):
                    nxt = []
                    for c in frontier:
                        for f in faces_of(c):
                            if f in surface_faces:
                                continue
                            o = neighbour(c, f)
                            if o is not None and o not in cavity and mesh.g[o] == grain:
                                cavity.add(o); nxt.append(o)
                    frontier = nxt
                cav = sorted(cavity)
                # boundary faces with outward-consistent orientation (from each owning tet)
                count = defaultdict(int)
                oriented = {}
                for c in cav:
                    t = mesh.t[c]
                    for i in range(4):
                        f = [t[j] for j in range(4) if j != i]
                        # orient face so that the opposite vertex t[i] lies on its positive side
                        if _signed_volume(X[[f[0], f[1], f[2], t[i]]]) < 0:
                            f = [f[0], f[2], f[1]]
                        key = tuple(sorted(f))
                        count[key] += 1
                        oriented[key] = f
                boundary = [oriented[f] for f, c in count.items() if c == 1]
                old_x = X[np.asarray([mesh.t[c] for c in cav])]
                omn, _, oq = measure_xyz(old_x)
                old_vol = sum(_signed_volume(x) for x in old_x)
                bx = X[np.asarray(boundary)]                     # (m, 3, 3)

                def score(v):
                    x = np.concatenate((bx[None].repeat(len(v), 0), v[:, None, None, :].repeat(len(boundary), 1)), axis=2)
                    vol = np.einsum('bmi,bmi->bm', x[:, :, 1] - x[:, :, 0],
                                    np.cross(x[:, :, 2] - x[:, :, 0], x[:, :, 3] - x[:, :, 0])) / 6
                    mn, mx, qq = measure_xyz(x)
                    valid = np.all(vol > 0, axis=1) & np.isclose(vol.sum(1), old_vol, rtol=1e-9,
                                                                   atol=1e-12 * max(1., abs(old_vol)))
                    return np.where(valid, qq.min(1), -np.inf), mn

                cav_nodes = np.unique(np.asarray([mesh.t[c] for c in cav]))
                h = float(np.linalg.norm(X[cav_nodes] - X[cav_nodes].mean(0), axis=1).mean())
                # Start from the best of many points spread through the cavity tets:
                # a cavity is star-shaped only from part of its interior (its kernel).
                bary = rng.dirichlet(np.ones(4), size=samples)
                which = rng.integers(0, len(cav), size=samples)
                starts = np.einsum('sk,skd->sd', bary, old_x[which])
                inner_faces = [f for f, c in count.items() if c == 2]
                extra = [old_x.mean(1), X[cav_nodes].mean(0)[None]]
                if inner_faces:
                    fx = X[np.asarray(inner_faces)]
                    extra += [fx.mean(1), (fx[:, [0, 1, 2]] + fx[:, [1, 2, 0]]).reshape(-1, 3) / 2]
                if isinstance(ring, tuple):
                    a, b = X[ring[1]], X[ring[2]]
                    extra.append(a + np.linspace(.2, .8, 7)[:, None] * (b - a))
                starts = np.vstack([starts] + extra)
                vals = score(starts)[0]
                j = int(np.argmax(vals))
                v, val = starts[j], vals[j]
                if not np.isfinite(val):
                    continue
                step = .25 * h
                for _ in range(search_iterations):
                    trials = v + step * _DIRS
                    vals = score(trials)[0]
                    j = int(np.argmax(vals))
                    if vals[j] > val + 1e-12:
                        v, val = trials[j], vals[j]
                    else:
                        step *= .5
                        if step < 1e-4 * h:
                            break
                if not np.isfinite(val) or val <= oq.min() + 1e-9:
                    continue
                new_x = np.concatenate((bx, np.repeat(v[None, None, :], len(boundary), 0)), axis=1)
                if np.sum(measure_xyz(new_x)[2] < 1.) > np.sum(oq < 1.):   # tets outside either limit
                    continue
                if best is None or val > best[0]:
                    best = (val, cav, boundary, v)
            if best is None:
                continue
            _, cav, boundary, v = best
            index = n_points
            n_points += 1
            points_list.append(v[None])
            X = coords()
            mesh.offsets = np.append(mesh.offsets, mesh.offsets[-1])   # new node has no base tets
            for c in cav:
                mesh.alive[c] = False
            new = [list(f) + [index] for f in boundary]
            _, _, nq = measure_xyz(X[np.asarray(new)])
            for x, qq in zip(new, nq):
                mesh.add(x, grain)
                quality.append(float(qq))
            inserted += 1
            done += 1
        alive = np.flatnonzero(mesh.alive)
        amn = measure_xyz(coords()[np.asarray(mesh.t)[alive]])[0]
        entry = dict(pass_number=pass_number + 1, inserted=done, below_target=int(np.sum(amn < target)),
                     below_15=int(np.sum(amn < 15)), below_10=int(np.sum(amn < 10)),
                     minimum_dihedral=float(amn.min()))
        history.append(entry)
        if progress is not None:
            progress(entry)
        if not done:
            break
    X = coords()
    alive = np.flatnonzero(mesh.alive)
    tets = np.asarray(mesh.t, dtype=t0.dtype)[alive]
    grains = np.asarray(mesh.g, dtype=np.asarray(grain_ids).dtype)[alive]
    # drop interior nodes no tet uses any more; surface nodes keep their numbers
    keep = np.zeros(len(X), bool)
    keep[np.unique(tets)] = True
    keep[:len(on_surface)][on_surface] = True
    removed = int(np.sum(~keep))
    if removed:
        remap = np.cumsum(keep) - 1
        X, tets = X[keep], remap[tets].astype(t0.dtype)
    mn = measure_xyz(X[tets])[0]
    after = dict(minimum_dihedral=float(mn.min()), below_target=int(np.sum(mn < target)),
                 below_15=int(np.sum(mn < 15)), below_10=int(np.sum(mn < 10)))
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  cavity_rings=list(rings), inserted_nodes=inserted, removed_nodes=removed, before=before,
                  after=after, history=history)
    return X, tets, grains, report


def insert_grain_tetrahedra(tets, surface, *, enabled=True, target=30., max_angle=150., cavity_rings=(1, 2, 3),
                            max_passes=3, progress=None):
    """Insert nodes into a verified GrainTetrahedra result and re-verify it.

    New nodes are appended after all existing nodes, so the grain-surface
    nodes keep their numbering. See insert_nodes for the acceptance rule.
    """
    points, new_t, new_g, insertion = insert_nodes(
        tets.points, tets.tetrahedra, tets.grain_ids, surface.triangles, enabled=enabled, target=target,
        max_angle=max_angle, cavity_rings=cavity_rings, max_passes=max_passes, progress=progress)
    report = dict(tets.report)
    report['node_insertion'] = insertion
    if not enabled or not insertion['inserted_nodes']:
        return replace(tets, report=report)
    return _reverify(tets, points, new_t, new_g, surface, report, 'node insertion')
