"""Parallel surface-angle repair with the same result as d3v2p0.

Each pass works in rounds. A round evaluates the best operation of every
pending triangle in parallel (d3v2p0's candidate function and the same
no-sharpening arithmetic) on the current state. The greedy loop then applies
them worst-first, as d3v2p0 does, but defers a triangle whose evaluation has
gone stale (a node it read was changed by an operation applied after it was
made) to the next round instead of evaluating it again on the spot. Every
applied operation was therefore evaluated on the exact current state and
passes the same rules and the same verification as in d3v2p0; only the order
differs, so the result is close to, not identical with, d3v2p0's. It does not
depend on the worker count.
"""
import math
import time
from dataclasses import replace
import numpy as np
from multiprocessing import shared_memory
from ..d3v2p0.surface_angles import (triangle_angles, _Topology, _apply, _undo, _angle_summary,
                                     _snap, _grain_fans_ok)
from ..d3v2p0.surface_intersections import find_surface_intersections
from ..d3v2p0.facet_angles import small_facet_angles
from .backend import plan, Plan, WorkerPool
from .parallel import single_thread_worker


def _cross(a, b):
    """np.cross for (..., 3) arrays without its axis bookkeeping."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    c = np.empty(a.shape)
    c[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    c[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    c[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return c


def _norm(x, axis=None):
    """np.linalg.norm for one vector (axis None) or rows of a 2-D array (axis=1)."""
    x = np.asarray(x, float)
    if axis is None:
        return math.sqrt(float(np.dot(x.ravel(), x.ravel())))
    return np.sqrt(np.einsum('ij,ij->i', x, x))


def _opening(key, tris, coord):
    """Smallest pairwise opening (degrees) at edge key between the given
    triangles (node triples); same construction as facet_angles.edge_openings."""
    if len(tris) < 2:
        return 180.
    a0, a1, a2 = coord(key[0])
    b0, b1, b2 = coord(key[1])
    x, y, z = b0 - a0, b1 - a1, b2 - a2
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0:
        return 0.
    x, y, z = x / length, y / length, z / length
    radial = []
    for t in tris:
        o = [n for n in t if n != key[0] and n != key[1]][0]
        c0, c1, c2 = coord(o)
        v0, v1, v2 = c0 - a0, c1 - a1, c2 - a2
        d = v0 * x + v1 * y + v2 * z
        v0, v1, v2 = v0 - d * x, v1 - d * y, v2 - d * z
        n = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        radial.append((v0 / n, v1 / n, v2 / n) if n > 0 else (v0, v1, v2))
    best = 180.
    for i in range(len(radial)):
        u0, u1, u2 = radial[i]
        for j in range(i + 1, len(radial)):
            w0, w1, w2 = radial[j]
            cx, cy, cz = u1 * w2 - u2 * w1, u2 * w0 - u0 * w2, u0 * w1 - u1 * w0
            best = min(best, math.degrees(math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz),
                                                     u0 * w0 + u1 * w1 + u2 * w2)))
    return best


def _keeps_openings(op, p, f, topo, limit):
    """No-sharpening rule at every edge the operation changes (as
    d3v2p0's _keeps_openings, with scalar arithmetic and no array copies)."""
    rows = [int(r) for r in op['rows']]
    new = [tuple(int(v) for v in t) for t in op['new']]
    cache = {}
    if op['kind'] == 'relocation':
        cache[int(op['node'])] = tuple(float(v) for v in op['point'])

    def coord(n):
        c = cache.get(n)
        if c is None:
            c = cache[n] = tuple(p[n].tolist())
        return c

    def edges(tris):
        return {(min(a, b), max(a, b)) for t in tris for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    old = [tuple(int(v) for v in f[r]) for r in rows]
    before_of = {}
    for e in edges(old):
        before_of[e] = _opening(e, [tuple(int(v) for v in f[r]) for r in topo.edge_tri.get(e, [])], coord)
    before = min(before_of.values())
    removed = set(rows)
    for e in sorted(edges(new)):
        tris = [tuple(int(v) for v in f[r]) for r in topo.edge_tri.get(e, []) if r not in removed]
        tris += [t for t in new if e[0] in t and e[1] in t]
        after = _opening(e, tris, coord)
        if after < min(before_of.get(e, before), limit) - 1e-6:
            return False
    return True


def _normals(x):
    return _cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])


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
        inside &= np.einsum('ij,ij->i', _cross(e, proj - x[:, k]), n) >= 0
    best = np.where(inside[:, None], proj, np.nan)
    for k in range(3):
        c = _closest_on_segments(y, x[:, k], x[:, (k + 1) % 3])
        better = ~inside & (np.isnan(best[:, 0]) |
                            (_norm(c - y, axis=1) < _norm(best - y, axis=1)))
        best[better] = c[better]
    return best


def _distance_to_triangles(y, x):
    if not len(x):
        return np.inf
    return float(_norm(_closest_on_triangles(y, x) - y, axis=1).min())


def _project_to_triangles(y, x):
    c = _closest_on_triangles(y, x)
    return c[int(np.argmin(_norm(c - y, axis=1)))]


def _orientation_ok(old_x, new_x, cos_normal):
    no, nn = _normals(old_x), _normals(new_x)
    lo, ln = _norm(no, axis=1), _norm(nn, axis=1)
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
            if np.any(nn @ ref <= cos_normal * _norm(nn, axis=1) * _norm(ref)):
                continue
            height = abs(np.dot(p[d] - p[c], _cross(p[a] - p[c], p[b] - p[c]))) / max(_norm(ref), 1e-300)
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
                    y = segs[int(np.argmin(_norm(segs - y, axis=1)))]
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
                    dev = float(_norm(_closest_on_segments(
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
        lengths = [_norm(x[(k + 1) % 3] - x[k]) for k in range(3)]
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


class _FastTopology(_Topology):
    """d3v2p0's _Topology built with array operations; the same attributes,
    with the same dict insertion and list orders."""

    def __init__(self, p, f, alive, patch, frozen, exterior, rve_face):
        ids = np.flatnonzero(alive)
        self.f, self.patch, self.exterior, self.rve_face = f, patch, exterior, rve_face
        rows = np.repeat(ids, 3)
        nodes = f[ids].ravel()
        order = np.argsort(nodes, kind='stable')
        self.node_tri = rows[order]
        self.offsets = np.r_[0, np.cumsum(np.bincount(nodes, minlength=len(p)))]
        e = np.sort(f[ids][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        width = int(len(p)) + 1
        code = e[:, 0].astype(np.int64) * width + e[:, 1]
        _, first, inverse, counts = np.unique(code, return_index=True, return_inverse=True, return_counts=True)
        rank = np.argsort(np.argsort(first, kind='stable'), kind='stable')      # unique id -> insertion rank
        slot = rank[inverse.ravel()]
        grouped = np.argsort(slot, kind='stable')                                 # flat rows by edge, t ascending
        owners = np.repeat(ids, 3)[grouped]
        starts = np.r_[0, np.cumsum(counts[np.argsort(rank)])]
        edge_list = e[first[np.argsort(rank)]]
        keys = list(zip(edge_list[:, 0].tolist(), edge_list[:, 1].tolist()))
        owner_list = owners.tolist()
        st = starts.tolist()
        self.edge_tri = {k: owner_list[st[i]:st[i + 1]] for i, k in enumerate(keys)}
        self.fixed = np.zeros(len(p), bool)
        self.fixed[f[frozen & alive].ravel()] = True
        cnt = np.diff(starts)
        two = cnt == 2
        a_ = owners[starts[:-1]]
        b_ = owners[np.minimum(starts[:-1] + 1, len(owners) - 1)]
        curve_edge = ~two | (patch[a_] != patch[b_])
        self.curve = {}
        for i in np.flatnonzero(curve_edge).tolist():
            k = keys[i]
            ts = owner_list[st[i]:st[i + 1]]
            key = frozenset(int(patch[t]) for t in ts)
            for n in k:
                self.curve.setdefault(n, []).append((k, key))


def _best(t, p, f, topo, ang, settings):
    """d3v2p0's choice for triangle t: best candidate passing the opening rule."""
    flips, relocation, collapses, max_dev, cos_normal, extent, wedge_limit = settings
    best = None
    for op in _candidates(t, p, f, topo, ang, flips, relocation, collapses, max_dev, cos_normal, extent):
        if wedge_limit is not None and not _keeps_openings(op, p, f, topo, wedge_limit):
            continue
        if best is None or op['gain'] > best['gain']:
            best = op
    return best


def _read_nodes(t, f, topo):
    """Every node whose coordinates _best(t) can read through the rows of f."""
    near = np.unique(np.concatenate([f[t]] + [topo.neighbours(n) for n in f[t]]))
    rows = np.concatenate([topo.ring(n) for n in near])
    return np.unique(f[rows])


_W = {}


def _attach(spec):
    """name -> array view on shared memory; keeps the handles alive."""
    arrays, handles = {}, []
    for name, (shm_name, shape, dtype) in spec.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        handles.append(shm)
        arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arrays, handles


def _init(spec, statics):
    _W['limits'] = single_thread_worker()
    arrays, handles = _attach(spec)
    _W.update(arrays, handles=handles, statics=statics, key=None)


def _evaluate(task):
    """Worker: evaluate triangles ids on the live shared state. The topology is
    that of the pass start (as in d3v2p0), built once per pass and worker,
    with its row lookups pointing at the live triangle array."""
    key, ids, settings = task
    p, f = _W['p'], _W['f']
    if _W['key'] != key:
        patch, frozen, exterior, rve_face, pairs = _W['statics']
        topo = _FastTopology(p, _W['f0'], _W['alive0'], patch, frozen, exterior, rve_face)
        topo.pairs = pairs
        topo.f = f
        _W.update(key=key, topo=topo, pass_ang=_W['ang'].copy())
    topo, ang, f0 = _W['topo'], _W['pass_ang'], _W['f0']
    out = []
    for t in ids:
        reads = np.union1d(_read_nodes(t, f0, topo), _read_nodes(t, f, topo))
        out.append((_best(t, p, f, topo, ang, settings), reads))
    return out


def _shared(arrays):
    """Copy arrays into new shared memory blocks: (views, handles, spec)."""
    views, handles, spec = {}, [], {}
    for name, a in arrays.items():
        shm = shared_memory.SharedMemory(create=True, size=max(1, a.nbytes))
        handles.append(shm)
        views[name] = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        views[name][...] = a
        spec[name] = (shm.name, a.shape, a.dtype.str)
    return views, handles, spec


def improve_surface_angles(surface, enabled=True, minimum_angle=30., max_passes=10,
                           max_deviation=None, minimum_facet_angle=.1, frozen_grain_ids=(),
                           flips=True, relocation=True, collapses=True,
                           max_normal_change=60., wedge_limit=None, n_workers=None, max_rounds=8,
                           backend='auto'):
    """d3v2p0.surface_angles.improve_surface_angles with parallel candidate
    evaluation in rounds. backend / n_workers: see d3v2p1.backend (None =
    automatic); the result does not depend on them. max_rounds: evaluation rounds per pass;
    triangles still stale after it are worked on again in the next pass."""
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, (int, np.integer)) or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
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
    chosen = plan(backend, n_workers)
    if not (enabled and max_passes):
        chosen = Plan(chosen.requested, 'numpy', 1)
    p = np.array(surface.points, dtype=float)
    f = np.array(surface.triangles)
    pairs = np.asarray(surface.grain_pairs)
    exterior = np.asarray(surface.exterior, dtype=bool)
    rve_face = np.asarray(surface.rve_face)
    extent = surface.report.get('rve_dimensions')
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
    settings = (flips, relocation, collapses, max_deviation, cos_normal, extent, wedge_limit)

    a0 = triangle_angles(p, f).min(axis=1)
    before = _angle_summary(a0, minimum_angle)
    history = []
    totals = dict(flips=0, relocations=0, collapses=0)
    handles, views = [], None
    pool = WorkerPool(chosen)
    if chosen.parallel:
        views, handles, spec = _shared(dict(p=p, f=f, f0=f, alive0=alive, ang=np.zeros(len(f))))
        p, f = views['p'], views['f']
        pool = WorkerPool(chosen, initializer=_init, initargs=(spec, (patch, frozen, exterior, rve_face, pairs)))
    # A triangle with no valid operation keeps that result while no node it read
    # changes (the evaluation depends only on those nodes and the rows around them).
    none_cache = {}                          # t -> (read nodes, version)
    state = dict(version=0, rounds=0)
    changed_at = np.zeros(len(p), np.int64)
    try:
        for iteration in range(max_passes if enabled else 0):
            clock = dict(setup=-time.perf_counter(), evaluate=0., apply=0., verify=0.)
            ang = np.full(len(f), 180.)
            ang[alive] = triangle_angles(p, f[alive]).min(axis=1)
            work = np.flatnonzero(alive & (ang < minimum_angle) & ~frozen)
            if not len(work):
                break
            order = work[np.argsort(ang[work])]
            topo = _FastTopology(p, f, alive, patch, frozen, exterior, rve_face)
            topo.pairs = pairs
            if views is not None:
                views['f0'][...] = f
                views['alive0'][...] = alive
                views['ang'][...] = ang
            key = iteration
            f0 = f.copy()
            pre = {}                         # t -> (best, read nodes, version evaluated at)
            state['rounds'] = 0

            def evaluate(ids):
                t_start = time.perf_counter()
                todo = []
                for t in ids:
                    c = none_cache.get(int(t))
                    if c is not None and changed_at[c[0]].max() <= c[1]:
                        pre[int(t)] = (None, c[0], c[1])
                    else:
                        todo.append(t)
                ids = np.array(todo, dtype=np.int64)
                if not len(ids):
                    state['rounds'] += 1
                    return
                def serial(chunks):
                    return [[(_best(t, p, f, topo, ang, settings),
                              np.union1d(_read_nodes(t, f0, topo), _read_nodes(t, f, topo))) for t in c]
                            for c in chunks]
                if chosen.parallel:
                    w = chosen.workers
                    chunks = [c for c in (ids[k::4 * w] for k in range(4 * w)) if len(c)]
                    pairs_ = zip(chunks, pool.map(_evaluate, [(key, c, settings) for c in chunks],
                                                  serial=lambda: serial(chunks)))
                else:
                    pairs_ = zip([ids], serial([ids]))
                for c, res in pairs_:
                    for t, (b, reads) in zip(c, res):
                        pre[int(t)] = (b, reads, state['version'])
                        if b is None:
                            none_cache[int(t)] = (reads, state['version'])
                state['rounds'] += 1
                clock['evaluate'] += time.perf_counter() - t_start

            clock['setup'] += time.perf_counter()
            t_loop = time.perf_counter()
            reserved = np.zeros(len(p), bool)
            ops = []
            pending = order
            while len(pending) and state['rounds'] < max_rounds:
                evaluate(pending)
                deferred = []
                for t in pending:
                    if reserved[f[t]].any() or not alive[t]:
                        continue
                    best, reads, stamp = pre[int(t)]
                    if changed_at[reads].max() > stamp:
                        deferred.append(t)       # stale: evaluate again next round
                        continue
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
                    state['version'] += 1
                    changed_at[touched] = state['version']
                    _apply(best, p, f, alive)
                    ops.append(best)
                pending = np.array(deferred, dtype=order.dtype)
            clock['apply'] = time.perf_counter() - t_loop - clock['evaluate']
            if not ops:
                break
            t_verify = time.perf_counter()
            active = np.ones(len(ops), bool)
            while active.any():
                ids = np.flatnonzero(alive)
                local = np.full(len(f), -1)
                local[ids] = np.arange(len(ids))
                changed = np.concatenate([o['rows'][alive[o['rows']]] for o, a in zip(ops, active) if a])
                hits = find_surface_intersections(p, f[ids], triangle_ids=local[changed])
                near = np.unique(np.concatenate([topo.edge_tri.get((min(a, b), max(a, b)), [])
                                                 for r in changed for a, b in ((f[r, 0], f[r, 1]), (f[r, 1], f[r, 2]),
                                                                               (f[r, 2], f[r, 0]))]
                                                + [changed]).astype(np.int64))
                near = near[alive[near]]
                folds, _ = small_facet_angles(p, f[near], minimum_facet_angle)
                folds = local[near[folds]] if len(folds) else folds
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
            clock['verify'] = time.perf_counter() - t_verify
            counts = {k: sum(o['kind'] == k for o in accepted) for k in ('flip', 'relocation', 'collapse')}
            totals['flips'] += counts['flip']; totals['relocations'] += counts['relocation']
            totals['collapses'] += counts['collapse']
            history.append(dict(pass_number=iteration + 1, worked=int(len(work)), proposed=len(ops),
                                accepted=len(accepted), rejected_geometry=len(ops) - len(accepted),
                                evaluation_rounds=int(state['rounds']),
                                seconds={k: round(v, 3) for k, v in clock.items()}, **counts))
            if not accepted:
                break
    finally:
        pool.close()
        if views is not None:
            p, f = p.copy(), f.copy()
            views = None
        for h in handles:
            h.close()
            h.unlink()

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
        n_workers=int(chosen.workers), backend=chosen.report(), full_validation_required=True)
    return replace(surface, points=p, triangles=f, grain_pairs=pairs, exterior=exterior,
                   rve_face=rve_face, report=report)
