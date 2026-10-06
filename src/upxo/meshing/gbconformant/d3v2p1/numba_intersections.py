"""numba kernels for d3v2p0's triangle intersection search.

pairs_intersect(points, triangles, pairs, tolerance) gives the same result
as d3v2p0.surface_intersections._intersecting_pairs: the same predicates,
tolerances and order of arithmetic, evaluated per pair in a compiled loop
(threads over pairs) that stops at the first detected contact. Import this
module only after backend.numba_available() is True.

find_pairs() runs the whole search (candidate grid, d3v2p0's candidate
rule and the pair test) in one parallel kernel, without candidate arrays.
"""
import math
import numpy as np
import numba
from numba import njit, prange


@njit(cache=True, inline='always')
def _sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


@njit(cache=True, inline='always')
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True, inline='always')
def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


@njit(cache=True, inline='always')
def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


@njit(cache=True)
def _inside(x, t0, t1, t2, tol):
    """(closed inside, strict inside) of x against triangle t, as d3v2p0's inside()."""
    u = _sub(t1, t0)
    v = _sub(t2, t0)
    w = _sub(x, t0)
    uu = _dot(u, u)
    uv = _dot(u, v)
    vv = _dot(v, v)
    wu = _dot(w, u)
    wv = _dot(w, v)
    den = uu * vv - uv * uv
    if den > 0:
        s = (vv * wu - uv * wv) / den
        q = (uu * wv - uv * wu) / den
    else:
        s = -np.inf
        q = -np.inf
    eps = tol / max(math.sqrt(max(uu, vv)), tol)
    closed = (s >= -eps) and (q >= -eps) and (s + q <= 1 + eps)
    strict = (s > eps) and (q > eps) and (s + q < 1 - eps)
    return closed, strict


@njit(cache=True)
def _allowed(x, a, shared_a, tol):
    """Contact at x is a shared vertex, or lies on a shared edge (of triangle a)."""
    for i in range(3):
        if shared_a[i] and _norm(_sub(a[i], x)) <= tol:
            return True
    for e in range(3):
        i, j = e, (e + 1) % 3
        if not (shared_a[i] and shared_a[j]):
            continue
        edge = _sub(a[j], a[i])
        l2 = _dot(edge, edge)
        t = _dot(_sub(x, a[i]), edge) / l2 if l2 > 0 else 0.
        t = min(max(t, 0.), 1.)
        proj = (a[i][0] + t * edge[0], a[i][1] + t * edge[1], a[i][2] + t * edge[2])
        if _norm(_sub(x, proj)) <= tol:
            return True
    return False


@njit(cache=True)
def _pair_hit(a, b, shared_a, tol):
    for side in range(2):
        source = a if side == 0 else b
        target = b if side == 0 else a
        t0, t1, t2 = target[0], target[1], target[2]
        normal = _cross(_sub(t1, t0), _sub(t2, t0))
        length = _norm(normal)
        if length > 0:
            normal = (normal[0] / length, normal[1] / length, normal[2] / length)
        else:
            normal = (0., 0., 0.)
        d = (_dot(_sub(source[0], t0), normal), _dot(_sub(source[1], t0), normal),
             _dot(_sub(source[2], t0), normal))
        for e in range(3):
            i, j = e, (e + 1) % 3
            d0, d1 = d[i], d[j]
            den = d0 - d1
            if abs(den) > tol:
                ratio = d0 / den
                if 0 <= ratio <= 1:
                    x = (source[i][0] + ratio * (source[j][0] - source[i][0]),
                         source[i][1] + ratio * (source[j][1] - source[i][1]),
                         source[i][2] + ratio * (source[j][2] - source[i][2]))
                    if _inside(x, t0, t1, t2, tol)[0] and not _allowed(x, a, shared_a, tol):
                        return True
            # nonadjacent vertex-on-facet and coplanar contacts
            if abs(d0) <= tol:
                x = source[i]
                if _inside(x, t0, t1, t2, tol)[0] and not _allowed(x, a, shared_a, tol):
                    return True
        if abs(d[0]) <= tol and abs(d[1]) <= tol and abs(d[2]) <= tol:
            c = ((source[0][0] + source[1][0] + source[2][0]) / 3,
                 (source[0][1] + source[1][1] + source[2][1]) / 3,
                 (source[0][2] + source[1][2] + source[2][2]) / 3)
            if _inside(c, t0, t1, t2, tol)[1]:
                return True
    # coplanar proper edge crossings
    na = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    norm = _norm(na)
    unit = (na[0] / norm, na[1] / norm, na[2] / norm) if norm > 0 else (0., 0., 0.)
    for k in range(3):
        if abs(_dot(_sub(b[k], a[0]), unit)) > tol:
            return False
    tol2 = tol * tol
    for ea in range(3):
        i, j = ea, (ea + 1) % 3
        for eb in range(3):
            k, l = eb, (eb + 1) % 3
            s1 = _dot(_cross(_sub(a[j], a[i]), _sub(b[k], a[i])), unit)
            s2 = _dot(_cross(_sub(a[j], a[i]), _sub(b[l], a[i])), unit)
            r1 = _dot(_cross(_sub(b[l], b[k]), _sub(a[i], b[k])), unit)
            r2 = _dot(_cross(_sub(b[l], b[k]), _sub(a[j], b[k])), unit)
            if s1 * s2 < -tol2 and r1 * r2 < -tol2:
                return True
    return False


@njit(cache=True, parallel=True)
def pairs_intersect(points, triangles, pairs, tolerance):
    """Boolean per candidate pair: True where the two triangles intersect
    other than at a shared vertex or along a shared edge."""
    out = np.zeros(len(pairs), dtype=np.bool_)
    for k in prange(len(pairs)):
        fa = triangles[pairs[k, 0]]
        fb = triangles[pairs[k, 1]]
        a = ((points[fa[0], 0], points[fa[0], 1], points[fa[0], 2]),
             (points[fa[1], 0], points[fa[1], 1], points[fa[1], 2]),
             (points[fa[2], 0], points[fa[2], 1], points[fa[2], 2]))
        b = ((points[fb[0], 0], points[fb[0], 1], points[fb[0], 2]),
             (points[fb[1], 0], points[fb[1], 1], points[fb[1], 2]),
             (points[fb[2], 0], points[fb[2], 1], points[fb[2], 2]))
        shared_a = (fa[0] == fb[0] or fa[0] == fb[1] or fa[0] == fb[2],
                    fa[1] == fb[0] or fa[1] == fb[1] or fa[1] == fb[2],
                    fa[2] == fb[0] or fa[2] == fb[1] or fa[2] == fb[2])
        out[k] = _pair_hit(a, b, shared_a, tolerance)
    return out


# ---------------------------------------------------------------- candidate search
# The candidate rule is d3v2p0's: bounding boxes overlap within the tolerance,
# and the centres are no farther apart than the query triangle's radius plus
# the largest radius in the other triangle's radius band plus the tolerance.

@njit(cache=True)
def _cell_range(lo, hi, pad, origin, h, dims):
    i0 = min(max(int((lo[0] - pad - origin[0]) / h), 0), dims[0] - 1)
    j0 = min(max(int((lo[1] - pad - origin[1]) / h), 0), dims[1] - 1)
    k0 = min(max(int((lo[2] - pad - origin[2]) / h), 0), dims[2] - 1)
    i1 = min(max(int((hi[0] + pad - origin[0]) / h), 0), dims[0] - 1)
    j1 = min(max(int((hi[1] + pad - origin[1]) / h), 0), dims[1] - 1)
    k1 = min(max(int((hi[2] + pad - origin[2]) / h), 0), dims[2] - 1)
    return i0, j0, k0, i1, j1, k1


@njit(cache=True)
def _grid(lo, hi, origin, h, dims):
    """Cell -> triangle CSR: every triangle is listed in each cell its box covers."""
    nx, ny = dims[0], dims[1]
    counts = np.zeros(dims[0] * dims[1] * dims[2] + 1, dtype=np.int64)
    for t in range(len(lo)):
        i0, j0, k0, i1, j1, k1 = _cell_range(lo[t], hi[t], 0., origin, h, dims)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    counts[1 + i + nx * (j + ny * k)] += 1
    ptr = np.cumsum(counts)
    fill = ptr[:-1].copy()
    items = np.empty(ptr[-1], dtype=np.int64)
    for t in range(len(lo)):
        i0, j0, k0, i1, j1, k1 = _cell_range(lo[t], hi[t], 0., origin, h, dims)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    c = i + nx * (j + ny * k)
                    items[fill[c]] = t
                    fill[c] += 1
    return ptr, items


@njit(cache=True)
def _vertices(points, f):
    return ((points[f[0], 0], points[f[0], 1], points[f[0], 2]),
            (points[f[1], 0], points[f[1], 1], points[f[1], 2]),
            (points[f[2], 0], points[f[2], 1], points[f[2], 2]))


@njit(cache=True)
def _query(a, points, triangles, centres, radii, band_max, lo, hi, ptr, items, origin, h, dims, full, tol,
           seen, out, start):
    """Hits of query triangle a, written from out[start] when start >= 0.
    Returns the number of hits. seen is this thread's marker array."""
    nx, ny = dims[0], dims[1]
    i0, j0, k0, i1, j1, k1 = _cell_range(lo[a], hi[a], tol, origin, h, dims)
    hits = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                c = i + nx * (j + ny * k)
                for m in range(ptr[c], ptr[c + 1]):
                    b = items[m]
                    if b == a or (full and b < a) or seen[b] == a:
                        continue
                    seen[b] = a
                    if not (lo[a, 0] <= hi[b, 0] + tol and lo[a, 1] <= hi[b, 1] + tol
                            and lo[a, 2] <= hi[b, 2] + tol and lo[b, 0] <= hi[a, 0] + tol
                            and lo[b, 1] <= hi[a, 1] + tol and lo[b, 2] <= hi[a, 2] + tol):
                        continue
                    dx = centres[a, 0] - centres[b, 0]
                    dy = centres[a, 1] - centres[b, 1]
                    dz = centres[a, 2] - centres[b, 2]
                    if math.sqrt(dx * dx + dy * dy + dz * dz) > radii[a] + band_max[b] + tol:
                        continue
                    # d3v2p0 tests every candidate as (smaller id, larger id)
                    s, t = (a, b) if a < b else (b, a)
                    fs, ft = triangles[s], triangles[t]
                    shared = (fs[0] == ft[0] or fs[0] == ft[1] or fs[0] == ft[2],
                              fs[1] == ft[0] or fs[1] == ft[1] or fs[1] == ft[2],
                              fs[2] == ft[0] or fs[2] == ft[1] or fs[2] == ft[2])
                    if _pair_hit(_vertices(points, fs), _vertices(points, ft), shared, tol):
                        if start >= 0:
                            out[start + hits, 0] = s
                            out[start + hits, 1] = t
                        hits += 1
    return hits


@njit(cache=True, parallel=True)
def _search_pass(query_ids, points, triangles, centres, radii, band_max, lo, hi, ptr, items, origin, h, dims,
                 full, tol, offsets, out, n_threads):
    """Per query: count hits (offsets[q] < 0) or write them from offsets[q]."""
    counts = np.zeros(len(query_ids), dtype=np.int64)
    seen = np.full((n_threads, len(triangles)), -1, dtype=np.int64)
    for q in prange(len(query_ids)):
        tid = numba.get_thread_id()
        counts[q] = _query(query_ids[q], points, triangles, centres, radii, band_max, lo, hi, ptr, items, origin,
                           h, dims, full, tol, seen[tid], out, offsets[q])
    return counts


def find_pairs(points, triangles, query_ids, full, tolerance, band_max):
    """Intersecting pairs (smaller id, larger id), possibly repeated, for the
    query triangles: d3v2p0's candidate rule and pair test, in numba.
    band_max: per triangle, the largest radius in its radius band."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    triangles = np.ascontiguousarray(triangles, dtype=np.int64)
    query_ids = np.ascontiguousarray(query_ids, dtype=np.int64)
    band_max = np.ascontiguousarray(band_max, dtype=np.float64)
    xyz = points[triangles]
    centres = xyz.mean(axis=1)
    radii = np.linalg.norm(xyz - centres[:, None], axis=2).max(axis=1)
    lo, hi = xyz.min(axis=1), xyz.max(axis=1)
    h = float(np.median((hi - lo).max(axis=1)))
    span = (hi.max(axis=0) - lo.min(axis=0)) + 2 * tolerance
    if not np.isfinite(h) or h <= 0:
        h = float(max(span.max(), 1.))
    while np.prod(np.floor(span / h) + 1) > 4e7:           # bounded grid size
        h *= 2
    dims = (np.floor(span / h) + 1).astype(np.int64)
    origin = lo.min(axis=0) - tolerance
    ptr, items = _grid(lo, hi, origin, h, dims)
    empty = np.empty((0, 2), dtype=np.int64)
    n_threads = numba.config.NUMBA_NUM_THREADS
    args = (query_ids, points, triangles, centres, radii, band_max, lo, hi, ptr, items, origin, h, dims,
            bool(full), float(tolerance))
    counts = _search_pass(*args, np.full(len(query_ids), -1, dtype=np.int64), empty, n_threads)
    total = int(counts.sum())
    if not total:
        return empty
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    out = np.empty((total, 2), dtype=np.int64)
    _search_pass(*args, offsets, out, n_threads)
    return out
