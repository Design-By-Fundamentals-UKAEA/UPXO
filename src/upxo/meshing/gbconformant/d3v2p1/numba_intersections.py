"""numba kernel for d3v2p0's triangle-pair intersection test.

pairs_intersect(points, triangles, pairs, tolerance) gives the same result
as d3v2p0.surface_intersections._intersecting_pairs: the same predicates,
tolerances and order of arithmetic, evaluated per pair in a compiled loop
(threads over pairs) that stops at the first detected contact. Import this
module only after backend.numba_available() is True.
"""
import math
import numpy as np
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
