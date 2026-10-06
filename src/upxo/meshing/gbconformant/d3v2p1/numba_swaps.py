"""numba kernels for tet face and edge swaps.

evaluate_swap() orients the new tets and applies d3v2p0.tet_swaps'
acceptance test (positive volumes, unchanged total volume, higher worst
quality, no more tets outside the limits) in one compiled call;
measure_tets() gives smallest/largest dihedral and quality for many tets.
Import this module only after backend.numba_available() is True.
"""
import numpy as np
from numba import njit, prange
from .numba_smoothing import _tet_angles, _FACES, _EDGE_FACES


@njit(cache=True, inline='always')
def _volume(p, a, b, c, d):
    """d3v2p0 _signed_volume of tet (a, b, c, d)."""
    ax, ay, az = p[b, 0] - p[a, 0], p[b, 1] - p[a, 1], p[b, 2] - p[a, 2]
    bx, by, bz = p[c, 0] - p[a, 0], p[c, 1] - p[a, 1], p[c, 2] - p[a, 2]
    dx, dy, dz = p[d, 0] - p[a, 0], p[d, 1] - p[a, 1], p[d, 2] - p[a, 2]
    return (ax * (by * dz - bz * dy) + ay * (bz * dx - bx * dz) + az * (bx * dy - by * dx)) / 6


@njit(cache=True)
def _quality(p, tet, target, span, x, faces, edge_faces):
    for v in range(4):
        for k in range(3):
            x[v, k] = p[tet[v], k]
    lo, hi, _ = _tet_angles(x, faces, edge_faces)
    return min(lo / target, (180. - hi) / span)


@njit(cache=True)
def _evaluate(p, old, new, target, span, faces, edge_faces):
    """Orient new in place; return (accepted, worst new quality)."""
    x = np.empty((4, 3))
    total_new = 0.
    smallest = np.inf
    for i in range(new.shape[0]):
        v = _volume(p, new[i, 0], new[i, 1], new[i, 2], new[i, 3])
        if v < 0:                                   # d3v2p0 _orient: swap the first two nodes
            new[i, 0], new[i, 1] = new[i, 1], new[i, 0]
            v = _volume(p, new[i, 0], new[i, 1], new[i, 2], new[i, 3])
        smallest = min(smallest, v)
        total_new += v
    if smallest <= 0:
        return False, 0.
    total_old = 0.
    for i in range(old.shape[0]):
        total_old += _volume(p, old[i, 0], old[i, 1], old[i, 2], old[i, 3])
    atol = 1e-12 * max(1., abs(total_old))
    if not abs(total_new - total_old) <= atol + 1e-9 * abs(total_old):
        return False, 0.
    old_min, old_out = np.inf, 0
    for i in range(old.shape[0]):
        q = _quality(p, old[i], target, span, x, faces, edge_faces)
        old_min = min(old_min, q)
        old_out += q < 1.
    new_min, new_out = np.inf, 0
    for i in range(new.shape[0]):
        q = _quality(p, new[i], target, span, x, faces, edge_faces)
        new_min = min(new_min, q)
        new_out += q < 1.
    if new_min <= old_min + 1e-9 or new_out > old_out:
        return False, 0.
    return True, new_min


def evaluate_swap(p, old, new, target, span):
    """(accepted, worst new quality, oriented new tets) for replacing tets
    old (k, 4) by new (m, 4); same test as d3v2p0.tet_swaps.acceptable."""
    new = np.array(new, dtype=np.int64)
    ok, value = _evaluate(p, np.asarray(old, dtype=np.int64), new, float(target), float(span),
                          _FACES, _EDGE_FACES)
    return ok, value, new


@njit(cache=True, parallel=True)
def _measure(p, tets, target, span, faces, edge_faces):
    n = tets.shape[0]
    mn, mx, q = np.empty(n), np.empty(n), np.empty(n)
    for i in prange(n):
        x = np.empty((4, 3))
        for v in range(4):
            for k in range(3):
                x[v, k] = p[tets[i, v], k]
        lo, hi, _ = _tet_angles(x, faces, edge_faces)
        mn[i], mx[i] = lo, hi
        q[i] = min(lo / target, (180. - hi) / span)
    return mn, mx, q


def measure_tets(p, tets, target, span):
    """(smallest dihedral, largest dihedral, quality) per tet."""
    tets = np.ascontiguousarray(tets, dtype=np.int64).reshape(-1, 4)
    return _measure(p, tets, float(target), float(span), _FACES, _EDGE_FACES)
