"""numba kernels for tet dihedral smoothing.

star_trials(x0, slot, trials) evaluates every trial position of one node
against the tets of its star: smallest and largest dihedral angle (degrees)
and signed volume per (trial, tet), as d3v2p0.tet_smoothing._improve_node's
evaluate() does with dihedral_angles_xyz and _volumes, in one compiled loop.
Import this module only after backend.numba_available() is True.
"""
import math
import numpy as np
from numba import njit

# faces of a tet (vertex triples) and, per edge, the two faces sharing it,
# in d3v2p0.tet_angles order
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
_EDGE_FACES = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]], dtype=np.int64)
_DEGREES = 180.0 / np.pi


@njit(cache=True, inline='always')
def _face_normal(x, a, b, c, cx, cy, cz):
    """Outward unit normal of face (a, b, c), as dihedral_angles_xyz builds it."""
    ux, uy, uz = x[b, 0] - x[a, 0], x[b, 1] - x[a, 1], x[b, 2] - x[a, 2]
    vx, vy, vz = x[c, 0] - x[a, 0], x[c, 1] - x[a, 1], x[c, 2] - x[a, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 0:
        nx, ny, nz = nx / length, ny / length, nz / length
    else:
        nx = ny = nz = 0.
    s = (x[a, 0] - cx) * nx + (x[a, 1] - cy) * ny + (x[a, 2] - cz) * nz
    side = 1. if s > 0 else (-1. if s < 0 else 0.)
    return nx * side, ny * side, nz * side


@njit(cache=True, inline='always')
def _edge_angle(n1, n2):
    cosine = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    cosine = min(max(cosine, -1.), 1.)
    return 180. - math.acos(cosine) * _DEGREES


@njit(cache=True)
def _tet_angles(x, faces, edge_faces):
    """(smallest dihedral, largest dihedral, volume) of one tet x (4, 3).
    Faces (0,1,2), (0,1,3), (0,2,3), (1,2,3); edges in d3v2p0.tet_angles order."""
    cx = (x[0, 0] + x[1, 0] + x[2, 0] + x[3, 0]) / 4
    cy = (x[0, 1] + x[1, 1] + x[2, 1] + x[3, 1]) / 4
    cz = (x[0, 2] + x[1, 2] + x[2, 2] + x[3, 2]) / 4
    f0 = _face_normal(x, 0, 1, 2, cx, cy, cz)
    f1 = _face_normal(x, 0, 1, 3, cx, cy, cz)
    f2 = _face_normal(x, 0, 2, 3, cx, cy, cz)
    f3 = _face_normal(x, 1, 2, 3, cx, cy, cz)
    e0 = _edge_angle(f0, f1)
    e1 = _edge_angle(f0, f2)
    e2 = _edge_angle(f1, f2)
    e3 = _edge_angle(f0, f3)
    e4 = _edge_angle(f1, f3)
    e5 = _edge_angle(f2, f3)
    lo = min(min(min(e0, e1), min(e2, e3)), min(e4, e5))
    hi = max(max(max(e0, e1), max(e2, e3)), max(e4, e5))
    ax, ay, az = x[1, 0] - x[0, 0], x[1, 1] - x[0, 1], x[1, 2] - x[0, 2]
    bx, by, bz = x[2, 0] - x[0, 0], x[2, 1] - x[0, 1], x[2, 2] - x[0, 2]
    dx, dy, dz = x[3, 0] - x[0, 0], x[3, 1] - x[0, 1], x[3, 2] - x[0, 2]
    volume = (ax * (by * dz - bz * dy) + ay * (bz * dx - bx * dz) + az * (bx * dy - by * dx)) / 6
    return lo, hi, volume


@njit(cache=True)
def _star_trials(x0, slot, trials, faces, edge_faces):
    k, b = x0.shape[0], trials.shape[0]
    amin = np.empty((b, k))
    amax = np.empty((b, k))
    vol = np.empty((b, k))
    x = np.empty((4, 3))
    for i in range(b):
        for j in range(k):
            for v in range(4):
                for d in range(3):
                    x[v, d] = x0[j, v, d]
            for d in range(3):
                x[slot[j], d] = trials[i, d]
            amin[i, j], amax[i, j], vol[i, j] = _tet_angles(x, faces, edge_faces)
    return amin, amax, vol


def star_trials(x0, slot, trials):
    """(amin, amax, volume), each (len(trials), len(x0)): x0 (k, 4, 3) star
    tets, slot (k,) the node's position in each tet, trials (B, 3)."""
    return _star_trials(np.ascontiguousarray(x0, dtype=np.float64), np.ascontiguousarray(slot, dtype=np.int64),
                        np.ascontiguousarray(trials, dtype=np.float64), _FACES, _EDGE_FACES)


@njit(cache=True)
def triangle_min_angles(x):
    """Smallest angle (degrees) of triangles x (N, 3, 3), as d3v2p0's
    _triangle_min_angles."""
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        best = np.inf
        for k in range(3):
            k1, k2 = (k + 1) % 3, (k + 2) % 3
            ux, uy, uz = x[i, k1, 0] - x[i, k, 0], x[i, k1, 1] - x[i, k, 1], x[i, k1, 2] - x[i, k, 2]
            vx, vy, vz = x[i, k2, 0] - x[i, k, 0], x[i, k2, 1] - x[i, k, 1], x[i, k2, 2] - x[i, k, 2]
            nu = math.sqrt(ux * ux + uy * uy + uz * uz)
            nv = math.sqrt(vx * vx + vy * vy + vz * vz)
            c = (ux * vx + uy * vy + uz * vz) / (nu * nv) if nu * nv > 0 else 1.
            c = min(max(c, -1.), 1.)
            best = min(best, math.acos(c) * _DEGREES)
        out[i] = best
    return out


@njit(cache=True)
def closest_on_segments(y, a, b):
    """Closest points from y (B, 3) to segments a-b (m, 3): (B, m, 3)."""
    out = np.empty((y.shape[0], a.shape[0], 3))
    for s in range(a.shape[0]):
        dx, dy, dz = b[s, 0] - a[s, 0], b[s, 1] - a[s, 1], b[s, 2] - a[s, 2]
        l2 = dx * dx + dy * dy + dz * dz
        for i in range(y.shape[0]):
            t = ((y[i, 0] - a[s, 0]) * dx + (y[i, 1] - a[s, 1]) * dy + (y[i, 2] - a[s, 2]) * dz) / l2 \
                if l2 > 0 else 0.
            t = min(max(t, 0.), 1.)
            out[i, s, 0] = a[s, 0] + t * dx
            out[i, s, 1] = a[s, 1] + t * dy
            out[i, s, 2] = a[s, 2] + t * dz
    return out


@njit(cache=True)
def closest_on_triangles(y, x):
    """Closest points from y (B, 3) to triangles x (m, 3, 3) and their
    distances: ((B, m, 3), (B, m)), as d3v2p0's _closest_on_triangles_batch."""
    nb, m = y.shape[0], x.shape[0]
    best = np.empty((nb, m, 3))
    best_d = np.empty((nb, m))
    for j in range(m):
        ux, uy, uz = x[j, 1, 0] - x[j, 0, 0], x[j, 1, 1] - x[j, 0, 1], x[j, 1, 2] - x[j, 0, 2]
        vx, vy, vz = x[j, 2, 0] - x[j, 0, 0], x[j, 2, 1] - x[j, 0, 1], x[j, 2, 2] - x[j, 0, 2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nn = nx * nx + ny * ny + nz * nz
        for i in range(nb):
            dist = ((y[i, 0] - x[j, 0, 0]) * nx + (y[i, 1] - x[j, 0, 1]) * ny + (y[i, 2] - x[j, 0, 2]) * nz) / nn \
                if nn > 0 else 0.
            px, py, pz = y[i, 0] - dist * nx, y[i, 1] - dist * ny, y[i, 2] - dist * nz
            inside = nn > 0
            for k in range(3):
                k1 = (k + 1) % 3
                ex, ey, ez = x[j, k1, 0] - x[j, k, 0], x[j, k1, 1] - x[j, k, 1], x[j, k1, 2] - x[j, k, 2]
                wx, wy, wz = px - x[j, k, 0], py - x[j, k, 1], pz - x[j, k, 2]
                cx, cy, cz = ey * wz - ez * wy, ez * wx - ex * wz, ex * wy - ey * wx
                if not (cx * nx + cy * ny + cz * nz >= 0):
                    inside = False
            bx, by, bz = px, py, pz
            if inside:
                ddx, ddy, ddz = px - y[i, 0], py - y[i, 1], pz - y[i, 2]
                bd = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            else:
                bd = np.inf
                for k in range(3):
                    k1 = (k + 1) % 3
                    ax, ay, az = x[j, k, 0], x[j, k, 1], x[j, k, 2]
                    sx, sy, sz = x[j, k1, 0] - ax, x[j, k1, 1] - ay, x[j, k1, 2] - az
                    l2 = sx * sx + sy * sy + sz * sz
                    t = ((y[i, 0] - ax) * sx + (y[i, 1] - ay) * sy + (y[i, 2] - az) * sz) / l2 if l2 > 0 else 0.
                    t = min(max(t, 0.), 1.)
                    qx, qy, qz = ax + t * sx, ay + t * sy, az + t * sz
                    ddx, ddy, ddz = qx - y[i, 0], qy - y[i, 1], qz - y[i, 2]
                    dc = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    if dc < bd:
                        bd = dc
                        bx, by, bz = qx, qy, qz
            best[i, j, 0], best[i, j, 1], best[i, j, 2] = bx, by, bz
            best_d[i, j] = bd
    return best, best_d
