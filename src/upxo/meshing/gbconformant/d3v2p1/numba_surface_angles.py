"""Compiled candidate search for surface-angle repair (numba).

Opt-in tier of d3v2p1.surface_angles.improve_surface_angles: that function runs it only when the compiled tier
is asked for explicitly (backend='numba', the pipeline's BACKEND setting or UPXO_BACKEND) and ``compiled_ready()``
says it works here; otherwise, and on any machine without numba, the numpy / parallel tiers run. 'auto' does not
select it. Importing this module needs numba.

``evaluate`` is d3v2p0.surface_angles._candidates + _keeps_openings + the choice of the best operation, for one
triangle, compiled. It reads the same things d3v2p0 reads: the live points and triangles, the angles and the
adjacency of the start of the pass (stale during the pass by design: the reservation of touched nodes keeps
the operations of one pass apart), and it chooses the same way (candidates in the same order, the first of
equal gains wins). The no-sharpening rule is d3v2p0's (the opening before an operation is measured at the old
position of a moved node).

The arithmetic is the same formulas as numpy's but not always in the same order of additions, so a choice that
hangs on the last digit of a float can differ from d3v2p0's; the checks that accept an operation afterwards
(intersections, facet openings, orientation, volumes) are the unchanged d3v2p1 ones.
"""
import math
import time
from dataclasses import replace

import numpy as np
from numba import njit

from ..d3v2p0.surface_angles import triangle_angles, _apply, _undo, _angle_summary
from .backend import check_backend, check_workers
from .facet_angles import small_facet_angles
from .mesh_tables import unique_rows
from .surface_intersections import find_surface_intersections

KIND_NONE, KIND_FLIP, KIND_RELOCATION, KIND_COLLAPSE = 0, 1, 2, 3
TOLERANCE = 1e-6


# --- small geometry ------------------------------------------------------------------------------------------
@njit(cache=True)
def _min_angle(ax, ay, az, bx, by, bz, cx, cy, cz):
    """Smallest interior angle (degrees) of a triangle, as d3v2p0.surface_angles.triangle_angles(...).min()."""
    best = 180.0
    for k in range(3):
        if k == 0:
            ox, oy, oz, px, py, pz, qx, qy, qz = ax, ay, az, bx, by, bz, cx, cy, cz
        elif k == 1:
            ox, oy, oz, px, py, pz, qx, qy, qz = bx, by, bz, cx, cy, cz, ax, ay, az
        else:
            ox, oy, oz, px, py, pz, qx, qy, qz = cx, cy, cz, ax, ay, az, bx, by, bz
        ux, uy, uz = px - ox, py - oy, pz - oz
        vx, vy, vz = qx - ox, qy - oy, qz - oz
        den = math.sqrt(ux * ux + uy * uy + uz * uz) * math.sqrt(vx * vx + vy * vy + vz * vz)
        c = (ux * vx + uy * vy + uz * vz) / den if den > 0 else 1.0
        c = min(1.0, max(-1.0, c))
        angle = math.degrees(math.acos(c))
        if angle < best:
            best = angle
    return best


@njit(cache=True)
def _closest_on_segment(yx, yy, yz, ax, ay, az, bx, by, bz):
    dx, dy, dz = bx - ax, by - ay, bz - az
    l2 = dx * dx + dy * dy + dz * dz
    t = ((yx - ax) * dx + (yy - ay) * dy + (yz - az) * dz) / l2 if l2 > 0 else 0.0
    t = min(1.0, max(0.0, t))
    return ax + t * dx, ay + t * dy, az + t * dz


@njit(cache=True)
def _closest_on_triangle(yx, yy, yz, x):
    """Closest point from y to the triangle x (3, 3), as d3v2p0.surface_angles._closest_on_triangles."""
    e1x, e1y, e1z = x[1, 0] - x[0, 0], x[1, 1] - x[0, 1], x[1, 2] - x[0, 2]
    e2x, e2y, e2z = x[2, 0] - x[0, 0], x[2, 1] - x[0, 1], x[2, 2] - x[0, 2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    nn = nx * nx + ny * ny + nz * nz
    s = ((yx - x[0, 0]) * nx + (yy - x[0, 1]) * ny + (yz - x[0, 2]) * nz) / nn if nn > 0 else 0.0
    qx, qy, qz = yx - s * nx, yy - s * ny, yz - s * nz
    inside = nn > 0
    for k in range(3):
        k2 = (k + 1) % 3
        ex, ey, ez = x[k2, 0] - x[k, 0], x[k2, 1] - x[k, 1], x[k2, 2] - x[k, 2]
        wx, wy, wz = qx - x[k, 0], qy - x[k, 1], qz - x[k, 2]
        cx, cy, cz = ey * wz - ez * wy, ez * wx - ex * wz, ex * wy - ey * wx
        if not (cx * nx + cy * ny + cz * nz >= 0):
            inside = False
    if inside:
        return qx, qy, qz
    bx, by, bz = np.nan, np.nan, np.nan
    for k in range(3):
        k2 = (k + 1) % 3
        cx, cy, cz = _closest_on_segment(yx, yy, yz, x[k, 0], x[k, 1], x[k, 2], x[k2, 0], x[k2, 1], x[k2, 2])
        if math.isnan(bx):
            bx, by, bz = cx, cy, cz
        else:
            dc = math.sqrt((cx - yx) ** 2 + (cy - yy) ** 2 + (cz - yz) ** 2)
            db = math.sqrt((bx - yx) ** 2 + (by - yy) ** 2 + (bz - yz) ** 2)
            if dc < db:
                bx, by, bz = cx, cy, cz
    return bx, by, bz


@njit(cache=True)
def _project_to_ring(yx, yy, yz, x):
    """The closest point to y on any triangle of x (r, 3, 3); the first of equal distances."""
    bx, by, bz, bd = 0.0, 0.0, 0.0, np.inf
    for i in range(x.shape[0]):
        cx, cy, cz = _closest_on_triangle(yx, yy, yz, x[i])
        d = math.sqrt((cx - yx) ** 2 + (cy - yy) ** 2 + (cz - yz) ** 2)
        if i == 0 or d < bd:
            bx, by, bz, bd = cx, cy, cz, d
    return bx, by, bz


@njit(cache=True)
def _distance_to_ring(yx, yy, yz, x):
    best = np.inf
    for i in range(x.shape[0]):
        cx, cy, cz = _closest_on_triangle(yx, yy, yz, x[i])
        d = math.sqrt((cx - yx) ** 2 + (cy - yy) ** 2 + (cz - yz) ** 2)
        if d < best:
            best = d
    return best


@njit(cache=True)
def _ring_coordinates(p, f, rows):
    x = np.empty((len(rows), 3, 3))
    for i in range(len(rows)):
        for k in range(3):
            v = f[rows[i], k]
            x[i, k, 0] = p[v, 0]
            x[i, k, 1] = p[v, 1]
            x[i, k, 2] = p[v, 2]
    return x


@njit(cache=True)
def _normal_of(x, i):
    ax, ay, az = x[i, 1, 0] - x[i, 0, 0], x[i, 1, 1] - x[i, 0, 1], x[i, 1, 2] - x[i, 0, 2]
    bx, by, bz = x[i, 2, 0] - x[i, 0, 0], x[i, 2, 1] - x[i, 0, 1], x[i, 2, 2] - x[i, 0, 2]
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@njit(cache=True)
def _orientation_ok(old_x, new_x, cos_normal):
    """d3v2p0.surface_angles._orientation_ok."""
    m = old_x.shape[0]
    lo = np.empty(m)
    ln = np.empty(m)
    dot = np.empty(m)
    longest = 0.0
    for i in range(m):
        ox, oy, oz = _normal_of(old_x, i)
        nx, ny, nz = _normal_of(new_x, i)
        lo[i] = math.sqrt(ox * ox + oy * oy + oz * oz)
        ln[i] = math.sqrt(nx * nx + ny * ny + nz * nz)
        dot[i] = ox * nx + oy * ny + oz * nz
        if lo[i] > longest:
            longest = lo[i]
    floor = 1e-14 * max(1.0, longest)
    for i in range(m):
        if ln[i] <= floor:
            return False
    for i in range(m):
        if not (dot[i] > cos_normal * lo[i] * ln[i]):
            return False
    return True


@njit(cache=True)
def _min_angle_of(x):
    best = 180.0
    for i in range(x.shape[0]):
        a = _min_angle(x[i, 0, 0], x[i, 0, 1], x[i, 0, 2], x[i, 1, 0], x[i, 1, 1], x[i, 1, 2],
                       x[i, 2, 0], x[i, 2, 1], x[i, 2, 2])
        if a < best:
            best = a
    return best


# --- adjacency of the start of the pass -------------------------------------------------------------------------
@njit(cache=True)
def _edge_id(a, b, ekey_sorted, esort, width):
    lo, hi = (a, b) if a < b else (b, a)
    key = lo * width + hi
    i = np.searchsorted(ekey_sorted, key)
    if i < len(ekey_sorted) and ekey_sorted[i] == key:
        return esort[i]
    return -1


@njit(cache=True)
def _node_kind(n, fixed, curve_off, curve_edge, curve_key, edge_u, edge_v):
    """(0, ., .) fixed; (1, ., .) interior; (2, n1, n2) on a curve. d3v2p0 _Topology.kind."""
    if fixed[n]:
        return 0, -1, -1
    c0, c1 = curve_off[n], curve_off[n + 1]
    if c1 == c0:
        return 1, -1, -1
    if c1 - c0 == 2 and curve_key[c0] == curve_key[c0 + 1]:
        e1, e2 = curve_edge[c0], curve_edge[c0 + 1]
        n1 = edge_u[e1] if edge_v[e1] == n else edge_v[e1]
        n2 = edge_u[e2] if edge_v[e2] == n else edge_v[e2]
        return 2, n1, n2
    return 0, -1, -1


@njit(cache=True)
def _neighbours(n, f, node_off, node_rows):
    """Sorted distinct nodes of the triangles around n, without n (live triangles, ring of the pass start)."""
    r0, r1 = node_off[n], node_off[n + 1]
    buf = np.empty(3 * (r1 - r0), np.int64)
    c = 0
    for q in range(r0, r1):
        row = node_rows[q]
        for k in range(3):
            v = f[row, k]
            if v != n:
                buf[c] = v
                c += 1
    return np.unique(buf[:c])


@njit(cache=True)
def _plane_faces(n, node_off, node_rows, exterior, rve_face):
    """RVE faces (ascending) of the cap triangles around n, as a 6-flag array."""
    flags = np.zeros(6, np.bool_)
    for q in range(node_off[n], node_off[n + 1]):
        row = node_rows[q]
        if exterior[row]:
            flags[rve_face[row]] = True
    return flags


# --- openings (no-sharpening rule) --------------------------------------------------------------------------------
@njit(cache=True)
def _opening(a, b, tris, count, p, ov_node, ov_x, ov_y, ov_z):
    """Smallest pairwise opening (degrees) at the edge a-b (a < b) between the triangles tris[:count].

    d3v2p0.facet_angles.edge_openings; the node ov_node (if >= 0) sits at (ov_x, ov_y, ov_z)."""
    if count < 2:
        return 180.0
    if a == ov_node:
        ax, ay, az = ov_x, ov_y, ov_z
    else:
        ax, ay, az = p[a, 0], p[a, 1], p[a, 2]
    if b == ov_node:
        bx, by, bz = ov_x, ov_y, ov_z
    else:
        bx, by, bz = p[b, 0], p[b, 1], p[b, 2]
    x, y, z = bx - ax, by - ay, bz - az
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0:
        return 0.0
    x, y, z = x / length, y / length, z / length
    radial = np.empty((count, 3))
    for i in range(count):
        o = -1
        for k in range(3):
            if tris[i, k] != a and tris[i, k] != b:
                o = tris[i, k]
                break
        if o == ov_node:
            vx, vy, vz = ov_x - ax, ov_y - ay, ov_z - az
        else:
            vx, vy, vz = p[o, 0] - ax, p[o, 1] - ay, p[o, 2] - az
        d = vx * x + vy * y + vz * z
        vx, vy, vz = vx - d * x, vy - d * y, vz - d * z
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        if n > 0:
            radial[i, 0], radial[i, 1], radial[i, 2] = vx / n, vy / n, vz / n
        else:
            radial[i, 0], radial[i, 1], radial[i, 2] = vx, vy, vz
    best = 180.0
    for i in range(count):
        for j in range(i + 1, count):
            cx = radial[i, 1] * radial[j, 2] - radial[i, 2] * radial[j, 1]
            cy = radial[i, 2] * radial[j, 0] - radial[i, 0] * radial[j, 2]
            cz = radial[i, 0] * radial[j, 1] - radial[i, 1] * radial[j, 0]
            angle = math.degrees(math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz),
                                            radial[i, 0] * radial[j, 0] + radial[i, 1] * radial[j, 1]
                                            + radial[i, 2] * radial[j, 2]))
            if angle < best:
                best = angle
    return best


@njit(cache=True)
def _edges_of(tris):
    """Distinct sorted edges (lo, hi) of the triangles, in ascending order."""
    m = tris.shape[0]
    keys = np.empty(3 * m, np.int64)
    width = np.int64(1) << 31
    c = 0
    for i in range(m):
        for k in range(3):
            u, v = tris[i, k], tris[i, (k + 1) % 3]
            lo, hi = (u, v) if u < v else (v, u)
            keys[c] = lo * width + hi
            c += 1
    keys = np.unique(keys)
    out = np.empty((len(keys), 2), np.int64)
    for i in range(len(keys)):
        out[i, 0] = keys[i] // width
        out[i, 1] = keys[i] % width
    return out


@njit(cache=True)
def _keeps_openings(kind, rows, new, node, px, py, pz, limit, p, f, ekey_sorted, esort, width, eoff, erows, capacity):
    """d3v2p0.surface_angles._keeps_openings: no opening may get sharper than the limit allows."""
    old_tris = np.empty((len(rows), 3), np.int64)
    for i in range(len(rows)):
        for k in range(3):
            old_tris[i, k] = f[rows[i], k]
    old_edges = _edges_of(old_tris)
    before_value = np.empty(len(old_edges))
    before = np.inf
    scratch = np.empty((capacity, 3), np.int64)             # capacity: most owners of an edge + the new triangles
    for i in range(len(old_edges)):
        e = _edge_id(old_edges[i, 0], old_edges[i, 1], ekey_sorted, esort, width)
        count = 0
        if e >= 0:
            for q in range(eoff[e], eoff[e + 1]):
                for k in range(3):
                    scratch[count, k] = f[erows[q], k]
                count += 1
        value = _opening(old_edges[i, 0], old_edges[i, 1], scratch, count, p, -1, 0.0, 0.0, 0.0)
        before_value[i] = value
        if value < before:
            before = value
    ov_node = node if kind == KIND_RELOCATION else -1
    new_edges = _edges_of(new)
    for i in range(len(new_edges)):
        a, b = new_edges[i, 0], new_edges[i, 1]
        e = _edge_id(a, b, ekey_sorted, esort, width)
        count = 0
        if e >= 0:
            for q in range(eoff[e], eoff[e + 1]):
                row = erows[q]
                removed = False
                for j in range(len(rows)):
                    if rows[j] == row:
                        removed = True
                        break
                if not removed:
                    for k in range(3):
                        scratch[count, k] = f[row, k]
                    count += 1
        for j in range(new.shape[0]):
            has_a = False
            has_b = False
            for k in range(3):
                if new[j, k] == a:
                    has_a = True
                if new[j, k] == b:
                    has_b = True
            if has_a and has_b:
                for k in range(3):
                    scratch[count, k] = new[j, k]
                count += 1
        after = _opening(a, b, scratch, count, p, ov_node, px, py, pz)
        previous = before
        for j in range(len(old_edges)):
            if old_edges[j, 0] == a and old_edges[j, 1] == b:
                previous = before_value[j]
                break
        if after < min(previous, limit) - TOLERANCE:
            return False
    return True


# --- collapse: every grain's shell stays a single closed fan --------------------------------------------------------
@njit(cache=True)
def _fans_ok(node, tris, pairs, exterior):
    """d3v2p0.surface_angles._grain_fans_ok."""
    m = tris.shape[0]
    labels = np.empty(2 * m, np.int64)
    c = 0
    for i in range(m):
        labels[c] = pairs[i, 0]
        c += 1
        if not exterior[i]:
            labels[c] = pairs[i, 1]
            c += 1
    grains = np.unique(labels[:c])
    for gi in range(len(grains)):
        g = grains[gi]
        member = np.empty(m, np.int64)
        count = 0
        for i in range(m):
            if pairs[i, 0] == g or ((not exterior[i]) and pairs[i, 1] == g):
                member[count] = i
                count += 1
        spoke_node = np.empty(3 * count, np.int64)
        spoke_row = np.empty(3 * count, np.int64)
        s = 0
        for q in range(count):
            for k in range(3):
                if tris[member[q], k] != node:
                    spoke_node[s] = tris[member[q], k]
                    spoke_row[s] = q
                    s += 1
        order = np.argsort(spoke_node[:s], kind='mergesort')
        parent = np.arange(count)
        i = 0
        while i < s:
            j = i
            while j < s and spoke_node[order[j]] == spoke_node[order[i]]:
                j += 1
            if j - i != 2:
                return False
            a = spoke_row[order[i]]
            b = spoke_row[order[i + 1]]
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[b] = a
            i = j
        root = 0
        while parent[root] != root:
            root = parent[root]
        for q in range(count):
            r = q
            while parent[r] != r:
                r = parent[r]
            if r != root:
                return False
    return True


# --- the candidate search for one triangle ---------------------------------------------------------------------------
@njit(cache=True)
def _sorted_intersection_equals(a, b, c):
    """True when the sorted distinct arrays a and b have as common elements exactly the sorted distinct array c."""
    common = np.empty(min(len(a), len(b)), np.int64)
    n = 0
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            common[n] = a[i]
            n += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    if n != len(c):
        return False
    for k in range(n):
        if common[k] != c[k]:
            return False
    return True


@njit(cache=True)
def evaluate(t, p, f, ang, node_off, node_rows, ekey_sorted, esort, width, eoff, erows, edge_u, edge_v,
             fixed, curve_off, curve_edge, curve_key, patch, pairs, exterior, rve_face, extent, has_extent,
             flips, relocation, collapses, max_dev, cos_normal, wedge_limit, max_owner):
    """The best operation for triangle t, as d3v2p0 chooses it. Returns
    (kind, rows, new, node, x, y, z, gain, keep, gone); kind is KIND_NONE when no candidate passes.

    wedge_limit < 0 switches the no-sharpening rule off. Ties between candidates go to the first generated."""
    best_kind = KIND_NONE
    best_gain = 0.0
    best_rows = np.empty(0, np.int64)
    best_new = np.empty((0, 3), np.int64)
    best_node = -1
    best_x = best_y = best_z = 0.0
    best_keep = np.empty(0, np.int64)
    best_gone = np.empty(0, np.int64)
    tri = (f[t, 0], f[t, 1], f[t, 2])

    if flips:
        for k in range(3):
            a = tri[k]
            b = tri[(k + 1) % 3]
            e = _edge_id(a, b, ekey_sorted, esort, width)
            if e < 0 or eoff[e + 1] - eoff[e] != 2:
                continue
            t0 = erows[eoff[e]]
            t1 = erows[eoff[e] + 1]
            if patch[t0] != patch[t1]:
                continue
            o = t0 if t1 == t else t1
            c = tri[(k + 2) % 3]
            d = -1
            for j in range(3):
                if f[o, j] != a and f[o, j] != b:
                    d = f[o, j]
                    break
            if d < 0 or c == d:
                continue
            if _edge_id(c, d, ekey_sorted, esort, width) >= 0:
                continue
            old_min = min(ang[t], ang[o])
            new_min = min(_min_angle(p[c, 0], p[c, 1], p[c, 2], p[a, 0], p[a, 1], p[a, 2], p[d, 0], p[d, 1], p[d, 2]),
                          _min_angle(p[c, 0], p[c, 1], p[c, 2], p[d, 0], p[d, 1], p[d, 2], p[b, 0], p[b, 1], p[b, 2]))
            if new_min <= old_min + TOLERANCE:
                continue
            # the summed normal of the two old triangles, and the new triangles' normals against it
            rx = ry = rz = 0.0
            for row in (t, o):
                ax, ay, az = p[f[row, 1], 0] - p[f[row, 0], 0], p[f[row, 1], 1] - p[f[row, 0], 1], p[f[row, 1], 2] - p[f[row, 0], 2]
                bx, by, bz = p[f[row, 2], 0] - p[f[row, 0], 0], p[f[row, 2], 1] - p[f[row, 0], 1], p[f[row, 2], 2] - p[f[row, 0], 2]
                rx += ay * bz - az * by
                ry += az * bx - ax * bz
                rz += ax * by - ay * bx
            rn = math.sqrt(rx * rx + ry * ry + rz * rz)
            new = np.empty((2, 3), np.int64)
            new[0, 0], new[0, 1], new[0, 2] = c, a, d
            new[1, 0], new[1, 1], new[1, 2] = c, d, b
            reject = False
            for i in range(2):
                ax, ay, az = p[new[i, 1], 0] - p[new[i, 0], 0], p[new[i, 1], 1] - p[new[i, 0], 1], p[new[i, 1], 2] - p[new[i, 0], 2]
                bx, by, bz = p[new[i, 2], 0] - p[new[i, 0], 0], p[new[i, 2], 1] - p[new[i, 0], 1], p[new[i, 2], 2] - p[new[i, 0], 2]
                nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
                if nx * rx + ny * ry + nz * rz <= cos_normal * math.sqrt(nx * nx + ny * ny + nz * nz) * rn:
                    reject = True
            if reject:
                continue
            ux, uy, uz = p[a, 0] - p[c, 0], p[a, 1] - p[c, 1], p[a, 2] - p[c, 2]
            vx, vy, vz = p[b, 0] - p[c, 0], p[b, 1] - p[c, 1], p[b, 2] - p[c, 2]
            cx_, cy_, cz_ = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            height = abs((p[d, 0] - p[c, 0]) * cx_ + (p[d, 1] - p[c, 1]) * cy_ + (p[d, 2] - p[c, 2]) * cz_) / max(rn, 1e-300)
            if height > max_dev:
                continue
            rows = np.empty(2, np.int64)
            rows[0] = t
            rows[1] = o
            if wedge_limit >= 0 and not _keeps_openings(KIND_FLIP, rows, new, -1, 0.0, 0.0, 0.0, wedge_limit, p, f,
                                                        ekey_sorted, esort, width, eoff, erows, max_owner + 4):
                continue
            gain = new_min - old_min
            if best_kind == KIND_NONE or gain > best_gain:
                best_kind, best_gain, best_rows, best_new = KIND_FLIP, gain, rows, new
                best_node, best_x, best_y, best_z = -1, 0.0, 0.0, 0.0

    if relocation:
        for v in range(3):
            n = tri[v]
            code, n1, n2 = _node_kind(n, fixed, curve_off, curve_edge, curve_key, edge_u, edge_v)
            if code == 0:
                continue
            ring = node_rows[node_off[n]:node_off[n + 1]].copy()
            old_x = _ring_coordinates(p, f, ring)
            old_min = ang[ring[0]]
            for i in range(1, len(ring)):
                if ang[ring[i]] < old_min:
                    old_min = ang[ring[i]]
            faces = _plane_faces(n, node_off, node_rows, exterior, rve_face)
            if faces.any() and not has_extent:
                continue
            px, py, pz = p[n, 0], p[n, 1], p[n, 2]
            if code == 1:
                nb = _neighbours(n, f, node_off, node_rows)
                sx = sy = sz = 0.0
                for q in range(len(nb)):
                    sx += p[nb[q], 0]
                    sy += p[nb[q], 1]
                    sz += p[nb[q], 2]
                tx, ty, tz = sx / len(nb), sy / len(nb), sz / len(nb)
            else:
                tx, ty, tz = .5 * (p[n1, 0] + p[n2, 0]), .5 * (p[n1, 1] + p[n2, 1]), .5 * (p[n1, 2] + p[n2, 2])
            have = False
            loc_min = 0.0
            loc_x = loc_y = loc_z = 0.0
            for s in range(3):
                step = 1.0 if s == 0 else (0.5 if s == 1 else 0.25)
                yx, yy, yz = px + step * (tx - px), py + step * (ty - py), pz + step * (tz - pz)
                if code == 1:
                    yx, yy, yz = _project_to_ring(yx, yy, yz, old_x)
                else:
                    c1x, c1y, c1z = _closest_on_segment(yx, yy, yz, p[n1, 0], p[n1, 1], p[n1, 2], px, py, pz)
                    c2x, c2y, c2z = _closest_on_segment(yx, yy, yz, p[n2, 0], p[n2, 1], p[n2, 2], px, py, pz)
                    d1 = math.sqrt((c1x - yx) ** 2 + (c1y - yy) ** 2 + (c1z - yz) ** 2)
                    d2 = math.sqrt((c2x - yx) ** 2 + (c2y - yy) ** 2 + (c2z - yz) ** 2)
                    if d1 <= d2:
                        yx, yy, yz = c1x, c1y, c1z
                    else:
                        yx, yy, yz = c2x, c2y, c2z
                for face in range(6):
                    if faces[face]:
                        axis = face // 2
                        value = (face % 2) * extent[axis]
                        if axis == 0:
                            yx = value
                        elif axis == 1:
                            yy = value
                        else:
                            yz = value
                new_x = old_x.copy()
                for i in range(len(ring)):
                    for k in range(3):
                        if f[ring[i], k] == n:
                            new_x[i, k, 0], new_x[i, k, 1], new_x[i, k, 2] = yx, yy, yz
                if not _orientation_ok(old_x, new_x, cos_normal):
                    continue
                new_min = _min_angle_of(new_x)
                if new_min <= old_min + TOLERANCE:
                    continue
                if code == 1:
                    dev = _distance_to_ring(px, py, pz, new_x)
                else:
                    q1x, q1y, q1z = _closest_on_segment(px, py, pz, p[n1, 0], p[n1, 1], p[n1, 2], yx, yy, yz)
                    q2x, q2y, q2z = _closest_on_segment(px, py, pz, yx, yy, yz, p[n2, 0], p[n2, 1], p[n2, 2])
                    dev = min(math.sqrt((q1x - px) ** 2 + (q1y - py) ** 2 + (q1z - pz) ** 2),
                              math.sqrt((q2x - px) ** 2 + (q2y - py) ** 2 + (q2z - pz) ** 2))
                if dev > max_dev:
                    continue
                if (not have) or new_min > loc_min:
                    have = True
                    loc_min, loc_x, loc_y, loc_z = new_min, yx, yy, yz
            if not have:
                continue
            new = np.empty((len(ring), 3), np.int64)
            for i in range(len(ring)):
                for k in range(3):
                    new[i, k] = f[ring[i], k]
            if wedge_limit >= 0 and not _keeps_openings(KIND_RELOCATION, ring, new, n, loc_x, loc_y, loc_z,
                                                        wedge_limit, p, f, ekey_sorted, esort, width, eoff, erows,
                                                        max_owner + len(ring) + 1):
                continue
            gain = loc_min - old_min
            if best_kind == KIND_NONE or gain > best_gain:
                best_kind, best_gain, best_rows, best_new = KIND_RELOCATION, gain, ring, new
                best_node, best_x, best_y, best_z = n, loc_x, loc_y, loc_z

    if collapses:
        l0 = math.sqrt((p[tri[1], 0] - p[tri[0], 0]) ** 2 + (p[tri[1], 1] - p[tri[0], 1]) ** 2 + (p[tri[1], 2] - p[tri[0], 2]) ** 2)
        l1 = math.sqrt((p[tri[2], 0] - p[tri[1], 0]) ** 2 + (p[tri[2], 1] - p[tri[1], 1]) ** 2 + (p[tri[2], 2] - p[tri[1], 2]) ** 2)
        l2 = math.sqrt((p[tri[0], 0] - p[tri[2], 0]) ** 2 + (p[tri[0], 1] - p[tri[2], 1]) ** 2 + (p[tri[0], 2] - p[tri[2], 2]) ** 2)
        k = 0
        if l1 < l0:
            k = 1
        if l2 < min(l0, l1):
            k = 2
        a = tri[k]
        b = tri[(k + 1) % 3]
        e_ab = _edge_id(a, b, ekey_sorted, esort, width)
        for direction in range(2):
            gone = a if direction == 0 else b
            stay = b if direction == 0 else a
            code, n1, n2 = _node_kind(gone, fixed, curve_off, curve_edge, curve_key, edge_u, edge_v)
            if code == 0:
                continue
            if code != 1 and stay != n1 and stay != n2:
                continue
            if e_ab >= 0:
                shared = erows[eoff[e_ab]:eoff[e_ab + 1]].copy()
            else:
                shared = np.empty(0, np.int64)
            ob = np.empty(3 * len(shared), np.int64)
            no = 0
            for q in range(len(shared)):
                for j in range(3):
                    v = f[shared[q], j]
                    if v != a and v != b:
                        ob[no] = v
                        no += 1
            opposite = np.unique(ob[:no])
            if not _sorted_intersection_equals(_neighbours(gone, f, node_off, node_rows),
                                               _neighbours(stay, f, node_off, node_rows), opposite):
                continue                                           # link condition
            ring = node_rows[node_off[gone]:node_off[gone + 1]]
            keep = np.empty(len(ring), np.int64)
            nk = 0
            for q in range(len(ring)):
                is_shared = False
                for j in range(len(shared)):
                    if shared[j] == ring[q]:
                        is_shared = True
                        break
                if not is_shared:
                    keep[nk] = ring[q]
                    nk += 1
            if nk == 0:
                continue
            keep = keep[:nk].copy()
            new = np.empty((nk, 3), np.int64)
            for i in range(nk):
                for j in range(3):
                    v = f[keep[i], j]
                    new[i, j] = stay if v == gone else v
            ring_stay = node_rows[node_off[stay]:node_off[stay + 1]]
            old_min = 180.0
            for q in range(len(ring)):
                if ang[ring[q]] < old_min:
                    old_min = ang[ring[q]]
            for q in range(len(ring_stay)):
                if ang[ring_stay[q]] < old_min:
                    old_min = ang[ring_stay[q]]
            new_x = _ring_coordinates(p, new, np.arange(nk))
            new_min = _min_angle_of(new_x)
            if new_min <= old_min + TOLERANCE:
                continue
            old_x = _ring_coordinates(p, f, keep)
            if not _orientation_ok(old_x, new_x, cos_normal):
                continue
            if _distance_to_ring(p[gone, 0], p[gone, 1], p[gone, 2], new_x) > max_dev:
                continue
            # every grain's shell must stay a single closed fan around the kept node
            others = np.empty(len(ring_stay), np.int64)
            no = 0
            for q in range(len(ring_stay)):
                is_shared = False
                for j in range(len(shared)):
                    if shared[j] == ring_stay[q]:
                        is_shared = True
                        break
                if not is_shared:
                    others[no] = ring_stay[q]
                    no += 1
            others = others[:no]
            total = nk + no
            fan_tris = np.empty((total, 3), np.int64)
            fan_pairs = np.empty((total, 2), pairs.dtype)
            fan_ext = np.empty(total, np.bool_)
            for i in range(nk):
                for j in range(3):
                    fan_tris[i, j] = new[i, j]
                fan_pairs[i, 0], fan_pairs[i, 1] = pairs[keep[i], 0], pairs[keep[i], 1]
                fan_ext[i] = exterior[keep[i]]
            for i in range(no):
                for j in range(3):
                    fan_tris[nk + i, j] = f[others[i], j]
                fan_pairs[nk + i, 0], fan_pairs[nk + i, 1] = pairs[others[i], 0], pairs[others[i], 1]
                fan_ext[nk + i] = exterior[others[i]]
            if not _fans_ok(stay, fan_tris, fan_pairs, fan_ext):
                continue
            rows = ring.copy()
            if wedge_limit >= 0 and not _keeps_openings(KIND_COLLAPSE, rows, new, -1, 0.0, 0.0, 0.0, wedge_limit, p, f,
                                                        ekey_sorted, esort, width, eoff, erows,
                                                        max_owner + nk + 1):
                continue
            gain = new_min - old_min
            if best_kind == KIND_NONE or gain > best_gain:
                best_kind, best_gain, best_rows, best_new = KIND_COLLAPSE, gain, rows, new
                best_node, best_x, best_y, best_z = -1, 0.0, 0.0, 0.0
                best_keep, best_gone = keep, shared
    return best_kind, best_rows, best_new, best_node, best_x, best_y, best_z, best_gain, best_keep, best_gone


# --- adjacency arrays of the start of a pass ------------------------------------------------------------------------
class Adjacency:
    """d3v2p0's per-pass _Topology as arrays, in the same orders: the triangles around each node (ascending), the
    edges in order of first appearance with their triangles (ascending), the curve edges around each node in the order
    of the edges, and the nodes that touch a frozen triangle."""

    def __init__(self, n_points, f, alive, patch, frozen):
        ids = np.flatnonzero(alive)
        fa = f[ids]
        nodes = fa.ravel()
        order = np.argsort(nodes, kind='stable')
        self.node_rows = np.repeat(ids, 3)[order].astype(np.int64)
        self.node_off = np.r_[0, np.cumsum(np.bincount(nodes, minlength=n_points))].astype(np.int64)
        self.width = np.int64(n_points + 1)
        e = np.sort(fa[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).astype(np.int64)
        code = e[:, 0] * self.width + e[:, 1]
        unique, first, inverse, counts = np.unique(code, return_index=True, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        insertion = np.argsort(first, kind='stable')                  # unique edges in order of first appearance
        rank = np.empty(len(insertion), np.int64)
        rank[insertion] = np.arange(len(insertion))
        grouped = np.argsort(rank[inverse], kind='stable')
        self.erows = np.repeat(ids, 3)[grouped].astype(np.int64)
        self.ecount = counts[insertion]
        self.eoff = np.r_[0, np.cumsum(self.ecount)].astype(np.int64)
        ekey = unique[insertion]
        self.edge_u = ekey // self.width
        self.edge_v = ekey % self.width
        self.esort = np.argsort(ekey).astype(np.int64)
        self.ekey_sorted = ekey[self.esort]
        self.max_owner = int(self.ecount.max()) if len(self.ecount) else 0
        self.fixed = np.zeros(n_points, bool)
        self.fixed[f[frozen & alive].ravel()] = True
        # curve edges: not exactly two owners in one patch
        first_owner = self.erows[self.eoff[:-1]]
        second_owner = self.erows[np.minimum(self.eoff[:-1] + 1, max(len(self.erows) - 1, 0))]
        is_curve = (self.ecount != 2) | (patch[first_owner] != patch[second_owner])
        curve = np.flatnonzero(is_curve)
        if len(curve):
            kmax = int(self.ecount[curve].max())
            owners = np.full((len(curve), kmax), -1, np.int64)
            for j in range(kmax):
                has = self.ecount[curve] > j
                owners[has, j] = patch[self.erows[self.eoff[curve[has]] + j]]
            owners.sort(axis=1)
            repeated = np.zeros(owners.shape, bool)
            repeated[:, 1:] = owners[:, 1:] == owners[:, :-1]
            owners[repeated] = -1
            owners.sort(axis=1)                                         # the set of patches around the edge
            key = unique_rows(owners, True)[1].astype(np.int64)
        else:
            key = np.empty(0, np.int64)
        at = np.stack((self.edge_u[curve], self.edge_v[curve]), axis=1).ravel()
        by_node = np.argsort(at, kind='stable')
        self.curve_edge = np.repeat(curve, 2)[by_node].astype(np.int64)
        self.curve_key = np.repeat(key, 2)[by_node]
        self.curve_off = np.r_[0, np.cumsum(np.bincount(at, minlength=n_points))].astype(np.int64)

    def edge_rows(self, a, b):
        """Triangles on the edge a-b at the start of the pass (a list), as d3v2p0's topo.edge_tri.get."""
        lo, hi = (a, b) if a < b else (b, a)
        i = np.searchsorted(self.ekey_sorted, lo * self.width + hi)
        if i < len(self.ekey_sorted) and self.ekey_sorted[i] == lo * self.width + hi:
            e = self.esort[i]
            return self.erows[self.eoff[e]:self.eoff[e + 1]].tolist()
        return []

    def rows_around_edges_of(self, f, rows):
        """The triangles that shared an edge with any of ``rows`` at the start of the pass, and ``rows`` themselves."""
        if len(rows) == 0:
            return np.asarray(rows, dtype=np.int64)
        e = np.sort(f[rows][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).astype(np.int64)
        code = e[:, 0] * self.width + e[:, 1]
        pos = np.searchsorted(self.ekey_sorted, code)
        pos = np.minimum(pos, max(len(self.ekey_sorted) - 1, 0))
        found = (self.ekey_sorted[pos] == code) if len(self.ekey_sorted) else np.zeros(len(code), bool)
        edge = self.esort[pos[found]]
        starts, ends = self.eoff[edge], self.eoff[edge + 1]
        lengths = ends - starts
        index = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(lengths.sum())
        return np.unique(np.concatenate((self.erows[index], np.asarray(rows, dtype=np.int64))))


# --- the repair, in d3v2p0's order ---------------------------------------------------------------------------------
_NAMES = {KIND_FLIP: 'flip', KIND_RELOCATION: 'relocation', KIND_COLLAPSE: 'collapse'}


def improve_surface_angles_compiled(surface, enabled=True, minimum_angle=30., max_passes=10,
                                    max_deviation=None, minimum_facet_angle=.1, frozen_grain_ids=(),
                                    flips=True, relocation=True, collapses=True,
                                    max_normal_change=60., wedge_limit=None, backend='auto', n_workers=None,
                                    progress=None, plan_report=None):
    """d3v2p0.surface_angles.improve_surface_angles with the candidate search compiled.

    The same arguments, rules, order of work (worst triangle first, one pass at a time, touched nodes reserved
    for the rest of the pass) and result as d3v2p0 (coordinates agree to the last digits of a float); the
    operations a pass proposes are checked afterwards for intersections and facet openings exactly as
    d3v2p1.surface_angles does, and rejected ones are rolled back. The candidate search runs in this process;
    backend / n_workers only steer the intersection search of the checks (d3v2p1.backend).
    plan_report: the backend report to store in the result (d3v2p1.surface_angles passes its plan's).
    Needs numba.
    """
    check_backend(backend)
    check_workers(n_workers)
    search = {} if backend == 'auto' and n_workers is None else dict(backend=backend, n_workers=n_workers)
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
    f = np.array(surface.triangles).astype(np.int64)
    pairs = np.asarray(surface.grain_pairs)
    pairs_int = pairs.astype(np.int64)                     # what the compiled search reads (one compiled signature)
    exterior = np.asarray(surface.exterior, dtype=bool)
    rve_face = np.asarray(surface.rve_face).astype(np.int64)
    extent = surface.report.get('rve_dimensions')
    has_extent = extent is not None
    extent = np.zeros(3) if extent is None else np.asarray(extent, dtype=float)
    edges0 = np.linalg.norm(p[f[:, 1]] - p[f[:, 0]], axis=1)
    if max_deviation is None:
        max_deviation = .1 * float(np.median(edges0))
    if not np.isfinite(max_deviation) or max_deviation < 0:
        raise ValueError('max_deviation must be finite and nonnegative')
    cos_normal = float(np.cos(np.radians(max_normal_change)))
    frozen = np.isin(pairs, np.asarray(list(frozen_grain_ids), dtype=pairs.dtype)).any(axis=1) \
        if len(frozen_grain_ids) else np.zeros(len(f), bool)
    _, patch = unique_rows(np.column_stack((pairs, rve_face)), True)
    patch = patch.astype(np.int64)
    alive = np.ones(len(f), bool)
    wedge = -1.0 if wedge_limit is None else float(wedge_limit)

    a0 = triangle_angles(p, f).min(axis=1)
    before = _angle_summary(a0, minimum_angle)
    history = []
    totals = dict(flips=0, relocations=0, collapses=0)
    for iteration in range(max_passes if enabled else 0):
        if progress is not None:
            try:
                progress(f'Surface angle repair pass {iteration+1}/{max_passes}', iteration + 1, max_passes)
            except Exception:
                pass
        clock = dict(setup=-time.perf_counter(), evaluate=0., apply=0., verify=0.)
        ang = np.full(len(f), 180.)
        ang[alive] = triangle_angles(p, f[alive]).min(axis=1)
        work = np.flatnonzero(alive & (ang < minimum_angle) & ~frozen)
        if not len(work):
            break
        adj = Adjacency(len(p), f, alive, patch, frozen)
        reserved = np.zeros(len(p), bool)
        ops = []
        clock['setup'] += time.perf_counter()
        t_loop = time.perf_counter()
        arrays = (adj.node_off, adj.node_rows, adj.ekey_sorted, adj.esort, adj.width, adj.eoff, adj.erows, adj.edge_u,
                  adj.edge_v, adj.fixed, adj.curve_off, adj.curve_edge, adj.curve_key, patch, pairs_int, exterior, rve_face,
                  extent, has_extent)
        for t in work[np.argsort(ang[work])]:
            t = int(t)
            if reserved[f[t]].any() or not alive[t]:
                continue
            kind, rows, new, node, x, y, z, gain, keep, gone = evaluate(
                t, p, f, ang, *arrays, flips, relocation, collapses, max_deviation, cos_normal, wedge, adj.max_owner)
            if kind == KIND_NONE:
                continue
            touched = np.unique(np.concatenate([f[rows].ravel(), new.ravel()]))
            if reserved[touched].any():
                continue
            reserved[touched] = True
            op = dict(kind=_NAMES[kind], rows=rows, new=new, gain=gain)
            if kind == KIND_RELOCATION:
                reserved[_neighbours(node, f, adj.node_off, adj.node_rows)] = True
                op.update(node=int(node), point=np.array([x, y, z]))
            elif kind == KIND_COLLAPSE:
                op.update(keep=keep, gone=gone)
            op['old_rows'] = f[rows].copy()
            op['old_alive'] = alive[rows].copy()
            op['old_point'] = p[int(node)].copy() if kind == KIND_RELOCATION else None
            _apply(op, p, f, alive)
            ops.append(op)
        clock['evaluate'] = time.perf_counter() - t_loop
        if not ops:
            break
        t_verify = time.perf_counter()
        active = np.ones(len(ops), bool)
        while active.any():
            ids = np.flatnonzero(alive)
            local = np.full(len(f), -1)
            local[ids] = np.arange(len(ids))
            changed = np.concatenate([o['rows'][alive[o['rows']]] for o, a in zip(ops, active) if a])
            hits = find_surface_intersections(p, f[ids], triangle_ids=local[changed], **search)
            near = adj.rows_around_edges_of(f, changed)
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
        totals['flips'] += counts['flip']
        totals['relocations'] += counts['relocation']
        totals['collapses'] += counts['collapse']
        history.append(dict(pass_number=iteration + 1, worked=int(len(work)), proposed=len(ops), accepted=len(accepted),
                            rejected_geometry=len(ops) - len(accepted), evaluation_rounds=1,
                            seconds={k: round(v, 3) for k, v in clock.items()}, **counts))
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
        n_workers=1, backend=dict(plan_report) if plan_report is not None else
        dict(requested='numba', used='numba', n_workers=1, fallback=None),
        full_validation_required=True)
    return replace(surface, points=p, triangles=f, grain_pairs=pairs, exterior=exterior,
                   rve_face=rve_face, report=report)


# --- is the compiled tier usable here? --------------------------------------------------------------------------------
_READY = {}


def compiled_ready():
    """(True, None) when the kernels compile and repair a small test surface correctly, else (False, reason).

    Checked once per process. numba_available() only compiles a one-line probe; this compiles the kernels of the
    tier (from the on-disk cache when there is one), so a machine where they cannot compile or cache falls back to
    the other tiers before any real work starts."""
    if 'result' not in _READY:
        try:
            from ..d3v2p0.rve_caps import ClosedRVE
            n = 5
            xy = np.array([(i, j) for j in range(n) for i in range(n)], float)
            points = np.column_stack((xy, np.zeros(len(xy))))
            points[12, :2] = [2.93, 2.02]                         # one node pushed next to its neighbour
            faces = []
            for j in range(n - 1):
                for i in range(n - 1):
                    a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
                    faces += [[a, b, c], [a, c, d]]
            k = len(faces)
            surface = ClosedRVE(points, np.array(faces), np.tile((1, 2), (k, 1)), np.zeros(k, bool), np.full(k, -1), {})
            repaired = improve_surface_angles_compiled(surface, max_deviation=0.)
            if triangle_angles(repaired.points, repaired.triangles).min() > 30:
                _READY['result'] = (True, None)
            else:
                _READY['result'] = (False, 'the compiled search failed its self-test')
        except Exception as error:
            _READY['result'] = (False, f'{type(error).__name__}: {error}')
    return _READY['result']
