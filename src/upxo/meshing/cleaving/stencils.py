"""
stencils.py
===========
Per-tet cleaving: cut one lattice tetrahedron along a grain-boundary label
split and produce the resulting sub-tets, each tagged with its grain label.

Design goal (this is the part the whole module lives or dies on): when two
lattice tets share a face, and that face is cut by the same grain boundary,
both tets MUST independently produce the identical triangulation of that
shared face -- otherwise the output mesh has a crack exactly where two
grains meet. Classic marching-tetrahedra implementations solve this with a
hand-built lookup table per topological case (1-vs-3, 2-vs-2, ...) and a lot
of care about which diagonal each case's table uses. We solve it with one
general, uniform rule instead:

* Every point in the output -- an original lattice vertex, or a new "cut"
  point introduced where a grain-boundary edge crosses -- gets a canonical
  key. An original vertex's key is its own global id; a cut point's key is
  its defining edge, i.e. the (min, max) global ids of the two lattice
  vertices it lies between. This key is a pure function of global vertex
  ids, so both tets sharing a face always compute the SAME key for the SAME
  physical point, regardless of which tet is asking.
* Any polygon that needs splitting into triangles (a cut tet-face's
  quadrilateral piece, or the new internal cutting face separating the two
  labels) is triangulated by FANNING from its own lowest-keyed point.
* Each label's sub-region within the tet (always convex -- it's a tet cut by
  planes through some of its edges) is tetrahedralized by CONING from the
  lowest-keyed point in that whole sub-region to every boundary triangle
  that doesn't already contain it.

This one rule is verified (see tests) to conserve volume exactly across all
14 non-trivial 2-label corner patterns and all 12 non-trivial 3-label
(2+1+1, triple-junction) corner patterns, for arbitrary global-id orderings
-- including reproducing an unviolated tet (all 4 corners one label)
unchanged, with no special-casing needed: coning from any of its own
vertices to the one opposite face IS just the original tet.

Reference for why an explicit ordering rule (rather than a fixed per-case
table) is the right fix for cross-tet consistency: Bronson et al., the
"Stencil Consistency and Generalization" discussion of cyclic virtual-cut
ordering.

Extending to 3 distinct corner labels (a triple junction) needs one new kind
of point: a tet FACE can have all 3 of its corners differently labelled (a
tet has only 4 faces and exactly 2 of them do, in this case -- the two
containing both "minority" corners), and that face then needs a new "face
triple point" where all 3 labels meet on it. It gets the same canonical-key
treatment as a cut point: keyed by that face's 3 sorted global vertex ids
(a pure function of the shared face's identity, so two tets sharing that
exact face independently compute the identical point), and it sorts after
every cut point (tier 2), same reasoning as cut points sorting after
original vertices. The 2 triple points a 2+1+1 tet produces are connected by
one new internal edge (the "triple line"), which every pairwise interface
between the 3 labels shares as one of its boundary edges -- exactly the
geometric picture of 3 grain boundaries meeting along a line.

Extending once more to all 4 corners distinctly labelled (a quadruple
junction -- point-like, where 4 grains meet at a single point) needs one
final new kind of point: every edge is cut, every face has its own triple
point (unchanged logic -- a face's triangulation only ever depends on its
own 3 corners, never on how many labels the WHOLE tet has), and there is
one new tet-INTERIOR "quadruple point" Q, keyed by the tet's own 4 sorted
global ids. Q is never shared with a neighbouring tet -- two different tets
never share all 4 corners -- so, unlike cut points and triple points, it
needs no cross-tet-consistency treatment at all; it sorts last (tier 3), so
an original vertex is always available as its region's coning apex instead.
Each of the 6 corner PAIRS gets a pairwise interface quad {cut(i,j), T_ijk,
Q, T_ijl} (k,l = the other two corners, so T_ijk/T_ijl are the 2 faces
containing edge i-j) -- Q playing the role the "triple line" played for 3
labels, generalised from a line to a point.

Scope: this module resolves tets with 1, 2, 3, or 4 distinct corner labels
-- i.e. every possible case for a tetrahedron's 4 corners.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

# A "point" is either ('v', global_vertex_id) -- an original lattice vertex,
# ('c', lo_gid, hi_gid) -- a cut point on the edge between those two original
# lattice vertices (lo_gid < hi_gid), ('t', g1, g2, g3) -- a face triple
# point (g1<g2<g3, the 3 sorted global ids of the face it's on), or
# ('q', g1, g2, g3, g4) -- a tet-interior quadruple point (the tet's own 4
# sorted global ids).
Point = Tuple

_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class TooManyLabelsError(Exception):
    """Raised when a tet has more than 4 distinct corner labels (impossible
    for an actual tetrahedron -- kept as a defensive guard)."""


def _vpoint(gid: int) -> Point:
    return ('v', int(gid))


def _cpoint(gid_a: int, gid_b: int) -> Point:
    a, b = int(gid_a), int(gid_b)
    return ('c', min(a, b), max(a, b))


def _tpoint(gid_a: int, gid_b: int, gid_c: int) -> Point:
    g = sorted((int(gid_a), int(gid_b), int(gid_c)))
    return ('t', g[0], g[1], g[2])


def _qpoint(gid_a: int, gid_b: int, gid_c: int, gid_d: int) -> Point:
    g = sorted((int(gid_a), int(gid_b), int(gid_c), int(gid_d)))
    return ('q', g[0], g[1], g[2], g[3])


def point_key(point: Point):
    """Canonical sort key: original vertex (tier 0) < cut point (tier 1) <
    face triple point (tier 2) < tet quadruple point (tier 3); within a
    tier, compare the defining ids."""
    if point[0] == 'v':
        return (0, point[1], 0, 0, 0)
    if point[0] == 'c':
        return (1, point[1], point[2], 0, 0)
    if point[0] == 't':
        return (2, point[1], point[2], point[3], 0)
    return (3, point[1], point[2], point[3], point[4])


def fan_triangulate(cyclic_points: Sequence[Point]) -> List[Tuple[Point, Point, Point]]:
    """
    Triangulate a polygon given in correct cyclic (boundary-walk) order, by
    fanning from its own lowest-keyed point. A triangle passes through
    unchanged (fanning a 3-gon from any of its points gives itself).
    """
    # Manual scan instead of min(..., key=lambda ...): avoids allocating a
    # fresh lambda + range object on every one of this function's ~700k
    # calls for what cyclic_points (always length 3 or 4 here) makes a
    # trivially short comparison anyway.
    apex_idx = 0
    best_key = point_key(cyclic_points[0])
    for i in range(1, len(cyclic_points)):
        k = point_key(cyclic_points[i])
        if k < best_key:
            best_key = k
            apex_idx = i
    ordered = cyclic_points[apex_idx:] + cyclic_points[:apex_idx]
    return [(ordered[0], ordered[k], ordered[k + 1])
            for k in range(1, len(ordered) - 1)]


def _trace_cutting_polygon(mixed_edges: List[Tuple[Point, Point]]) -> List[Point]:
    """
    The new internal face separating the two labels is bounded by exactly
    one edge per "mixed" original tet-face (the segment where the boundary
    crosses that face, connecting the 2 cut points on it). Trace those
    edges into a single cyclic point order.
    """
    adjacency: Dict[Point, List[Point]] = {}
    for a, b in mixed_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    start = next(iter(adjacency))
    cycle = [start]
    prev, cur = None, start
    while True:
        candidates = [n for n in adjacency[cur] if n != prev]
        nxt = candidates[0] if candidates else adjacency[cur][0]
        if nxt == start:
            break
        cycle.append(nxt)
        prev, cur = cur, nxt
    return cycle


def cleave_tet(vertex_ids: Sequence[int],
               vertex_labels: Sequence[int]) -> Dict[int, List[Tuple[Point, Point, Point, Point]]]:
    """
    Cleave one tet given its 4 corner global vertex ids and their sampled
    labels. Returns {label: [tet, ...]}, each tet a 4-tuple of Points (not
    yet resolved to final output vertex ids/coordinates -- see cleave.py).

    Handles every possible case: 1, 2, 3, or 4 distinct labels among the 4
    corners (an unviolated tet with 1 label is returned as itself, no
    cutting). TooManyLabelsError is unreachable for an actual tetrahedron
    (only 4 corners exist) -- kept as a defensive guard.
    """
    # Computed from a single pass over vertex_labels (labels_present used to
    # be a separate genexpr over the same input, doubling the int() calls
    # for every tet that reaches the general path below).
    labs = [int(l) for l in vertex_labels]
    labels_present = sorted(set(labs))
    if len(labels_present) > 4:
        raise TooManyLabelsError(
            f'tet has {len(labels_present)} distinct corner labels '
            f'{labels_present} -- impossible for a tetrahedron')

    if len(labels_present) == 1:
        # No cutting needed at all. This is provably what the general path
        # below also computes -- traced directly: with every face a single
        # triangle and the apex fanned from the 4 original vertices, the
        # ONE face not containing the apex is exactly the original tet's
        # opposite face, so the output is the same 4 corners back again --
        # just far more directly, and worth a dedicated path because this
        # case is common: confirmed directly on a real multi-grain mesh,
        # over a third of all tets are entirely inside one grain.
        return {labels_present[0]: [tuple(_vpoint(g) for g in vertex_ids)]}

    # No int() coercion needed here -- every gids[x] use below is passed
    # straight into _vpoint/_cpoint/_tpoint/_qpoint, which already coerce
    # to int themselves, so pre-converting here was pure duplicated work.
    gids = vertex_ids

    # Cut points, keyed by the local edge that produced them.
    cut_point: Dict[Tuple[int, int], Point] = {}
    for i, j in _TET_EDGES:
        if labs[i] != labs[j]:
            cut_point[(i, j)] = _cpoint(gids[i], gids[j])

    def cut_of(i: int, j: int) -> Point:
        # _TET_EDGES always inserts with i < j, so the correctly-ordered
        # key is always (min, max) -- one dict lookup instead of an "in"
        # check plus a fallback lookup (each a full hash+compare).
        return cut_point[(i, j)] if i < j else cut_point[(j, i)]

    region_points: Dict[int, set] = {lab: set() for lab in labels_present}
    region_tris: Dict[int, List[Tuple[Point, ...]]] = {lab: [] for lab in labels_present}
    mixed_edges: List[Tuple[Point, Point]] = []

    def add(lab: int, tri: Tuple[Point, ...]):
        # Measured, not assumed: an earlier attempt to track a running min
        # here directly (avoiding the set) made this SLOWER, not faster --
        # point_key() ended up being called on every point of every
        # triangle added (no dedup), instead of once per unique point at
        # the end via min(). set.update()'s native tuple hashing dedupes
        # essentially for free (fan_triangulate output repeats the same
        # apex across many triangles), so the real cost driver is the
        # per-point-key work, not the set itself -- confirmed by profiling
        # both versions, not reasoned from first principles alone.
        region_tris[lab].append(tri)
        region_points[lab].update(tri)

    for (i, j, k) in _TET_FACES:
        local = (i, j, k)
        face_labs = [labs[x] for x in local]
        distinct = set(face_labs)

        if len(distinct) == 1:
            add(face_labs[0], tuple(_vpoint(gids[x]) for x in local))

        elif len(distinct) == 2:
            # Exactly one mixed edge on this face (the segment where the
            # boundary crosses it) -- collect once, regardless of label
            # (used below only when the WHOLE tet has 2 distinct labels).
            counts = {lv: face_labs.count(lv) for lv in distinct}
            duo_lab = next(lv for lv, c in counts.items() if c == 2)
            solo_lab = next(lv for lv in distinct if lv != duo_lab)
            duo = [x for x in local if labs[x] == duo_lab]
            other = next(x for x in local if x not in duo)
            a, b = duo
            mixed_edges.append((cut_of(a, other), cut_of(b, other)))

            add(solo_lab, (_vpoint(gids[other]), cut_of(other, a), cut_of(other, b)))
            quad = [_vpoint(gids[a]), _vpoint(gids[b]), cut_of(b, other), cut_of(a, other)]
            for t in fan_triangulate(quad):
                add(duo_lab, t)

        else:  # len(distinct) == 3: this face has a triple point on it
            T = _tpoint(gids[i], gids[j], gids[k])
            for idx, corner in enumerate(local):
                nxt = local[(idx + 1) % 3]
                prv = local[(idx - 1) % 3]
                quad = [_vpoint(gids[corner]), cut_of(corner, nxt), T, cut_of(corner, prv)]
                for t in fan_triangulate(quad):
                    add(labs[corner], t)

    if len(labels_present) == 2:
        cycle = _trace_cutting_polygon(mixed_edges)
        cutting_tris = [tuple(cycle)] if len(cycle) == 3 else fan_triangulate(cycle)
        for lab in labels_present:
            for t in cutting_tris:
                add(lab, t)

    elif len(labels_present) == 3:
        # Exactly one label has 2 corners (majority); the other 2 labels
        # each have exactly 1 corner. 2 of the tet's 4 faces then contain
        # both minority corners together with a majority one, and each of
        # those got a triple point above. The 3 labels' pairwise interfaces
        # all meet along the segment connecting those 2 triple points.
        counts = Counter(labs)
        maj_label = next(lv for lv, c in counts.items() if c == 2)
        maj_corners = [x for x in range(4) if labs[x] == maj_label]
        min_labels = [lv for lv in labels_present if lv != maj_label]
        min_corner = {lv: next(x for x in range(4) if labs[x] == lv) for lv in min_labels}

        m0, m1 = maj_corners
        lab_b, lab_c = min_labels
        c_b, c_c = min_corner[lab_b], min_corner[lab_c]

        t1 = _tpoint(gids[m0], gids[c_b], gids[c_c])
        t2 = _tpoint(gids[m1], gids[c_b], gids[c_c])

        ab_quad = [cut_of(m0, c_b), t1, t2, cut_of(m1, c_b)]
        for t in fan_triangulate(ab_quad):
            add(maj_label, t)
            add(lab_b, t)

        ac_quad = [cut_of(m0, c_c), t1, t2, cut_of(m1, c_c)]
        for t in fan_triangulate(ac_quad):
            add(maj_label, t)
            add(lab_c, t)

        bc_tri = (cut_of(c_b, c_c), t1, t2)
        add(lab_b, bc_tri)
        add(lab_c, bc_tri)

    elif len(labels_present) == 4:
        # All 4 corners distinctly labelled. Every face (any 3 of 4
        # all-different corners) got its own triple point in the face loop
        # above -- recompute the same deterministic values here to wire up
        # the 6 pairwise interfaces, one per corner pair (i,j), each a quad
        # {cut(i,j), T_ijk, Q, T_ijl} where k,l are the other two corners.
        q_point = _qpoint(*gids)
        face_triple = {frozenset(f): _tpoint(gids[f[0]], gids[f[1]], gids[f[2]])
                        for f in _TET_FACES}

        for (i, j) in _TET_EDGES:
            k, l = (x for x in range(4) if x not in (i, j))
            t_ijk = face_triple[frozenset((i, j, k))]
            t_ijl = face_triple[frozenset((i, j, l))]
            quad = [cut_of(i, j), t_ijk, q_point, t_ijl]
            for t in fan_triangulate(quad):
                add(labs[i], t)
                add(labs[j], t)

    output: Dict[int, List[Tuple[Point, Point, Point, Point]]] = {}
    for lab in labels_present:
        pts = region_points[lab]
        apex = min(pts, key=point_key)
        tets = []
        for tri in region_tris[lab]:
            if apex in tri:
                continue
            tets.append((apex,) + tri)
        output[lab] = tets
    return output
