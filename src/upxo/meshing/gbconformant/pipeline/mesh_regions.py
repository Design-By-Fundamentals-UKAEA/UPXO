"""Where the grain boundary surface, the junction lines and the junction points of a tetrahedral mesh are, and
how many layers of elements every tetrahedron and surface triangle lies from each of them.

A *feature* is one of three things a grain structure has:

* ``gb_surface``: the faces that separate two grains,
* ``junction_lines``: the mesh edges shared by three or more regions (grains, and the box exterior when asked),
* ``junction_points``: the nodes where junction lines end, branch or change the set of regions they separate (for a
  generic grain structure, the points where four grains meet).

The *layer* of an element is 1 when it touches the feature, 2 when it shares a node with a layer-1 element and is
not itself in layer 1, 3 when it shares a node with a layer-2 element and is not in layers 1 or 2, and so on; 0 when
it is farther than the asked-for order. What counts as touching is a choice (:data:`TOUCH_CHOICES`): sharing a face
(with the grain boundary surface) or an edge (with a junction line), or sharing any node. Junction points are always
touched by a node. Triangles are only ever grown along the surface they belong to.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus.junctions import (
    PAIRS, box_face_bits, junction_lines, junction_point_nodes, tet_junction_edges)

GB_SURFACE, JUNCTION_LINES, JUNCTION_POINTS = "gb_surface", "junction_lines", "junction_points"
FEATURES = (GB_SURFACE, JUNCTION_LINES, JUNCTION_POINTS)
FEATURE_TITLES = {GB_SURFACE: "Grain boundary surface", JUNCTION_LINES: "Junction lines",
                  JUNCTION_POINTS: "Junction points"}
FACE_OR_EDGE, NODE = "face or edge", "node"
TOUCH_CHOICES = (FACE_OR_EDGE, NODE)
MAX_ORDER = 50
TRIANGLE_PAIRS = ((0, 1), (0, 2), (1, 2))
_FACE_CORNERS = np.array(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)))


@dataclass
class MeshSurface:
    """The triangles that separate regions: faces between two grains and, when asked, faces on the box exterior."""

    nodes: np.ndarray
    """(F, 3) node indices of every triangle, ascending in each row."""
    grains: np.ndarray
    """(F, 2) the two grain labels of a grain boundary triangle (lower first); (0, grain) on the exterior."""
    tets: np.ndarray
    """(F, 2) the tetrahedra on either side; -1 where there is none."""
    between_grains: np.ndarray
    """(F,) True for a triangle that separates two grains."""
    nonmanifold: int = 0
    """Faces shared by more than two tetrahedra (a broken mesh; they are left out)."""


@dataclass
class RegionLayers:
    """The layer of every tetrahedron and surface triangle for each feature (0 = beyond the asked-for order)."""

    order: int
    touch: str
    include_rve_faces: bool
    surface: MeshSurface
    tet: dict = field(default_factory=dict)
    """{feature: (T,) int8}"""
    triangle: dict = field(default_factory=dict)
    """{feature: (F,) int8}"""
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    """(E, 2) node pairs of the junction edges."""
    line_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    point_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    counts: dict = field(default_factory=dict)


def tet_faces(tetrahedra, n_nodes):
    """Every distinct face of the mesh: ``(nodes (F,3), first tet, second tet or -1, number of tets on the face)``."""
    t = np.asarray(tetrahedra, dtype=np.int64)
    if not len(t):
        return (np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    faces = np.sort(t[:, _FACE_CORNERS].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(t)), 4)
    n = int(n_nodes)
    if n ** 3 < 2 ** 62:
        key = (faces[:, 0] * n + faces[:, 1]) * n + faces[:, 2]
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    else:
        order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
        sorted_faces = faces[order]
        starts = np.flatnonzero(np.r_[True, np.any(sorted_faces[1:] != sorted_faces[:-1], axis=1)])
    count = np.diff(np.r_[starts, len(order)])
    first = owner[order[starts]]
    second = np.where(count >= 2, owner[order[np.minimum(starts + 1, len(order) - 1)]], -1)
    return faces[order[starts]], first, second, count


def mesh_surface(tetrahedra, grains, n_nodes, include_rve_faces=False) -> MeshSurface:
    """The grain boundary triangles of a mesh, with the box exterior's triangles when ``include_rve_faces``."""
    g = np.asarray(grains, dtype=np.int64)
    nodes, first, second, count = tet_faces(tetrahedra, n_nodes)
    two = count == 2
    other = np.where(two, g[np.maximum(second, 0)], 0)
    mine = g[first] if len(first) else np.zeros(0, dtype=np.int64)
    between = two & (mine != other)
    exterior = (count == 1) if include_rve_faces else np.zeros(len(count), dtype=bool)
    keep = between | exterior
    pair = np.column_stack((np.where(between, np.minimum(mine, other), 0), np.where(between, np.maximum(mine, other), mine)))
    return MeshSurface(nodes[keep], pair[keep], np.column_stack((first, second))[keep], between[keep],
                       int(np.sum(count > 2)))


def _edge_keys(elements, pairs, n_nodes):
    a, b = elements[:, [p[0] for p in pairs]], elements[:, [p[1] for p in pairs]]
    return np.minimum(a, b) * n_nodes + np.maximum(a, b)


def _grow(elements, seed, n_nodes, order):
    """Layer of every element: 1 on ``seed``, then one more for each ring of elements sharing a node."""
    layer = np.zeros(len(elements), dtype=np.int8)
    layer[seed] = 1
    for step in range(1, order):
        front = layer == step
        if not front.any():
            break
        reached = np.zeros(n_nodes, dtype=bool)
        reached[elements[front]] = True
        layer[reached[elements].any(axis=1) & (layer == 0)] = step + 1
    return layer


def _check_order(order):
    if int(order) != order or not 1 <= int(order) <= MAX_ORDER:
        raise ValueError(f"The order must be a whole number from 1 to {MAX_ORDER}, not {order!r}.")
    return int(order)


def feature_layers(points, tetrahedra, grains, *, order=1, touch=FACE_OR_EDGE, include_rve_faces=False,
                   features=FEATURES) -> RegionLayers:
    """Layer of every tetrahedron and surface triangle for each of ``features`` up to ``order`` layers away."""
    order = _check_order(order)
    if touch not in TOUCH_CHOICES:
        raise ValueError(f"touch must be one of {', '.join(TOUCH_CHOICES)}, not {touch!r}.")
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"{unknown[0]!r} is not a feature; the features are {', '.join(FEATURES)}.")
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64)
    g = np.asarray(grains, dtype=np.int64)
    n = len(p)
    surface = mesh_surface(t, g, n, include_rve_faces)
    tri = surface.nodes
    out = RegionLayers(order, touch, bool(include_rve_faces), surface)

    edges = tet_junction_edges(p, t, g, include_rve_faces)
    out.edges = np.asarray(edges.nodes, dtype=np.int64).reshape(-1, 2)
    out.line_nodes = np.zeros(n, dtype=bool)
    out.line_nodes[out.edges.ravel()] = True
    out.point_nodes = junction_point_nodes(edges, n, None if include_rve_faces else box_face_bits(p) != 0)
    n_lines = int(len(junction_lines(edges).length)) if len(out.edges) else 0
    out.counts = {"tetrahedra": int(len(t)), "triangles": int(len(tri)),
                  "grain_boundary_triangles": int(surface.between_grains.sum()),
                  "junction_edges": int(len(out.edges)), "junction_lines": n_lines,
                  "junction_line_nodes": int(out.line_nodes.sum()), "junction_points": int(out.point_nodes.sum()),
                  "nonmanifold_faces": surface.nonmanifold}

    if GB_SURFACE in features:
        if touch == FACE_OR_EDGE:
            seed = np.zeros(len(t), dtype=bool)
            on = surface.tets[surface.between_grains]
            seed[on[on >= 0]] = True
        else:
            nodes = np.zeros(n, dtype=bool)
            nodes[tri[surface.between_grains]] = True
            seed = nodes[t].any(axis=1)
        out.tet[GB_SURFACE] = _grow(t, seed, n, order)
        out.triangle[GB_SURFACE] = surface.between_grains.astype(np.int8)
    if JUNCTION_LINES in features:
        if touch == FACE_OR_EDGE and len(out.edges):
            keys = np.unique(np.minimum(out.edges[:, 0], out.edges[:, 1]) * n + np.maximum(out.edges[:, 0], out.edges[:, 1]))
            seed_t = np.isin(_edge_keys(t, PAIRS, n), keys).any(axis=1)
            seed_f = np.isin(_edge_keys(tri, TRIANGLE_PAIRS, n), keys).any(axis=1)
        else:
            seed_t, seed_f = out.line_nodes[t].any(axis=1), out.line_nodes[tri].any(axis=1)
        out.tet[JUNCTION_LINES] = _grow(t, seed_t, n, order)
        out.triangle[JUNCTION_LINES] = _grow(tri, seed_f, n, order)
    if JUNCTION_POINTS in features:
        out.tet[JUNCTION_POINTS] = _grow(t, out.point_nodes[t].any(axis=1), n, order)
        out.triangle[JUNCTION_POINTS] = _grow(tri, out.point_nodes[tri].any(axis=1), n, order)
    return out


def select_layers(layer, order, mode_only: bool):
    """Elements in layers ``1..order`` (or only in layer ``order`` when ``mode_only``)."""
    layer = np.asarray(layer)
    return (layer == order) if mode_only else ((layer >= 1) & (layer <= order))
