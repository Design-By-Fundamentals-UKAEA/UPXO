"""Grain-boundary junction lines of a voxel label array and of a tetrahedral mesh.

A junction edge is a segment shared by three or more regions. Each edge
carries the set of regions meeting along it (labels sorted ascending and
padded on the left with -1; the box exterior, when included, is label 0).
Edges with the same region set that touch end to end form one junction line.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SET_WIDTH = 6
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass
class JunctionEdges:
    """Junction edges: node pairs, region sets and lengths."""
    nodes: np.ndarray
    sets: np.ndarray
    lengths: np.ndarray
    truncated: int = 0


@dataclass
class JunctionLines:
    """Junction lines: connected junction edges sharing one region set."""
    sets: np.ndarray        # (S, SET_WIDTH) distinct region sets
    set_index: np.ndarray   # (L,) index into ``sets`` for every line
    length: np.ndarray      # (L,)
    n_edges: np.ndarray     # (L,)


def _empty_edges():
    return JunctionEdges(np.zeros((0, 2), dtype=np.int64),
                         np.full((0, SET_WIDTH), -1, dtype=np.int64), np.zeros(0))


def voxel_junction_edges(labels, include_rve_faces=False):
    """Junction edges of a voxel label array; each edge is one voxel edge long.

    With ``include_rve_faces`` the box exterior counts as a region (label 0),
    so grain-boundary traces on the box faces are included. Without it only
    edges inside the box are used.
    """
    a = np.asarray(labels)
    if a.ndim != 3 or a.dtype.kind not in 'iu' or (a.size and a.min() < 0):
        raise ValueError('labels must be a 3D array of nonnegative integers')
    if include_rve_faces:
        a = np.pad(a, 1, constant_values=0)
    a = a.astype(np.int64, copy=False)
    stride_i, stride_j = (a.shape[1] + 1) * (a.shape[2] + 1), a.shape[2] + 1
    node_parts, set_parts = [], []
    for edge_axis in range(3):
        pa, pb = [ax for ax in range(3) if ax != edge_axis]
        quadrants = []
        for lo_a in (True, False):
            for lo_b in (True, False):
                window = [slice(None)] * 3
                window[pa] = slice(None, -1) if lo_a else slice(1, None)
                window[pb] = slice(None, -1) if lo_b else slice(1, None)
                quadrants.append(a[tuple(window)])
        shape = quadrants[0].shape
        if quadrants[0].size == 0:
            continue
        around = np.sort(np.stack([q.ravel() for q in quadrants], axis=1), axis=1)
        repeated = np.concatenate(
            [np.zeros((len(around), 1), dtype=bool), np.diff(around, axis=1) == 0], axis=1)
        selected = np.flatnonzero((~repeated).sum(axis=1) >= 3)
        if not len(selected):
            continue
        index = np.unravel_index(selected, shape)
        start = np.stack([index[ax] if ax == edge_axis else index[ax] + 1 for ax in range(3)], axis=1)
        end = start.copy()
        end[:, edge_axis] += 1
        node_parts.append(np.column_stack((start @ [stride_i, stride_j, 1], end @ [stride_i, stride_j, 1])))
        row = np.sort(np.where(repeated[selected], -1, around[selected]), axis=1)
        set_parts.append(np.hstack([np.full((len(row), SET_WIDTH - 4), -1, dtype=np.int64), row]))
    if not node_parts:
        return _empty_edges()
    nodes = np.vstack(node_parts)
    return JunctionEdges(nodes, np.vstack(set_parts), np.ones(len(nodes)))


def box_face_bits(points, face_tolerance=1e-8):
    """One bit per box face for every node: bit ``2 axis`` is the low face of the axis, ``2 axis + 1`` the high one."""
    p = np.asarray(points, dtype=float)
    bits = np.zeros(len(p), dtype=np.int64)
    if not len(p):
        return bits
    tol = face_tolerance * float((p.max(axis=0) - p.min(axis=0)).max())
    for axis in range(3):
        bits |= (np.abs(p[:, axis] - p[:, axis].min()) <= tol).astype(np.int64) << (2 * axis)
        bits |= (np.abs(p[:, axis] - p[:, axis].max()) <= tol).astype(np.int64) << (2 * axis + 1)
    return bits


def junction_point_nodes(edges, n_nodes, ignore_ends=None):
    """Nodes where junction lines end, branch or change their region set: the junction points.

    A node is a junction point when its junction edges carry two or more different region sets, when three or
    more junction edges meet at it, or when only one does (a line that ends). ``ignore_ends`` is a boolean mask of
    nodes whose line ends do not count (nodes on the box faces, when the box exterior is not a region).
    """
    point = np.zeros(int(n_nodes), dtype=bool)
    if not len(edges.nodes):
        return point
    sets, set_index = np.unique(edges.sets, axis=0, return_inverse=True)
    ends = np.asarray(edges.nodes, dtype=np.int64).ravel()
    degree = np.bincount(ends, minlength=int(n_nodes))
    pair = np.unique(ends * len(sets) + np.repeat(set_index.ravel(), 2))
    kinds = np.bincount(pair // len(sets), minlength=int(n_nodes))
    dangling = degree == 1
    if ignore_ends is not None:
        dangling &= ~np.asarray(ignore_ends, dtype=bool)
    return (kinds >= 2) | (degree >= 3) | dangling


def tet_junction_edges(points, tetrahedra, grains, include_rve_faces=False,
                       face_tolerance=1e-8):
    """Junction edges of a tetrahedral mesh; ``grains`` is one label per tetrahedron.

    ``grains`` must be positive. With ``include_rve_faces`` an edge lying in a
    box face also belongs to the exterior region (label 0).
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64)
    g = np.asarray(grains, dtype=np.int64)
    if g.shape != (len(t),) or (len(g) and g.min() <= 0):
        raise ValueError('grains must hold one positive label per tetrahedron')
    n_nodes, stride = len(p), int(g.max()) + 2
    face_bits = box_face_bits(p, face_tolerance) if include_rve_faces else np.zeros(n_nodes, dtype=np.int64)
    node_pairs = np.unique(np.concatenate((
        t.ravel() * stride + np.repeat(g, 4) + 1,
        np.flatnonzero(face_bits) * stride + 1)))
    regions_at_node = np.bincount(node_pairs // stride, minlength=n_nodes)
    candidate = regions_at_node >= 3
    edge_keys, edge_grains = [], []
    for i, j in PAIRS:
        a, b = t[:, i], t[:, j]
        keep = candidate[a] & candidate[b]
        if not keep.any():
            continue
        a, b, gk = a[keep], b[keep], g[keep]
        key = np.minimum(a, b) * n_nodes + np.maximum(a, b)
        edge_keys.append(key)
        edge_grains.append(gk + 1)
        if include_rve_faces:
            on_face = (face_bits[a] & face_bits[b]) != 0
            edge_keys.append(key[on_face])
            edge_grains.append(np.ones(int(on_face.sum()), dtype=np.int64))
    if not edge_keys:
        return _empty_edges()
    pair = np.unique(np.concatenate(edge_keys) * stride + np.concatenate(edge_grains))
    edge, region = pair // stride, pair % stride - 1
    unique_edges, inverse = np.unique(edge, return_inverse=True)
    counts = np.bincount(inverse)
    first = np.concatenate(([0], np.flatnonzero(np.diff(edge)) + 1))
    rank = np.arange(len(edge)) - first[inverse]
    is_junction = counts[inverse] >= 3
    fill = is_junction & (rank < SET_WIDTH)
    sets = np.full((len(unique_edges), SET_WIDTH), -1, dtype=np.int64)
    sets[inverse[fill], rank[fill]] = region[fill]
    selected = counts >= 3
    if not selected.any():
        return _empty_edges()
    keys = unique_edges[selected]
    nodes = np.column_stack((keys // n_nodes, keys % n_nodes))
    lengths = np.linalg.norm(p[nodes[:, 0]] - p[nodes[:, 1]], axis=1)
    return JunctionEdges(nodes, np.sort(sets[selected], axis=1), lengths,
                         truncated=int((counts[selected] > SET_WIDTH).sum()))


def junction_lines(edges):
    """Group junction edges into lines: connected edges with the same region set."""
    if not len(edges.nodes):
        empty = np.zeros(0)
        return JunctionLines(np.full((0, SET_WIDTH), -1, dtype=np.int64),
                             empty.astype(np.int64), empty, empty.astype(np.int64))
    sets, set_index = np.unique(edges.sets, axis=0, return_inverse=True)
    set_index = set_index.ravel()
    n_sets = len(sets)
    keys = np.concatenate((edges.nodes[:, 0] * n_sets + set_index,
                           edges.nodes[:, 1] * n_sets + set_index))
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n = len(edges.nodes)
    a, b = inverse[:n], inverse[n:]
    graph = coo_matrix((np.ones(n), (a, b)), shape=(len(unique_keys),) * 2)
    count, component = connected_components(graph, directed=False)
    edge_component = component[a]
    length = np.bincount(edge_component, weights=edges.lengths, minlength=count)
    n_edges = np.bincount(edge_component, minlength=count)
    line_set = np.zeros(count, dtype=np.int64)
    line_set[edge_component] = set_index
    return JunctionLines(sets, line_set, length, n_edges)


def regions_of(set_row):
    """Positive labels of one region-set row (drops padding and the exterior)."""
    return tuple(int(v) for v in set_row if v > 0)


def totals_per_grain(lines):
    """{label: total length of the lines that border it}."""
    totals = {}
    for column in range(lines.sets.shape[1]):
        labels = lines.sets[lines.set_index, column]
        valid = labels > 0
        if not valid.any():
            continue
        values = np.bincount(labels[valid], weights=lines.length[valid])
        for label in np.flatnonzero(values):
            totals[int(label)] = totals.get(int(label), 0.0) + float(values[label])
    return totals


def totals_per_set(lines):
    """{region set (tuple of labels incl. exterior 0): total length}."""
    sums = np.bincount(lines.set_index, weights=lines.length, minlength=len(lines.sets))
    return {tuple(int(v) for v in row if v >= 0): float(total)
            for row, total in zip(lines.sets, sums)}
