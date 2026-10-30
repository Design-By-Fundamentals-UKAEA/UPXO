"""The two element types a deck can hold: linear tetrahedra (C3D4) and linear hexahedra (C3D8).

The type is told by the number of corners per element (4 or 8). Corner, edge and face numbering
follow Abaqus: a C3D8 has nodes 1-4 on one face and 5-8 on the opposite face, and faces S1..S6.
A voxel at grid position ``(i, j, k)`` has corners ``(0,0,0) (1,0,0) (1,1,0) (0,1,0)`` then the same
four at ``z + 1``, which gives a positive volume.
"""
from dataclasses import dataclass

import numpy as np

_CHUNK = 250_000


@dataclass(frozen=True)
class ElementKind:
    abaqus: str
    n_nodes: int
    edges: tuple
    """Corner pairs of the element edges."""
    faces: tuple
    """Corners of every face, in cyclic order, pointing out of the element."""
    face_edges: tuple
    """Corner pairs of the edges of one face, in face-local numbering."""
    vtk_type: int
    gmsh_type: int
    noun: str


TET4 = ElementKind(
    abaqus='C3D4', n_nodes=4,
    edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    faces=((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)),
    face_edges=((0, 1), (0, 2), (1, 2)),
    vtk_type=10, gmsh_type=4, noun='tetrahedra')

HEX8 = ElementKind(
    abaqus='C3D8', n_nodes=8,
    edges=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    faces=((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    face_edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    vtk_type=12, gmsh_type=5, noun='hexahedra')

KINDS = {4: TET4, 8: HEX8}
# The six tetrahedra a hexahedron splits into around its 0-6 diagonal (positive for a positive hexahedron).
_HEX_TETS = np.array(((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)))


def element_kind(elements) -> ElementKind:
    """The :class:`ElementKind` of an ``(M, 4)`` or ``(M, 8)`` connectivity array."""
    shape = np.shape(elements)
    if len(shape) != 2 or shape[1] not in KINDS:
        raise ValueError(f'Expected Mx4 tetrahedra or Mx8 hexahedra, not shape {shape}.')
    return KINDS[shape[1]]


def _tet_volume(v):
    return np.einsum('ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])) / 6


def element_volumes(points, elements, chunk=_CHUNK):
    """Signed volume of every element (positive for a correctly numbered element)."""
    p = np.asarray(points, dtype=float)
    e = np.asarray(elements)
    kind = element_kind(e)
    out = np.empty(len(e))
    for start in range(0, len(e), chunk):
        v = p[e[start:start + chunk]]
        if kind is TET4:
            out[start:start + chunk] = _tet_volume(v)
        else:
            out[start:start + chunk] = sum(_tet_volume(v[:, corners]) for corners in _HEX_TETS)
    return out


def element_faces(elements, n_nodes, node_mask=None, chunk=200_000):
    """Every distinct face: ``(nodes (F, k), first element, second element or -1, elements on the face)``.

    Tetrahedron faces come with their nodes ascending; hexahedron faces keep the cyclic order of the
    first element that has them (a quadrilateral needs it to stay a quadrilateral). Faces are ordered
    by their ascending node triple.

    ``node_mask`` (a boolean per node) keeps only the faces whose nodes are all in it: the other faces are
    never built into the tables that are sorted, which is what makes the call cheap for a large mesh when
    only some faces matter. The faces that are kept, and their order, are the same as without it.
    """
    e = np.asarray(elements)
    kind = element_kind(e) if e.ndim == 2 and e.shape[1] in KINDS else TET4
    width = len(kind.faces[0])
    if not len(e):
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, width), dtype=np.int64), empty, empty.copy(), empty.copy()
    corners, per = np.array(kind.faces), len(kind.faces)
    cyclic_parts, owner_parts = [], []
    for start in range(0, len(e), chunk):
        block = e[start:start + chunk].astype(np.int64, copy=False)
        cyc = block[:, corners].reshape(-1, width)
        own = np.repeat(np.arange(start, start + len(block)), per)
        if node_mask is not None:
            keep = node_mask[cyc].all(axis=1)
            cyc, own = cyc[keep], own[keep]
        cyclic_parts.append(cyc)
        owner_parts.append(own)
    cyclic = np.concatenate(cyclic_parts)
    owner = np.concatenate(owner_parts)
    del cyclic_parts, owner_parts
    ordered = np.sort(cyclic, axis=1)
    if not len(ordered):
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, width), dtype=np.int64), empty, empty.copy(), empty.copy()
    # Three corners of a face name it: two faces of a conforming mesh never share three nodes.
    triple = ordered[:, :3]
    n = int(n_nodes)
    if n ** 3 < 2 ** 62:
        key = (triple[:, 0] * n + triple[:, 1]) * n + triple[:, 2]
        order = np.argsort(key, kind='stable')
        sorted_key = key[order]
        starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    else:
        order = np.lexsort((triple[:, 2], triple[:, 1], triple[:, 0]))
        sorted_triple = triple[order]
        starts = np.flatnonzero(np.r_[True, np.any(sorted_triple[1:] != sorted_triple[:-1], axis=1)])
    count = np.diff(np.r_[starts, len(order)])
    first = owner[order[starts]]
    second = np.where(count >= 2, owner[order[np.minimum(starts + 1, len(order) - 1)]], -1)
    nodes = ordered if width == 3 else cyclic
    return nodes[order[starts]], first, second, count


def face_areas(points, faces):
    """Area of every triangle or (planar) quadrilateral; quadrilateral corners in cyclic order."""
    p = np.asarray(points, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return np.zeros(0)
    v = p[f]
    if f.shape[1] == 3:
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    return 0.5 * np.linalg.norm(np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]), axis=1)


def grid_hexahedra(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Nodes ``(N, 3)`` and C3D8 connectivity ``(M, 8)`` of a voxel grid of ``shape``.

    Node ``(i, j, k)`` has index ``(i * (ny + 1) + j) * (nz + 1) + k``; voxel ``(i, j, k)`` is element
    ``(i * ny + j) * nz + k`` (C order, the order of ``labels.ravel()``).
    """
    nx, ny, nz = (int(n) for n in shape)
    if min(nx, ny, nz) < 1:
        raise ValueError(f'A voxel grid needs at least one voxel along each axis, not {shape}.')
    sx, sy, sz = (float(s) for s in spacing)
    shape_n = (nx + 1, ny + 1, nz + 1)
    points = np.empty((shape_n[0] * shape_n[1] * shape_n[2], 3))
    points[:, 0] = np.broadcast_to((np.arange(nx + 1) * sx)[:, None, None], shape_n).ravel()
    points[:, 1] = np.broadcast_to((np.arange(ny + 1) * sy)[None, :, None], shape_n).ravel()
    points[:, 2] = np.broadcast_to((np.arange(nz + 1) * sz)[None, None, :], shape_n).ravel()
    points += np.asarray(origin, dtype=float)
    stride_i, stride_j = (ny + 1) * (nz + 1), nz + 1
    # node numbers fit in 32 bits for any grid of up to about 1290 voxels along each axis
    dtype = np.int32 if len(points) < 2 ** 31 - 1 else np.int64
    base = (np.arange(nx)[:, None, None] * stride_i + np.arange(ny)[None, :, None] * stride_j
            + np.arange(nz)[None, None, :]).ravel().astype(dtype)
    offsets = np.array([0, stride_i, stride_i + stride_j, stride_j], dtype=dtype)
    hexahedra = np.empty((len(base), 8), dtype=dtype)
    hexahedra[:, :4] = base[:, None] + offsets[None, :]
    hexahedra[:, 4:] = hexahedra[:, :4] + dtype(1)
    return points, hexahedra
