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
    e = np.asarray(elements, dtype=np.int64)
    kind = element_kind(e)
    out = np.empty(len(e))
    for start in range(0, len(e), chunk):
        v = p[e[start:start + chunk]]
        if kind is TET4:
            out[start:start + chunk] = _tet_volume(v)
        else:
            out[start:start + chunk] = sum(_tet_volume(v[:, corners]) for corners in _HEX_TETS)
    return out


def element_faces(elements, n_nodes):
    """Every distinct face: ``(nodes (F, k), first element, second element or -1, elements on the face)``.

    Tetrahedron faces come with their nodes ascending; hexahedron faces keep the cyclic order of the
    first element that has them (a quadrilateral needs it to stay a quadrilateral). Faces are ordered
    by their ascending node triple.
    """
    e = np.asarray(elements, dtype=np.int64)
    kind = element_kind(e) if e.ndim == 2 and e.shape[1] in KINDS else TET4
    width = len(kind.faces[0])
    if not len(e):
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, width), dtype=np.int64), empty, empty.copy(), empty.copy()
    cyclic = e[:, np.array(kind.faces)].reshape(-1, width)
    ordered = np.sort(cyclic, axis=1)
    # Three corners of a face name it: two faces of a conforming mesh never share three nodes.
    triple = ordered[:, :3]
    owner = np.repeat(np.arange(len(e)), len(kind.faces))
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
    i, j, k = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij')
    points = np.column_stack((i.ravel() * sx, j.ravel() * sy, k.ravel() * sz)) + np.asarray(origin, dtype=float)
    stride_i, stride_j = (ny + 1) * (nz + 1), nz + 1
    vi, vj, vk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    base = (vi.ravel() * stride_i + vj.ravel() * stride_j + vk.ravel()).astype(np.int64)
    offsets = np.array([0, stride_i, stride_i + stride_j, stride_j], dtype=np.int64)
    bottom = base[:, None] + offsets[None, :]
    return points, np.hstack((bottom, bottom + 1))
