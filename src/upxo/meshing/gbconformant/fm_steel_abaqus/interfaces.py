"""The triangles of a tetrahedral mesh that separate one feature from another.

A triangle is an interface when the two tetrahedra that share it carry different feature labels.
Triangles of the box faces (one tetrahedron only) are not interfaces. Areas are in mesh coordinates.
A hexahedral mesh (``(M, 8)`` connectivity) gives quadrilaterals, ``faces`` ``(K, 4)`` in cyclic order.
"""
from dataclasses import dataclass
import numpy as np

_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


@dataclass
class Interfaces:
    """``faces`` (K, 3) node indices, ``labels`` (K, 2) the two features (ascending), ``area`` (K,)."""
    faces: np.ndarray
    labels: np.ndarray
    area: np.ndarray

    def pairs(self):
        """``(pairs, inverse)``: the distinct label pairs (P, 2) and, for every face, the index of its pair."""
        if not len(self.faces):
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
        pairs, inverse = np.unique(self.labels, axis=0, return_inverse=True)
        return pairs, inverse.ravel()


def interface_faces(points, tetrahedra, tet_label):
    """The :class:`Interfaces` of a mesh; ``tet_label`` is one feature label per tetrahedron."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    label = np.asarray(tet_label, dtype=np.int64)
    if label.shape != (len(t),):
        raise ValueError('tet_label needs one label per tetrahedron.')
    if t.ndim == 2 and t.shape[1] == 8:
        return _hexahedron_interfaces(p, t, label)
    tri = np.sort(t[:, _FACES].reshape(-1, 3), axis=1)
    owner = np.repeat(label, 4)
    order = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    tri, owner = tri[order], owner[order]
    same_as_next = np.all(tri[1:] == tri[:-1], axis=1)
    if len(same_as_next) > 1 and np.any(same_as_next[1:] & same_as_next[:-1]):
        raise ValueError('A triangle belongs to three tetrahedra: the mesh is not conforming.')
    first = np.flatnonzero(same_as_next)
    differ = first[owner[first] != owner[first + 1]]
    faces = tri[differ]
    labels = np.sort(np.column_stack((owner[differ], owner[differ + 1])), axis=1)
    v = p[faces]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) if len(faces) else np.zeros(0)
    return Interfaces(faces=faces, labels=labels, area=area)


def _hexahedron_interfaces(points, hexahedra, label):
    from .elements import element_faces, face_areas

    nodes, first, second, count = element_faces(hexahedra, len(points))
    if np.any(count > 2):
        raise ValueError('A face belongs to three hexahedra: the mesh is not conforming.')
    two = count == 2
    differ = two & (label[first] != label[np.maximum(second, 0)])
    faces = nodes[differ]
    labels = np.sort(np.column_stack((label[first[differ]], label[second[differ]])), axis=1)
    return Interfaces(faces=faces, labels=labels, area=face_areas(points, faces))
