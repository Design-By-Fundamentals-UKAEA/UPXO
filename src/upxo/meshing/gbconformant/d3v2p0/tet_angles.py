"""Dihedral angles of linear tetrahedra."""
import numpy as np

_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def dihedral_angles_xyz(x):
    """Interior dihedral angles (degrees) of tets given as coordinates.

    x has shape (..., 4, 3); the result has shape (..., 6), one angle per edge
    in the order (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    """
    x = np.asarray(x, dtype=float)
    centre = x.mean(axis=-2)
    normals = {}
    for face in _FACES:
        a, b, c = x[..., face[0], :], x[..., face[1], :], x[..., face[2], :]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)
        side = np.sign(np.sum((a - centre) * n, axis=-1, keepdims=True))
        normals[face] = n * side                               # outward
    out = np.empty(x.shape[:-2] + (6,))
    for k, edge in enumerate(_EDGES):
        f1, f2 = [face for face in _FACES if edge[0] in face and edge[1] in face]
        cosine = np.clip(np.sum(normals[f1] * normals[f2], axis=-1), -1, 1)
        out[..., k] = 180. - np.degrees(np.arccos(cosine))
    return out


def dihedral_angles(points, tetrahedra, batch_size=500000):
    """Interior dihedral angles (degrees), shape (n, 6), one per tet edge.

    The angle at an edge is the angle between the two faces sharing it,
    measured inside the tetrahedron. A regular tetrahedron gives 70.53 degrees
    at every edge; slivers and needles give angles near 0 or 180.
    """
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra)
    out = np.empty((len(tetrahedra), 6))
    for start in range(0, len(tetrahedra), batch_size):
        out[start:start + batch_size] = dihedral_angles_xyz(points[tetrahedra[start:start + batch_size]])
    return out


def dihedral_summary(points, tetrahedra, thresholds=(5., 10., 15., 20., 30.),
                     upper_thresholds=(150., 160., 170.)):
    """Smallest and largest dihedral angle per tet, summarised for reports."""
    angles = dihedral_angles(points, tetrahedra)
    smallest, largest = angles.min(axis=1), angles.max(axis=1)
    if not len(smallest):
        return dict(minimum=None, maximum=None)
    return dict(minimum=float(smallest.min()), maximum=float(largest.max()),
                percentiles_minimum_0p1_1_50=np.percentile(smallest, [.1, 1, 50]).tolist(),
                **{f'tets_below_{t:g}_deg': int(np.sum(smallest < t)) for t in thresholds},
                **{f'tets_above_{t:g}_deg': int(np.sum(largest > t)) for t in upper_thresholds})
