"""d3v2p0.facet_angles.small_facet_angles with the edge table built from integer keys.

The result is d3v2p0's, element for element. The edge table (see mesh_tables.EdgeIncidence) can be
passed in by a caller that checks many coordinate sets on one connectivity, so it is built once.
"""
from itertools import combinations
import numpy as np
from .mesh_tables import EdgeIncidence
from ..d3v2p0.facet_angles import edge_openings, sharpens  # noqa: F401  (unchanged helpers, re-exported)


def small_facet_angles(points, triangles, minimum_degrees=.1, incidence=None):
    """Return facet pairs and opening angles below the requested threshold.

    Uses opposite-vertex rays projected perpendicular to each shared edge.
    Coplanar neighbouring triangles on opposite sides have a valid 180 degree
    opening; coincident same-side sheets have zero opening. Junction edges with
    three or more incident facets are checked pairwise as well.
    incidence: the EdgeIncidence of ``triangles`` when the caller has it (None builds it).
    """
    if not np.isfinite(minimum_degrees) or not 0 <= minimum_degrees < 180:
        raise ValueError('minimum_degrees must lie in [0, 180)')
    f = np.asarray(triangles); p = np.asarray(points)
    if not len(f) or minimum_degrees == 0:
        return np.empty((0, 2), int), np.empty(0)
    if incidence is None:
        incidence = EdgeIncidence(f)
    elif incidence.n_triangles != len(f):
        raise ValueError('incidence does not belong to these triangles')
    unique, counts, order, offsets = incidence.edges, incidence.counts, incidence.order, incidence.offsets
    opposite = f[:, [2, 0, 1]].ravel(); hits = []; angles = []
    for count in np.unique(counts[counts >= 2]):
        ids = np.flatnonzero(counts == count)
        at = order[offsets[ids, None] + np.arange(count)]
        e = unique[ids]; axis = p[e[:, 1]] - p[e[:, 0]]
        length = np.linalg.norm(axis, axis=1)
        axis = np.divide(axis, length[:, None], out=np.zeros_like(axis), where=length[:, None] > 0)
        radial = p[opposite[at]] - p[e[:, 0], None]
        radial -= np.einsum('nki,ni->nk', radial, axis)[:, :, None] * axis[:, None]
        norms = np.linalg.norm(radial, axis=2)
        unit = np.divide(radial, norms[:, :, None], out=np.zeros_like(radial), where=norms[:, :, None] > 0)
        for a, b in combinations(range(int(count)), 2):
            angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(unit[:, a], unit[:, b]), axis=1),
                                          np.einsum('ij,ij->i', unit[:, a], unit[:, b])))
            bad = (angle < minimum_degrees) & (norms[:, a] > 0) & (norms[:, b] > 0) & (length > 0)
            hits.append(at[bad][:, [a, b]] // 3); angles.append(angle[bad])
    return (np.vstack(hits), np.concatenate(angles)) if hits else (np.empty((0, 2), int), np.empty(0))
