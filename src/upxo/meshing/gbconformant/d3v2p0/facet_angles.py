"""Detect almost closed wedges between facets meeting along a shared edge."""
from itertools import combinations
import numpy as np


def small_facet_angles(points, triangles, minimum_degrees=.1):
    """Return facet pairs and opening angles below the requested threshold.

    Uses opposite-vertex rays projected perpendicular to each shared edge.
    Coplanar neighbouring triangles on opposite sides have a valid 180 degree
    opening; coincident same-side sheets have zero opening. Junction edges with
    three or more incident facets are checked pairwise as well.
    """
    if not np.isfinite(minimum_degrees) or not 0 <= minimum_degrees < 180:
        raise ValueError('minimum_degrees must lie in [0, 180)')
    f=np.asarray(triangles);p=np.asarray(points)
    if not len(f) or minimum_degrees==0:return np.empty((0,2),int),np.empty(0)
    edges=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');offsets=np.r_[0,np.cumsum(counts)]
    opposite=f[:,[2,0,1]].ravel();hits=[];angles=[]
    for count in np.unique(counts[counts>=2]):
        ids=np.flatnonzero(counts==count)
        incidence=order[offsets[ids,None]+np.arange(count)]
        e=unique[ids];axis=p[e[:,1]]-p[e[:,0]]
        length=np.linalg.norm(axis,axis=1)
        axis=np.divide(axis,length[:,None],out=np.zeros_like(axis),where=length[:,None]>0)
        radial=p[opposite[incidence]]-p[e[:,0],None]
        radial-=np.einsum('nki,ni->nk',radial,axis)[:,:,None]*axis[:,None]
        norms=np.linalg.norm(radial,axis=2)
        unit=np.divide(radial,norms[:,:,None],out=np.zeros_like(radial),where=norms[:,:,None]>0)
        for a,b in combinations(range(int(count)),2):
            angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(unit[:,a],unit[:,b]),axis=1),
                                       np.einsum('ij,ij->i',unit[:,a],unit[:,b])))
            bad=(angle<minimum_degrees)&(norms[:,a]>0)&(norms[:,b]>0)&(length>0)
            hits.append(incidence[bad][:,[a,b]]//3);angles.append(angle[bad])
    return (np.vstack(hits),np.concatenate(angles)) if hits else (np.empty((0,2),int),np.empty(0))


def edge_openings(points, triangles, edges, edge_triangles):
    """Smallest pairwise opening (degrees) at each given edge.

    edges: iterable of (a, b) node pairs; edge_triangles: mapping from the
    sorted pair to the triangle rows sharing that edge. Edges with fewer
    than two triangles give 180.
    """
    p = np.asarray(points)
    f = np.asarray(triangles)
    out = []
    for a, b in edges:
        key = (min(a, b), max(a, b))
        rows = edge_triangles.get(key, [])
        if len(rows) < 2:
            out.append(180.)
            continue
        axis = p[key[1]] - p[key[0]]
        length = np.linalg.norm(axis)
        if length == 0:
            out.append(0.)
            continue
        axis = axis / length
        radial = []
        for r in rows:
            o = [n for n in f[r] if n not in key]
            v = p[o[0]] - p[key[0]]
            v = v - (v @ axis) * axis
            n = np.linalg.norm(v)
            radial.append(v / n if n > 0 else v)
        best = 180.
        for i in range(len(radial)):
            for j in range(i + 1, len(radial)):
                best = min(best, float(np.degrees(np.arctan2(np.linalg.norm(np.cross(radial[i], radial[j])),
                                                              radial[i] @ radial[j]))))
        out.append(best)
    return np.asarray(out)


def sharpens(before, after, limit, tolerance=1e-6):
    """True where an opening breaks the no-sharpening rule: below ``limit`` it
    may not decrease, at or above ``limit`` it may not fall below it."""
    before, after = np.asarray(before, float), np.asarray(after, float)
    return after < np.minimum(before, limit) - tolerance
