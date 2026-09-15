"""Geometric triangle intersections, excluding shared mesh vertices/edges."""
import numpy as np
from scipy.spatial import cKDTree


def _intersecting_pairs(points, faces, pairs, tolerance):
    if not len(pairs):return np.zeros(0,dtype=bool)
    a,b=points[faces[pairs[:,0]]],points[faces[pairs[:,1]]]
    shared=faces[pairs[:,0],:,None]==faces[pairs[:,1],None,:]
    shared_a=shared.any(axis=2)
    hit=np.zeros(len(pairs),bool)
    def inside(x,t):
        u,v=t[:,1]-t[:,0],t[:,2]-t[:,0];w=x-t[:,0]
        uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
        wu=np.einsum('ij,ij->i',w,u);wv=np.einsum('ij,ij->i',w,v)
        den=uu*vv-uv*uv
        s=np.divide(vv*wu-uv*wv,den,out=np.full(len(t),-np.inf),where=den>0)
        q=np.divide(uu*wv-uv*wu,den,out=np.full(len(t),-np.inf),where=den>0)
        eps=tolerance/np.maximum(np.sqrt(np.maximum(uu,vv)),tolerance)
        return (s>=-eps)&(q>=-eps)&(s+q<=1+eps), (s>eps)&(q>eps)&(s+q<1-eps)
    def allowed(x):
        # Shared vertices are legal. A shared edge is legal along its full length.
        legal=np.any(shared_a & (np.linalg.norm(a-x[:,None],axis=2)<=tolerance),axis=1)
        for i,j in ((0,1),(1,2),(2,0)):
            edge=a[:,j]-a[:,i];length2=np.einsum('ij,ij->i',edge,edge)
            t=np.divide(np.einsum('ij,ij->i',x-a[:,i],edge),length2,out=np.zeros(len(a)),where=length2>0)
            projection=a[:,i]+np.clip(t,0,1)[:,None]*edge
            legal |= shared_a[:,i]&shared_a[:,j]&(np.linalg.norm(x-projection,axis=1)<=tolerance)
        return legal
    for source,target in ((a,b),(b,a)):
        normal=np.cross(target[:,1]-target[:,0],target[:,2]-target[:,0])
        length=np.linalg.norm(normal,axis=1)
        normal=np.divide(normal,length[:,None],out=np.zeros_like(normal),where=length[:,None]>0)
        distances=np.einsum('nki,ni->nk',source-target[:,0,None],normal)
        for i,j in ((0,1),(1,2),(2,0)):
            d0,d1=distances[:,i],distances[:,j]
            den=d0-d1
            ratio=np.divide(d0,den,out=np.zeros(len(a)),where=np.abs(den)>tolerance)
            crossing=(np.abs(den)>tolerance)&(ratio>=0)&(ratio<=1)
            x=source[:,i]+ratio[:,None]*(source[:,j]-source[:,i])
            hit |= crossing & inside(x,target)[0] & ~allowed(x)
            # Includes nonadjacent vertex-on-facet and coplanar contacts.
            x=source[:,i]
            hit |= (np.abs(d0)<=tolerance)&inside(x,target)[0]&~allowed(x)
        coplanar=np.all(np.abs(distances)<=tolerance,axis=1)
        hit |= coplanar & inside(source.mean(axis=1),target)[1]
    # Coplanar proper edge crossings (no vertex need lie inside the other face).
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);norm=np.linalg.norm(na,axis=1)
    unit=np.divide(na,norm[:,None],out=np.zeros_like(na),where=norm[:,None]>0)
    coplanar=np.all(np.abs(np.einsum('nki,ni->nk',b-a[:,0,None],unit))<=tolerance,axis=1)
    def orient(u,v,w):return np.einsum('ij,ij->i',np.cross(v-u,w-u),unit)
    for i,j in ((0,1),(1,2),(2,0)):
        for k,l in ((0,1),(1,2),(2,0)):
            s1,s2=orient(a[:,i],a[:,j],b[:,k]),orient(a[:,i],a[:,j],b[:,l])
            t1,t2=orient(b[:,k],b[:,l],a[:,i]),orient(b[:,k],b[:,l],a[:,j])
            hit |= coplanar&(s1*s2 < -tolerance**2)&(t1*t2 < -tolerance**2)
    return hit


def find_surface_intersections(points, triangles, tolerance=None, batch_size=256, triangle_ids=None):
    """Return intersecting triangle index pairs, using a bounded-memory search.

    Adjacent triangles are still checked: sharing one node does not permit them
    to cross elsewhere. Tests include coplanar overlap and vertex/facet contact.
    This uses floating-point tolerances, not exact arithmetic predicates.
    """
    points=np.asarray(points);triangles=np.asarray(triangles)
    if not len(triangles):return np.empty((0,2),int)
    tolerance=1e-9*max(1.,float(np.ptp(points,axis=0).max())) if tolerance is None else tolerance
    xyz=points[triangles];centres=xyz.mean(axis=1)
    radii=np.linalg.norm(xyz-centres[:,None],axis=2).max(axis=1)
    lo,hi=xyz.min(axis=1),xyz.max(axis=1)
    # One unusually long triangle must not enlarge every query over the dense
    # fine mesh. Separate radius bands retain conservative sphere bounds.
    bands=np.floor(np.log2(np.maximum(radii,max(tolerance,1e-30)))).astype(int)
    trees=[]
    for band in np.unique(bands):
        members=np.flatnonzero(bands==band)
        trees.append((members,cKDTree(centres[members]),float(radii[members].max())))
    found=[]
    query_ids=np.arange(len(triangles)) if triangle_ids is None else np.unique(triangle_ids)
    for start in range(0,len(query_ids),batch_size):
        ids=query_ids[start:start+batch_size]
        for members,tree,maximum_radius in trees:
            neighbours=tree.query_ball_point(centres[ids],radii[ids]+maximum_radius+tolerance)
            sizes=np.array([len(n) for n in neighbours])
            if not sizes.sum():continue
            a=np.repeat(ids,sizes);b=members[np.concatenate(neighbours).astype(int)]
            keep=((a<b) if triangle_ids is None else (a!=b))&np.all(lo[a]<=hi[b]+tolerance,axis=1)&np.all(lo[b]<=hi[a]+tolerance,axis=1)
            candidates=np.unique(np.sort(np.column_stack((a[keep],b[keep])),axis=1),axis=0)
            found.append(candidates[_intersecting_pairs(points,triangles,candidates,tolerance)])
    return np.unique(np.vstack(found),axis=0) if found else np.empty((0,2),int)
