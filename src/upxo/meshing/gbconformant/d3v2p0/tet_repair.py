"""Separate touching interface sheets before volume meshing."""
from dataclasses import replace
import numpy as np


def repair_tet_contacts(surface, separation=.15):
    """Split disconnected local surface fans and open their contact slightly.

    Only ambiguous edges with two incident triangles per grain-pair sheet are
    separable here. Genuine multi-grain junction edges remain shared. The
    operation changes local morphology; all changes are made on a copy.
    Planar RVE coordinates remain fixed. Revalidation is mandatory afterwards.
    """
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError('separation must be positive')
    p, f = surface.points.copy(), surface.triangles.copy()
    pairs = np.asarray(surface.grain_pairs)
    edges = np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    unique, inverse = np.unique(edges,axis=0,return_inverse=True)
    owners = [[] for _ in unique]
    for i,e in enumerate(inverse):owners[e].append(i//3)
    cut = set()
    for e,ids in enumerate(owners):
        if len(ids) < 4:continue
        keys,counts=np.unique(pairs[ids],axis=0,return_counts=True)
        if np.all(counts==2) and any(np.sum(np.any(pairs[ids]==g,axis=1))>2 for g in np.unique(keys)):
            cut.add(e)
    incident=[[] for _ in p]
    for i,tri in enumerate(f):
        for v in tri:incident[v].append(i)
    node_edges=[[] for _ in p]
    for e,(a,b) in enumerate(unique):node_edges[a].append(e);node_edges[b].append(e)
    points=p.tolist();splits=[]
    extent=np.asarray(surface.report['rve_dimensions']);tol=1e-9*max(extent)
    for v,ids in enumerate(incident):
        if not ids:continue
        adjacency={i:set() for i in ids}
        for e in node_edges[v]:
            groups={}
            for i in owners[e]:
                key=tuple(pairs[i]) if e in cut else None
                groups.setdefault(key,[]).append(i)
            for group in groups.values():
                for i in group:adjacency[i].update(group)
        unseen=set(ids);components=[]
        while unseen:
            stack=[unseen.pop()];component=[]
            while stack:
                i=stack.pop();component.append(i)
                nxt=adjacency[i]&unseen;unseen.difference_update(nxt);stack.extend(nxt)
            components.append(component)
        if len(components)<2:continue
        fixed=(np.abs(p[v])<tol)|(np.abs(p[v]-extent)<tol)
        for j,ids in enumerate(components):
            neighbours=np.setdiff1d(np.unique(surface.triangles[ids]),[v])
            delta=p[neighbours].mean(axis=0)-p[v];delta[fixed]=0
            norm=np.linalg.norm(delta)
            if norm:delta*=min(separation/norm,.35)
            pos=np.clip(p[v]+delta,0,extent)
            node=v if j==0 else len(points)
            if j==0:points[v]=pos.tolist()
            else:points.append(pos.tolist())
            for i in ids:f[i,f[i]==v]=node
        splits.append(int(v))
    out=replace(surface,points=np.asarray(points),triangles=f,
                report={'rve_dimensions':extent.tolist(),'rve_volume':float(np.prod(extent))})
    before=p[surface.triangles];after=out.points[f]
    normal_before=np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])
    normal_after=np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
    if np.any(np.einsum('ij,ij->i',normal_before,normal_after) <= 0):
        raise RuntimeError('Contact separation inverted/collapsed a triangle; reduce separation')
    out.report['contact_repair']={'split_source_nodes':splits,'added_nodes':len(points)-len(p),
                                  'separation_limit':float(separation),'separable_contact_edges':len(cut)}
    return out
