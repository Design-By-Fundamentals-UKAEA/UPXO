"""Bounded separation of crossing interfaces beside a shared junction line."""
from dataclasses import replace
import numpy as np
from .surface_intersections import find_surface_intersections


def repair_interface_crossings(surface, max_displacement=.35, radius=1., opening=.1,
                               max_passes=3):
    """Open intersecting interface wedges while freezing their shared junction.

    Only free vertices on intersecting triangles move. Each pair of interfaces
    must share a grain and at least two junction nodes at the crossing. A local
    separating plane passes through that line; outward normals determine the
    sides. opening controls separation relative to distance from the line.
    All patch junction nodes and RVE-plane coordinates remain fixed. Unsupported
    intersections or failed checks raise rather than returning a claimed fix.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (max_displacement,radius,opening)):
        raise ValueError('Displacement, radius and opening must be positive')
    if not isinstance(max_passes,int) or max_passes < 1:raise ValueError('max_passes must be a positive integer')
    original=np.asarray(surface.points);p=original.copy();f=np.asarray(surface.triangles)
    keys,patch=np.unique(np.column_stack((surface.grain_pairs,surface.rve_face)),axis=0,return_inverse=True)
    memberships=np.unique(np.column_stack((f.ravel(),np.repeat(patch,3))),axis=0)
    fixed=np.bincount(memberships[:,0],minlength=len(p))>1
    extent=np.asarray(surface.report['rve_dimensions']);tol=1e-9*max(1.,float(extent.max()))
    fixed |= np.any((np.abs(p)<=tol)|(np.abs(p-extent)<=tol),axis=1)
    xyz=original[f];old_normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    crossings=find_surface_intersections(p,f)
    initial=len(crossings);passes=0
    for passes in range(1,max_passes+1):
        if not len(crossings):break
        xyz=p[f];normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        for a,b in np.unique(np.sort(patch[crossings],axis=1),axis=0):
            hits=crossings[np.all(np.sort(patch[crossings],axis=1)==[a,b],axis=1)]
            common=np.intersect1d(keys[a,:2],keys[b,:2])
            if a==b or len(common)!=1 or keys[a,2]>=0 or keys[b,2]>=0:
                raise RuntimeError(f'Unsupported crossing between patches {keys[a].tolist()} and {keys[b].tolist()}')
            # Separate distant crossing sites of the same patch pair.
            adjacency={int(i):set() for i in np.unique(hits)}
            node_owner={}
            for i in adjacency:
                for node in f[i]:
                    if node in node_owner:
                        j=node_owner[node];adjacency[i].add(j);adjacency[j].add(i)
                    node_owner[node]=i
            for i,j in hits:adjacency[int(i)].add(int(j));adjacency[int(j)].add(int(i))
            remaining=set(adjacency)
            while remaining:
                stack=[remaining.pop()];ids=[]
                while stack:
                    i=stack.pop();ids.append(i);added=adjacency[i]&remaining
                    remaining-=added;stack.extend(added)
                ids=np.asarray(ids)
                pa=np.unique(f[ids[patch[ids]==a]]);pb=np.unique(f[ids[patch[ids]==b]])
                shared=np.intersect1d(pa,pb)
                if len(shared)<2:
                    raise RuntimeError(f'Crossing faces {ids.tolist()} do not share a repairable junction line')
                anchor=p[shared].mean(axis=0)
                _,_,vectors=np.linalg.svd(p[shared]-anchor,full_matrices=False)
                line=vectors[0]
                centre=p[np.unique(f[ids])].mean(axis=0)
                near=np.linalg.norm(p-centre,axis=1)<radius
                na=normals[(patch==a)&np.any(near[f],axis=1)].sum(axis=0)*(1 if keys[a,0]==common[0] else -1)
                nb=normals[(patch==b)&np.any(near[f],axis=1)].sum(axis=0)*(1 if keys[b,0]==common[0] else -1)
                if min(np.linalg.norm(na),np.linalg.norm(nb))<=tol**2:
                    raise RuntimeError('Cannot estimate normals at an interface crossing')
                normal=na/np.linalg.norm(na)-nb/np.linalg.norm(nb)
                normal-=np.dot(normal,line)*line
                if np.linalg.norm(normal)<1e-10:raise RuntimeError('Ambiguous crossing separation direction')
                normal/=np.linalg.norm(normal)
                for nodes,sign in ((pa,1),(pb,-1)):
                    nodes=nodes[~fixed[nodes]]
                    rel=p[nodes]-anchor;distance=rel@normal
                    radial=np.linalg.norm(rel-np.outer(rel@line,line),axis=1)
                    push=np.maximum(0,opening*radial-sign*distance)
                    p[nodes]+=np.outer(push*sign,normal)
        displacement=np.linalg.norm(p-original,axis=1)
        if displacement.max()>max_displacement+tol:
            raise RuntimeError(f'Intersection repair needs displacement {displacement.max():.6g}, above limit {max_displacement}')
        xyz=p[f];new_normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        if np.any(np.einsum('ij,ij->i',old_normal,new_normal)<=0):
            raise RuntimeError('Intersection repair would invert a triangle; adjust radius/opening or repair upstream')
        crossings=find_surface_intersections(p,f)
        if not len(crossings):break
    if len(crossings):
        raise RuntimeError(f'{len(crossings)} surface intersections remain after repair; example faces {crossings[:5].tolist()}')
    report=dict(surface.report)
    displacement=np.linalg.norm(p-original,axis=1)
    report['intersection_repair']={'initial_intersections':initial,'remaining_intersections':0,
        'moved_nodes':np.flatnonzero(displacement>tol).tolist(),
        'maximum_displacement':float(displacement.max()),'junction_nodes_fixed':True,
        'passes':passes if initial else 0}
    return replace(surface,points=p,report=report)
