"""Bounded local rollback of smoothing that creates intersections or cap tangencies."""
from dataclasses import replace
import numpy as np
from .surface_intersections import find_surface_intersections
from .facet_angles import small_facet_angles


def stabilize_interfaces(surface, dimensions, enabled=True, boundary_clearance=.025,
                         max_passes=12, relaxation=.5, max_correction=None,
                         minimum_facet_angle=1.):
    """Pull defective neighbourhoods toward their extracted voxel coordinates.

    Checks all triangle intersections, source-relative normal reversal/collapse,
    and clearance of originally interior nodes from each box plane. This restores
    local voxel detail where required; it does not split connectivity or delete
    grains. Frozen junctions and original RVE coordinates remain exact. Failure
    raises instead of returning a geometry claimed to be suitable for meshing.
    ``max_correction`` bounds movement from the incoming smoothed coordinates;
    None permits rollback all the way to the extracted geometry.
    """
    extent=np.asarray(dimensions,dtype=float)
    if extent.shape!=(3,) or np.any(~np.isfinite(extent)) or np.any(extent<=0):raise ValueError('Invalid dimensions')
    if not 0<relaxation<1 or not isinstance(max_passes,int) or max_passes<1:raise ValueError('Invalid rollback controls')
    if not np.isfinite(boundary_clearance) or boundary_clearance<=0:raise ValueError('Clearance must be positive')
    if max_correction is not None and (not np.isfinite(max_correction) or max_correction<=0):raise ValueError('Invalid correction limit')
    original=surface.original_points;target=surface.points;f=surface.triangles
    if np.any(original<0) or np.any(original>extent):raise ValueError('Reference geometry outside RVE')
    p=target.copy();weight=np.ones(len(p));history=[]
    if not enabled:return replace(surface,points=p),dict(enabled=False,verified=False)
    edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    xyz=original[f];normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    area=np.linalg.norm(normals,axis=1)
    # A requested clearance cannot exceed a node's original distance to a face.
    lower=np.minimum(original,boundary_clearance)
    upper=extent-np.minimum(extent-original,boundary_clearance)
    query_ids=None;hits=np.empty((0,2),dtype=int)
    for step in range(max_passes+1):
        retained=hits if query_ids is None else hits[~np.any(np.isin(hits,query_ids),axis=1)]
        new_hits=find_surface_intersections(p,f,triangle_ids=query_ids)
        hits=np.unique(np.vstack((retained,new_hits)),axis=0)
        xyz=p[f];n=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        bad_face=(np.linalg.norm(n,axis=1)<.05*area)|(np.einsum('ij,ij->i',n,normals)<=0)
        folds,angles=small_facet_angles(p,f,minimum_facet_angle)
        if len(folds):bad_face[np.unique(folds)]=True
        bad_node=np.any((p<lower-1e-12)|(p>upper+1e-12),axis=1)
        if len(hits):bad_face[np.unique(hits)]=True
        bad_node[np.unique(f[bad_face])]=True
        history.append(dict(pass_number=step,intersections=len(hits),small_facet_angles=len(folds),affected_nodes=int(bad_node.sum())))
        if not np.any(bad_node):break
        if step==max_passes:raise RuntimeError(f'Geometry guard exhausted passes: {history[-1]}')
        # One ring spreads the transition and avoids moving only a sharp tip.
        affected=bad_node.copy()
        affected[np.unique(edges[np.any(bad_node[edges],axis=1)])]=True
        weight[affected]*=relaxation
        if step>=max_passes-2:weight[affected]=0.
        previous=p
        p=original+weight[:,None]*(target-original)
        p[surface.node_kind==3]=target[surface.node_kind==3]
        p[surface.fixed_axes]=target[surface.fixed_axes]
        query_ids=np.flatnonzero(np.any(np.any(p!=previous,axis=1)[f],axis=1))
        movement=np.linalg.norm(p-target,axis=1)
        if max_correction is not None and movement.max()>max_correction+1e-12:
            raise RuntimeError('Geometry rollback exceeds max_correction; increase the bound or reduce smoothing')
    # Independent full scan confirms the incremental neighbourhood checks.
    final_hits=find_surface_intersections(p,f)
    if len(final_hits):raise RuntimeError('Final full intersection check failed after rollback')
    movement=np.linalg.norm(p-target,axis=1)
    return replace(surface,points=p),dict(enabled=True,verified=True,remaining_intersections=0,
        moved_nodes=int(np.count_nonzero(movement>1e-12)),maximum_correction=float(movement.max()),
        boundary_clearance=float(boundary_clearance),history=history,
        minimum_facet_angle=float(minimum_facet_angle),remaining_small_facet_angles=0,
        frozen_junctions_preserved=bool(np.array_equal(p[surface.node_kind==3],target[surface.node_kind==3])),
        rve_coordinates_preserved=bool(np.array_equal(p[surface.fixed_axes],target[surface.fixed_axes])))
