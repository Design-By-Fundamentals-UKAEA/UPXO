"""
Junction-point detection on grain-boundary segment maps.

``findJP`` collects coordinates shared by two or more boundary segments
per grain; ``separate_junctions_by_order`` bins them by junction order
(T-junction, quadruple point, …). Used with 2D MC GB pipelines
(:mod:`upxo.gbops.mcgb2dops`).

Import::

    import upxo.jpops.jpops as jpOps
"""

import numpy as np

def findJP(segments):
    """
    Find multi-segment junction points for each central grain.

    Parameters
    ----------
    segments : dict
        ``cg → {neighbour_gid: Nx2 coordinate array}`` of shared boundary
        points between central grain ``cg`` and each neighbour.

    Returns
    -------
    dict
        ``cg → array of [x, y, order]`` where order is how many segments
        meet at that point (plus bookkeeping offset used historically).
    """
    junctions = {}
    for cg, neighbors in segments.items():
        all_pts = [pts for pts in neighbors.values() if pts.size > 0]
        if not all_pts:
            continue
        stacked_pts = np.vstack(all_pts)
        unique_pts, counts = np.unique(stacked_pts, axis=0, return_counts=True)
        junction_mask = counts >= 2
        if np.any(junction_mask):
            res = np.column_stack((unique_pts[junction_mask], counts[junction_mask]+1))
            junctions[cg] = res
    return junctions

def separate_junctions_by_order(junctions, include_empty=True):
    """
    Partition junction points by order (2, 3, 4, …).

    Parameters
    ----------
    junctions : dict
        Output of :func:`findJP` (or same shape).
    include_empty : bool, optional
        If True, include empty order keys between min and max order.

    Returns
    -------
    dict
        ``order → {gid: Mx2 coordinates for junctions of that order}``.
    """

    if not junctions:
        return {}
    valid_orders = np.unique(np.concatenate([jp[:, 2]
                    for jp in junctions.values() if jp.size > 0])).astype(int)
    if valid_orders.size == 0:
        return {}
    orders = (range(valid_orders.min(), valid_orders.max()+1) if include_empty else valid_orders)
    jps_by_order = {int(order): {int(gid): jp[jp[:, 2] == order][:, :2]
                         for gid, jp in junctions.items() 
                         if jp.size > 0 and np.any(jp[:, 2] == order) } 
                         for order in orders}
    return jps_by_order