import numpy as np

def findJP(segments):
    """
    Import
    ------
    import upxo.jpops as jpOps
    Use as: jpOps.findJP
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
    Import
    ------
    import upxo.jpops as jpOps
    Use as: jpOps.separate_junctions_by_order
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