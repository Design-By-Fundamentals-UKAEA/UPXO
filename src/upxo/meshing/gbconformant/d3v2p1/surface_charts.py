"""d3v2p0.surface_charts.is_disk_chart with the edge table from integer keys.

The answer is d3v2p0's for every chart; disk_charts is d3v2p0's.
"""
import numpy as np
from .mesh_tables import unique_rows, triangle_edges
from ..d3v2p0.surface_charts import disk_charts  # noqa: F401  (unchanged, re-exported)


def is_disk_chart(triangles):
    """Require a manifold disk with one non-branching boundary loop."""
    edges,counts=unique_rows(triangle_edges(triangles),False,True)
    if np.any(counts>2) or len(np.unique(triangles))-len(edges)+len(triangles)!=1:return False
    boundary=edges[counts==1]
    if not len(boundary):return False
    nodes,degree=np.unique(boundary,return_counts=True)
    if np.any(degree!=2):return False
    neighbors={int(v):[] for v in nodes}
    for a,b in boundary:neighbors[int(a)].append(int(b));neighbors[int(b)].append(int(a))
    visited=set();stack=[int(nodes[0])]
    while stack:
        v=stack.pop()
        if v not in visited:visited.add(v);stack.extend(neighbors[v])
    return len(visited)==len(nodes)
