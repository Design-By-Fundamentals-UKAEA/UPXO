"""Partition difficult discrete surfaces into disk-shaped parametrization charts."""
import numpy as np


def is_disk_chart(triangles):
    """Require a manifold disk with one non-branching boundary loop."""
    edges,counts=np.unique(np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
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


def disk_charts(triangles, max_triangles=64):
    """Cover triangles once with disks, cutting only along existing mesh edges.

    Growth attaches a triangle along one boundary edge with a new vertex, or
    fills an ear along two boundary edges. Both operations preserve disk
    topology. No coordinates, triangles or grain labels are changed.
    """
    triangles=np.asarray(triangles)
    edges=[{tuple(sorted((int(a),int(b)))) for a,b in zip(t,t[[1,2,0]])} for t in triangles]
    owners={}
    for i,e in enumerate(edges):
        for edge in e:owners.setdefault(edge,[]).append(i)
    remaining=set(range(len(triangles)));charts=[]
    while remaining:
        seed=min(remaining);remaining.remove(seed)
        chart=[seed];boundary=set(edges[seed]);used=set(boundary);vertices=set(triangles[seed])
        pending=set(j for e in edges[seed] for j in owners[e])&remaining
        while pending and len(chart)<max_triangles:
            i=min(pending);pending.remove(i)
            shared=edges[i]&boundary
            if edges[i]&(used-boundary):continue
            if len(shared)==1 and len(set(triangles[i])-vertices)!=1:continue
            if len(shared) not in (1,2):continue
            chart.append(i);remaining.remove(i)
            boundary.symmetric_difference_update(edges[i]);used.update(edges[i]);vertices.update(triangles[i])
            pending.update(j for e in edges[i] for j in owners[e] if j in remaining)
        charts.append(np.asarray(chart,dtype=int))
    return charts
