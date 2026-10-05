"""d3v2p0.discrete_topology with the .msh text written by a few bulk
formatting calls; the file content is byte-identical to d3v2p0's."""
import io
from collections import defaultdict
import numpy as np



def _lines(values, fmt):
    """np.savetxt(values, fmt) as a list of lines (newline included)."""
    if not len(values):
        return []
    buffer = io.StringIO()
    np.savetxt(buffer, values, fmt=fmt)
    return buffer.getvalue().splitlines(keepends=True)


def add_chart_topology(gmsh, points, triangles, groups):
    """Import charts with explicit shared curves, retaining source element tags.

    A curve joins consecutive edges having the same incident chart set. Curves
    end at branches; closed chains are split in two to give ordinary endpoints.
    This avoids Gmsh's expensive rediscovery of thousands of small chart seams.
    """
    owners = defaultdict(list)
    for chart, ids in enumerate(groups):
        edges, counts = np.unique(np.sort(triangles[ids][:, [[0,1],[1,2],[2,0]]]
                                         .reshape(-1,2),axis=1),axis=0,return_counts=True)
        if np.any(counts>2):
            raise ValueError('Nonmanifold edge inside a discrete chart')
        for a,b in edges[counts==1]:
            owners[(int(a),int(b))].append(chart)
    by_incidence = defaultdict(list)
    node_incidence = defaultdict(set)
    for edge, charts in owners.items():
        by_incidence[tuple(charts)].append(edge)
        for v in edge:node_incidence[v].add(tuple(charts))
    curves=[]
    for charts, edges in by_incidence.items():
        neighbors=defaultdict(list)
        for a,b in edges:
            neighbors[a].append(b);neighbors[b].append(a)
        remaining=set(edges)
        def walk(a,b):
            chain=[a,b];remaining.remove(tuple(sorted((a,b))))
            while len(neighbors[b])==2 and len(node_incidence[b])==1 and b!=chain[0]:
                c=neighbors[b][0] if neighbors[b][0]!=a else neighbors[b][1]
                edge=tuple(sorted((b,c)))
                if edge not in remaining:break
                remaining.remove(edge);chain.append(c);a,b=b,c
            return chain
        for a in sorted(neighbors):
            if len(neighbors[a])!=2 or len(node_incidence[a])>1:
                for b in neighbors[a]:
                    if tuple(sorted((a,b))) in remaining:
                        curves.append((charts,walk(a,b)))
        while remaining:
            a,b=min(remaining);chain=walk(a,b)
            if chain[0]==chain[-1]:
                middle=(len(chain)-1)//2
                curves.extend([(charts,chain[:middle+1]),(charts,chain[middle:])])
            else:curves.append((charts,chain))
    endpoints=sorted({v for _,chain in curves for v in (chain[0],chain[-1])})
    for v in endpoints:
        gmsh.model.addDiscreteEntity(0,v+1)
    boundaries=[[] for _ in groups]
    for tag,(charts,chain) in enumerate(curves,1):
        gmsh.model.addDiscreteEntity(1,tag,[chain[0]+1,chain[-1]+1])
        for chart in charts:boundaries[chart].append(tag)
    for chart in range(len(groups)):
        gmsh.model.addDiscreteEntity(2,chart+1,boundaries[chart])
    # Classify each node at import instead of repeatedly rebuilding Gmsh's
    # global node cache while inserting individual point elements.
    assigned=np.zeros(len(points),dtype=bool);node_blocks=[]
    for v in endpoints:
        node_blocks.append((0,v+1,np.array([v],dtype=int)))
        assigned[v]=True
    for tag,(_,chain) in enumerate(curves,1):
        nodes=np.asarray(chain[1:-1],dtype=int)
        if len(nodes):
            if np.any(assigned[nodes]):raise RuntimeError('Curve interior has multiple owners')
            node_blocks.append((1,tag,nodes))
            assigned[nodes]=True
    for chart,ids in enumerate(groups):
        nodes=np.unique(triangles[ids]);nodes=nodes[~assigned[nodes]]
        if len(nodes):
            node_blocks.append((2,chart+1,nodes))
            assigned[nodes]=True
    # Bulk import avoids rebuilding native mesh caches once per entity. All
    # geometry entities and their boundary incidence already exist above.
    import tempfile
    from pathlib import Path
    path=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',suffix='.msh',delete=False) as stream:
            path=Path(stream.name)
            stream.write('$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n')
            used=np.flatnonzero(assigned)
            stream.write(f'{len(node_blocks)} {len(used)} {used.min()+1} {used.max()+1}\n')
            all_nodes=np.concatenate([nodes for _,_,nodes in node_blocks])
            tag_lines=_lines(all_nodes+1,'%d')
            coord_lines=_lines(points[all_nodes],'%.17g')
            start=0
            for dim,tag,nodes in node_blocks:
                end=start+len(nodes)
                stream.write(f'{dim} {tag} 0 {len(nodes)}\n')
                stream.write(''.join(tag_lines[start:end]))
                stream.write(''.join(coord_lines[start:end]))
                start=end
            total=len(triangles)+len(endpoints)+sum(len(chain)-1 for _,chain in curves)
            stream.write(f'$EndNodes\n$Elements\n{len(groups)+len(endpoints)+len(curves)} {total} 1 {total}\n')
            all_ids=np.concatenate([np.asarray(ids) for ids in groups]) if len(groups) else np.zeros(0,int)
            tri_lines=_lines(np.column_stack((all_ids+1,triangles[all_ids]+1)),'%d')
            start=0
            for chart,ids in enumerate(groups):
                stream.write(f'2 {chart+1} 2 {len(ids)}\n')
                stream.write(''.join(tri_lines[start:start+len(ids)]));start+=len(ids)
            next_element=len(triangles)+1
            for v in endpoints:
                stream.write(f'0 {v+1} 15 1\n{next_element} {v+1}\n');next_element+=1
            rows=[];heads=[]
            for tag,(_,chain) in enumerate(curves,1):
                line=np.column_stack((chain[:-1],chain[1:]))+1
                heads.append(f'1 {tag} 1 {len(line)}\n')
                rows.append(np.column_stack((np.arange(next_element,next_element+len(line)),line)))
                next_element+=len(line)
            curve_lines=_lines(np.vstack(rows),'%d') if rows else []
            start=0
            for head,r in zip(heads,rows):
                stream.write(head);stream.write(''.join(curve_lines[start:start+len(r)]));start+=len(r)
            stream.write('$EndElements\n')
        gmsh.merge(str(path))
    finally:
        if path is not None:path.unlink(missing_ok=True)
