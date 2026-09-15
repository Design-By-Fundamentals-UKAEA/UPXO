"""Grain adjacency from actual shared triangular interfaces, with area weights."""
import numpy as np
import pyvista as pv


def grain_adjacency(surface):
    """Return all grain nodes and undirected positive-area interface edges.

    RVE caps, point contacts and edge-only contacts do not introduce neighbours.
    Disconnected pieces sharing a grain ID are represented by one graph node.
    """
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(map(int, np.unique(surface.grain_pairs)))
    mask = ~surface.exterior
    pairs = np.sort(surface.grain_pairs[mask], axis=1)
    xyz = surface.points[surface.triangles[mask]]
    areas = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)/2
    if np.any(~np.isfinite(areas)) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError('Invalid internal-interface geometry or grain ownership')
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    totals = np.bincount(inverse, weights=areas, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    for pair, area, count in zip(unique, totals, counts):
        if area > 0:
            graph.add_edge(int(pair[0]), int(pair[1]), area=float(area), triangles=int(count))
    return graph


def adjacency_graph_view(surface, *, focus_grain=None, hops=1, full_graph=False,
                          minimum_area=0., max_nodes=150, layout='spring', seed=42,
                          cmap='tab20c', node_size=400., show_labels=True,
                          weight_edge_width=True, edge_width=1.5,
                          surface_opacity=.45, show_surface_edges=False,
                          show=True, window_size=(1600, 800)):
    """Graph and corresponding 3D grain boundaries, using a shared grain-ID cmap.

    Area filtering affects only the displayed graph. The returned full graph
    retains every grain and positive-area adjacency. Selection is not silently
    truncated: an oversized display asks for fewer hops or a higher node limit.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from .thickness_view import _poly
    if not isinstance(hops, (int, np.integer)) or hops < 0: raise ValueError('hops must be nonnegative')
    if not isinstance(max_nodes, (int, np.integer)) or max_nodes < 1: raise ValueError('max_nodes must be positive')
    if not np.isfinite(minimum_area) or minimum_area < 0: raise ValueError('minimum_area must be nonnegative')
    if layout not in ('spring','circular'): raise ValueError('layout must be spring or circular')
    graph = grain_adjacency(surface)
    filtered = nx.Graph()
    filtered.add_nodes_from(graph.nodes)
    filtered.add_edges_from((a,b,data) for a,b,data in graph.edges(data=True) if data['area'] >= minimum_area)
    if focus_grain is None:
        focus_grain = min(filtered.nodes, key=lambda g:(-filtered.degree[g],g))
    if focus_grain not in filtered: raise ValueError('Focus grain ID is absent')
    nodes = sorted(filtered.nodes if full_graph else nx.single_source_shortest_path_length(filtered, focus_grain, cutoff=hops))
    if len(nodes) > max_nodes:
        raise ValueError(f'Display has {len(nodes)} nodes, above max_nodes={max_nodes}; reduce hops or increase max_nodes.')
    displayed = filtered.subgraph(nodes).copy()
    pos = nx.spring_layout(displayed, seed=seed, weight=None, iterations=75) if layout=='spring' else nx.circular_layout(displayed)
    norm = Normalize(vmin=min(graph.nodes), vmax=max(graph.nodes) if len(graph)>1 else min(graph.nodes)+1)
    colormap = plt.get_cmap(cmap)
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    areas=np.array([d['area'] for _,_,d in displayed.edges(data=True)])
    widths = edge_width*(.5+2.5*np.sqrt(areas/areas.max())) if weight_edge_width and len(areas) else edge_width
    nx.draw_networkx_edges(displayed,pos,ax=ax,width=widths,alpha=.45,edge_color='gray')
    nx.draw_networkx_nodes(displayed,pos,ax=ax,node_size=node_size,
        node_color=[colormap(norm(g)) for g in nodes],
        edgecolors=['red' if g==focus_grain else 'black' for g in nodes],
        linewidths=[2.5 if g==focus_grain else .7 for g in nodes])
    if show_labels:
        nx.draw_networkx_labels(displayed,pos,ax=ax,font_size=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.3))
    ax.set_title(f'Grain adjacency | focus {focus_grain}\n{len(nodes)} grains, {displayed.number_of_edges()} interfaces')
    ax.axis('off');ax.margins(.15)
    ax.text(.01,.01,'Edges = shared surface interfaces\nRed outline = focus grain',transform=ax.transAxes,fontsize=9)
    plotter=pv.Plotter(shape=(1,2),notebook=False,window_size=window_size)
    plotter.subplot(0,0);plotter.add_chart(pv.ChartMPL(fig));plt.close(fig)
    plotter.subplot(0,1)
    selected=np.any(np.isin(surface.grain_pairs,nodes),axis=1)
    ids=np.flatnonzero(selected)
    mesh=_poly(surface.points,surface.triangles[ids])
    # A shared face is drawn once: focus gets priority, then a selected owner.
    pairs=surface.grain_pairs[ids]
    owner=np.where(np.any(pairs==focus_grain,axis=1),focus_grain,
                   np.where(np.isin(pairs[:,0],nodes),pairs[:,0],pairs[:,1]))
    mesh.cell_data['grain_id']=owner
    plotter.add_mesh(mesh,scalars='grain_id',cmap=cmap,clim=(norm.vmin,norm.vmax),
        opacity=surface_opacity,show_edges=show_surface_edges,show_scalar_bar=False)
    plotter.add_text(f'Boundaries of displayed grains | focus {focus_grain}',font_size=11)
    plotter.add_axes();plotter.view_isometric();plotter.reset_camera()
    report=dict(total_grains=len(graph),total_interfaces=graph.number_of_edges(),
        connected_components=nx.number_connected_components(graph),
        isolated_grain_ids=list(nx.isolates(graph)),focus_grain=int(focus_grain),
        displayed_grains=nodes,displayed_interfaces=displayed.number_of_edges(),
        minimum_display_area=float(minimum_area))
    print(report)
    print('Focus neighbours (all interfaces, before display filtering):')
    for neighbour in sorted(graph.neighbors(focus_grain)):
        print(f"  grain {neighbour}: shared area {graph[focus_grain][neighbour]['area']:.8g}")
    if show:plotter.show()
    return plotter,graph,report
