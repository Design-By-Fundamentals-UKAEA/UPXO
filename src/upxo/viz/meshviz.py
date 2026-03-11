import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np

def see_femesh(points_2d, lines, triangles, quads, figsize=(8, 8), dpi=150,
               tri_edgecolor='blue', tri_facecolor='none', tri_linewidth=1.0, tri_alpha=1.0,
               quad_edgecolor='green', quad_facecolor='none', quad_linewidth=1.0, quad_alpha=1.0,
               line_color='black', line_linewidth=2.0, line_alpha=1.0,
               show_nodes=False, node_color='red', node_size=3, node_marker='o',
               show_node_numbers=False, node_number_fontsize=6, node_number_color='darkred',
               show_elem_numbers=False, elem_number_fontsize=7, elem_number_color='darkblue',
               colorby_quality=None, quality_data=None, cmap='viridis', 
               clim=None, show_colorbar=False, colorbar_label='Quality',
               title='FE Mesh Visualization', show_title=True,
               show_grid=False, grid_alpha=0.3, grid_linestyle='--',
               show_axis=False, show_legend=True,
               show_stats=True, stats_loc='upper right', stats_fontsize=8):
    """
    Enhanced finite element mesh visualization with extensive customization options.
    
    Parameters
    ----------
    points_2d : ndarray, shape (N, 2)
        Node coordinates
    lines : ndarray, shape (M, 2), or None
        Line element connectivity. If None, no lines are plotted.
    triangles : ndarray, shape (P, 3), or None
        Triangle element connectivity. If None, no triangles are plotted.
    quads : ndarray, shape (Q, 4), or None
        Quad element connectivity. If None, no quads are plotted.
    figsize : tuple, optional
        Figure size. Default (8, 8)
    dpi : int, optional
        Figure DPI. Default 150
    tri_edgecolor : str, optional
        Triangle edge color. Default 'blue'
    tri_facecolor : str, optional
        Triangle face color. Default 'none'
    tri_linewidth : float, optional
        Triangle edge width. Default 1.0
    tri_alpha : float, optional
        Triangle transparency. Default 1.0
    quad_edgecolor : str, optional
        Quad edge color. Default 'green'
    quad_facecolor : str, optional
        Quad face color. Default 'none'
    quad_linewidth : float, optional
        Quad edge width. Default 1.0
    quad_alpha : float, optional
        Quad transparency. Default 1.0
    line_color : str, optional
        Feature line color. Default 'black'
    line_linewidth : float, optional
        Feature line width. Default 2.0
    line_alpha : float, optional
        Feature line transparency. Default 1.0
    show_nodes : bool, optional
        Show node markers. Default False
    node_color : str, optional
        Node marker color. Default 'red'
    node_size : float, optional
        Node marker size. Default 3
    node_marker : str, optional
        Node marker style. Default 'o'
    show_node_numbers : bool, optional
        Annotate node numbers. Default False
    node_number_fontsize : int, optional
        Node number font size. Default 6
    node_number_color : str, optional
        Node number color. Default 'darkred'
    show_elem_numbers : bool, optional
        Annotate element numbers at centroids. Default False
    elem_number_fontsize : int, optional
        Element number font size. Default 7
    elem_number_color : str, optional
        Element number color. Default 'darkblue'
    colorby_quality : str or None, optional
        Element quality metric to color by ('tri', 'quad', 'all'). Default None
    quality_data : ndarray or None, optional
        Element quality values (one per element). Default None
    cmap : str, optional
        Colormap for quality visualization. Default 'viridis'
    clim : tuple or None, optional
        Color limits (min, max) for quality colormap. Default None (auto)
    show_colorbar : bool, optional
        Show colorbar for quality. Default False
    colorbar_label : str, optional
        Colorbar label. Default 'Quality'
    title : str, optional
        Plot title. Default 'FE Mesh Visualization'
    show_title : bool, optional
        Display title. Default True
    show_grid : bool, optional
        Show background grid. Default False
    grid_alpha : float, optional
        Grid transparency. Default 0.3
    grid_linestyle : str, optional
        Grid line style. Default '--'
    show_axis : bool, optional
        Show axis. Default False
    show_legend : bool, optional
        Show legend. Default True
    show_stats : bool, optional
        Show mesh statistics. Default True
    stats_loc : str, optional
        Statistics text location. Default 'upper right'
    stats_fontsize : int, optional
        Statistics font size. Default 8
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Import
    ------
    from upxo.viz.meshviz import see_femesh
    
    Examples
    --------
    >>> # Basic usage
    >>> fig, ax = see_femesh(points_2d, lines, triangles, quads)
    
    >>> # With nodes and element numbers
    >>> fig, ax = see_femesh(points_2d, lines, triangles, quads,
    ...                      show_nodes=True, show_elem_numbers=True)
    
    >>> # Color by quality
    >>> fig, ax = see_femesh(points_2d, lines, triangles, quads,
    ...                      colorby_quality='all', quality_data=quality_values,
    ...                      show_colorbar=True, cmap='coolwarm')
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Handle None inputs
    if triangles is None:
        triangles = np.array([], dtype=int).reshape(0, 3)
    if quads is None:
        quads = np.array([], dtype=int).reshape(0, 4)
    if lines is None:
        lines = np.array([], dtype=int).reshape(0, 2)
    
    # Prepare for quality coloring
    tri_facecolors = None
    quad_facecolors = None
    
    if colorby_quality is not None and quality_data is not None:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
        
        if clim is None:
            vmin, vmax = np.min(quality_data), np.max(quality_data)
        else:
            vmin, vmax = clim
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        if colorby_quality in ['tri', 'all'] and len(triangles) > 0:
            tri_qual = quality_data[:len(triangles)]
            tri_facecolors = cmap_obj(norm(tri_qual))
            tri_facecolor = tri_facecolors
        
        if colorby_quality in ['quad', 'all'] and len(quads) > 0:
            quad_start = len(triangles) if colorby_quality == 'all' else 0
            quad_qual = quality_data[quad_start:quad_start+len(quads)]
            quad_facecolors = cmap_obj(norm(quad_qual))
            quad_facecolor = quad_facecolors
    
    # Plot triangle elements
    if len(triangles) > 0:
        tri_verts = points_2d[triangles]
        tri_col = PolyCollection(tri_verts, edgecolors=tri_edgecolor, 
                                facecolors=tri_facecolor if tri_facecolors is None else tri_facecolors,
                                linewidths=tri_linewidth, alpha=tri_alpha,
                                label=f'Triangles ({len(triangles)})')
        ax.add_collection(tri_col)
        
        # Element numbers for triangles
        if show_elem_numbers:
            for elem_idx, tri_nodes in enumerate(triangles):
                centroid = points_2d[tri_nodes].mean(axis=0)
                ax.text(centroid[0], centroid[1], str(elem_idx),
                       fontsize=elem_number_fontsize, color=elem_number_color,
                       ha='center', va='center')
    
    # Plot quadrilateral elements
    if len(quads) > 0:
        quad_verts = points_2d[quads]
        quad_col = PolyCollection(quad_verts, edgecolors=quad_edgecolor,
                                 facecolors=quad_facecolor if quad_facecolors is None else quad_facecolors,
                                 linewidths=quad_linewidth, alpha=quad_alpha,
                                 label=f'Quads ({len(quads)})')
        ax.add_collection(quad_col)
        
        # Element numbers for quads
        if show_elem_numbers:
            for elem_idx, quad_nodes in enumerate(quads):
                centroid = points_2d[quad_nodes].mean(axis=0)
                ax.text(centroid[0], centroid[1], str(len(triangles) + elem_idx),
                       fontsize=elem_number_fontsize, color=elem_number_color,
                       ha='center', va='center')
    
    # Plot feature boundary lines
    if len(lines) > 0:
        line_verts = points_2d[lines]
        line_col = LineCollection(line_verts, colors=line_color,
                                  linewidths=line_linewidth, alpha=line_alpha,
                                  label=f'Boundaries ({len(lines)})')
        ax.add_collection(line_col)
    
    # Plot nodes
    if show_nodes:
        ax.scatter(points_2d[:, 0], points_2d[:, 1],
                  c=node_color, s=node_size, marker=node_marker,
                  zorder=10, label=f'Nodes ({len(points_2d)})')
    
    # Node numbers
    if show_node_numbers:
        for node_idx, (x, y) in enumerate(points_2d):
            ax.text(x, y, str(node_idx),
                   fontsize=node_number_fontsize, color=node_number_color,
                   ha='right', va='bottom')
    
    # Colorbar for quality
    if show_colorbar and colorby_quality is not None and quality_data is not None:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=colorbar_label)
    
    # Statistics text
    if show_stats:
        stats_text = f"Mesh Statistics:\n"
        stats_text += f"Nodes: {len(points_2d)}\n"
        stats_text += f"Triangles: {len(triangles)}\n"
        stats_text += f"Quads: {len(quads)}\n"
        stats_text += f"Boundary lines: {len(lines)}\n"
        stats_text += f"Total elements: {len(triangles) + len(quads)}"
        
        # Position mapping
        loc_map = {
            'upper right': (0.98, 0.98),
            'upper left': (0.02, 0.98),
            'lower right': (0.98, 0.02),
            'lower left': (0.02, 0.02)
        }
        x, y = loc_map.get(stats_loc, (0.98, 0.98))
        ha = 'right' if 'right' in stats_loc else 'left'
        va = 'top' if 'upper' in stats_loc else 'bottom'
        
        ax.text(x, y, stats_text, transform=ax.transAxes,
               fontsize=stats_fontsize, verticalalignment=va,
               horizontalalignment=ha, bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))
    
    # Final setup
    ax.autoscale()
    ax.set_aspect('equal')
    
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
    
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    if show_title:
        ax.set_title(title)
    
    if show_legend and (len(triangles) > 0 or len(quads) > 0 or len(lines) > 0 or show_nodes):
        ax.legend(loc='best', fontsize=8)
    
    return fig, ax


def plot_elements_by_ids_geomMesh(element_ids, eltype, elConn, nodes, GBlines,
                         fig=None, ax=None,
                         gblines_by_grain=None, local_view=True,
                         figsize=(6, 6), dpi=100,
                         tri_edgecolor='tab:green', quad_edgecolor='tab:blue',
                         tri_facecolor='none', quad_facecolor='none',
                         elem_linewidth=0.7, gb_linewidth=2.0,
                         quality_by_eltype=None, cmap='cubehelix_r',
                         vmin=None, vmax=None, show_colorbar=True,
                         title='Selected elements'):
    """
    Import
    ------
    from upxo.viz import meshviz
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sm = None
    if quality_by_eltype is not None:
        all_quality = []
        for etype, qvals in quality_by_eltype.items():
            if qvals is not None and len(qvals) > 0:
                all_quality.extend(np.asarray(qvals).ravel().tolist())
        if all_quality:
            vmin = np.nanmin(all_quality) if vmin is None else vmin
            vmax = np.nanmax(all_quality) if vmax is None else vmax
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])

    if eltype is None:
        eltypes = ['triangle', 'quad']
    elif isinstance(eltype, (list, tuple, set)):
        eltypes = list(eltype)
    else:
        eltypes = [eltype]

    element_ids = np.asarray(element_ids, dtype=int)
    element_ids = element_ids[element_ids >= 0]
    if element_ids.size == 0:
        return fig, ax

    grain_nodes = np.array([], dtype=int)

    for et in eltypes:
        if et not in elConn:
            continue
        conn = elConn[et]
        qvals = None
        if quality_by_eltype is not None and et in quality_by_eltype:
            qvals = np.asarray(quality_by_eltype[et])

        for elem_idx in element_ids:
            if elem_idx >= len(conn):
                continue
            poly = nodes[conn[int(elem_idx)]][:, :2]
            if qvals is not None and sm is not None:
                qval = qvals[int(elem_idx)] if int(elem_idx) < len(qvals) else np.nan
                facecolor = sm.to_rgba(qval) if np.isfinite(qval) else 'none'
                edgecolor = 'k'
            else:
                facecolor = tri_facecolor if et == 'triangle' else quad_facecolor
                edgecolor = tri_edgecolor if et == 'triangle' else quad_edgecolor
            ax.fill(poly[:, 0], poly[:, 1], facecolor=facecolor,
                    edgecolor=edgecolor, linewidth=elem_linewidth)
            grain_nodes = np.unique(np.concatenate((grain_nodes, conn[[int(elem_idx)]].ravel())))

    if local_view:
        mask = np.isin(GBlines, grain_nodes).all(axis=1)
        gblines = GBlines[mask]
        for i, j in gblines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=gb_linewidth, alpha=0.6)
    else:
        for i, j in GBlines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=0.4, alpha=0.4)

    if sm is not None and show_colorbar:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Element Quality')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def plot_elements_geometric_grain(grain_name, elsets_eltype=None,
                        elConn=None, nodes=None, GBlines=None, fig=None, ax=None,
                        gblines_by_grain=None, local_view=True, figsize=(6, 6), dpi=100,
                        tri_edgecolor='tab:green', quad_edgecolor='tab:blue',
                        elem_linewidth=0.7, gb_linewidth=2.0, plot_triangles=True,
                        plot_quads=True, quality_by_eltype=None, cmap='cubehelix_r',
                        vmin=None, vmax=None, show_colorbar=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sm = None
    if quality_by_eltype is not None:
        all_quality = []
        for eltype, qvals in quality_by_eltype.items():
            if qvals is not None and len(qvals) > 0:
                all_quality.extend(np.asarray(qvals).ravel().tolist())
        if all_quality:
            vmin = np.nanmin(all_quality) if vmin is None else vmin
            vmax = np.nanmax(all_quality) if vmax is None else vmax
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])

    grain_nodes = np.array([], dtype=int)
    for eltype, conn in elConn.items():
        if eltype == 'triangle' and not plot_triangles:
            continue
        if eltype == 'quad' and not plot_quads:
            continue
        if eltype not in elsets_eltype or grain_name not in elsets_eltype[eltype]:
            continue
        elem_ids = elsets_eltype[eltype][grain_name]
        if elem_ids.size == 0:
            continue
        qvals = None
        if quality_by_eltype is not None and eltype in quality_by_eltype:
            qvals = np.asarray(quality_by_eltype[eltype])
        for elem_idx in elem_ids:
            poly = nodes[conn[int(elem_idx)]][:, :2]
            if qvals is not None and sm is not None:
                qval = qvals[int(elem_idx)] if int(elem_idx) < len(qvals) else np.nan
                facecolor = sm.to_rgba(qval) if np.isfinite(qval) else 'none'
                edgecolor = 'k'
            else:
                facecolor = 'none'
                edgecolor = tri_edgecolor if eltype == 'triangle' else quad_edgecolor
            ax.fill(poly[:, 0], poly[:, 1], facecolor=facecolor,
                    edgecolor=edgecolor, linewidth=elem_linewidth)
        grain_nodes = np.unique(np.concatenate((grain_nodes, conn[elem_ids].ravel())))

    if local_view:
        if gblines_by_grain and grain_name in gblines_by_grain:
            gblines = gblines_by_grain[grain_name]
        else:
            mask = np.isin(GBlines, grain_nodes).all(axis=1)
            gblines = GBlines[mask]
        for i, j in gblines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=gb_linewidth, alpha=0.6)
    else:
        for i, j in GBlines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=0.4, alpha=0.4)

    if sm is not None and show_colorbar:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Element Quality')

    ax.set_aspect('equal')
    ax.set_title(f'Elements in {grain_name}')
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax


def plot_elements_by_elIDs(element_ids, eltype=None, elConn=None,
                         nodes=None, GBlines=None, fig=None, ax=None,
                         gblines_by_grain=None, local_view=True,
                         figsize=(6, 6), dpi=100,
                         tri_edgecolor='tab:green', quad_edgecolor='tab:blue',
                         elem_linewidth=0.7, gb_linewidth=2.0,
                         quality_by_eltype=None, cmap='cubehelix_r',
                         vmin=None, vmax=None, show_colorbar=True,
                         title='Selected elements'):
    """
    Import
    ------
    from upxo.viz import meshviz

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sm = None
    if quality_by_eltype is not None:
        all_quality = []
        for etype, qvals in quality_by_eltype.items():
            if qvals is not None and len(qvals) > 0:
                all_quality.extend(np.asarray(qvals).ravel().tolist())
        if all_quality:
            vmin = np.nanmin(all_quality) if vmin is None else vmin
            vmax = np.nanmax(all_quality) if vmax is None else vmax
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])

    if eltype is None:
        eltypes = ['triangle', 'quad']
    elif isinstance(eltype, (list, tuple, set)):
        eltypes = list(eltype)
    else:
        eltypes = [eltype]

    element_ids = np.asarray(element_ids, dtype=int)
    element_ids = element_ids[element_ids >= 0]
    if element_ids.size == 0:
        return fig, ax

    grain_nodes = np.array([], dtype=int)

    for et in eltypes:
        if et not in elConn:
            continue
        conn = elConn[et]
        qvals = None
        if quality_by_eltype is not None and et in quality_by_eltype:
            qvals = np.asarray(quality_by_eltype[et])

        for elem_idx in element_ids:
            if elem_idx >= len(conn):
                continue
            poly = nodes[conn[int(elem_idx)]][:, :2]
            if qvals is not None and sm is not None:
                qval = qvals[int(elem_idx)] if int(elem_idx) < len(qvals) else np.nan
                facecolor = sm.to_rgba(qval) if np.isfinite(qval) else 'none'
                edgecolor = 'k'
            else:
                facecolor = 'none'
                edgecolor = tri_edgecolor if et == 'triangle' else quad_edgecolor
            ax.fill(poly[:, 0], poly[:, 1], facecolor=facecolor,
                    edgecolor=edgecolor, linewidth=elem_linewidth)
            grain_nodes = np.unique(np.concatenate((grain_nodes, conn[[int(elem_idx)]].ravel())))

    if local_view:
        mask = np.isin(GBlines, grain_nodes).all(axis=1)
        gblines = GBlines[mask]
        for i, j in gblines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=gb_linewidth, alpha=0.6)
    else:
        for i, j in GBlines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
       color='k', linewidth=0.4, alpha=0.4)

    if sm is not None and show_colorbar:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Element Quality')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def plot_elements_geometric_grains(grain_names=None, grain_ids=None, prefix='grain.',
                              elsets_eltype=None, elConn=None, nodes=None, GBlines=None,
                              fig=None, ax=None,
                              gblines_by_grain=None, local_view=True,
                              figsize=(6, 6), dpi=100,
                              tri_edgecolor='tab:green', quad_edgecolor='tab:blue',
                              elem_linewidth=0.7, gb_linewidth=2.0,
                              plot_triangles=True, plot_quads=True,
                              quality_by_eltype=None, cmap='cubehelix_r',
                              vmin=None, vmax=None, show_colorbar=True):
    if grain_names is None and grain_ids is None:
        raise ValueError("Provide grain_names or grain_ids.")
        
    if grain_names is None:
        if type(grain_ids) in [int, np.int16, np.int32, np.int64]: grain_ids = [grain_ids]
        grain_names = [f"{prefix}{gid}" for gid in grain_ids]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sm = None
    if quality_by_eltype is not None:
        all_quality = []
        for eltype, qvals in quality_by_eltype.items():
            if qvals is not None and len(qvals) > 0:
                all_quality.extend(np.asarray(qvals).ravel().tolist())
        if all_quality:
            vmin = np.nanmin(all_quality) if vmin is None else vmin
            vmax = np.nanmax(all_quality) if vmax is None else vmax
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])

    for grain_name in grain_names:
        grain_nodes = np.array([], dtype=int)
        for eltype, conn in elConn.items():
            if eltype == 'triangle' and not plot_triangles:
                continue
            if eltype == 'quad' and not plot_quads:
                continue
            if eltype not in elsets_eltype or grain_name not in elsets_eltype[eltype]:
                continue
            elem_ids = elsets_eltype[eltype][grain_name]
            if elem_ids.size == 0:
                continue
            qvals = None
            if quality_by_eltype is not None and eltype in quality_by_eltype:
                qvals = np.asarray(quality_by_eltype[eltype])
            for elem_idx in elem_ids:
                poly = nodes[conn[int(elem_idx)]][:, :2]
                if qvals is not None and sm is not None:
                    qval = qvals[int(elem_idx)] if int(elem_idx) < len(qvals) else np.nan
                    facecolor = sm.to_rgba(qval) if np.isfinite(qval) else 'none'
                    edgecolor = 'k'
                else:
                    facecolor = 'none'
                    edgecolor = tri_edgecolor if eltype == 'triangle' else quad_edgecolor
                ax.fill(poly[:, 0], poly[:, 1], facecolor=facecolor,
                        edgecolor=edgecolor, linewidth=elem_linewidth)
            grain_nodes = np.unique(np.concatenate((grain_nodes, conn[elem_ids].ravel())))

        if local_view:
            if gblines_by_grain and grain_name in gblines_by_grain:
                gblines = gblines_by_grain[grain_name]
            else:
                mask = np.isin(GBlines, grain_nodes).all(axis=1)
                gblines = GBlines[mask]
            for i, j in gblines:
                ax.plot([nodes[i, 0], nodes[j, 0]],
                        [nodes[i, 1], nodes[j, 1]],
                        color='k', linewidth=gb_linewidth, alpha=0.6)

    if not local_view:
        for i, j in GBlines:
            ax.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    color='k', linewidth=0.4, alpha=0.4)

    if sm is not None and show_colorbar:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Element Quality')

    ax.set_aspect('equal')
    ax.set_title('Elements in grains')
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax


def pick_contrasting_colours_from_cmap(n_colours, cmap_name='nipy_spectral'):
    cmap = plt.cm.get_cmap(cmap_name)
    if n_colours <= 1:
        return [cmap(0.55)]

    candidates_u = np.linspace(0.06, 0.94, 256)
    candidates_rgb = np.array([cmap(u)[:3] for u in candidates_u])

    selected_idx = [np.argmax(np.linalg.norm(candidates_rgb - np.array([0.5, 0.5, 0.5]), axis=1))]
    while len(selected_idx) < n_colours:
        selected_rgb = candidates_rgb[selected_idx]
        dist_to_selected = np.linalg.norm(
            candidates_rgb[:, None, :] - selected_rgb[None, :, :], axis=2
        )
        min_dist = dist_to_selected.min(axis=1)
        min_dist[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(min_dist)))

    return [cmap(candidates_u[i]) for i in selected_idx]


def resolve_band_colours(bands, band_colours=None, auto_cmap='nipy_spectral'):
    if band_colours is not None and len(band_colours) < len(bands):
        raise ValueError('Length of band_colours must be >= number of bands.')
    if band_colours is not None:
        return band_colours
    return pick_contrasting_colours_from_cmap(len(bands), cmap_name=auto_cmap)


def resolve_plot_eltype(eltypes, plot_eltype=None):
    if len(eltypes) == 0:
        return None
    if plot_eltype is None:
        return 'quad' if 'quad' in eltypes else eltypes[0]
    if plot_eltype not in eltypes:
        raise ValueError(f"plot_eltype '{plot_eltype}' is not in selected eltypes: {eltypes}")
    return plot_eltype


def plot_band_elements(element_ids_by_band, bands, plot_eltype, gbcoords,
                       gsConfMesh, gblines_by_grain=None, colours_to_use=None,
                       title='Selected elements by band', band_facecolors=False):
    fig = ax = None
    if plot_eltype is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_title(title)
        ax.plot(gbcoords[:, 0], gbcoords[:, 1], 'ro', label='GB nodes')
        ax.legend(loc='best')
        return fig, ax

    if colours_to_use is None:
        colours_to_use = resolve_band_colours(bands)

    if isinstance(plot_eltype, (list, tuple, set)):
        plot_eltypes = list(plot_eltype)
    else:
        plot_eltypes = [plot_eltype]

    for idx, band in enumerate(bands):
        band_min = min(band)
        band_max = max(band)
        band_key = (band_min, band_max)
        band_color = colours_to_use[idx]
        band_facecolor = band_color if band_facecolors else 'none'

        plotted_any = False
        for etype in plot_eltypes:
            band_ids = element_ids_by_band[band_key].get(etype, np.array([], dtype=int))
            if band_ids.size == 0:
                continue

            fig, ax = plot_elements_by_ids_geomMesh(
                band_ids, etype, gsConfMesh.elConn, gsConfMesh.nodes, gsConfMesh.GBlines,
                fig=fig, ax=ax, gblines_by_grain=gblines_by_grain, local_view=False,
                figsize=(6, 6), dpi=100,
                tri_edgecolor=band_color if etype == 'triangle' else 'tab:green',
                quad_edgecolor=band_color if etype == 'quad' else 'tab:blue',
                tri_facecolor=band_facecolor if etype == 'triangle' else 'none',
                quad_facecolor=band_facecolor if etype == 'quad' else 'none',
                elem_linewidth=0.7, gb_linewidth=2.0, quality_by_eltype=None, cmap='cubehelix_r',
                vmin=None, vmax=None, show_colorbar=False, title=title
            )
            plotted_any = True

        if plotted_any:
            if len(plot_eltypes) == 1:
                ax.plot([], [], color=band_color, linewidth=2, label=f'Band [{band_min}, {band_max}]')
            else:
                ax.plot([], [], color=band_color, linewidth=2,
                        label=f'Band [{band_min}, {band_max}] ({", ".join(plot_eltypes)})')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_title(title)
    ax.plot(gbcoords[:, 0], gbcoords[:, 1], 'ro', label='GB nodes')
    ax.legend(loc='best')
    return fig, ax

def see_gbElements_grains(gb_elements_grain, eltypes=None, elConn=None,
                         nodes=None, GBlines=None, fig=None, ax=None,
                         gblines_by_grain=None, local_view=True,
                         figsize=(6, 6), dpi=100,
                         tri_edgecolor='tab:green', quad_edgecolor='tab:blue',
                         elem_linewidth=0.7, gb_linewidth=2.0,
                         quality_by_eltype=None, cmap='cubehelix_r',
                         vmin=None, vmax=None, show_colorbar=True,
                         title='Selected elements'):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    for idx, eltype in enumerate(eltypes):
        elem_ids = gb_elements_grain.get(eltype, {}).get('elem_ids', np.array([], dtype=int))
        fig, ax = plot_elements_by_elIDs(element_ids=elem_ids, eltype=eltype, elConn=elConn,
            nodes=nodes, GBlines=GBlines, fig=fig, ax=ax,  tri_edgecolor=tri_edgecolor, 
            quad_edgecolor=quad_edgecolor, elem_linewidth=elem_linewidth, gb_linewidth=gb_linewidth,
            vmin=vmin, vmax=vmax, gblines_by_grain=gblines_by_grain, local_view=local_view, 
            figsize=figsize, dpi=dpi, quality_by_eltype=quality_by_eltype, cmap=cmap, 
            show_colorbar=(idx == len(eltypes)-1), title=title)
    plt.tight_layout()
    return fig, ax