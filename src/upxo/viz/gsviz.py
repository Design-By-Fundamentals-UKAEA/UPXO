import pyvista as pv
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import vtk
from upxo._sup import dataTypeHandlers as dth
from shapely.geometry import MultiPolygon
"""
Visualization utilities for grain structure data.

Import
------
from upxo.viz import gsviz
"""

# Visualization presets for different use cases
PLOT_PRESETS = {
    'publication': {
        'figsize': (6, 4.5), 'dpi': 300, 'cmap': 'viridis',
        'colorbar': True, 'interpolation': None
    },
    'presentation': {
        'figsize': (10, 8), 'dpi': 150, 'cmap': 'plasma',
        'colorbar': True, 'interpolation': None
    },
    'quick': {
        'figsize': (6, 6), 'dpi': 100, 'cmap': 'viridis',
        'colorbar': True, 'interpolation': None
    },
    'minimal': {
        'figsize': (5, 5), 'dpi': 80, 'cmap': 'gray',
        'colorbar': False, 'interpolation': None
    }
}

def see_map(lfi, fids=None, cmap='viridis', 
            figsize=(8, 6), dpi=100, vmin=None, vmax=None,
            title='Map View', xlabel='X-axis', ylabel='Y-axis',
            cmlabel='Value', preset='minimal', ax=None, show=True, 
            cbar_ticks=None,  mbar=False, mbar_length=10, mbar_loc='bot_left',
            mbar_units='μm',
            **imshow_kwargs):
    """
    Visualize 2D map data with flexible styling options.
    
    Parameters
    ----------
    mapdata : array_like
        2D data array to visualize
    cmap : str, optional
        Colormap name. Default is 'viridis'.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    dpi : int, optional
        Figure resolution. Default is 100.
    vmin, vmax : float, optional
        Color scale limits
    title : str, optional
        Plot title. Default is 'Map View'.
    xlabel, ylabel : str, optional
        Axis labels. Default is 'X-axis', 'Y-axis'.
    cmlabel : str, optional
        Colorbar label. Default is 'Value'.
    preset : str, optional
        Use predefined style preset ('publication', 'presentation', 'quick', 'minimal').
        Overrides figsize, dpi, cmap if specified. Individual parameters override preset.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates new figure.
    show : bool, optional
        Whether to call plt.show(). Default is True. Set False for further customization.
    **kwargs : dict
        Additional arguments passed to imshow (e.g., interpolation, alpha, extent)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object for further customization

    Import
    ------
    from upxo.viz.gsviz import see_map
    
    Examples
    --------
    >>> # Quick visualization
    >>> see_map(data)
    
    >>> # Publication-ready figure
    >>> see_map(data, preset='publication', title='Grain Structure')
    
    >>> # Custom styling with preset base
    >>> ax = see_map(data, preset='presentation', cmap='coolwarm', show=False)
    >>> ax.set_aspect('equal')
    >>> plt.show()
    """
    # Apply preset if specified
    params = {}
    if preset is not None:
        if preset not in PLOT_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PLOT_PRESETS.keys())}")
        params = PLOT_PRESETS[preset].copy()
    # --------------------------------------------------------
    # Override preset with explicit parameters
    if cmap != 'viridis' or not preset:
        params['cmap'] = cmap
    if figsize != (8, 6) or not preset:
        params['figsize'] = figsize
    if dpi != 100 or not preset:
        params['dpi'] = dpi
    # --------------------------------------------------------
    # Extract colorbar setting from preset or default to True
    show_colorbar = params.pop('colorbar', True)
    # --------------------------------------------------------
    # Merge remaining kwargs
    imshow_params = {k: v for k, v in params.items() if k not in ['figsize', 'dpi', 'cmap']}
    imshow_params.update(imshow_kwargs)
    # --------------------------------------------------------
    # Create figure/axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=params.get('figsize', figsize), 
                               dpi=params.get('dpi', dpi))
        created_fig = True
    else:
        created_fig = False
    # --------------------------------------------------------
    # Process lfi
    if fids is not None:
        lfi = np.where(np.isin(lfi, fids), lfi, np.nan)
    # --------------------------------------------------------
    im = ax.imshow(lfi, cmap=params.get('cmap', cmap), 
                   vmin=vmin, vmax=vmax, **imshow_params)
    # --------------------------------------------------------
    # Add scale bar if requested
    if mbar:
        xstart, ystart = 0, 0
        xsize, ysize = lfi.shape[1], lfi.shape[0]
        if mbar_loc == 'bot_left':
            mbar_xstart = xstart+0.05*min(xsize, ysize)
            mbar_ystart = ystart+0.95*min(xsize, ysize)
        elif mbar_loc == 'top_left':
            mbar_xstart = xstart+0.05*min(xsize, ysize)
            mbar_ystart = ystart+0.05*min(xsize, ysize)
        elif mbar_loc == 'bot_right':
            mbar_xstart = xstart+0.95*min(xsize, ysize)
            mbar_ystart = ystart+0.95*min(xsize, ysize)
        elif mbar_loc == 'top_right':
            mbar_xstart = xstart+0.95*min(xsize, ysize)
            mbar_ystart = ystart+0.05*min(xsize, ysize)
        mbar_xend = mbar_xstart+mbar_length
        mbar_ends = [[mbar_xstart, mbar_xend], [mbar_ystart, mbar_ystart]]
        xtext = 0.2*sum(mbar_ends[0])
        ytext = 0.5*sum(mbar_ends[1])-0.2*mbar_length
        ax.plot(mbar_ends[0], mbar_ends[1], c='k', linewidth=4)
        ax.text(xtext, ytext, f'{mbar_length} {mbar_units}',
                fontsize=10, bbox={'color': 'white', 'alpha': 0.75})
    # --------------------------------------------------------------------
    if show_colorbar:
        plt.colorbar(im, ax=ax, label=cmlabel, ticks=cbar_ticks)
    # --------------------------------------------------------
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Only show if we created the figure and show=True
    if created_fig and show:
        plt.show()
    
    return ax

def see_bsegs_gid(lfi, pcid, dim=2, bsegGeomType='pix', 
                  figsize=(5, 5), dpi=75):
    """
    Orchastrator function to visualize boundary segments.

    Parameters
    ----------
    lfi : ndarray
        Local feature ID array. here, this should be the lfi of segments and not 
        the usual lfi/lgi of MCGS2D/MCGS3D.
    dim: int
        Dimensionality
    pcid : int
        Parent cell ID. This could be the grain ID you are interested in.
    bsegGeomType : str
        Boundary segment geometry type. Options include: 'pix' (for 2D),
        'vox' (for 3D) and 'geom' (for 2D and 3D: MCGS or VTGS).
        Currently only 'pix' (pixel) is supported.
    figsize : tuple, optional
        Figure size. Default is (5, 5).
    dpi : int, optional
        Figure resolution. Default is 75.

    Returns
    -------
    None

    Functions orchastrated
    ----------------------
    see_bsegs_gid_pix(lfi, pcid)

    Examples
    --------
    see_bsegs_gid(localSegIDMasked_lfi, pcid, dim=2, bsegGeomType='pix')
    """
    if dim == 2 and bsegGeomType=='pix':
        see_bsegs_gid_pix(lfi, pcid, figsize=figsize, dpi=dpi)
    if dim == 2 and bsegGeomType=='geom':
        raise NotImplementedError("2D geometric boundary segment visualization not yet implemented.")
    if dim == 3 and bsegGeomType=='vox':
        raise NotImplementedError("3D voxel boundary segment visualization not yet implemented.")
    if dim == 3 and bsegGeomType=='geom':
        raise NotImplementedError("3D geometric boundary segment visualization not yet implemented.")

def see_bsegs_gid_pix(lfi, pcid, figsize=(8, 6), dpi=100, vmin=None, vmax=None,
            title='Map View', xlabel='X-axis', ylabel='Y-axis',
            cmlabel='Value', preset='minimal', ax=None, show=True, 
            cbar_ticks=None, **imshow_kwargs):
    """
    Visualize boundary segments for a given parent cell ID in 2D pixel data.

    Parameters
    ----------
    lfi : ndarray
        Local feature ID array. here, this should be the lfi of segments and not 
        the usual lfi/lgi of MCGS2D/MCGS3D.
    pcid : int
        Parent cell ID. This could be the grain ID you are interested in.
    figsize : tuple, optional
        Figure size. Default is (5, 5).
    dpi : int, optional
        Figure resolution. Default is 75.

    Returns
    -------
    None

    Examples
    --------
    >>> # Visualize boundary segments for parent cell ID of largest grain
    >>> pcid = np.argmax(pxt.gs[3].prop.npixels.to_numpy())+1
    >>> see_bsegs_gid_pix(localSegIDMasked_lfi, pcid, figsize=(5,5), dpi=75)
    """
    scaled_mask = np.asarray(lfi >= pcid, dtype=float)
    scaled_mask = scaled_mask*np.asarray(lfi < pcid+1, dtype=float)
    scaled_mask = lfi*scaled_mask
    '''scaled_mask = lfi*np.asarray(lfi >= pcid, dtype=float)*np.asarray(lfi < pcid+1, dtype=float)'''
    scaled_mask[scaled_mask < pcid] = np.nan
    # ----------------------------------------------
    see_map(scaled_mask, cmap='viridis', 
            figsize=(figsize), dpi=dpi, vmin=vmin, vmax=vmax,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cmlabel=cmlabel, preset=preset, ax=ax, show=show, 
            cbar_ticks=cbar_ticks, **imshow_kwargs)

def plot_multipolygon_geometric(gs_geometric, fig=None, ax=None, cmap='tab20', edgecolor='black', 
                        alpha=0.7, lw=1, figsize=(10, 10), dpi=100, 
                        points=None, point_color='red', point_size=20, 
                        point_marker='o', point_alpha=0.8, point_label='Points'):
    """
    Plot a Shapely MultiPolygon object using matplotlib with unique colors per polygon.
    
    Parameters
    ----------
    gs_geometric : shapely.geometry.MultiPolygon
        The MultiPolygon object to plot
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for polygon colors. Default 'tab20'
    edgecolor : str, optional
        Edge color for polygons. Default 'black'
    alpha : float, optional
        Transparency level (0-1). Default 0.7
    lw : float, optional
        Line width for edges. Default 1
    figsize : tuple, optional
        Figure size if creating new figure. Default (10, 10)
    dpi : int, optional
        DPI for figure. Default 100
    points : ndarray or None, optional
        N×2 array of coordinate points to plot on top of polygons. 
        Format: [[x1, y1], [x2, y2], ...]. Default None (no points plotted)
    point_color : str, optional
        Color for plotted points. Default 'red'
    point_size : float, optional
        Size of plotted points. Default 20
    point_marker : str, optional
        Marker style for points. Default 'o' (circle)
    point_alpha : float, optional
        Transparency for points (0-1). Default 0.8
    point_label : str, optional
        Label for points in legend. Default 'Points'
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Import
    ------
    from upxo.viz.gsviz import plot_multipolygon_geometric

    Function signature
    ------------------
    plot_multipolygon_geometric(gs_geometric, fig=None, ax=None, cmap='tab20', edgecolor='black', 
                        alpha=0.7, lw=1, figsize=(10, 10), dpi=100, 
                        points=None, point_color='red', point_size=20, 
                        point_marker='o', point_alpha=0.8, point_label='Points')
    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.cm as cm

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Get colormap
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    num_polygons = len(gs_geometric.geoms)
    colors = [cmap(i / num_polygons) for i in range(num_polygons)]
    
    # Plot each polygon individually with unique color
    for idx, poly in enumerate(gs_geometric.geoms):
        # Get exterior coordinates
        exterior_coords = np.array(poly.exterior.coords)
        polygon_patch = Polygon(exterior_coords, closed=True, 
                            facecolor=colors[idx], edgecolor=edgecolor,
                            linewidth=lw, alpha=alpha)
        ax.add_patch(polygon_patch)
        
        # Plot holes (interiors) if any - use white or transparent
        for interior in poly.interiors:
            interior_coords = np.array(interior.coords)
            hole_patch = Polygon(interior_coords, closed=True,
                            facecolor='white', edgecolor=edgecolor,
                            linewidth=lw, alpha=1.0)
            ax.add_patch(hole_patch)
    
    # Plot coordinate points if provided
    if points is not None:
        # Validate that points is a numpy array with shape (N, 2)
        if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2:
            ax.scatter(points[:, 0], points[:, 1], 
                      c=point_color, s=point_size, marker=point_marker,
                      alpha=point_alpha, label=point_label, zorder=10)
            
            # Add legend if points are plotted
            if points.shape[0] > 0:
                ax.legend(loc='best')
    
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('')
    
    return fig, ax

def make_pvgrid(lfi, scalar_name='lfi', origin=(0, 0, 0), spacing=(1, 1, 1)):
    pvgrid = pv.ImageData()
    pvgrid.dimensions = np.array(lfi.shape) + 1
    pvgrid.origin = origin
    pvgrid.spacing = spacing
    pvgrid.cell_data[str(scalar_name)] = lfi.flatten(order="F")
    return pvgrid

def plot_pvgrid(pvgrid, scalar_name='lfi', show_edges=False, alpha=1.0, title='',
                cmap='nipy_spectral', _xname_='', _yname_='', _zname_=''):
    """
    gsviz.plot_pvgrid(gsviz.make_pvgrid(lfi, scalar_name='lfi'),
                scalar_name, show_edges=False, alpha=1.0, title='',
                cmap='nipy_spectral', _xname_='', _yname_='', _zname_='')
    Import
    ------
    from upxo.viz import gsviz
    """
    pvp = pv.Plotter()
    pvp.add_mesh(pvgrid, scalars=scalar_name,
                 show_edges=show_edges, opacity=alpha, cmap=cmap)
    pvp.add_text(f"{title}", font_size=10)
    _ = pvp.add_axes(line_width=5, cone_radius=0.6,
                     shaft_length=0.7, tip_length=0.3,
                     ambient=0.5, label_size=(0.4, 0.16),
                     xlabel=_xname_, ylabel=_yname_, zlabel=_zname_,
                     viewport=(0, 0, 0.25, 0.25))
    pvp.show()

def grain_viewer(lfi):
    """
    Create an interactive 3D visualization with a slider to explore individual grains.
    
    Parameters
    ----------
    lfi : np.ndarray
        3D labeled image array where each voxel contains a grain ID.

    Returns
    -------
    None

    Import
    ------
    from upxo.viz import gsviz
    Use as: gsviz.grain_viewer(lfi)
    """
    max_gid = int(np.max(lfi))

    grid = pv.ImageData(dimensions=np.array(lfi.shape)+1,
        spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),)
    grid.cell_data["gid"] = lfi.ravel(order="F")

    plotter = pv.Plotter()
    actor = {"mesh": None}

    def show_grain(gid):
        if actor["mesh"] is not None:
            plotter.remove_actor(actor["mesh"])
            actor["mesh"] = None

        grain = grid.threshold([gid - 0.5, gid + 0.5], scalars="gid")
        if grain.n_cells == 0:
            plotter.render()
            return

        actor["mesh"] = plotter.add_mesh(grain, show_edges=True, color="tan", opacity=1.0)
        plotter.set_focus(grain.center)
        plotter.reset_camera()
        plotter.render()

    plotter.add_slider_widget(callback=lambda v: show_grain(int(v)),
                rng=[1, max_gid], value=1, title="Grain ID", fmt="%0.f",)

    show_grain(1)
    plotter.show()

def view_selected_grain_boundary_voxels(lfi, grain_ids, viewInternalOnly=True, spacing=(1.0, 1.0, 1.0),
                    origin=(0.0, 0.0, 0.0), cmap="tab20", opacity=1.0, point_size=6.0,
                    show_as_cubes=True, show=True):
    """
    Visualize boundary voxels of selected grain IDs in 3D.
    Parameters
    ----------
    lfi : ndarray[int]
        3D grain ID array.
    grain_ids : iterable of int
        Grain IDs to visualize. Should be a list, set, or array of integers corresponding to 
        grain IDs in the LFI.
    viewInternalOnly : bool, optional
        If True, only visualize internal grain boundaries. If False, include outer RVE 
        boundaries. Default is True.
    spacing : tuple of float, optional
        Voxel spacing in (x, y, z) directions. Default is (1.0, 1.0, 1.0).
    origin : tuple of float, optional
        Origin coordinates for the grid. Default is (0.0, 0.0, 0.0).
    cmap : str, optional
        Colormap for visualizing different grain IDs. Default is "tab20".
    opacity : float, optional
        Opacity for the visualized voxels (0.0 to 1.0). Default is 1.0 (fully opaque).
    point_size : float, optional
        Size of the points representing boundary voxels. Default is 6.0.
    show_as_cubes : bool, optional
        If True, visualize boundary voxels as cubes. If False, visualize as points. 
        Default is True.
    show : bool, optional
        Whether to display the plot immediately. Default is True. Set to False for 
        further customization before showing.
    Returns
    -------
    pl : pyvista.Plotter
        The PyVista plotter object containing the visualization. Can be used for further
          customization or saving the plot.
    pdata : pyvista.PolyData
        The PolyData object containing the boundary voxel points and their associated grain IDs. 
        Can be used for further analysis or custom visualization.
    
    Import
    ------
    from upxo.viz import gsviz
    Example usage:
    >>> # Visualize boundary voxels for grain IDs 1, 2, and 3
    >>> gsviz.view_selected_grain_boundary_voxels(lfi, grain_ids=[1, 2, 3], viewInternalOnly=True, spacing=(1.0, 1.0, 1.0),
    ...     origin=(0.0, 0.0, 0.0), cmap="tab20", opacity=0.9, point_size=6.0, show_as_cubes=True, show=True)
    """
    import upxo.gbops.grainBoundOps3d as gbOps
    lfi = np.asarray(lfi)
    if lfi.ndim != 3:
        raise ValueError("lfi must be a 3D array.")
    if grain_ids is None:
        raise ValueError("grain_ids must be a non-empty iterable of grain IDs.")

    gids = np.asarray(list(grain_ids), dtype=lfi.dtype)
    if gids.size == 0:
        raise ValueError("grain_ids must be non-empty.")

    if viewInternalOnly:
        boundary = gbOps.compute_gb_boundary_mask_interiorVoxels(lfi)
    else:
        boundary = gbOps.compute_gb_boundary_mask(lfi)
    mask = boundary & np.isin(lfi, gids)

    ijk = np.argwhere(mask)
    if ijk.size == 0:
        raise ValueError("No boundary voxels found for the selected grain IDs.")

    pts = ijk.astype(np.float64)
    pts[:, 0] = origin[0] + (pts[:, 0]+0.5) * spacing[0]
    pts[:, 1] = origin[1] + (pts[:, 1]+0.5) * spacing[1]
    pts[:, 2] = origin[2] + (pts[:, 2]+0.5) * spacing[2]

    pdata = pv.PolyData(pts)
    pdata["gid"] = lfi[mask].astype(np.int32)

    pl = pv.Plotter()
    if show_as_cubes:
        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=spacing[0],
                       y_length=spacing[1], z_length=spacing[2],)
        voxels = pdata.glyph(geom=cube, scale=False, orient=False)
        pl.add_mesh(voxels, scalars="gid", cmap=cmap, opacity=opacity, 
            show_scalar_bar=True, show_edges=True, edge_color='black', lighting=True)
    else:
        pl.add_mesh(pdata, scalars="gid", cmap=cmap, opacity=opacity,
            point_size=point_size, render_points_as_spheres=True,
            show_scalar_bar=True,)

    pl.add_axes()
    pl.show_grid()
    if show:
        pl.show()

    return pl, pdata

def viz_clip_plane(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalarName='lfi',
                   cmap='viridis', invert=True, crinkle=True,
                   normal_rotation=True, add_outline=False, throw=False,
                   pvp=None):
    """
    Visualize grain structure along a clip plane.

    Parameters
    ----------
    normal : str or dth.dt.ITERABLE(float), optional
        Normal specification of clipping plane. Default value is 'x'.

    origin : dth.dt.ITERABLE(float), optional
        Specification of origin, that is clip plane centre coordinate.

    scalarName : str, optional
        self.pvgrid cell_data scalar specification. Default value is 'lfi'.

    cmap : str, optional
        Colour map specification. Default value is 'viridis'.
        Recommended values:
            * viridis
            * nipy_spectral

    invert : bool, optional
        Invert clip sense if True, dont if False. Default value is True.

    crinkle : bool, optional
        Crinkle view voxels if True, section view if False. Default value
        is True.

    normal_rotation : bool, optional
        Rotation specification of normal. Default value is True.
        NOTE: To be implemented completely.

    add_outline : bool, optional
        Add an outline around the grain structure. Default value is False.

    throw : bool, optional
        Throw the pvp if True, dont if False. Default value is False.

    pvp : pv.Plotter, optional
        PyVista plotter object to plot over. If no pvp has been provided,
        new pvp shall be created. Default value is None.

    Import
    ------
    from upxo.viz import gsviz
    Use as: gsviz.viz_clip_plane(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalarName='lfi',
                   cmap='viridis', invert=True, crinkle=True,
                     normal_rotation=True, add_outline=False, throw=False, pvp=None)

    """
    pvgrid = make_pvgrid(lfi, scalar_name=scalarName, origin=(0, 0, 0), spacing=(1, 1, 1))
    if pvp is None or not isinstance(pvp, pv.Plotter):
        pvp = pv.Plotter()
    # -------------------------------------
    if add_outline:
        pvp.add_mesh(pvgrid.outline())
    # -------------------------------------
    pvp.add_mesh_clip_plane(pvgrid, normal=normal, origin=origin,
                            scalars=scalarName, cmap=cmap, invert=invert,
                            crinkle=crinkle,
                            normal_rotation=normal_rotation, tubing=False,
                            interaction_event=vtk.vtkCommand.InteractionEvent)
    # -------------------------------------
    if throw:
        return pvp
    else:
        pvp.show()

def viz_mesh_slice(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalarName='lgi',
                    cmap='viridis', normal_rotation=True, add_outline=False,
                    throw=False, pvp=None):
    """
    Visualize grain structure along a slice plane.

    Parameters
    ----------
    lfi : object
        The grain structure object to visualize.

    normal : str or dth.dt.ITERABLE(float), optional
        Normal specification of clipping plane. Default value is 'x'.

    origin : dth.dt.ITERABLE(float), optional
        Specification of origin, that is clip plane centre coordinate.

    scalarName : str, optional
        self.pvgrid cell_data scalar specification. Default value is 'lgi'.

    cmap : str, optional
        Colour map specification. Default value is 'viridis'.
        Recommended values:
            * viridis
            * nipy_spectral

    add_outline : bool, optional
        Add an outline around the grain structure. Default value is False.

    throw : bool, optional
        Throw the pvp if True, dont if False. Default value is False.

    pvp : bool, optional
        PyVista plotter object to plot over. If no pvp has been provided,
        new pvp shall be created. Default value is None.

    Import
    ------
    from upxo.viz import gsviz
    gsviz.viz_mesh_slice(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalar='lgi',
                    cmap='viridis', normal_rotation=True, add_outline=False,
                    throw=False, pvp=None)
    """
    pvgrid = make_pvgrid(lfi, scalar_name=scalarName, origin=(0, 0, 0), spacing=(1, 1, 1))

    if pvp is None or not isinstance(pvp, pv.Plotter):
        pvp = pv.Plotter()
    # -------------------------------------
    if add_outline:
        pvp.add_mesh(pvgrid.outline())
    # -------------------------------------
    pvp.add_mesh_slice(pvgrid, scalars=scalarName,
                        normal=normal, origin=origin, cmap=cmap,
                        normal_rotation=False,
                        interaction_event=vtk.vtkCommand.InteractionEvent)
    # -------------------------------------
    if throw:
        return pvp
    else:
        pvp.show()


def viz_mesh_slice_ortho(lfi, scalarName='lfi', cmap='viridis',
                         style='surface', add_outline=False,
                         throw=False, pvp=None):
    """
    Viz. grain str. along three fundamental mutually orthogonal planes.

    Parameters
    ----------
    lfi : object
        The grain structure object to visualize.

    scalarName : str, optional
        self.pvgrid cell_data scalar specification. Default value is 'lgi'.

    cmap : str, optional
        Colour map specification. Default value is 'viridis'.
        Recommended values:
            * viridis
            * nipy_spectral

    add_outline : bool, optional
        Add an outline around the grain structure. Default value is False.

    throw : bool, optional
        Throw the pvp if True, dont if False. Default value is False.

    pvp : bool, optional
        PyVista plotter object to plot over. If no pvp has been provided,
        new pvp shall be created. Default value is None.

    Import
    ------
    from upxo.viz import gsviz
    gsviz.viz_mesh_slice_ortho(lfi, scalarName='lfi', cmap='viridis',
                         style='surface', add_outline=False,
                         throw=False, pvp=None)
    """
    pvgrid = make_pvgrid(lfi, scalar_name=scalarName, origin=(0, 0, 0), spacing=(1, 1, 1))

    if pvp is None or not isinstance(pvp, pv.Plotter):
        pvp = pv.Plotter()
    # -------------------------------------
    if add_outline:
        pvp.add_mesh(pvgrid.outline())
    # -------------------------------------
    pvp.add_mesh_slice_orthogonal(pvgrid, scalars=scalarName,
                                    style=style, cmap=cmap,
                                    interaction_event=vtk.vtkCommand.InteractionEvent)
    # -------------------------------------
    if throw:
        return pvp
    else:
        pvp.show()
# ===============================================================================================
def vox2geom_plots(plotType, **kwargs):
    if plotType == 1:
        view_grains(**kwargs)
    elif plotType == 2:
        view_boundary_voxels(**kwargs)
    elif plotType == 3:
        see_clip_plane(**kwargs)
    elif plotType == 4:
        see_mesh_slice(**kwargs)
    elif plotType == 5:
        see_mesh_slice_ortho(**kwargs)
    else:
        raise ValueError(f"Unknown plotType '{plotType}'. Available options: 'grain_viewer'\n",
                         " 'view_boundary_voxels', 'see_clip_plane', 'see_mesh_slice', 'see_mesh_slice_ortho'.")

def view_grains(lfi=None, **kwargs):
    grain_viewer(lfi, **kwargs)

def view_boundary_voxels(lfi=None, **kwargs):
    view_selected_grain_boundary_voxels(lfi[::1, ::1, ::1], np.unique(lfi),
                        viewInternalOnly=True, spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0), cmap="gist_ncar", opacity=1.0, point_size=6.0,
                        show_as_cubes=True, show=True)

def see_clip_plane(lfi=None, **kwargs):
    viz_clip_plane(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalarName='lfi',
                    cmap='gist_ncar', invert=True, crinkle=True,
                        normal_rotation=True, add_outline=False, throw=False, pvp=None)

def see_mesh_slice(lfi=None, **kwargs):
    viz_mesh_slice(lfi, normal='x', origin=[5.0, 5.0, 5.0], scalarName='lfi',
                        cmap='gist_ncar', normal_rotation=True, add_outline=False,
                        throw=False, pvp=None)
    
def see_mesh_slice_ortho(lfi=None, **kwargs):
    viz_mesh_slice_ortho(lfi, scalarName='lfi', cmap='gist_ncar',
            style='surface', add_outline=False,
            throw=False, pvp=None)

# ===============================================================================================
def plot_manifold_geom(cells_dict_list, figsize=(12, 6), dpi=100, inlude_legend=True):
    """
    Helper to handle MultiPolygon and GeometryCollection iteration for plotting.

    Parameters
    ----------
    cells_dict_list : list of dict
        List of dictionaries, each mapping grain IDs to Shapely geometries 
        (which may be Polygons, MultiPolygons, or GeometryCollections).
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : ndarray of matplotlib.axes.Axes
        Array of axes objects

    Import
    ------
    from upxo.viz import gsviz
    Use as: gsviz.plot_manifold_geom([cells_dict_1, cells_dict_2], figsize=(12, 6))
    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    
    def get_polygons(g):
        """Recursively extract only Polygon parts from any geometry type."""
        if isinstance(g, Polygon):
            return [g]
        elif isinstance(g, (MultiPolygon, GeometryCollection)):
            res = []
            for part in g.geoms:
                res.extend(get_polygons(part))
            return res
        return [] # Filter out Points and LineStrings

    num_plots = len(cells_dict_list)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, dpi=dpi)
    
    # Handle single subplot case (axes is not an array)
    if num_plots == 1:
        axes = [axes]
    
    for idx, (ax, cells_dict) in enumerate(zip(axes, cells_dict_list)):
        for gid, geom in cells_dict.items():
            # Clean the geometry to ensure we only have polygons
            polys = get_polygons(geom)

            # Plot each valid polygonal part
            first_part = True
            for part in polys:
                x, y = part.exterior.xy
                # Maintain consistent coloring for all parts of the same Grain ID
                if first_part:
                    line, = ax.plot(x, y, linewidth=1.5, label=f'Grain {gid}')
                    color = line.get_color()
                    first_part = False
                else:
                    ax.plot(x, y, linewidth=1.5, color=color)
                
                ax.fill(x, y, alpha=0.2, color=color)
        
        ax.set_aspect('equal')
        if inlude_legend:
            ax.legend(loc='best')
        ax.set_title(f'Subplot {idx + 1}')
    
    plt.tight_layout()
    return fig, axes

def see_2dPoints(points, figsize=(6, 6), dpi=100, title='2D Points', xlabel='X-axis', ylabel='Y-axis',
                point_color='black', point_size=20, point_marker='.', point_alpha=0.8, label='Points',
                plot_legend=True):
    """
    Visualize a set of 2D points using matplotlib.

    Parameters
    ----------
    points : ndarray
        N×2 array of coordinate points to visualize. Format: [[x1, y1], [x2, y2], ...].
    figsize : tuple, optional
        Figure size (width, height). Default is (6, 6).
    dpi : int, optional
        Figure resolution. Default is 100.
    title : str, optional
        Plot title. Default is '2D Points'.
    xlabel : str, optional
        X-axis label. Default is 'X-axis'.
    ylabel : str, optional
        Y-axis label. Default is 'Y-axis'.
    point_color : str or list of str, optional
        Color for the points. Can be a single color or a list of colors for each point. Default is 'black'.
    point_size : float or list of float, optional
        Size of the points. Can be a single size or a list of sizes for each point. Default is 20.
    point_marker : str, optional
        Marker style for the points (e.g., '.' for dots). Default is '.'.
    point_alpha : float or list of float, optional
        Transparency level for the points (0-1). Can be a single value or a list for each point. Default is 0.8.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Import
    ------
    from upxo.viz import gsviz
    Use as: gsviz.see_2dPoints(points_array)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Validate that points is a numpy array with shape (N, 2)
    if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2:
        ax.scatter(points[:, 0], points[:, 1], 
                   c=point_color, s=point_size, marker=point_marker,
                   alpha=point_alpha, label=label, zorder=10)
        if plot_legend:
            ax.legend(loc='best')
    else:
        raise ValueError("points must be a 2D array with shape (N, 2).")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax