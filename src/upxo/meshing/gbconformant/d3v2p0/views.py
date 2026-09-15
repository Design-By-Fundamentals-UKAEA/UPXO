"""Read-only PyVista views of the final shared grain complex."""
import numpy as np
import pyvista as pv


def grain_volume_changes(labels, spacing, surface):
    """Compare labelled voxel volumes with oriented closed grain-shell volumes."""
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, float), (3,))
    if labels.ndim != 3 or not labels.size or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Provide a nonempty 3D label array and positive finite spacing')
    grains, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(grains, np.unique(surface.grain_pairs)):
        raise ValueError('Reference and final grain IDs differ')
    extent = np.asarray(surface.report['rve_dimensions'])
    if not np.allclose(np.asarray(labels.shape)*spacing, extent):
        raise ValueError('Reference and final RVE dimensions differ')
    reference = counts.astype(float)*float(np.prod(spacing))
    xyz = surface.points[surface.triangles]-extent/2
    contributions = np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2]))/6
    pairs = surface.grain_pairs
    final = np.bincount(np.searchsorted(grains, pairs[:, 0]), weights=contributions, minlength=len(grains))
    internal = ~surface.exterior
    final -= np.bincount(np.searchsorted(grains, pairs[internal, 1]),
                         weights=contributions[internal], minlength=len(grains))
    if np.any(~np.isfinite(final)) or np.any(final <= 0):
        raise ValueError('Final closed surfaces contain nonpositive or nonfinite grain volumes')
    if not np.isclose(final.sum(), reference.sum(), rtol=1e-8):
        raise ValueError('Final closed volumes do not sum to the reference RVE volume')
    change = final-reference
    return dict(grain_id=grains, reference_volume=reference, final_volume=final,
                absolute_change=change, percent_change=100*change/reference)


def volume_change_view(labels, spacing, surface, *, grain_id=None, mode='percent',
                       cmap='coolwarm', limits=None, voxel_edges=True,
                       voxel_opacity=1., internal_opacity=.35, rve_opacity=.35,
                       show_internal=True, show_rve=True, show_edges=False,
                       show_lines=False, line_color='red', line_width=3.,
                       show_points=False, point_color='black', point_size=10.,
                       top_n=20, window_size=(1500, 700), show=True):
    """Linked voxel/final-surface maps; negative is shrinkage, positive expansion.

    Each shared interface is displayed once, using its first owner unless a
    particular grain is selected. No averaging of two grains' volume changes.
    """
    data = grain_volume_changes(labels, spacing, surface)
    grains = data['grain_id']
    if grain_id is not None and grain_id not in grains: raise ValueError('Unknown grain ID')
    if mode not in ('percent', 'absolute'): raise ValueError('mode must be percent or absolute')
    values = data['percent_change' if mode == 'percent' else 'absolute_change']
    if limits is None:
        bound = max(float(np.abs(values).max()), 1e-12)
        limits = (-bound, bound)
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError('limits must contain two increasing finite values')
    voxel_grid = pv.ImageData(dimensions=np.asarray(labels.shape)+1,
                             spacing=np.broadcast_to(np.asarray(spacing, float), (3,)))
    voxel_ids = np.asarray(labels).ravel(order='F')
    voxel_grid.cell_data['grain_id'] = voxel_ids
    voxel_grid.cell_data['volume_change'] = values[np.searchsorted(grains, voxel_ids)]
    if grain_id is not None: voxel_grid = voxel_grid.extract_cells(voxel_ids == grain_id)
    plotter = pv.Plotter(shape=(1, 2), notebook=False, window_size=window_size)
    title = 'Volume change (%)' if mode == 'percent' else 'Volume change (physical units cubed)'
    plotter.subplot(0, 0)
    plotter.add_mesh(voxel_grid, scalars='volume_change', cmap=cmap, clim=limits,
        show_edges=voxel_edges, opacity=voxel_opacity, scalar_bar_args={'title': title})
    plotter.add_text('Reference voxels | per-grain volume change', font_size=11)
    plotter.add_axes()
    plotter.subplot(0, 1)
    pairs = surface.grain_pairs
    selected = np.ones(len(pairs), bool) if grain_id is None else np.any(pairs == grain_id, axis=1)
    for exterior, visible, opacity in ((False, show_internal, internal_opacity), (True, show_rve, rve_opacity)):
        mask = selected & (surface.exterior == exterior)
        if not visible or not mask.any(): continue
        faces = surface.triangles[mask]
        used, local = np.unique(faces, return_inverse=True)
        mesh = pv.PolyData(surface.points[used],
            np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())
        owner = pairs[mask, 0] if grain_id is None else np.full(len(faces), grain_id)
        mesh.cell_data['grain_id'] = owner
        mesh.cell_data['volume_change'] = values[np.searchsorted(grains, owner)]
        plotter.add_mesh(mesh, scalars='volume_change', cmap=cmap, clim=limits,
            opacity=opacity, show_edges=show_edges, scalar_bar_args={'title': title})
    if show_lines or show_points:
        lines, points = junction_geometry(surface, None if grain_id is None else [grain_id])
        _overlays(plotter, lines, points, show_lines, line_color, line_width,
                  show_points, point_color, point_size)
    plotter.add_text('Final closed surfaces | per-grain volume change', font_size=11)
    plotter.add_axes()
    plotter.link_views([0, 1])
    plotter.subplot(0, 0); plotter.view_isometric(); plotter.reset_camera()
    print('Negative = shrinkage; positive = expansion. Volumes in physical units cubed.')
    print(f"Reference total: {data['reference_volume'].sum():.9g}; final total: {data['final_volume'].sum():.9g}")
    print('Grain ID    Reference volume      Final volume     Absolute change      Change (%)')
    order = np.argsort(-np.abs(data['percent_change']))[:max(0, int(top_n))]
    if grain_id is not None: order = np.flatnonzero(grains == grain_id)
    for i in order:
        print(f"{int(grains[i]):8d} {data['reference_volume'][i]:19.8g} {data['final_volume'][i]:17.8g} "
              f"{data['absolute_change'][i]:19.8g} {data['percent_change'][i]:15.6g}")
    if show: plotter.show()
    return plotter, data


def displacement_view(interfaces, *, grain_id=None, component='magnitude',
                      cmap='viridis', limits=None, opacity=1., show_edges=False,
                      show_lines=True, line_color='red', line_width=3.,
                      show_points=False, point_color='black', point_size=10.,
                      window_size=(1500, 700), show=True):
    """Linked original/current surfaces with exact pre-Gmsh node displacement."""
    original, current = np.asarray(interfaces.original_points), np.asarray(interfaces.points)
    if original.shape != current.shape or not np.all(np.isfinite([original, current])):
        raise ValueError('Finite original/current coordinates with matching node correspondence required')
    delta = current-original
    magnitude = np.linalg.norm(delta, axis=1)
    if component == 'magnitude': values = magnitude
    elif component in ('x', 'y', 'z'): values = delta[:, 'xyz'.index(component)]
    else: raise ValueError('component must be magnitude, x, y, or z')
    if limits is None:
        if component == 'magnitude': limits = (0., max(float(values.max()), 1e-12))
        else:
            bound = max(float(np.abs(values).max()), 1e-12)
            limits = (-bound, bound)
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError('limits must contain two increasing finite values')
    pairs = interfaces.grain_pairs
    mask = np.ones(len(pairs), bool) if grain_id is None else np.any(pairs == grain_id, axis=1)
    if not mask.any(): raise ValueError('No internal interface triangles for the selected grain')
    faces = interfaces.triangles[mask]
    used, local = np.unique(faces, return_inverse=True)
    cells = np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel()
    edges = np.asarray(interfaces.junction_edges)
    owned_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    n = len(current)
    edges = np.sort(edges, axis=1)
    edges = edges[np.isin(edges[:, 0]*n+edges[:, 1], owned_edges[:, 0]*n+owned_edges[:, 1])]
    line_nodes, line_local = np.unique(edges, return_inverse=True)
    point_ids = used[interfaces.node_kind[used] == 3]
    plotter = pv.Plotter(shape=(1, 2), notebook=False, window_size=window_size)
    title = 'Displacement magnitude' if component == 'magnitude' else f'Displacement {component}'
    for panel, coordinates in enumerate((original, current)):
        plotter.subplot(0, panel)
        mesh = pv.PolyData(coordinates[used], cells)
        mesh.point_data['displacement'] = values[used]
        mesh.point_data['displacement_vector'] = delta[used]
        plotter.add_mesh(mesh, scalars='displacement', preference='point', cmap=cmap,
            clim=limits, opacity=opacity, show_edges=show_edges,
            scalar_bar_args={'title': title+' (physical units)'}, name='displacement_surface')
        lines = pv.PolyData(coordinates[line_nodes])
        if len(edges):
            lines.lines = np.column_stack((np.full(len(edges), 2), line_local.reshape(-1, 2))).ravel()
            lines.verts = np.empty(0, int)
        _overlays(plotter, lines, coordinates[point_ids], show_lines, line_color, line_width,
                  show_points, point_color, point_size)
        plotter.add_text('Original voxel interfaces' if panel == 0 else 'Moved interfaces (pre-Gmsh)', font_size=11)
        plotter.add_axes()
    plotter.link_views([0, 1])
    plotter.subplot(0, 0); plotter.view_isometric(); plotter.reset_camera()
    print('Exact node displacements, in SPACING physical units.')
    print(f'Selected nodes: {len(used):,}; mean magnitude: {magnitude[used].mean():.6g}; '
          f'maximum magnitude: {magnitude[used].max():.6g}; colour limits: {limits}')
    if show: plotter.show()
    return plotter


def _group_indices(values):
    keys, inverse = np.unique(values, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    return dict(zip(map(int, keys), np.split(order, np.cumsum(np.bincount(inverse))[:-1])))


def _surface_segments(faces, pairs, rve_face):
    """Connected components by shared edge within grain-pair/RVE-face patches."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    keys, inverse = np.unique(np.column_stack((np.sort(pairs, axis=1), rve_face)),
                              axis=0, return_inverse=True)
    result = np.empty(len(faces), dtype=int)
    next_id = 1
    for ids in _group_indices(inverse).values():
        tri = faces[ids]
        _, edge_inv = np.unique(np.sort(tri[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                                axis=0, return_inverse=True)
        incidence = coo_matrix((np.ones(len(edge_inv)),
            (np.repeat(np.arange(len(tri)), 3), edge_inv))).tocsr()
        count, components = connected_components(incidence @ incidence.T, directed=False)
        result[ids] = components + next_id
        next_id += count
    return result


def grain_selector(labels, spacing, surface, grid, *, initial_grain=None,
                   grain_cmap='nipy_spectral', segment_cmap='tab20',
                   voxel_opacity=1., tet_opacity=1., surface_opacity=.65,
                   voxel_edges=True, tet_edges=False, surface_edges=True, show_rve=True,
                   show_lines=True, line_color='red', line_width=3.,
                   show_points=False, point_color='black', point_size=10.,
                   include_rve_junctions=True, reset_camera=True,
                   window_size=(1800, 700), show=True, curvature_type=None,
                   curvature_cmap='tab20c', curvature_limits=None):
    """Interactive linked voxel/tet/interface view with grain-index slider.

    Membership indexes and physical junctions are built once. Surface segment
    IDs are local to the selected grain, separated by neighbour, cap face, and
    edge-connected component; artificial Gmsh chart seams do not split segments.
    Returns the plotter and an exact-ID selection callback.
    """
    from scipy.spatial import cKDTree
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, float), (3,))
    grains = np.unique(surface.grain_pairs)
    if not np.array_equal(np.unique(labels), grains) or not np.array_equal(
            np.unique(grid.cell_data['grain_id']), grains):
        raise ValueError('Voxel, surface and tetrahedral grain IDs must match')
    if not np.allclose(np.asarray(labels.shape)*spacing, surface.report['rve_dimensions']):
        raise ValueError('Voxel and surface physical extents differ')
    if initial_grain is None: initial_grain = int(grains[0])
    if initial_grain not in grains: raise ValueError('Initial grain ID is absent')
    voxels = pv.ImageData(dimensions=np.asarray(labels.shape)+1, spacing=spacing)
    voxels.cell_data['grain_id'] = labels.ravel(order='F')
    voxel_groups = _group_indices(voxels.cell_data['grain_id'])
    tet_groups = _group_indices(grid.cell_data['grain_id'])
    face_records = np.unique(np.vstack((np.column_stack((surface.grain_pairs[:, 0], np.arange(len(surface.triangles)))),
                                        np.column_stack((surface.grain_pairs[:, 1], np.arange(len(surface.triangles)))))), axis=0)
    face_groups = {gid: face_records[ids, 1] for gid, ids in _group_indices(face_records[:, 0]).items()}
    curvature_cache = {}
    if curvature_type is not None:
        if curvature_type not in ('mean', 'gaussian'):
            raise ValueError('curvature_type must be mean, gaussian, or None')
        print('Computing per-grain boundary curvature and one fixed colour range...', flush=True)
        for gid, ids in face_groups.items():
            faces = surface.triangles[ids].copy()
            reverse = (~surface.exterior[ids]) & (surface.grain_pairs[ids, 1] == gid)
            faces[reverse] = faces[reverse][:, [0, 2, 1]]
            used, local = np.unique(faces, return_inverse=True)
            boundary = pv.PolyData(surface.points[used],
                np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())
            values = boundary.curvature(curv_type=curvature_type)
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Nonfinite curvature for grain {gid}')
            curvature_cache[gid] = (used, values)
        if curvature_limits is None:
            curvature_limits = (min(float(v.min()) for _, v in curvature_cache.values()),
                                max(float(v.max()) for _, v in curvature_cache.values()))
            if curvature_limits[0] == curvature_limits[1]:
                pad = max(1., abs(curvature_limits[0])) * 1e-6
                curvature_limits = (curvature_limits[0]-pad, curvature_limits[1]+pad)
        if (len(curvature_limits) != 2 or not np.all(np.isfinite(curvature_limits))
                or curvature_limits[0] >= curvature_limits[1]):
            raise ValueError('Curvature limits must be finite and increasing')
        print('Fixed curvature limits:', curvature_limits, flush=True)
    junction_lines, junction_points = junction_geometry(surface, include_rve=include_rve_junctions)
    tree = cKDTree(surface.points)
    line_nodes = tree.query(junction_lines.points)[1] if junction_lines.n_points else np.empty(0, int)
    point_nodes = tree.query(junction_points)[1] if len(junction_points) else np.empty(0, int)
    global_edges = line_nodes[junction_lines.lines.reshape(-1, 3)[:, 1:]] if junction_lines.n_lines else np.empty((0, 2), int)
    plotter = pv.Plotter(shape=(1, 3), notebook=False, window_size=window_size)
    for panel in range(3):
        plotter.subplot(0, panel); plotter.add_axes()
    plotter.link_views([0, 1, 2])
    clim = (float(grains.min()), float(grains.max()))
    state = {'index': None, 'slider': None}

    def select_grain(gid):
        if gid not in face_groups: raise ValueError(f'Unknown grain ID: {gid}')
        gid = int(gid)
        index = int(np.searchsorted(grains, gid))
        if state['index'] == index: return
        state['index'] = index
        ids = face_groups[gid]
        all_nodes = np.unique(surface.triangles[ids])
        owned_edges = np.sort(surface.triangles[ids][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        npoints = len(surface.points)
        edges = global_edges[np.isin(global_edges[:, 0]*npoints+global_edges[:, 1],
                                     owned_edges[:, 0]*npoints+owned_edges[:, 1])]
        used, local = np.unique(edges, return_inverse=True)
        lines = pv.PolyData(surface.points[used])
        if len(edges):
            lines.lines = np.column_stack((np.full(len(edges), 2), local.reshape(-1, 2))).ravel()
            lines.verts = np.empty(0, int)
        points = junction_points[np.isin(point_nodes, all_nodes)]
        if not show_rve: ids = ids[~surface.exterior[ids]]
        faces = surface.triangles[ids]
        segments = _surface_segments(faces, surface.grain_pairs[ids], surface.rve_face[ids]) if len(ids) else np.empty(0, int)
        used, local = np.unique(faces, return_inverse=True)
        boundary = pv.PolyData(surface.points[used],
            np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())
        boundary.cell_data['segment_id'] = segments
        boundary.cell_data['neighbour_grain_id'] = np.where(surface.grain_pairs[ids, 0] == gid,
            surface.grain_pairs[ids, 1], surface.grain_pairs[ids, 0])
        boundary.cell_data['rve_face'] = surface.rve_face[ids]
        selected_tets = grid.extract_cells(tet_groups[gid])
        if curvature_type is not None:
            curvature_nodes, values = curvature_cache[gid]
            boundary.point_data['curvature'] = values[np.searchsorted(curvature_nodes, used)]
            tet_boundary = selected_tets.extract_surface()
            distance, nearest = cKDTree(surface.points[curvature_nodes]).query(tet_boundary.points)
            tol = max(1., float(np.max(surface.report['rve_dimensions']))) * 1e-9
            if np.any(distance > tol):
                raise ValueError('Tet boundary does not match the curvature source; use matching results')
            tet_boundary.point_data['curvature'] = values[nearest]
            selected_tets = tet_boundary
        selected = (voxels.extract_cells(voxel_groups[gid]), selected_tets, boundary)
        titles = ('Voxellated grain', 'Tetrahedral mesh + junctions', 'Boundary segments + junctions')
        if curvature_type is not None:
            titles = ('Voxellated grain', 'Tet boundary curvature + junctions', 'Boundary curvature + junctions')
        for panel, mesh in enumerate(selected):
            plotter.subplot(0, panel)
            for name in ('selected_grain', 'junction_lines', 'junction_points'):
                plotter.renderer.remove_actor(name, reset_camera=False, render=False)
            if panel == 2 and 'Surface segment ID' in plotter.scalar_bars:
                plotter.remove_scalar_bar('Surface segment ID', render=False)
            if mesh.n_cells:
                kwargs = dict(scalars='segment_id', cmap=segment_cmap, opacity=surface_opacity,
                              show_edges=surface_edges, scalar_bar_args={'title': 'Surface segment ID',
                              'fmt': '%.0f', 'n_labels': min(5, len(np.unique(segments)))}) if panel == 2 else dict(
                    scalars='grain_id', cmap=grain_cmap, clim=clim, show_scalar_bar=False,
                    opacity=voxel_opacity if panel == 0 else tet_opacity,
                    show_edges=voxel_edges if panel == 0 else tet_edges)
                if curvature_type is not None and panel:
                    kwargs = dict(scalars='curvature', preference='point', cmap=curvature_cmap,
                        clim=curvature_limits, opacity=tet_opacity if panel == 1 else surface_opacity,
                        show_edges=tet_edges if panel == 1 else surface_edges,
                        scalar_bar_args={'title': f'{curvature_type.capitalize()} curvature (panel {panel+1})'})
                plotter.add_mesh(mesh, name='selected_grain', reset_camera=False, **kwargs)
            if panel:
                _overlays(plotter, lines, points, show_lines, line_color, line_width,
                          show_points, point_color, point_size)
            plotter.add_text(f'{titles[panel]} | grain {gid}', name='panel_title', font_size=11)
        plotter.subplot(0, 0)
        plotter.add_text('Slider: grain selection | Left/Right or Down/Up: previous/next',
                         position='lower_left', font_size=8, name='instructions')
        if reset_camera:
            plotter.view_isometric(); plotter.reset_camera()
        if state['slider'] is not None:
            rep = state['slider'].GetRepresentation()
            rep.SetValue(index); rep.SetTitleText(f'Grain ID: {gid}')
        plotter.render()

    select_grain(initial_grain)
    plotter.subplot(0, 0)
    if len(grains) > 1:
        def slider_changed(value):
            index = int(np.clip(round(value), 0, len(grains)-1))
            select_grain(int(grains[index]))
        state['slider'] = plotter.add_slider_widget(slider_changed, (0, len(grains)-1),
            value=state['index'], title=f'Grain ID: {initial_grain}',
            pointa=(.08, .12), pointb=(.92, .12), interaction_event='end')
        state['slider'].GetRepresentation().ShowSliderLabelOff()
    def step(amount):
        select_grain(int(grains[np.clip(state['index']+amount, 0, len(grains)-1)]))
    for key in ('Right', 'Up'): plotter.add_key_event(key, lambda: step(1))
    for key in ('Left', 'Down'): plotter.add_key_event(key, lambda: step(-1))
    if show: plotter.show()
    return plotter, select_grain


def linked_comparison(labels, spacing, surface, quality, *, cmap='nipy_spectral',
                      voxel_opacity=1., surface_opacity=1., transparent_opacity=.2,
                      show_rve=False, rve_opacity=.1, line_color='red', line_width=3.,
                      show_points=False, point_color='black', point_size=10.,
                      include_rve_junctions=True, bins=60, log_counts=False,
                      quality_threshold=.05, window_size=(2000, 650), show=True):
    """Three camera-linked geometry panels and an independent minSICN histogram."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, float), (3,))
    quality = np.asarray(quality)
    if labels.ndim != 3 or not labels.size or np.any(spacing <= 0):
        raise ValueError('Provide a nonempty 3D label array and positive spacing')
    if not quality.size or np.any(~np.isfinite(quality)):
        raise ValueError('Provide finite tetrahedral minSICN values')
    if not np.allclose(np.asarray(labels.shape)*spacing, surface.report['rve_dimensions']):
        raise ValueError('Voxel and surface physical dimensions differ')
    if not np.array_equal(np.unique(labels), np.unique(surface.grain_pairs)):
        raise ValueError('Voxel and surface grain IDs differ; use matching results')
    voxels = pv.ImageData(dimensions=np.asarray(labels.shape)+1, spacing=spacing)
    voxels.cell_data['grain_id'] = labels.ravel(order='F')
    clim = (float(labels.min()), float(labels.max()))
    meshes = []
    for exterior in (False, True):
        mask = surface.exterior == exterior
        faces = surface.triangles[mask]
        if not len(faces):
            meshes.append(None); continue
        used, local = np.unique(faces, return_inverse=True)
        mesh = pv.PolyData(surface.points[used],
            np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())
        mesh.cell_data['grain_id'] = surface.grain_pairs[mask, 0]
        meshes.append(mesh)
    plotter = pv.Plotter(shape=(1, 4), notebook=False, window_size=window_size)
    plotter.subplot(0, 0)
    plotter.add_mesh(voxels, scalars='grain_id', cmap=cmap, clim=clim,
                     opacity=voxel_opacity, show_edges=False, show_scalar_bar=False)
    plotter.add_text('Voxellated grains', font_size=11)
    plotter.add_axes()
    for panel, opacity, title in ((1, surface_opacity, 'Internal surface mesh'),
                                   (2, transparent_opacity, 'Transparent surfaces + junctions')):
        plotter.subplot(0, panel)
        if meshes[0] is not None:
            plotter.add_mesh(meshes[0], scalars='grain_id', cmap=cmap, clim=clim,
                             opacity=opacity, show_edges=False, show_scalar_bar=False)
        if show_rve and meshes[1] is not None:
            plotter.add_mesh(meshes[1], scalars='grain_id', cmap=cmap, clim=clim,
                             opacity=rve_opacity, show_edges=False, show_scalar_bar=False)
        plotter.add_text(title, font_size=11)
        plotter.add_axes()
    lines, points = junction_geometry(surface, include_rve=include_rve_junctions)
    _overlays(plotter, lines, points, True, line_color, line_width,
              show_points, point_color, point_size)
    plotter.subplot(0, 3)
    fig, ax = plt.subplots(figsize=(5, 6), layout='constrained')
    ax.hist(quality, bins=bins, color='steelblue', edgecolor='white', linewidth=.3,
            log=log_counts)
    ax.axvline(quality_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Target {quality_threshold:g}')
    ax.set(xlabel='Tetrahedron quality (Gmsh minSICN)', ylabel='Element count',
           title='Tet element quality distribution')
    ax.text(.03, .97, f'N = {quality.size:,}\nMinimum = {quality.min():.4g}\n'
            f'Median = {np.median(quality):.4g}\nBelow target = {(quality < quality_threshold).sum():,}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=.2)
    plotter.add_chart(pv.ChartMPL(fig))
    plt.close(fig)  # The VTK chart retains the figure; suppress a separate notebook figure.
    plotter.link_views(views=[0, 1, 2])
    plotter.subplot(0, 0)
    plotter.view_isometric()
    plotter.reset_camera()
    if show: plotter.show()
    return plotter


def junction_geometry(surface, grain_ids=None, include_rve=True):
    """Extract physical junctions, excluding artificial Gmsh chart seams.

    Three material regions define a line; the exterior counts as one region
    when requested. Points are line endpoints/branches or >=4-region nodes.
    Classify globally before restricting to the displayed grains.
    """
    f, pairs = surface.triangles, surface.grain_pairs
    edges, inv = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                           axis=0, return_inverse=True)
    ownership = np.unique(np.vstack((np.column_stack((inv, np.repeat(pairs[:, 0], 3))),
                                     np.column_stack((inv, np.repeat(pairs[:, 1], 3))))), axis=0)
    counts = np.bincount(ownership[:, 0], minlength=len(edges))
    if include_rve:
        counts[np.unique(inv.reshape(-1, 3)[surface.exterior])] += 1
    line_mask = counts >= 3
    node_owners = np.unique(np.vstack((np.column_stack((f.ravel(), np.repeat(pairs[:, 0], 3))),
                                       np.column_stack((f.ravel(), np.repeat(pairs[:, 1], 3))))), axis=0)
    node_counts = np.bincount(node_owners[:, 0], minlength=len(surface.points))
    if include_rve:
        node_counts[np.unique(f[surface.exterior])] += 1
    degree = np.bincount(edges[line_mask].ravel(), minlength=len(surface.points))
    point_mask = (node_counts >= 4) | ((degree > 0) & (degree != 2))
    if grain_ids is not None:
        chosen_faces = np.any(np.isin(pairs, grain_ids), axis=1)
        allowed_edges = np.zeros(len(edges), bool)
        allowed_edges[np.unique(inv.reshape(-1, 3)[chosen_faces])] = True
        line_mask &= allowed_edges
        allowed_nodes = np.zeros(len(surface.points), bool)
        allowed_nodes[np.unique(f[chosen_faces])] = True
        point_mask &= allowed_nodes
    line_edges = edges[line_mask]
    used, local = np.unique(line_edges, return_inverse=True)
    lines = pv.PolyData(surface.points[used])
    if len(line_edges):
        lines.lines = np.column_stack((np.full(len(line_edges), 2), local.reshape(-1, 2))).ravel()
        lines.verts = np.empty(0, dtype=int)
    return lines, surface.points[point_mask]


def non_touching_grains(surface, seed=42, maximum=None):
    """Seeded greedy maximal set: no selected grains share a surface node.

    This is not a maximum-cardinality independent-set solver. With maximum
    supplied the selection stops early. Touching includes face/edge/point contact.
    """
    if maximum is not None and (not isinstance(maximum, (int, np.integer)) or maximum < 1):
        raise ValueError('maximum must be a positive integer or None')
    f, pairs = surface.triangles, surface.grain_pairs
    records = np.unique(np.vstack((np.column_stack((np.repeat(pairs[:, 0], 3), f.ravel())),
                                   np.column_stack((np.repeat(pairs[:, 1], 3), f.ravel())))), axis=0)
    grains, starts = np.unique(records[:, 0], return_index=True)
    groups = np.split(records[:, 1], starts[1:])
    occupied = np.zeros(len(surface.points), bool)
    selected = []
    for i in np.random.default_rng(seed).permutation(len(grains)):
        nodes = groups[i]
        if not occupied[nodes].any():
            selected.append(grains[i]); occupied[nodes] = True
            if maximum is not None and len(selected) >= maximum: break
    return np.asarray(selected, dtype=pairs.dtype)


def _overlays(plotter, lines, points, show_lines, line_color, line_width,
              show_points, point_color, point_size):
    if show_lines and lines.n_lines:
        plotter.add_mesh(lines, color=line_color, line_width=line_width, name='junction_lines')
    if show_points and len(points):
        plotter.add_points(points, color=point_color, point_size=point_size,
                           render_points_as_spheres=True, name='junction_points')


def surface_view(surface, *, internal_grains_only=False, show_rve=True,
                 internal_opacity=.35, rve_opacity=.15, cmap='nipy_spectral',
                 show_lines=True, line_color='red', line_width=3.,
                 show_points=False, point_color='black', point_size=10.,
                 include_rve_junctions=True, show=True):
    grains = np.unique(surface.grain_pairs)
    if internal_grains_only:
        # Geometric contact includes an isolated RVE-plane node, not just caps.
        extent = np.asarray(surface.report['rve_dimensions'])
        tol = max(1., float(extent.max())) * 1e-9
        on_box = np.any((np.abs(surface.points) <= tol) |
                        (np.abs(surface.points-extent) <= tol), axis=1)
        touching = np.unique(surface.grain_pairs[np.any(on_box[surface.triangles], axis=1)])
        grains = np.setdiff1d(grains, touching)
    selected = np.any(np.isin(surface.grain_pairs, grains), axis=1)
    plotter = pv.Plotter(notebook=False)
    for cap, opacity in ((False, internal_opacity), (True, rve_opacity)):
        mask = selected & (surface.exterior == cap)
        if cap and (not show_rve or internal_grains_only): continue
        if not mask.any(): continue
        faces = surface.triangles[mask]
        used, local = np.unique(faces, return_inverse=True)
        mesh = pv.PolyData(surface.points[used],
            np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())
        pairs = surface.grain_pairs[mask]
        mesh.cell_data['grain_id'] = np.where(np.isin(pairs[:, 0], grains), pairs[:, 0], pairs[:, 1])
        plotter.add_mesh(mesh, scalars='grain_id', cmap=cmap, opacity=opacity,
                         show_edges=False, clim=(float(np.min(surface.grain_pairs)),
                                               float(np.max(surface.grain_pairs))),
                         show_scalar_bar=False)
    if show_lines or show_points:
        lines, points = junction_geometry(surface, grains, include_rve_junctions)
        _overlays(plotter, lines, points, show_lines, line_color, line_width,
                  show_points, point_color, point_size)
    plotter.add_axes()
    print(f'Displaying boundaries incident to {len(grains)} grains')
    if show: plotter.show()
    return plotter


def tet_view(surface, grid, *, seed=42, maximum=None, color_segments=False,
             cmap='nipy_spectral', opacity=1., show_edges=False,
             show_lines=True, line_color='red', line_width=3.,
             show_points=False, point_color='black', point_size=10.,
             include_rve_junctions=True, clip=False, crinkle=True,
             normal='x', origin=None, invert=False, show=True):
    ids = non_touching_grains(surface, seed, maximum)
    if not np.array_equal(np.unique(grid.cell_data['grain_id']), np.unique(surface.grain_pairs)):
        raise ValueError('Surface and tetrahedral grain IDs differ; use matching results.')
    selected = grid.extract_cells(np.isin(grid.cell_data['grain_id'], ids))
    scalar = 'grain_id'
    if color_segments and selected.n_cells:
        # Selected grains have no shared nodes, so connectivity regions cannot
        # merge different grains. Each disconnected piece gets a global ID.
        selected = selected.connectivity()
        selected.cell_data['segment_id'] = np.asarray(selected.cell_data['RegionId']) + 1
        scalar = 'segment_id'
    plotter = pv.Plotter(notebook=False)
    kwargs = dict(scalars=scalar, cmap=cmap, opacity=opacity, show_edges=show_edges)
    if selected.n_cells:
        if clip:
            plotter.add_mesh_clip_plane(selected, crinkle=crinkle, normal=normal,
                                       origin=origin, invert=invert, **kwargs)
        else: plotter.add_mesh(selected, **kwargs)
    if show_lines or show_points:
        lines, points = junction_geometry(surface, ids, include_rve_junctions)
        # Keep complete physical junctions as spatial context in the clip view.
        _overlays(plotter, lines, points, show_lines, line_color, line_width,
                  show_points, point_color, point_size)
    plotter.add_axes()
    print(f'{len(ids)} mutually non-touching grains; {selected.n_cells:,} tetrahedra')
    print('Selected grain IDs:', ids.tolist())
    if show: plotter.show()
    return plotter, ids
