"""Read-only island, orthogonal-section and exploded-grain viewers."""
import numpy as np
import pyvista as pv
from .views import _group_indices
from .thickness_view import _poly


def _grain_faces(surface):
    pairs = surface.grain_pairs
    records = np.unique(np.vstack((np.column_stack((pairs[:, 0], np.arange(len(pairs)))),
                                   np.column_stack((pairs[:, 1], np.arange(len(pairs)))))), axis=0)
    return {gid: records[ids, 1] for gid, ids in _group_indices(records[:, 0]).items()}


def nested_island_view(surface, *, host_grain_id=None, host_opacity=.08,
                       island_opacity=1., cmap='tab20c', color_by='grain_id',
                       show_edges=False, max_depth=None, show=True):
    """Classify positive closed shell envelopes by strict containment.

    Requires previously validated nonintersecting closed grain surfaces. Tests
    all child-envelope vertices after bounding-box pruning. Uses floating-point
    VTK containment, not an independent intersection or topology certificate.
    Disconnected grain components receive separate envelope records.
    """
    if color_by not in ('grain_id', 'depth'): raise ValueError('color_by must be grain_id or depth')
    envelopes = []
    tol = max(1., float(np.max(surface.report['rve_dimensions'])))*1e-9
    print('Finding closed grain envelopes and nesting...', flush=True)
    for gid, ids in _grain_faces(surface).items():
        faces = surface.triangles[ids].copy()
        reverse = ~surface.exterior[ids] & (surface.grain_pairs[ids, 1] == gid)
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
        mesh = _poly(surface.points, faces).connectivity()
        for local_ids in _group_indices(mesh.cell_data['RegionId']).values():
            shell = mesh.extract_cells(local_ids).extract_surface()
            if shell.n_open_edges:
                raise ValueError(f'Grain {gid} contains an open/nonmanifold shell')
            tri = shell.faces.reshape(-1, 4)[:, 1:]
            xyz = shell.points[tri]-shell.points.mean(axis=0)
            volume = float(np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()/6)
            if volume > tol**3:
                envelopes.append(dict(grain_id=gid, mesh=shell, volume=volume,
                    lower=shell.points.min(axis=0), upper=shell.points.max(axis=0), parent=None))
    for i, child in enumerate(envelopes):
        candidates = [j for j, host in enumerate(envelopes) if host['grain_id'] != child['grain_id']
            and host['volume'] > child['volume']
            and np.all(child['lower'] > host['lower']+tol)
            and np.all(child['upper'] < host['upper']-tol)]
        for j in sorted(candidates, key=lambda j: envelopes[j]['volume']):
            host = envelopes[j]
            enclosed = pv.PolyData(child['mesh'].points).select_enclosed_points(
                host['mesh'], tolerance=1e-9, check_surface=True)
            if np.all(enclosed.point_data['SelectedPoints']):
                child['parent'] = j; break
    def depth(i):
        d = 0
        while envelopes[i]['parent'] is not None:
            i = envelopes[i]['parent']; d += 1
        return d
    depths = [depth(i) for i in range(len(envelopes))]
    parent_ids = {e['parent'] for e in envelopes if e['parent'] is not None}
    if host_grain_id is None and parent_ids:
        root = min(parent_ids, key=lambda i:(depths[i], envelopes[i]['grain_id']))
        host_grain_id = envelopes[root]['grain_id']
    roots = {i for i,e in enumerate(envelopes) if e['grain_id'] == host_grain_id}
    selected = {}
    for i in range(len(envelopes)):
        j, relative = i, 0
        while j not in roots and envelopes[j]['parent'] is not None:
            j = envelopes[j]['parent']; relative += 1
        if j in roots and (max_depth is None or relative <= max_depth): selected[i] = relative
    plotter = pv.Plotter(notebook=False)
    for i, relative in selected.items():
        e = envelopes[i]; mesh = e['mesh'].copy()
        mesh.cell_data[color_by] = np.full(mesh.n_cells, e['grain_id'] if color_by == 'grain_id' else relative)
        clim = (float(np.min(surface.grain_pairs)), float(np.max(surface.grain_pairs))) if color_by == 'grain_id' else (0, max(1, max(selected.values())))
        plotter.add_mesh(mesh, scalars=color_by, cmap=cmap, clim=clim,
            opacity=host_opacity if relative == 0 or i in parent_ids else island_opacity, show_edges=show_edges,
            show_scalar_bar=True)
    records = [dict(component_id=i, grain_id=e['grain_id'], parent_component=e['parent'],
                    parent_grain_id=None if e['parent'] is None else envelopes[e['parent']]['grain_id'],
                    depth=depths[i]) for i,e in enumerate(envelopes)]
    print('Nested components:', [r for r in records if r['parent_component'] is not None])
    plotter.add_text(f'Host grain {host_grain_id} and nested islands' if selected else 'No nested islands found for this selection', font_size=11)
    plotter.add_axes(); plotter.view_isometric(); plotter.reset_camera()
    if show: plotter.show()
    return plotter, records


def orthogonal_slice_view(labels, spacing, grid=None, *, source='voxels', cmap='tab20c',
                           positions=None, opacity=1., show_edges=False,
                           window_size=(1500, 750), show=True):
    """Three axis-aligned cut planes, controlled by sliders updated on release."""
    if source not in ('voxels', 'tets', 'both'): raise ValueError('source must be voxels, tets or both')
    spacing = np.broadcast_to(np.asarray(spacing, float), (3,))
    extent = np.asarray(labels.shape)*spacing
    if positions is None: positions = extent/2
    positions = np.asarray(positions, float)
    if positions.shape != (3,) or np.any(positions <= 0) or np.any(positions >= extent):
        raise ValueError('Initial positions must lie strictly inside the RVE')
    voxel_grid = pv.ImageData(dimensions=np.asarray(labels.shape)+1, spacing=spacing)
    voxel_grid.cell_data['grain_id'] = np.asarray(labels).ravel(order='F')
    if source != 'voxels' and grid is None: raise ValueError('Tetrahedral grid required')
    if source != 'voxels':
        if not np.array_equal(np.unique(grid.cell_data['grain_id']), np.unique(labels)):
            raise ValueError('Voxel and tet grain IDs differ')
        if not np.allclose(np.asarray(grid.bounds)[[1,3,5]], extent) or not np.allclose(np.asarray(grid.bounds)[[0,2,4]], 0):
            raise ValueError('Voxel and tet extents differ')
    sources = [('Voxel slices', voxel_grid)] if source == 'voxels' else [('Tet slices', grid)]
    if source == 'both': sources = [('Voxel slices', voxel_grid), ('Tet slices', grid)]
    plotter = pv.Plotter(shape=(1, len(sources)), notebook=False, window_size=window_size)
    clim = (float(np.min(labels)), float(np.max(labels)))
    for panel,(title,mesh) in enumerate(sources):
        plotter.subplot(0,panel); plotter.add_mesh(mesh.outline(), color='gray')
        plotter.add_text(title+' | release sliders to update', font_size=11); plotter.add_axes()
    state = {'positions':positions.copy(), 'widgets':[]}
    def set_position(axis, value):
        if axis not in (0,1,2) or not np.isfinite(value) or not 0<float(value)<extent[axis]:
            raise ValueError('Slice axis/position is outside the RVE interior')
        state['positions'][axis] = float(value)
        for panel,(_,mesh) in enumerate(sources):
            plotter.subplot(0,panel)
            cut = mesh.slice(normal=np.eye(3)[axis], origin=state['positions'])
            plotter.renderer.remove_actor(f'slice_{axis}', reset_camera=False, render=False)
            if cut.n_cells:
                plotter.add_mesh(cut, scalars='grain_id', cmap=cmap, clim=clim, opacity=opacity,
                                 show_edges=show_edges, show_scalar_bar=False,
                                 name=f'slice_{axis}', reset_camera=False)
        if len(state['widgets']) > axis:
            state['widgets'][axis].GetRepresentation().SetValue(float(value))
        plotter.subplot(0,0); plotter.render()
    for axis in range(3): set_position(axis, positions[axis])
    if len(sources)>1: plotter.link_views(list(range(len(sources))))
    plotter.subplot(0,0)
    for axis in range(3):
        plotter.subplot(0,0)
        epsilon = extent[axis]*1e-6
        widget = plotter.add_slider_widget(lambda v,a=axis:set_position(a,v),
            (epsilon, extent[axis]-epsilon), value=positions[axis], title='XYZ'[axis]+' position',
            pointa=(.1,.23-axis*.075), pointb=(.9,.23-axis*.075), interaction_event='end',
            title_height=.02, slider_width=.03)
        widget.GetRepresentation().SetLabelHeight(.018)
        state['widgets'].append(widget)
    plotter.view_isometric(); plotter.reset_camera(bounds=voxel_grid.bounds)
    if show: plotter.show()
    return plotter, state, set_position


def exploded_grain_view(surface, *, grain_ids=None, factor=.25, max_factor=2.,
                         cmap='tab20c', opacity=1., show_edges=False,
                         show_outline=True, show=True):
    """Translate display-only grain shells radially from the RVE centre.

    Uses surface-vertex mean centres. Shared vertices are duplicated in the
    display, allowing shells to separate without changing source connectivity.
    """
    if not 0 <= factor <= max_factor or max_factor <= 0: raise ValueError('Invalid explosion factors')
    grouped = _grain_faces(surface)
    if grain_ids is None: grain_ids = sorted(grouped)
    grain_ids = np.unique(grain_ids)
    if not len(grain_ids) or any(int(g) not in grouped for g in grain_ids): raise ValueError('Unknown or empty grain selection')
    centre = np.asarray(surface.report['rve_dimensions'])/2
    point_blocks=[]; face_blocks=[]; shift_blocks=[]; labels=[]; count=0
    for gid in grain_ids:
        ids = grouped[int(gid)]
        faces = surface.triangles[ids].copy()
        reverse = ~surface.exterior[ids] & (surface.grain_pairs[ids,1] == gid)
        faces[reverse] = faces[reverse][:,[0,2,1]]
        used, local = np.unique(faces, return_inverse=True)
        coordinates = surface.points[used].copy()
        point_blocks.append(coordinates); face_blocks.append(local.reshape(-1,3)+count)
        shift_blocks.append(np.broadcast_to(coordinates.mean(axis=0)-centre, coordinates.shape))
        labels.append(np.full(len(faces),gid)); count += len(used)
    original = np.vstack(point_blocks); shifts=np.vstack(shift_blocks); faces=np.vstack(face_blocks)
    mesh = pv.PolyData(original+factor*shifts, np.column_stack((np.full(len(faces),3),faces)).ravel())
    mesh.cell_data['grain_id']=np.concatenate(labels)
    plotter=pv.Plotter(notebook=False)
    plotter.add_mesh(mesh,scalars='grain_id',cmap=cmap,opacity=opacity,show_edges=show_edges,
                     name='exploded_mesh',scalar_bar_args={'title':'Grain ID','position_y':.2,'height':.05})
    if show_outline:
        box=pv.Box(bounds=(0,centre[0]*2,0,centre[1]*2,0,centre[2]*2))
        plotter.add_mesh(box.outline(),color='gray')
    plotter.add_text('Exploded grain shells | display-only translations',font_size=11)
    plotter.add_axes()
    state={'factor':float(factor),'widget':None}
    def set_factor(value):
        if not 0<=value<=max_factor:raise ValueError('Explosion factor outside slider range')
        state['factor']=float(value);mesh.points=original+value*shifts
        if state['widget'] is not None: state['widget'].GetRepresentation().SetValue(float(value))
        plotter.render()
    state['widget']=plotter.add_slider_widget(set_factor,(0,max_factor),value=factor,title='Explosion factor',
                             pointa=(.15,.08),pointb=(.85,.08),interaction_event='end')
    plotter.view_isometric()
    all_positions=np.vstack((original,original+max_factor*shifts))
    lo,hi=all_positions.min(axis=0),all_positions.max(axis=0)
    plotter.reset_camera(bounds=(lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
    if show:plotter.show()
    return plotter,state,set_factor
