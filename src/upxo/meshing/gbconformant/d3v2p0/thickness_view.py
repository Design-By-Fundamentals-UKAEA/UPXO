"""Sampled normal-ray thickness and boundary spacing diagnostics (not guards)."""
import numpy as np
import pyvista as pv
import vtk


def _poly(points, faces):
    used, local = np.unique(faces, return_inverse=True)
    return pv.PolyData(points[used],
        np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3))).ravel())


def _locator(mesh):
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    return locator


def _ray_distances(locator, centres, directions, length, offset, tolerance):
    distances = np.full(len(centres), np.nan)
    cell = vtk.vtkGenericCell()
    for i, (centre, direction) in enumerate(zip(centres, directions)):
        t, sub_id, cell_id = vtk.mutable(0.), vtk.mutable(0), vtk.mutable(0)
        hit, param = [0., 0., 0.], [0., 0., 0.]
        start = centre+offset*direction
        end = centre+length*direction
        found = locator.IntersectWithLine(start, end, tolerance, t, hit, param, sub_id, cell_id, cell)
        if found:
            distance = float(np.dot(np.asarray(hit)-centre, direction))
            if distance > offset+tolerance: distances[i] = distance
    return distances


def sample_thickness_separation(surface, grain_id, *, max_samples=2000, seed=42,
                                ray_offset=None, max_distance=None):
    """Measure inward thickness and outward boundary-sheet spacing at centroids.

    The inward first hit uses the selected grain's complete closed boundary.
    The outward first hit uses the full shared surface complex. Outward rays
    from RVE caps are undefined and not measured. No-hit results remain NaN.
    These normal-ray distances are not Euclidean minimum thickness/separation,
    medial-axis thickness, or evidence of empty gaps between conforming grains.
    """
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError('max_samples must be a positive integer')
    ids = np.flatnonzero(np.any(surface.grain_pairs == grain_id, axis=1))
    if not len(ids): raise ValueError('Selected grain ID is absent')
    extent = np.asarray(surface.report['rve_dimensions'], float)
    diagonal = float(np.linalg.norm(extent))
    tolerance = max(1., diagonal)*1e-10
    offset = 10*tolerance if ray_offset is None else float(ray_offset)
    length = 2*diagonal if max_distance is None else float(max_distance)
    if not np.isfinite(offset) or offset <= tolerance:
        raise ValueError('ray_offset must exceed the numerical intersection tolerance')
    if not np.isfinite(length) or length <= offset:
        raise ValueError('max_distance must be finite and greater than ray_offset')
    faces = surface.triangles[ids].copy()
    reverse = ~surface.exterior[ids] & (surface.grain_pairs[ids, 1] == grain_id)
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    local_ids = np.arange(len(ids))
    if len(ids) > max_samples:
        local_ids = np.sort(np.random.default_rng(seed).choice(len(ids), max_samples, replace=False))
    sampled = ids[local_ids]
    xyz = surface.points[faces[local_ids]]
    normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms <= tolerance**2): raise ValueError('Degenerate sampled triangles')
    normals /= norms[:, None]
    centres = xyz.mean(axis=1)
    print(f'Building spatial indexes; tracing {len(sampled):,} sampled faces for grain {grain_id}...', flush=True)
    own_locator = _locator(_poly(surface.points, faces))
    thickness = _ray_distances(own_locator, centres, -normals, length, offset, tolerance)
    del own_locator
    separation = np.full(len(sampled), np.nan)
    internal = ~surface.exterior[sampled]
    if internal.any():
        full_locator = _locator(_poly(surface.points, surface.triangles))
        separation[internal] = _ray_distances(full_locator, centres[internal], normals[internal], length, offset, tolerance)
    report = dict(grain_id=int(grain_id), total_grain_triangles=len(ids), sampled_triangles=len(sampled),
        valid_thickness=int(np.isfinite(thickness).sum()),
        valid_separation=int(np.isfinite(separation).sum()),
        undefined_outward_rve_caps=int((~internal).sum()),
        thickness_no_hit=int(np.isnan(thickness).sum()),
        separation_no_hit=int(np.isnan(separation[internal]).sum()),
        ray_offset=offset, numerical_tolerance=tolerance, max_ray_distance=length,
        minimum_thickness_certified=False, minimum_separation_certified=False,
        sampling='seeded uniform triangle sampling; not area-weighted')
    return dict(triangle_ids=sampled, centres=centres, outward_normals=normals,
                thickness=thickness, separation=separation, report=report)


def thickness_separation_view(surface, grain_id, *, max_samples=2000, seed=42,
                               ray_offset=None, max_distance=None, cmap='viridis',
                               limits=None, context_opacity=.12, map_opacity=1.,
                               show_edges=False, show_lines=True, line_color='red',
                               line_width=3., show_points=False, point_color='black',
                               point_size=10., window_size=(1500, 700), show=True):
    """Two linked sampled surface maps; grey areas have no displayed measurement."""
    from .views import junction_geometry, _overlays
    data = sample_thickness_separation(surface, grain_id, max_samples=max_samples, seed=seed,
                                      ray_offset=ray_offset, max_distance=max_distance)
    if limits is None: limits = (0., float(np.linalg.norm(surface.report['rve_dimensions'])))
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError('limits must be finite and increasing')
    owned = np.any(surface.grain_pairs == grain_id, axis=1)
    context = _poly(surface.points, surface.triangles[owned])
    lines, points = junction_geometry(surface, [grain_id]) if show_lines or show_points else (None, None)
    plotter = pv.Plotter(shape=(1, 2), notebook=False, window_size=window_size)
    for panel, key, title in ((0, 'thickness', 'Inward normal-ray thickness'),
                              (1, 'separation', 'Outward normal-ray boundary spacing')):
        plotter.subplot(0, panel)
        finite = np.isfinite(data[key])
        # Exclude measured triangles from grey context to avoid coincident faces.
        background = owned.copy()
        background[data['triangle_ids'][finite]] = False
        if background.any():
            plotter.add_mesh(_poly(surface.points, surface.triangles[background]),
                             color='lightgray', opacity=context_opacity, show_edges=False)
        if finite.any():
            measured = _poly(surface.points, surface.triangles[data['triangle_ids'][finite]])
            measured.cell_data[key] = data[key][finite]
            plotter.add_mesh(measured, scalars=key, cmap=cmap, clim=limits,
                opacity=map_opacity, show_edges=show_edges,
                scalar_bar_args={'title': key.capitalize()+' (physical units)'}, name='measurement')
        if show_lines or show_points:
            _overlays(plotter, lines, points, show_lines, line_color, line_width,
                      show_points, point_color, point_size)
        plotter.add_text(f'{title} | grain {grain_id}\n{finite.sum():,}/{len(finite):,} valid samples', font_size=10)
        plotter.add_axes()
    plotter.link_views([0, 1]); plotter.subplot(0, 0); plotter.view_isometric()
    plotter.reset_camera(bounds=context.bounds)
    print(data['report'])
    for key in ('thickness', 'separation'):
        valid = data[key][np.isfinite(data[key])]
        print(key, {'minimum_sampled':float(valid.min()), 'median_sampled':float(np.median(valid)),
                    'maximum_sampled':float(valid.max())} if len(valid) else 'No valid measurements')
    if show: plotter.show()
    return plotter, data
