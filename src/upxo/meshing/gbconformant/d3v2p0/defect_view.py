"""Display saved surface flags and current quality warnings without repairing."""
import numpy as np
import pyvista as pv


def defect_only_view(surface, validation, grid=None, *, grain_id=None,
                     show_reported=True, show_intersections=True, show_small_angles=True,
                     show_surface_quality=True, show_coincident_nodes=True,
                     show_tet_quality=True, surface_quality_threshold=.05,
                     tet_quality_threshold=.05, show_context=False, context_opacity=.05,
                     surface_opacity=1., tet_opacity=1., show_edges=True,
                     point_size=12., colors=None, show_lines=False,
                     line_color='red', line_width=3., window_size=(1400, 800), show=True):
    """Use matching saved validation and current quality; no full intersection scan.

    The validator's bad_triangle_ids mix topology and quality flags: they are
    displayed as reported flags, not reclassified as confirmed topology defects.
    Coincident-node locations are recovered using the report's numerical tolerance.
    """
    from .surface_quality import triangle_quality
    from .thickness_view import _poly
    from .views import junction_geometry
    from scipy.spatial import cKDTree
    colors = {**dict(reported='royalblue', surface_quality='orange', small_angles='magenta',
                     intersections='red', coincident_nodes='cyan', tet_quality='gold',
                     invalid_tet_quality='black'), **(colors or {})}
    f = surface.triangles
    selected = np.ones(len(f), bool) if grain_id is None else np.any(surface.grain_pairs == grain_id, axis=1)
    if not selected.any(): raise ValueError('Selected grain has no surface triangles')
    if not 0 <= surface_quality_threshold <= 1 or not 0 <= tet_quality_threshold <= 1:
        raise ValueError('Quality thresholds must lie in [0, 1]')
    def ids_from_report(key):
        ids = np.asarray(validation.get(key, []), dtype=int).ravel()
        if np.any(ids < 0) or np.any(ids >= len(f)):
            raise ValueError('Validation triangle IDs are incompatible with this surface')
        ids = np.unique(ids)
        return ids[selected[ids]]
    categories = {
        'reported': ids_from_report('bad_triangle_ids'),
        'surface_quality': np.flatnonzero(selected & (triangle_quality(surface.points, f) < surface_quality_threshold)),
        'small_angles': ids_from_report('small_facet_angle_pairs'),
        'intersections': ids_from_report('intersecting_triangle_pairs'),
    }
    active = dict(reported=show_reported, surface_quality=show_surface_quality,
                  small_angles=show_small_angles, intersections=show_intersections)
    # Draw each surface face only once; more specific flags take precedence.
    category_order = list(categories)
    codes = np.full(len(f), -1, int)
    for i, key in enumerate(category_order):
        if active[key]: codes[categories[key]] = i
    plotter = pv.Plotter(notebook=False, window_size=window_size)
    if show_context:
        context_ids = np.flatnonzero(selected & (codes < 0))
        if len(context_ids):
            plotter.add_mesh(_poly(surface.points, f[context_ids]), color='lightgray',
                             opacity=context_opacity, name='context')
    labels = dict(reported='Saved flags', surface_quality='Poor triangles',
                  small_angles='Narrow openings', intersections='Intersections')
    counts = {key:len(ids) for key,ids in categories.items()}
    visible_count = 0
    for i, key in enumerate(category_order):
        ids = np.flatnonzero(codes == i)
        if len(ids):
            mesh = _poly(surface.points, f[ids]); mesh.cell_data['source_triangle_id'] = ids
            plotter.add_mesh(mesh, color=colors[key], opacity=surface_opacity, show_edges=show_edges,
                             label=labels[key], name=key)
            visible_count += len(ids)
    coincident_ids = np.empty(0, int)
    if show_coincident_nodes and validation.get('near_coincident_node_pairs', 0):
        tolerance = validation.get('coordinate_tolerance')
        if tolerance is None: raise ValueError('Saved coincident-node flags lack their tolerance')
        used = np.unique(f)
        pairs = cKDTree(surface.points[used]).query_pairs(tolerance, output_type='ndarray')
        coincident_ids = np.unique(used[pairs])
        if grain_id is not None:
            coincident_ids = coincident_ids[np.isin(coincident_ids, np.unique(f[selected]))]
        if len(coincident_ids):
            plotter.add_points(surface.points[coincident_ids], color=colors['coincident_nodes'],
                point_size=point_size, render_points_as_spheres=True, label='Coincident nodes', name='coincident_nodes')
            visible_count += len(coincident_ids)
    counts['coincident_nodes_displayed'] = len(coincident_ids)
    tet_ids = np.empty(0, int)
    if show_tet_quality and grid is not None:
        q = np.asarray(grid.cell_data['minSICN'])
        eligible = np.ones(len(q), bool) if grain_id is None else grid.cell_data['grain_id'] == grain_id
        invalid = ~np.isfinite(q) | (q <= 0)
        poor = np.isfinite(q) & (q > 0) & (q < tet_quality_threshold)
        tet_ids = np.flatnonzero(eligible & (invalid | poor))
        for key, mask, label in (('tet_quality', poor, 'Poor tets'),
                                  ('invalid_tet_quality', invalid, 'Invalid tet quality')):
            ids = np.flatnonzero(eligible & mask)
            counts[key] = len(ids)
            if len(ids):
                plotter.add_mesh(grid.extract_cells(ids), color=colors[key], opacity=tet_opacity,
                                 show_edges=show_edges, label=label, name=key)
                visible_count += len(ids)
    if show_lines:
        lines, _ = junction_geometry(surface, None if grain_id is None else [grain_id])
        if lines.n_lines:
            plotter.add_mesh(lines, color=line_color, line_width=line_width, name='junction_context')
    if visible_count: plotter.add_legend(size=(.3, .2))
    message = 'Defect / quality-warning view' if visible_count else 'No selected flags to display (not a certification)'
    plotter.add_text(message, font_size=12)
    plotter.add_axes(); plotter.view_isometric()
    plotter.reset_camera(bounds=_poly(surface.points, f[selected]).bounds)
    print('Saved validation:', validation.get('status', 'unknown'))
    print('Counts before display precedence; categories can overlap:', counts)
    for key in ('blockers', 'warnings', 'unchecked'):
        print(key.upper()+':', validation.get(key, []))
    if not validation.get('intersection_check_performed', False):
        print('Intersection scan was not performed in this saved report; absence of flags is not a pass.')
    if show_tet_quality and grid is None:
        print('No tetrahedral grid supplied: tetrahedral quality was not displayed.')
    print('Dedicated physical minimum-separation and short-edge guards are not provided by this view.')
    if show: plotter.show()
    return plotter, dict(surface_triangle_ids=categories, coincident_node_ids=coincident_ids,
                         tet_cell_ids=tet_ids, counts=counts)
