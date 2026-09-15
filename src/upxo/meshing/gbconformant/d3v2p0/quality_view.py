"""Interactive tetrahedral quality-range selection with a linked histogram."""
import numpy as np
import pyvista as pv


def quality_histogram_view(grid, surface=None, *, grain_id=None, initial_range=(0., .05),
                           bins=80, cmap='viridis', color_limits=(0., 1.),
                           selected_opacity=1., show_edges=True, show_context=True,
                           context_opacity=.08, log_counts=True, target=.05,
                           show_lines=False, line_color='red', line_width=3.,
                           show_points=False, point_color='black', point_size=10.,
                           window_size=(1600, 750), show=True):
    """Select whole tetrahedra by inclusive minSICN range; does not modify grid.

    Sliders update on release. Histogram bins/colour limits stay fixed. The
    returned state contains original grid cell IDs for the current selection.
    """
    import matplotlib.pyplot as plt
    from .views import junction_geometry, _overlays
    from .thickness_view import _poly
    q = np.asarray(grid.cell_data['minSICN'])
    if not len(q) or not np.all(np.isfinite(q)):
        raise ValueError('Provide a nonempty mesh with finite minSICN cell values')
    eligible = np.arange(len(q)) if grain_id is None else np.flatnonzero(grid.cell_data['grain_id'] == grain_id)
    if not len(eligible): raise ValueError('No tetrahedra for the selected grain')
    quality = q[eligible]
    extent = (min(0., float(quality.min())), max(1., float(quality.max())))
    if len(initial_range) != 2 or not extent[0] <= initial_range[0] <= initial_range[1] <= extent[1]:
        raise ValueError('initial_range must be increasing and inside the slider range')
    if not isinstance(bins, (int, np.integer)) or bins < 1: raise ValueError('bins must be positive')
    if len(color_limits) != 2 or not np.all(np.isfinite(color_limits)) or color_limits[0] >= color_limits[1]:
        raise ValueError('color_limits must be finite and increasing')
    counts, edges = np.histogram(quality, bins=bins, range=extent)
    plotter = pv.Plotter(shape=(1, 2), notebook=False, window_size=window_size)
    plotter.subplot(0, 0)
    if show_context:
        if surface is not None:
            mask = surface.exterior if grain_id is None else np.any(surface.grain_pairs == grain_id, axis=1)
            if mask.any():
                plotter.add_mesh(_poly(surface.points, surface.triangles[mask]),
                                 color='lightgray', opacity=context_opacity, name='context')
        else:
            plotter.add_mesh(grid.outline(), color='gray', name='context')
    if (show_lines or show_points) and surface is not None:
        lines, points = junction_geometry(surface, None if grain_id is None else [grain_id])
        _overlays(plotter, lines, points, show_lines, line_color, line_width,
                  show_points, point_color, point_size)
    plotter.add_axes()
    plotter.add_text('Release sliders to select whole tetrahedra', position='upper_left', font_size=11)
    plotter.subplot(0, 1)
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge', color='lightgray', label='All eligible tets')
    bars = ax.bar(edges[:-1], np.zeros_like(counts), width=np.diff(edges), align='edge',
                  color='darkorange', label='Selected tets')
    ax.axvline(target, color='red', linestyle='--', label=f'Quality target {target:g}')
    lower_line = ax.axvline(initial_range[0], color='black', linewidth=1)
    upper_line = ax.axvline(initial_range[1], color='black', linewidth=1)
    annotation = ax.text(.02, .98, '', transform=ax.transAxes, va='top', fontsize=10,
                         bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    ax.set(xlabel='Gmsh minSICN', ylabel='Tetrahedron count', title='Quality distribution and selection', xlim=extent)
    if log_counts:
        ax.set_yscale('log'); ax.set_ylim(.8, max(2., float(counts.max())*1.5))
    else: ax.set_ylim(0., max(1., float(counts.max())*1.2))
    ax.legend(loc='lower right', fontsize=9)
    plotter.add_chart(pv.ChartMPL(fig)); plt.close(fig)
    state = dict(lower=float(initial_range[0]), upper=float(initial_range[1]),
                 cell_ids=np.empty(0, int), count=0, eligible_count=len(eligible), widgets=[], last_range=None)

    def select_range(lower, upper):
        lower, upper = float(lower), float(upper)
        if not extent[0] <= lower <= upper <= extent[1]: raise ValueError('Invalid quality range')
        state['lower'], state['upper'] = lower, upper
        if state['last_range'] == (lower, upper): return
        state['last_range'] = (lower, upper)
        selected = (quality >= lower) & (quality <= upper)
        ids = eligible[selected]
        state['cell_ids'], state['count'] = ids, len(ids)
        plotter.subplot(0, 0)
        plotter.renderer.remove_actor('quality_selection', reset_camera=False, render=False)
        if len(ids):
            plotter.add_mesh(grid.extract_cells(ids), scalars='minSICN', cmap=cmap,
                clim=color_limits, opacity=selected_opacity, show_edges=show_edges,
                name='quality_selection', reset_camera=False,
                scalar_bar_args={'title':'minSICN (fixed scale)', 'position_y':.28, 'height':.05})
        text = f'Range [{lower:.5g}, {upper:.5g}]\n{len(ids):,} / {len(eligible):,} tets ({100*len(ids)/len(eligible):.4g}%)'
        plotter.add_text(text if len(ids) else text+'\nNo matching tetrahedra',
                         position=(10, int(window_size[1]*.36)), font_size=11, name='selection_count')
        selected_counts, _ = np.histogram(quality[selected], bins=edges)
        for bar, value in zip(bars, selected_counts): bar.set_height(int(value))
        lower_line.set_xdata([lower, lower]); upper_line.set_xdata([upper, upper])
        annotation.set_text(text)
        for widget, value in zip(state['widgets'], (lower, upper)):
            widget.GetRepresentation().SetValue(value)
        fig.canvas.draw()
        plotter.render()

    select_range(*initial_range)
    plotter.subplot(0, 0)
    def lower_changed(value): select_range(min(float(value), state['upper']), state['upper'])
    def upper_changed(value): select_range(state['lower'], max(float(value), state['lower']))
    for callback, value, title, height in ((lower_changed, state['lower'], 'Lower quality', .17),
                                           (upper_changed, state['upper'], 'Upper quality', .085)):
        widget = plotter.add_slider_widget(callback, extent, value=value, title=title,
            pointa=(.1, height), pointb=(.9, height), interaction_event='end', fmt='%.4f')
        state['widgets'].append(widget)
    plotter.view_isometric(); plotter.reset_camera(bounds=grid.bounds)
    if show: plotter.show()
    return plotter, state, select_range
