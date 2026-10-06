"""Clipped, crinkle view of tet elements filtered by dihedral angle.

A plane widget clips the mesh with whole tets kept (crinkle clip). Two sliders
filter the shown tets: the smallest dihedral angle at or below one value, and
the largest dihedral angle at or above another. At their defaults (180 and 0)
every tet is shown. Tets are coloured by their smallest dihedral angle.

launch_tet_quality_view opens the window in a separate process, so a notebook
does not wait for it. Command line:
    python -m upxo.meshing.gbconformant.d3v2p0.tet_quality_view TETS.vtu [--title T]
"""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def tet_angle_arrays(grid):
    """Per-tet smallest and largest dihedral angles (degrees) of a tet UnstructuredGrid."""
    import pyvista as pv
    from .tet_angles import dihedral_angles
    tets = np.asarray(grid.cells_dict[pv.CellType.TETRA])
    angles = dihedral_angles(np.asarray(grid.points), tets)
    return angles.min(axis=1), angles.max(axis=1)


def shown_cells(centres, smallest, largest, origin, normal, min_upper=180., max_lower=0.):
    """Mask of tets on the kept side of the plane (centre behind it, as a crinkle
    clip keeps whole cells) that pass both angle filters."""
    normal = np.asarray(normal, float)
    side = (np.asarray(centres) - np.asarray(origin, float)) @ normal <= 0
    return side & (np.asarray(smallest) <= min_upper) & (np.asarray(largest) >= max_lower)


def show_tet_quality(vtu, *, title='Tet quality', clim=(0., 70.5)):
    """Open the interactive clip-and-filter window (blocks until closed)."""
    import pyvista as pv
    grid = pv.read(str(vtu)) if isinstance(vtu, (str, Path)) else vtu
    smallest, largest = tet_angle_arrays(grid)
    grid.cell_data['min_dihedral'] = smallest
    grid.cell_data['max_dihedral'] = largest
    centres = grid.cell_centers().points
    state = dict(origin=np.asarray(grid.center), normal=np.array([1., 0., 0.]), min_upper=180., max_lower=0.)
    plotter = pv.Plotter(window_size=(1400, 1000), title=title)
    plotter.add_mesh(grid.outline(), color='grey')

    def update():
        mask = shown_cells(centres, smallest, largest, state['origin'], state['normal'],
                           state['min_upper'], state['max_lower'])
        if mask.any():
            part = grid.extract_cells(np.flatnonzero(mask))
            plotter.add_mesh(part, scalars='min_dihedral', cmap='viridis', clim=clim, show_edges=True,
                             edge_color='black', line_width=.2, name='tets',
                             scalar_bar_args=dict(title='smallest dihedral (deg)'))
        else:
            plotter.remove_actor('tets')
        plotter.add_text(f'{int(mask.sum())} of {len(mask)} tets shown', position='upper_left',
                         font_size=10, name='count')

    def on_plane(normal, origin):
        state['normal'], state['origin'] = np.asarray(normal), np.asarray(origin)
        update()

    def on_min(value):
        state['min_upper'] = float(value)
        update()

    def on_max(value):
        state['max_lower'] = float(value)
        update()

    plotter.add_plane_widget(on_plane, normal='x', origin=state['origin'])
    plotter.add_slider_widget(on_min, rng=[0., 180.], value=180., title='show smallest dihedral <=',
                              pointa=(.02, .1), pointb=(.32, .1), style='modern')
    plotter.add_slider_widget(on_max, rng=[0., 180.], value=0., title='show largest dihedral >=',
                              pointa=(.36, .1), pointb=(.66, .1), style='modern')
    update()
    plotter.add_axes()
    plotter.show()
    return plotter


def launch_tet_quality_view(vtu_path, *, title='Tet quality'):
    """Open show_tet_quality in a separate process and return the process."""
    vtu_path = Path(vtu_path)
    if not vtu_path.is_file():
        raise FileNotFoundError(vtu_path)
    return subprocess.Popen([sys.executable, '-m', __name__, str(vtu_path), '--title', title])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Clipped crinkle view of tets filtered by dihedral angle.')
    parser.add_argument('vtu')
    parser.add_argument('--title', default='Tet quality')
    args = parser.parse_args(argv)
    show_tet_quality(args.vtu, title=args.title)


if __name__ == '__main__':
    main()
