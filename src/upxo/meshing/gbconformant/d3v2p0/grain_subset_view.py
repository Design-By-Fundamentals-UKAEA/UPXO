"""View a chosen subset of grains of a saved closed RVE surface.

The view can open in a separate process (launch_grain_subset_view), so a
notebook continues without waiting for the window to close. Selection helpers
pick grains by volume percentile.

Command line:
    python -m upxo.meshing.gbconformant.d3v2p0.grain_subset_view SURFACE.vtp IDS.json [--title T] [--screenshot PNG]
IDS.json holds a list of grain IDs.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


def grain_subset_mesh(surface, grain_ids):
    """Triangles bounding the given grains, as a PolyData with a 'grain' label.

    surface: PolyData with cell arrays grain_a, grain_b and is_rve_cap (as saved
    by the CM pipeline). A triangle between two selected grains appears once
    for each, so every selected grain is a closed shell.
    """
    import pyvista as pv
    ids = np.unique(np.asarray(list(grain_ids)))
    ga = np.asarray(surface.cell_data['grain_a'])
    gb = np.asarray(surface.cell_data['grain_b'])
    cap = np.asarray(surface.cell_data['is_rve_cap']).astype(bool)
    in_a, in_b = np.isin(ga, ids), np.isin(gb, ids) & ~cap
    cells = np.concatenate((np.flatnonzero(in_a), np.flatnonzero(in_b)))
    labels = np.concatenate((ga[in_a], gb[in_b]))
    faces = np.asarray(surface.faces).reshape(-1, 4)[cells]
    mesh = pv.PolyData(np.asarray(surface.points), faces.ravel())
    mesh.cell_data['grain'] = labels
    return mesh


def grains_above_percentile(volumes, percentile=90.):
    """Grain IDs whose volume is above the given percentile of all grain volumes.

    volumes: {grain id: volume} (e.g. a tet report's per_grain volumes or a
    surface report's enclosed_grain_volumes; values may be dicts with 'volume').
    """
    if not np.isfinite(percentile) or not 0 <= percentile < 100:
        raise ValueError('percentile must lie in [0, 100)')
    ids = np.array([int(k) for k in volumes])
    v = np.array([float(x['volume'] if isinstance(x, dict) else x) for x in volumes.values()])
    if not len(v):
        return []
    return sorted(ids[v > np.percentile(v, percentile)].tolist())


def show_grain_subset(surface, grain_ids, *, title='Grain subset', note=None, rve_dimensions=None,
                      screenshot=None, seed=0):
    """Open an interactive window (blocks until it is closed)."""
    import pyvista as pv
    if isinstance(surface, (str, Path)):
        surface = pv.read(str(surface))
    ids = np.unique(np.asarray(list(grain_ids)))
    mesh = grain_subset_mesh(surface, ids)
    order = np.random.default_rng(seed).permutation(len(ids))         # scatter neighbouring colours
    colour = dict(zip(ids.tolist(), order.tolist()))
    mesh.cell_data['colour'] = np.array([colour[int(g)] for g in mesh.cell_data['grain']], float)
    bounds = surface.bounds if rve_dimensions is None else (0, rve_dimensions[0], 0, rve_dimensions[1],
                                                           0, rve_dimensions[2])
    plotter = pv.Plotter(window_size=(1400, 1000), title=title)
    if mesh.n_cells:
        plotter.add_mesh(mesh, scalars='colour', cmap='tab20', show_edges=True, edge_color='black',
                         line_width=.3, show_scalar_bar=False)
    plotter.add_mesh(pv.Box(bounds=bounds), style='wireframe', color='grey', line_width=2)
    plotter.add_text(note or f'{len(ids)} grains', font_size=11)
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.show(screenshot=None if screenshot is None else str(screenshot))
    return plotter


def launch_grain_subset_view(surface_path, grain_ids, *, title='Grain subset', note=None, screenshot=None):
    """Open show_grain_subset in a separate process and return the process.

    The caller does not wait; closing the window ends the process.
    """
    surface_path = Path(surface_path)
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    ids = sorted(int(g) for g in grain_ids)
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', prefix='grain_subset_', delete=False)
    with handle:
        json.dump(dict(ids=ids, note=note), handle)
    command = [sys.executable, '-m', __name__, str(surface_path), handle.name, '--title', title]
    if screenshot is not None:
        command += ['--screenshot', str(screenshot)]
    return subprocess.Popen(command)


def main(argv=None):
    parser = argparse.ArgumentParser(description='View a subset of grains of a saved RVE surface.')
    parser.add_argument('surface')
    parser.add_argument('ids')
    parser.add_argument('--title', default='Grain subset')
    parser.add_argument('--screenshot')
    args = parser.parse_args(argv)
    data = json.loads(Path(args.ids).read_text())
    ids, note = (data, None) if isinstance(data, list) else (data['ids'], data.get('note'))
    show_grain_subset(args.surface, ids, title=args.title, note=note, screenshot=args.screenshot)


if __name__ == '__main__':
    main()
