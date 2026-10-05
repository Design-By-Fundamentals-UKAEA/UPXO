"""Element-quality distribution plots and a grain-boundary surface view."""
import subprocess
import sys
from pathlib import Path
import numpy as np


def _worst_band(values, fraction, best, worst='low'):
    """Elements within ``fraction`` of the way from the worst value towards the
    best possible value.

    worst='low' (smallest angles, quality): band [lowest, lowest + fraction *
    (best - lowest)]. worst='high' (largest angles): band [highest - fraction *
    (highest - best), highest]. Returns (band_edge, count, worst_value).
    """
    values = np.asarray(values, float)
    if worst == 'low':
        extreme = float(values.min())
        edge = extreme + fraction * (best - extreme)
        return edge, int(np.sum(values <= edge)), extreme
    extreme = float(values.max())
    edge = extreme - fraction * (extreme - best)
    return edge, int(np.sum(values >= edge)), extreme


def _histogram(values, title, xlabel, unit, value_range, fraction, best, worst, limit=None, limit_label=None,
               groups=None, bins=60, color=None, element='elements'):
    """One histogram figure with the worst value and the worst band marked."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    if groups:
        for name, mask in groups:
            ax.hist(values[mask], bins=bins, range=value_range, alpha=.6, label=name)
    else:
        ax.hist(values, bins=bins, range=value_range, color=color)
    edge, count, extreme = _worst_band(values, fraction, best, worst)
    lo, hi = (extreme, edge) if worst == 'low' else (edge, extreme)
    ax.axvspan(lo, hi, color='tab:red', alpha=.15,
               label=f'[{lo:.3g}, {hi:.3g}]{unit}: {count} {element}')
    ax.axvline(extreme, color='tab:red', lw=1.5,
               label=f"{'lowest' if worst == 'low' else 'highest'}: {extreme:.3g}{unit} (best possible {best:.4g}{unit})")
    if limit is not None:
        ax.axvline(limit, color='k', ls='--', lw=1, label=limit_label)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(element)
    ax.set_title(title, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    return fig, dict(worst=extreme, best_possible=float(best), fraction=float(fraction),
                     band=[float(lo), float(hi)], count=count)


def triangle_quality_distribution(points, triangles, exterior=None, min_angle_limit=30., fraction=.05,
                                  bins=60):
    """Three separate histograms of surface-triangle quality: smallest angle,
    largest angle (degrees) and the 0-1 shape quality 4*sqrt(3)*area/sum(edge^2).

    Each marks the worst value in the mesh and counts the triangles within
    ``fraction`` of the way from it towards the best possible value (60 deg
    for both angles, 1 for quality): [lowest, lowest + fraction*(best -
    lowest)] for smallest angle and quality, [highest - fraction*(highest -
    best), highest] for largest angle. exterior, if given, splits internal
    grain boundaries from RVE faces. Returns (figures, stats).
    """
    from .surface_angles import triangle_angles
    from .surface_quality import triangle_quality
    if not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0, 1]')
    p, f = np.asarray(points, float), np.asarray(triangles)
    angles = triangle_angles(p, f)
    smallest, largest = angles.min(axis=1), angles.max(axis=1)
    quality = triangle_quality(p, f)
    groups = None
    if exterior is not None:
        ext = np.asarray(exterior, bool)
        groups = [('grain boundaries', ~ext), ('RVE faces', ext)]
    n = len(f)
    figures, stats = [], dict(triangles=int(n), below_min_angle_limit=int(np.sum(smallest < min_angle_limit)))
    for key, values, title, xlabel, unit, rng, best, worst, limit in (
            ('smallest_angle', smallest, 'Surface triangles: smallest angle', 'smallest angle (deg)', ' deg',
             (0, 60), 60., 'low', min_angle_limit),
            ('largest_angle', largest, 'Surface triangles: largest angle', 'largest angle (deg)', ' deg',
             (60, 180), 60., 'high', None),
            ('shape_quality', quality, 'Surface triangles: shape quality', 'shape quality (0-1)', '',
             (0, 1), 1., 'low', None)):
        fig, st = _histogram(values, f'{title} ({n} triangles)', xlabel, unit, rng, fraction, best, worst,
                             limit=limit, limit_label=None if limit is None else f'limit {limit:g} deg',
                             groups=groups, bins=bins, element='triangles')
        figures.append(fig)
        stats[key] = st
    return figures, stats


def tet_quality_distribution(points, tetrahedra, quality=None, min_dihedral_limit=15., max_dihedral_limit=150.,
                             fraction=.05, bins=60):
    """Separate histograms of tet quality: smallest and largest dihedral angle
    (degrees) and, if given, Gmsh minSICN. Each marks the worst value and counts
    the tets within ``fraction`` of the way from it towards the best possible
    value (70.53 deg, the regular tet, for both dihedral angles; 1 for
    minSICN). Returns (figures, stats)."""
    from .tet_angles import dihedral_angles
    if not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0, 1]')
    regular = float(np.degrees(np.arccos(1 / 3)))
    a = dihedral_angles(points, tetrahedra)
    smallest, largest = a.min(axis=1), a.max(axis=1)
    n = len(a)
    items = [('smallest_dihedral', smallest, 'Tetrahedra: smallest dihedral angle', 'smallest dihedral (deg)',
              ' deg', (0, 75), regular, 'low', min_dihedral_limit, '#4878CF'),
             ('largest_dihedral', largest, 'Tetrahedra: largest dihedral angle', 'largest dihedral (deg)',
              ' deg', (70, 180), regular, 'high', max_dihedral_limit, '#D65F5F')]
    if quality is not None:
        items.append(('minSICN', np.asarray(quality, float), 'Tetrahedra: Gmsh minSICN', 'minSICN', '',
                      (0, 1), 1., 'low', None, '#6ACC65'))
    figures, stats = [], dict(tetrahedra=int(n), below_min_limit=int(np.sum(smallest < min_dihedral_limit)),
                              above_max_limit=int(np.sum(largest > max_dihedral_limit)))
    for key, values, title, xlabel, unit, rng, best, worst, limit, color in items:
        fig, st = _histogram(values, f'{title} ({n} tets)', xlabel, unit, rng, fraction, best, worst,
                             limit=limit, limit_label=None if limit is None else f'limit {limit:g} deg',
                             bins=bins, color=color, element='tets')
        figures.append(fig)
        stats[key] = st
    return figures, stats


def show_surface(vtp, *, show_edges=True, internal_only=True, color_by='grain', title='Grain-boundary surface'):
    """Interactive view of a saved grain-boundary surface (blocks until closed).

    show_edges: draw triangle edges. internal_only: hide RVE-face triangles.
    color_by: 'grain' (grain_a), 'min_angle' (smallest triangle angle) or 'none'.
    """
    import pyvista as pv
    from .surface_angles import triangle_angles
    if color_by not in ('grain', 'min_angle', 'none'):
        raise ValueError("color_by must be 'grain', 'min_angle' or 'none'")
    s = pv.read(str(vtp))
    if internal_only and 'is_rve_cap' in s.cell_data.keys():
        s = s.extract_cells(np.flatnonzero(np.asarray(s.cell_data['is_rve_cap']) == 0)).extract_surface()
    pl = pv.Plotter(window_size=(1400, 1000), title=title)
    kw = dict(show_edges=show_edges, edge_color='black', line_width=.3)
    if color_by == 'grain':
        g = np.asarray(s.cell_data['grain_a'])
        s.cell_data['colour'] = np.random.default_rng(0).permutation(int(g.max()) + 1)[g].astype(float)
        pl.add_mesh(s, scalars='colour', cmap='tab20', show_scalar_bar=False, **kw)
    elif color_by == 'min_angle':
        f = np.asarray(s.faces).reshape(-1, 4)[:, 1:]
        s.cell_data['smallest angle'] = triangle_angles(np.asarray(s.points), f).min(axis=1)
        pl.add_mesh(s, scalars='smallest angle', cmap='viridis', clim=(0, 60), **kw)
    else:
        pl.add_mesh(s, color='lightgrey', **kw)
    pl.add_axes()
    pl.show()


def launch_surface_view(vtp, *, show_edges=True, internal_only=True, color_by='grain',
                        title='Grain-boundary surface'):
    """Open show_surface in a separate process and return the process."""
    vtp = Path(vtp)
    if not vtp.is_file():
        raise FileNotFoundError(vtp)
    cmd = [sys.executable, '-m', __name__, str(vtp), '--color-by', color_by, '--title', title]
    if not show_edges:
        cmd.append('--no-edges')
    if not internal_only:
        cmd.append('--with-rve-faces')
    return subprocess.Popen(cmd)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='View a saved grain-boundary surface.')
    ap.add_argument('vtp')
    ap.add_argument('--color-by', default='grain')
    ap.add_argument('--title', default='Grain-boundary surface')
    ap.add_argument('--no-edges', action='store_true')
    ap.add_argument('--with-rve-faces', action='store_true')
    a = ap.parse_args(argv)
    show_surface(a.vtp, show_edges=not a.no_edges, internal_only=not a.with_rve_faces,
                 color_by=a.color_by, title=a.title)


if __name__ == '__main__':
    main()
