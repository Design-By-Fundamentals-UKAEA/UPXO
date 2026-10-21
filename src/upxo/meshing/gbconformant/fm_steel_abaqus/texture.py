"""Pole figures of the raw features and of the meshed features, and their difference.

Orientations are Bunge-Euler angles in degrees, one per feature of the chosen
level, replicated through the selected sample symmetry exactly as the
hierarchy pole-figure views do. The difference is after minus before, each
field optionally divided by its own maximum first.
"""
from dataclasses import dataclass, field
import numpy as np

from upxo.viz.xphy.pole_figure import (
    PoleFigure, apply_sample_symmetry, compute_sample_symmetry_group)

CRYSTAL_SYMMETRIES = ('cubic', 'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic', 'triclinic')
PLOT_MODES = ('density', 'scatter', 'hybrid')


@dataclass
class TextureSettings:
    pole_family: object = '100'          # '100', '110', '111', 'hkl' text, or [h, k, l]
    crystal_symmetry: str = 'cubic'
    apply_sample_symmetry: bool = True
    use_rd: bool = True
    use_td: bool = True
    use_nd: bool = True
    half_width_deg: float = 7.5
    grid_points: object = 'auto'
    unit_normalize: bool = True
    plot_mode: str = 'density'
    projection: str = 'equal_area'
    hemisphere: str = 'upper'
    cmap: str = 'viridis'
    difference_cmap: str = 'RdBu_r'
    levels: int = 10
    contour_scale: str = 'linear'
    colorbar_decimals: int | None = None
    axis_labels: tuple = (None, None, None)

    def validate(self):
        if self.crystal_symmetry not in CRYSTAL_SYMMETRIES:
            raise ValueError(f'crystal_symmetry must be one of {CRYSTAL_SYMMETRIES}')
        if self.plot_mode not in PLOT_MODES:
            raise ValueError(f'plot_mode must be one of {PLOT_MODES}')
        if not self.half_width_deg > 0:
            raise ValueError('half_width_deg must be > 0')
        if self.grid_points != 'auto' and int(self.grid_points) < 10:
            raise ValueError("grid_points must be 'auto' or an integer >= 10")


def orientation_table(hierarchy, labels=None):
    """(labels, eulers) of the features in ``labels`` (default: all), ascending."""
    chosen = sorted(hierarchy.features if labels is None else labels)
    missing = [label for label in chosen if label not in hierarchy.features]
    if missing:
        raise ValueError(f'Labels without an orientation: {missing[:5]}')
    eulers = np.array([hierarchy.features[label].euler for label in chosen], dtype=float)
    return np.array(chosen, dtype=np.int64), eulers.reshape(-1, 3)


def _replicate(eulers, settings):
    group = compute_sample_symmetry_group(
        settings.apply_sample_symmetry, settings.use_rd, settings.use_td, settings.use_nd)
    if len(group) > 1 and len(eulers):
        return apply_sample_symmetry(eulers, 'euler_deg', group)
    return eulers


def pole_figure(eulers, settings):
    """:class:`PoleFigure` of ``eulers`` with the sample symmetry replicated."""
    settings.validate()
    if not len(eulers):
        raise ValueError('No orientations to plot.')
    return PoleFigure(_replicate(np.asarray(eulers, dtype=float), settings),
                      convention='euler_deg', symmetry=settings.crystal_symmetry)


def resolved_grid_points(settings):
    """One integer grid size for both sides, so their fields line up point by point."""
    if settings.grid_points == 'auto':
        return PoleFigure.auto_grid_points(settings.half_width_deg)
    return int(settings.grid_points)


def density_difference(eulers_before, eulers_after, settings):
    """After-minus-before MUD density difference on a shared grid.

    Returns ``Xi``, ``Yi``, ``zi_diff`` and ``iqr`` (interquartile range of the
    difference field; a single-number summary, not a verdict).
    """
    before = pole_figure(eulers_before, settings)
    after = pole_figure(eulers_after, settings)
    return after.compute_density_difference(
        before, pole_family=settings.pole_family, grid_points=resolved_grid_points(settings),
        half_width_deg=settings.half_width_deg, unit_normalize=settings.unit_normalize)
