"""
Visual Physics (xphy) subpackage for UPXO.
Exposes advanced pole figure plotting utilities.
"""

from upxo.viz.xphy.pole_figure import (
    PoleFigure,
    plot_pole_figure_from_3d,
    plot_components,
    plot_variants
)

__all__ = [
    'PoleFigure',
    'plot_pole_figure_from_3d',
    'plot_components',
    'plot_variants'
]

