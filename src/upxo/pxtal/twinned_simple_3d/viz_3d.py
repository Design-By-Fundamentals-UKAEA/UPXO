"""
viz_3d.py
=========
Visualization utilities for the twinned simple 3D pipeline.
"""

import numpy as np
from typing import Optional, Dict, Tuple


_DEFAULT_ROLE_OPACITY = {
    'non_host':      0.25,
    'host':          0.8,
    'primary_twin':  0.6,
    'secondary_twin': 0.6,
}

# Single solid colours per role — PyVista's add_legend() correctly
# picks up solid colours set via color=, whereas continuous cmaps
# (e.g. 'Blues') produce identical legend swatches for every role.
_DEFAULT_ROLE_COLOR = {
    'non_host':      'lightgray',
    'host':          'steelblue',
    'primary_twin':  'darkorange',
    'secondary_twin': 'crimson',
}


def plot_ipf_slice(
        lgi_3d: np.ndarray,
        all_quats: Dict[int, np.ndarray],
        axis: int = 2,
        slice_idx: Optional[int] = None,
        sample_direction: Tuple[float, float, float] = (0., 0., 1.),
        figsize: Tuple[float, float] = (7., 7.),
        dpi: int = 150,
        title: Optional[str] = None,
):
    """
    Plot an IPF-coloured 2D slice through a 3D grain structure.

    Parameters
    ----------
    lgi_3d : ndarray (nx, ny, nz), int
    all_quats : dict {int: ndarray(4,)}
    axis : int
        0=X, 1=Y, 2=Z.  Default 2.
    slice_idx : int or None
        Defaults to mid-slice if None.
    sample_direction : tuple (3,)
    figsize, dpi : figure size and resolution
    title : str or None
    """
    import matplotlib.pyplot as plt
    from upxo.gsdataops.grid_ops import section_from_3d
    from upxo.viz import ebsdviz

    if slice_idx is None:
        slice_idx = lgi_3d.shape[axis] // 2

    lgi_2d = section_from_3d(lgi_3d, axis=axis, location=slice_idx)
    rgb = ebsdviz.build_ipf_rgb(
        lgi_2d.astype(np.int64), all_quats,
        sample_direction=sample_direction,
    )

    axis_label = ['X', 'Y', 'Z'][axis]
    if title is None:
        title = f'IPF map - {axis_label}-normal slice, index {slice_idx}'

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig, ax


def render_3d(
        lgi_twinned: np.ndarray,
        twin_role: Dict[int, str],
        role_opacity: Optional[Dict[str, float]] = None,
        role_color: Optional[Dict[str, str]] = None,
):
    """
    3D PyVista voxel render with per-role solid colour and opacity.

    Each grain role is rendered as one ``pv.ImageData`` threshold mesh
    with a single solid colour, so ``pvp.add_legend()`` correctly shows
    a distinct colour swatch per role.

    Parameters
    ----------
    lgi_twinned : ndarray (nx, ny, nz), int
    twin_role : dict {int: str}
        Role map: 'host' | 'primary_twin' | 'secondary_twin' | 'non_host'.
    role_opacity : dict {str: float} or None
        Per-role opacity.  Defaults to ``_DEFAULT_ROLE_OPACITY``.
    role_color : dict {str: str} or None
        Per-role PyVista colour name.  Defaults to ``_DEFAULT_ROLE_COLOR``.
        Must be a single named colour (not a colormap) so the legend
        swatch is correct.
    """
    import pyvista as pv

    opacity = role_opacity if role_opacity is not None else _DEFAULT_ROLE_OPACITY
    color   = role_color   if role_color   is not None else _DEFAULT_ROLE_COLOR

    # Defensive: grains in lgi not present in twin_role default to 'non_host'
    all_lgi_gids = set(int(g) for g in np.unique(lgi_twinned) if g > 0)
    unlabelled   = all_lgi_gids - set(twin_role.keys())
    if unlabelled:
        twin_role = dict(twin_role)
        for gid in unlabelled:
            twin_role[gid] = 'non_host'

    pvp = pv.Plotter()

    for role, alpha in opacity.items():
        gids_role = [g for g, r in twin_role.items() if r == role]
        if not gids_role:
            continue

        masked = np.where(np.isin(lgi_twinned, gids_role), lgi_twinned, 0)

        grid = pv.ImageData()
        grid.dimensions = np.array(masked.shape) + 1
        grid.origin     = (0, 0, 0)
        grid.spacing    = (1, 1, 1)
        grid.cell_data['lgi'] = masked.flatten(order='F')
        grid_thresh = grid.threshold(0.5, scalars='lgi')

        pvp.add_mesh(
            grid_thresh,
            opacity=alpha,
            color=color.get(role, 'white'),   # single colour -> correct legend swatch
            show_edges=False,
            label=role,
        )

    pvp.add_legend()
    pvp.show()
