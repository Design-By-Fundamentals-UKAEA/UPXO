"""
EBSD visualisation helpers for UPXO.

All functions accept an :class:`upxo.interfaces.defdap.ebsd_reader.EBSDReader`
instance (or plain NumPy arrays where noted) and produce Matplotlib figures.

Import
------
import upxo.viz.ebsdviz as ebsdviz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gb_mask(lfi: np.ndarray) -> np.ndarray:
    """Return a boolean (ny, nx) array that is True at grain-boundary pixels.

    Boundaries are detected via Sobel edge detection on the integer label field.

    Parameters
    ----------
    lfi : np.ndarray, shape (ny, nx)
        Integer grain-label field.  Non-positive values are treated as
        non-grain pixels (boundaries / non-indexed).
    """
    lfi_f = lfi.astype(float)
    sx = sobel(lfi_f, axis=1)
    sy = sobel(lfi_f, axis=0)
    return np.hypot(sx, sy) > 0


def _overlay_boundaries(ax, gb_mask: np.ndarray,
                        color=(0, 0, 0), alpha: float = 1.0) -> None:
    """Overlay *gb_mask* as a solid-colour layer on *ax*."""
    rgba = np.zeros((*gb_mask.shape, 4), dtype=np.float32)
    rgba[gb_mask] = [*color, alpha]
    ax.imshow(rgba, interpolation='nearest')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_grain_labels(rdr,
                      show_boundaries: bool = True,
                      boundary_color=(0, 0, 0),
                      cmap: str = 'nipy_spectral',
                      figsize: tuple = (7, 6),
                      dpi: int = 100,
                      title: str = 'Grain label field',
                      ax=None):
    """Plot the grain label field (lfi_ebsd) with optional boundary overlay.

    Parameters
    ----------
    rdr : EBSDReader
        Loaded EBSD reader with ``lfi_ebsd`` populated.
    show_boundaries : bool
        Whether to overlay grain boundaries in black.
    boundary_color : tuple
        RGB tuple for boundary colour (values 0–1).
    cmap : str
        Matplotlib colormap for grain IDs.
    figsize, dpi : tuple, int
        Figure size and resolution.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes or None
        If provided, draw onto this axes; otherwise create a new figure.

    Returns
    -------
    fig, ax
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    lfi = rdr.lfi_ebsd.astype(float)
    lfi[lfi <= 0] = np.nan
    im = ax.imshow(lfi, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Grain ID', fraction=0.046, pad=0.04)

    if show_boundaries:
        _overlay_boundaries(ax, _gb_mask(rdr.lfi_ebsd), color=boundary_color)

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_euler_maps(rdr,
                    show_boundaries: bool = True,
                    boundary_color=(0, 0, 0),
                    cmaps=('hsv', 'viridis', 'hsv'),
                    figsize: tuple = (15, 5),
                    dpi: int = 100,
                    suptitle: str = 'Bunge Euler angle maps'):
    """Plot phi1, Phi, phi2 Euler maps side-by-side.

    Parameters
    ----------
    rdr : EBSDReader
        Loaded EBSD reader with ``euler_ebsd`` and ``lfi_ebsd`` populated.
    show_boundaries : bool
        Overlay grain boundaries on every panel.
    cmaps : tuple of str
        One colormap per channel (phi1, Phi, phi2).
    figsize, dpi : tuple, int
        Figure size and resolution.

    Returns
    -------
    fig, axes
    """
    euler_deg = np.degrees(rdr.euler_ebsd)
    channel_labels = ['phi1 (°)', 'Phi (°)', 'phi2 (°)']
    gb = _gb_mask(rdr.lfi_ebsd) if show_boundaries else None

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi,
                              sharex=True, sharey=True)
    for i, (ax, label, cmap) in enumerate(zip(axes, channel_labels, cmaps)):
        data = euler_deg[:, :, i].copy()
        data[rdr.lfi_ebsd <= 0] = np.nan
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='degrees', fraction=0.046, pad=0.04)
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if show_boundaries:
            _overlay_boundaries(ax, gb, color=boundary_color)

    plt.suptitle(suptitle, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_grain_size_histogram(rdr,
                               figsize: tuple = (7, 4),
                               dpi: int = 100,
                               bins: int = 40,
                               color: str = 'steelblue'):
    """Plot a grain area histogram in physical units (µm²).

    Parameters
    ----------
    rdr : EBSDReader
        Loaded EBSD reader with ``lfi_ebsd`` and ``step_size`` populated.

    Returns
    -------
    fig, ax
    """
    grain_ids = np.unique(rdr.lfi_ebsd[rdr.lfi_ebsd >= 1])
    grain_sizes = np.array([np.sum(rdr.lfi_ebsd == g) for g in grain_ids])
    areas = grain_sizes * rdr.step_size ** 2

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(areas, bins=bins, color=color, edgecolor='k', alpha=0.8)
    ax.set_xlabel('Grain area (µm²)')
    ax.set_ylabel('Count')
    ax.set_title(f'Grain size distribution  (n={len(grain_sizes)}, '
                 f'mean={areas.mean():.1f} µm²)')
    plt.tight_layout()
    plt.show()

    print(f'Grain area stats (µm²):  min={areas.min():.2f}  '
          f'max={areas.max():.2f}  mean={areas.mean():.2f}  '
          f'median={np.median(areas):.2f}')
    return fig, ax


def plot_grain_structure_with_boundaries(rdr,
                                          show_euler_panel: bool = True,
                                          euler_channel: int = 0,
                                          boundary_color=(0, 0, 0),
                                          figsize: tuple = (16, 7),
                                          dpi: int = 100,
                                          suptitle: str = 'Grain structure with grain boundaries'):
    """Two-panel figure: grain label map + Euler background, both with boundaries.

    Parameters
    ----------
    rdr : EBSDReader
        Loaded EBSD reader.
    show_euler_panel : bool
        If False, only the grain-label panel is shown (single axes).
    euler_channel : int
        Which Euler channel (0=phi1, 1=Phi, 2=phi2) to use as orientation
        background in the right panel.
    boundary_color : tuple
        RGB boundary overlay colour (values 0–1).
    figsize, dpi : tuple, int
        Figure size and resolution.
    suptitle : str
        Overall figure title.

    Returns
    -------
    fig, axes   (axes is a single Axes if show_euler_panel=False)
    """
    euler_channel_names = {0: 'phi1', 1: 'Phi', 2: 'phi2'}
    n_panels = 2 if show_euler_panel else 1
    fig, axes_arr = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi,
                                  sharex=True, sharey=True)
    if n_panels == 1:
        axes_arr = [axes_arr]

    gb = _gb_mask(rdr.lfi_ebsd)

    # --- left panel: grain label field ---
    ax0 = axes_arr[0]
    lfi_disp = rdr.lfi_ebsd.astype(float)
    lfi_disp[lfi_disp <= 0] = np.nan
    im0 = ax0.imshow(lfi_disp, cmap='nipy_spectral', interpolation='nearest')
    plt.colorbar(im0, ax=ax0, label='Grain ID', fraction=0.046, pad=0.04)
    ax0.set_title('Grain label field + boundaries')
    ax0.set_xlabel('x (pixels)')
    ax0.set_ylabel('y (pixels)')
    _overlay_boundaries(ax0, gb, color=boundary_color)

    # --- right panel: Euler orientation background ---
    if show_euler_panel:
        ax1 = axes_arr[1]
        euler_bg = np.degrees(rdr.euler_ebsd[:, :, euler_channel]).copy()
        euler_bg[rdr.lfi_ebsd <= 0] = np.nan
        im1 = ax1.imshow(euler_bg, cmap='hsv', interpolation='nearest',
                          vmin=0, vmax=360)
        plt.colorbar(im1, ax=ax1,
                     label=f'{euler_channel_names.get(euler_channel, "Euler")} (°)',
                     fraction=0.046, pad=0.04)
        ax1.set_title(f'{euler_channel_names.get(euler_channel, "Euler")} map + boundaries')
        ax1.set_xlabel('x (pixels)')
        ax1.set_ylabel('y (pixels)')
        _overlay_boundaries(ax1, gb, color=boundary_color)

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()

    n_gb_px = int(gb.sum())
    print(f'Grain-boundary pixels : {n_gb_px}  '
          f'({100 * n_gb_px / gb.size:.2f}% of map)')

    return fig, axes_arr if n_panels > 1 else axes_arr[0]
