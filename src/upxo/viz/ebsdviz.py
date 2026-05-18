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


def build_ipf_rgb(
        lfi: np.ndarray,
        quats_dict: dict,
        sample_direction=(0., 0., 1.),
        grey: tuple = (0.75, 0.75, 0.75),
) -> np.ndarray:
    """
    Build a pixel-level IPF RGB image from an integer label field and a
    grain-quaternion mapping.

    Parameters
    ----------
    lfi : ndarray (ny, nx), int
        Pixel label field; each value is a grain ID.
    quats_dict : dict {int gid -> ndarray (4,)}
        Quaternion ``(w, x, y, z)`` for each oriented grain.
    sample_direction : array-like (3,)
        Sample reference direction for IPF colouring. Default [001] (ND).
    grey : tuple of 3 floats
        RGB colour for pixels whose grain has no assigned orientation.

    Returns
    -------
    rgb : ndarray (ny, nx, 3), float32 in [0, 1]
    """
    sd = np.asarray(sample_direction, dtype=np.float64)
    sd /= np.linalg.norm(sd) + 1e-12

    max_gid = int(lfi.max())
    q_lut   = np.zeros((max_gid + 1, 4), dtype=np.float64)
    q_lut[:, 0] = 1.0                        # identity quaternion default
    valid   = np.zeros(max_gid + 1, dtype=bool)
    for gid, q in quats_dict.items():
        g = int(gid)
        if 0 <= g <= max_gid:
            q_lut[g] = q
            valid[g] = True

    qmap = q_lut[lfi]                        # (ny, nx, 4)
    w = qmap[..., 0]; x = qmap[..., 1]
    y = qmap[..., 2]; z = qmap[..., 3]

    # v = |R(q).T @ sd|  (standard IPF colour formula)
    # R(q) columns:
    #   col0 = [1-2(y²+z²),  2(xy+wz),   2(xz-wy)]
    #   col1 = [2(xy-wz),    1-2(x²+z²), 2(yz+wx)]
    #   col2 = [2(xz+wy),    2(yz-wx),   1-2(x²+y²)]
    s0, s1, s2 = sd
    v0 = (1 - 2*(y*y + z*z))*s0 + 2*(x*y + w*z)*s1 + 2*(x*z - w*y)*s2
    v1 = 2*(x*y - w*z)*s0 + (1 - 2*(x*x + z*z))*s1 + 2*(y*z + w*x)*s2
    v2 = 2*(x*z + w*y)*s0 + 2*(y*z - w*x)*s1 + (1 - 2*(x*x + y*y))*s2

    v   = np.stack([np.abs(v0), np.abs(v1), np.abs(v2)], axis=-1)
    vmx = v.max(axis=-1, keepdims=True)
    vmx = np.where(vmx == 0, 1.0, vmx)
    rgb = np.clip(v / vmx, 0.0, 1.0).astype(np.float32)
    rgb[~valid[lfi]] = np.asarray(grey, dtype=np.float32)
    return rgb


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


# ---------------------------------------------------------------------------
# MDF visualisation
# ---------------------------------------------------------------------------

def plot_mdf(
        mdf: dict,
        peaks: dict,
        figsize: tuple[float, float] = (9, 4),
        angle_max: float = 65.0,
) -> tuple:
    """
    Plot a grain-boundary MDF histogram with KDE overlay, CSL reference lines,
    and auto-detected peak annotations.

    Parameters
    ----------
    mdf : dict
        Output of ``crystal_orientation.compute_mdf_from_quats()``.
    peaks : dict
        Output of ``crystal_orientation.detect_mdf_peaks()``.
    figsize : (float, float)
        Figure size. Default (9, 4).
    angle_max : float
        Upper x-axis limit in degrees. Default 65.

    Returns
    -------
    fig, ax
    """
    centers    = mdf['hist_bin_centers']
    density    = mdf['hist_density']
    bw         = float(centers[1] - centers[0])
    theta_fine = peaks['theta_fine']
    kde_vals   = peaks['kde_vals']
    csl        = peaks['csl']
    csl_tol    = peaks['csl_tol']

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(centers, density, width=bw, color='steelblue', edgecolor='k',
           linewidth=0.3, alpha=0.55, label='MDF (histogram)')
    ax.plot(theta_fine, kde_vals, color='navy', lw=1.8, label='KDE')

    for csl_label, ang in csl.items():
        if ang <= angle_max:
            ax.axvline(ang, color='crimson', lw=0.9, ls='--', alpha=0.7)
            ax.text(ang + 0.3, ax.get_ylim()[1] * 0.92, csl_label,
                    fontsize=7, color='crimson', rotation=90, va='top')

    for pi, angle in zip(peaks['peak_indices'], peaks['peak_angles']):
        ax.annotate(
            f'{angle:.1f}°',
            xy=(angle, density[pi]),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.8),
        )

    ax.set_xlabel('Misorientation angle (°)')
    ax.set_ylabel('Probability density (°⁻¹)')
    ax.set_title(f'MDF with CSL annotations + KDE — {mdf["n_pairs"]} grain-boundary pairs')
    ax.set_xlim(0, angle_max)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


from upxo.interfaces.user_inputs.nbWidgets import (  # noqa: E402
    mdf_peak_selector,
    selectProps_twinGS,
    selectProps_twinGS as make_property_stats_widgets,  # legacy alias
    readProps_twinGS,
    readProps_twinGS as read_property_stats_widgets,    # legacy alias
)


def plot_mdf_selected(
        mdf: dict,
        peaks: dict,
        selected_peaks: dict,
        figsize: tuple[float, float] = (9, 4),
        angle_max: float = 65.0,
) -> tuple:
    """
    Replot the MDF histogram + KDE with only the user-selected peaks
    highlighted in colour (unselected bins are greyed out).

    Parameters
    ----------
    mdf : dict
        Output of ``crystal_orientation.compute_mdf_from_quats()``.
    peaks : dict
        Output of ``crystal_orientation.detect_mdf_peaks()``.
    selected_peaks : dict
        Output of ``mdf_peak_selector()`` after user confirmation.
        Keys: ``'angles'`` and ``'indices'``.
    figsize : (float, float)
    angle_max : float

    Returns
    -------
    fig, ax
    """
    centers    = mdf['hist_bin_centers']
    density    = mdf['hist_density']
    bw         = float(centers[1] - centers[0])
    theta_fine = peaks['theta_fine']
    kde_vals   = peaks['kde_vals']
    csl        = peaks['csl']
    csl_tol    = peaks['csl_tol']
    sel_angles = selected_peaks['angles']
    sel_set    = set(int(i) for i in selected_peaks['indices'])

    bar_colors = [
        'steelblue' if any(abs(centers[ci] - a) < bw * 0.5 + 1e-9 for a in sel_angles)
        else 'lightgrey'
        for ci in range(len(centers))
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(centers, density, width=bw, color=bar_colors, edgecolor='k',
           linewidth=0.3, alpha=0.75, label='MDF (histogram)')
    ax.plot(theta_fine, kde_vals, color='navy', lw=1.8, label='KDE')

    for csl_label, ang in csl.items():
        if ang <= angle_max:
            ax.axvline(ang, color='crimson', lw=0.9, ls='--', alpha=0.6)
            ax.text(ang + 0.3, ax.get_ylim()[1] * 0.92, csl_label,
                    fontsize=7, color='crimson', rotation=90, va='top')

    for pi_idx in selected_peaks['indices']:
        angle   = float(centers[pi_idx])
        nearest = min(csl, key=lambda k: abs(csl[k] - angle))
        delta   = angle - csl[nearest]
        csl_tag = f'\n≈ {nearest}' if abs(delta) <= csl_tol else ''
        ax.annotate(
            f'{angle:.1f}°{csl_tag}',
            xy=(angle, density[pi_idx]),
            xytext=(0, 12),
            textcoords='offset points',
            ha='center', fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.9, color='darkblue'),
            color='darkblue',
        )

    ax.set_xlabel('Misorientation angle (°)')
    ax.set_ylabel('Probability density (°⁻¹)')
    ax.set_title(f'MDF — {len(sel_angles)} selected peak(s) highlighted  '
                 f'(total pairs = {mdf["n_pairs"]})')
    ax.set_xlim(0, angle_max)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# CSL grain map
# ---------------------------------------------------------------------------

def plot_csl_grain_map(
        lfi: 'np.ndarray',
        csl_grains: dict,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 130,
        alpha_bg: float = 0.15,
        suptitle: str | None = None,
) -> tuple:
    """
    Colour-code the grain label field by CSL boundary participation.

    Each CSL type present in *csl_grains* gets its own colour.  Grains
    that touch boundaries of more than one CSL type have blended colours.
    Non-CSL grains are shown in semi-transparent light grey.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field (positive = grain ID, ≤0 = unindexed).
    csl_grains : dict
        Output of ``crystal_orientation.segregate_csl_pairs()``.
        Keys are CSL labels; each value must have ``'grains_all'``,
        ``'n_pairs'``, and ``'n_grains'``.
    figsize, dpi, alpha_bg, suptitle
        Standard figure parameters.

    Returns
    -------
    fig, ax
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    lfi = np.asarray(lfi, dtype=int)
    ny, nx = lfi.shape

    palette = list(mcolors.TABLEAU_COLORS.values())

    # grain_id → set of CSL indices
    label_map: dict[int, set] = {}
    for ci, (csl_label, info) in enumerate(csl_grains.items()):
        for gid in info['grains_all']:
            label_map.setdefault(int(gid), set()).add(ci)

    # Build RGBA image — start with light grey
    rgba      = np.ones((ny, nx, 4), dtype=float)
    rgba[..., :3] = 0.88
    rgba[lfi <= 0, 3] = 0.0   # transparent unindexed pixels

    gid_flat  = lfi.ravel()
    rgba_flat = rgba.reshape(-1, 4)

    for gid, csl_set in label_map.items():
        mask = gid_flat == gid
        if not mask.any():
            continue
        cols  = [mcolors.to_rgba(palette[ci % len(palette)]) for ci in csl_set]
        blend = np.mean(cols, axis=0)
        rgba_flat[mask] = blend

    rgba = rgba_flat.reshape(ny, nx, 4)

    # Dim non-CSL grains
    non_csl = set(int(g) for g in np.unique(lfi[lfi > 0])) - set(label_map)
    for gid in non_csl:
        mask = lfi == gid
        rgba[mask, :3] = 0.85
        rgba[mask,  3] = alpha_bg

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgba, interpolation='nearest', aspect='equal')

    patches = [
        mpatches.Patch(
            color=palette[ci % len(palette)],
            label=f'{lbl}  ({info["n_pairs"]} boundaries, {info["n_grains"]} grains)',
        )
        for ci, (lbl, info) in enumerate(csl_grains.items())
    ]
    patches.append(mpatches.Patch(color=(0.85, 0.85, 0.85, max(alpha_bg, 0.4)),
                                  label='Non-CSL grains'))
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_axis_off()
    if suptitle:
        ax.set_title(suptitle, fontsize=10)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Parent / twin grain summary
# ---------------------------------------------------------------------------

def print_parent_grain_summary(
        parent_info: dict,
        csl_grains: dict,
) -> None:
    """
    Print a formatted table of parent / twin / intermediate grain counts
    for each CSL type.

    Parameters
    ----------
    parent_info : dict
        Output of ``crystal_orientation.identify_parent_grains()``.
    csl_grains : dict
        Output of ``crystal_orientation.segregate_csl_pairs()`` — used to
        retrieve the number of pairs per CSL label.
    """
    header = (f"{'CSL type':<18} {'ref °':>6}  {'pairs':>6}  "
              f"{'pure parents':>13}  {'pure twins':>11}  {'intermediates':>14}")
    print(header)
    print('─' * 75)
    for lbl, v in parent_info.items():
        n_pairs = len(csl_grains[lbl]['pairs']) if lbl in csl_grains else 0
        print(f"{lbl:<18} {v['csl_angle']:>6.2f}  {n_pairs:>6}  "
              f"{v['n_pure_parents']:>13}  "
              f"{v['n_pure_twins']:>11}  "
              f"{v['n_intermediates']:>14}")


def plot_parent_grain_summary(
        parent_info: dict,
        figsize: tuple[float, float] = (8, 4),
        dpi: int = 120,
        title: str = 'Parent / twin / intermediate grain counts per CSL type',
        bar_width: float = 0.25,
        colors: tuple[str, str, str] = ('steelblue', 'coral', 'mediumseagreen'),
) -> tuple:
    """
    Grouped bar chart of pure-parent, pure-twin, and intermediate grain
    counts for each CSL type.

    Parameters
    ----------
    parent_info : dict
        Output of ``crystal_orientation.identify_parent_grains()``.
    figsize, dpi, title
        Standard figure parameters.
    bar_width : float
        Width of each bar group.
    colors : tuple of str
        Three colours for (pure parents, pure twins, intermediates).

    Returns
    -------
    fig, ax
    """
    labels   = list(parent_info.keys())
    x        = np.arange(len(labels))
    w        = bar_width
    pp_vals  = [parent_info[l]['n_pure_parents']  for l in labels]
    pt_vals  = [parent_info[l]['n_pure_twins']    for l in labels]
    im_vals  = [parent_info[l]['n_intermediates'] for l in labels]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    b1 = ax.bar(x - w, pp_vals, w, label='Pure parents',  color=colors[0], edgecolor='k', linewidth=0.5)
    b2 = ax.bar(x,     pt_vals, w, label='Pure twins',    color=colors[1], edgecolor='k', linewidth=0.5)
    b3 = ax.bar(x + w, im_vals, w, label='Intermediates', color=colors[2], edgecolor='k', linewidth=0.5)
    ax.bar_label(b1, padding=3, fontsize=8)
    ax.bar_label(b2, padding=3, fontsize=8)
    ax.bar_label(b3, padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of grains')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_parent_twin_map(
        lfi: 'np.ndarray',
        parent_info: dict,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 130,
        alpha_bg: float = 0.12,
        colors: dict | None = None,
        suptitle: str | None = None,
) -> tuple:
    """
    Colour-code the grain label field by parent / twin / intermediate role
    for each CSL type present in *parent_info*.

    One subplot is produced per CSL type.  Within each subplot:

    * **Pure parents**   – solid blue (default)
    * **Pure twins**     – solid coral / red
    * **Intermediates**  – solid green
    * **Non-role grains** – semi-transparent light grey

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field.
    parent_info : dict
        Output of ``crystal_orientation.identify_parent_grains()``.
    figsize, dpi
        Figure dimensions per subplot (total width scaled by n_csl).
    alpha_bg : float
        Opacity of non-role background grains.
    colors : dict, optional
        Override role colours.  Keys: ``'pure_parent'``, ``'pure_twin'``,
        ``'intermediate'``.  Values: any Matplotlib colour spec.
    suptitle : str, optional
        Figure-level title.

    Returns
    -------
    fig, axes
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    default_colors = {
        'pure_parent':  '#4878CF',   # steel blue
        'pure_twin':    '#D65F5F',   # coral red
        'intermediate': '#59A14F',   # medium green
    }
    if colors:
        default_colors.update(colors)

    lfi   = np.asarray(lfi, dtype=int)
    ny, nx = lfi.shape
    labels = list(parent_info.keys())
    n      = len(labels)

    fig_w  = figsize[0] * n
    fig, axes = plt.subplots(1, n, figsize=(fig_w, figsize[1]), dpi=dpi,
                             squeeze=False)

    for col, lbl in enumerate(labels):
        ax   = axes[0, col]
        info = parent_info[lbl]

        pp_set = set(info['pure_parents'].tolist())
        pt_set = set(info['pure_twins'].tolist())
        im_set = set(info['intermediates'].tolist())

        # Start with light grey background (non-role grains)
        rgba      = np.ones((ny, nx, 4), dtype=float)
        rgba[..., :3] = 0.88
        rgba[lfi <= 0, 3] = 0.0   # transparent unindexed

        gid_flat  = lfi.ravel()
        rgba_flat = rgba.reshape(-1, 4)

        role_map = [
            (pp_set, mcolors.to_rgba(default_colors['pure_parent'])),
            (pt_set, mcolors.to_rgba(default_colors['pure_twin'])),
            (im_set, mcolors.to_rgba(default_colors['intermediate'])),
        ]

        all_role_gids: set[int] = pp_set | pt_set | im_set
        # Dim non-role grains first
        for gid in np.unique(lfi[lfi > 0]):
            if int(gid) not in all_role_gids:
                mask = gid_flat == gid
                rgba_flat[mask, :3] = 0.85
                rgba_flat[mask,  3] = alpha_bg

        # Paint roles
        for gid_set, colour in role_map:
            c = np.array(colour, dtype=float)
            for gid in gid_set:
                mask = gid_flat == gid
                if mask.any():
                    rgba_flat[mask] = c

        rgba = rgba_flat.reshape(ny, nx, 4)
        ax.imshow(rgba, interpolation='nearest', aspect='equal')
        ax.set_axis_off()
        ax.set_title(lbl, fontsize=10)

        patches = [
            mpatches.Patch(color=default_colors['pure_parent'],
                           label=f"Pure parents ({info['n_pure_parents']})"),
            mpatches.Patch(color=default_colors['pure_twin'],
                           label=f"Pure twins ({info['n_pure_twins']})"),
            mpatches.Patch(color=default_colors['intermediate'],
                           label=f"Intermediates ({info['n_intermediates']})"),
            mpatches.Patch(color=(0.85, 0.85, 0.85, max(alpha_bg, 0.4)),
                           label='Non-role grains'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=7, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_combined_parent_twin_map(
        lfi: 'np.ndarray',
        parent_info: dict,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 130,
        alpha_bg: float = 0.10,
        color_parent: str = '#4878CF',
        color_twin: str = '#D65F5F',
        color_intermediate: str = '#59A14F',
        suptitle: str | None = 'Pure-parent and pure-twin grains (all CSL types combined)',
) -> tuple:
    """
    Single map showing pure-parent, pure-twin and intermediate grains
    aggregated across **all** CSL types in *parent_info*.

    A grain that is a pure-parent in *any* CSL type → coloured as parent.
    A grain that is a pure-twin   in *any* CSL type → coloured as twin.
    A grain that is intermediate  in *any* CSL type → coloured as intermediate.
    Priority when a grain appears in multiple roles: intermediate > parent > twin.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field.
    parent_info : dict
        Output of ``crystal_orientation.identify_parent_grains()``.
    figsize, dpi
        Figure size and resolution.
    alpha_bg : float
        Opacity of non-role grains.
    color_parent, color_twin, color_intermediate : str
        Matplotlib colour specs for each role.
    suptitle : str or None
        Figure title.

    Returns
    -------
    fig, ax
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    lfi    = np.asarray(lfi, dtype=int)
    ny, nx = lfi.shape

    # ── Aggregate grain sets across all CSL types ─────────────────────────────
    all_pp: set[int] = set()
    all_pt: set[int] = set()
    all_im: set[int] = set()

    for info in parent_info.values():
        all_pp.update(info['pure_parents'].tolist())
        all_pt.update(info['pure_twins'].tolist())
        all_im.update(info['intermediates'].tolist())

    # Resolve overlaps: intermediate > parent > twin
    only_pp = (all_pp | all_pt) - all_im   # in pp or pt but NOT intermediate
    # within only_pp, parent takes priority over twin
    pure_parent_final   = only_pp & all_pp
    pure_twin_final     = only_pp & all_pt - pure_parent_final
    intermediate_final  = all_im

    # ── Build RGBA image ──────────────────────────────────────────────────────
    rgba      = np.ones((ny, nx, 4), dtype=float)
    rgba[..., :3] = 0.88
    rgba[lfi <= 0, 3] = 0.0

    gid_flat  = lfi.ravel()
    rgba_flat = rgba.reshape(-1, 4)

    all_role_gids = pure_parent_final | pure_twin_final | intermediate_final

    # Dim non-role grains
    for gid in np.unique(lfi[lfi > 0]):
        if int(gid) not in all_role_gids:
            mask = gid_flat == gid
            rgba_flat[mask, :3] = 0.85
            rgba_flat[mask,  3] = alpha_bg

    for gid_set, colour_str in [
        (pure_twin_final,     color_twin),
        (pure_parent_final,   color_parent),
        (intermediate_final,  color_intermediate),
    ]:
        c = np.array(mcolors.to_rgba(colour_str), dtype=float)
        for gid in gid_set:
            mask = gid_flat == gid
            if mask.any():
                rgba_flat[mask] = c

    rgba = rgba_flat.reshape(ny, nx, 4)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgba, interpolation='nearest', aspect='equal')
    ax.set_axis_off()

    patches = [
        mpatches.Patch(color=color_parent,      label=f'Pure parents ({len(pure_parent_final)})'),
        mpatches.Patch(color=color_twin,         label=f'Pure twins ({len(pure_twin_final)})'),
        mpatches.Patch(color=color_intermediate, label=f'Intermediates ({len(intermediate_final)})'),
        mpatches.Patch(color=(0.85, 0.85, 0.85, max(alpha_bg, 0.4)), label='Non-role grains'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)

    if suptitle:
        ax.set_title(suptitle, fontsize=10)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Grain-role morphological / topological statistics
# ---------------------------------------------------------------------------

def plot_grain_role_property_stats(
        lfi: 'np.ndarray',
        parent_info: dict,
        prop: dict,
        neigh_gid: dict,
        selected_props: list | None = None,
        selected_groups: list | None = None,
        step_size: float = 1.0,
        bins: int = 40,
        bw_method: str = 'scott',
        peak_prominence: float = 0.01,
        figsize_per: tuple[float, float] = (5, 4),
        dpi: int = 110,
        suptitle: str = 'Grain morphological & topological statistics by role',
        ncols: int | None = None,
        fontsize: float = 9.0,
) -> tuple:
    """
    For each selected property, plot overlaid histograms + KDE + detected peaks
    for each selected grain-role group (pure parents, pure twins, intermediates,
    non-role).  Min / max / mean / std are embedded in the legend labels.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field.
    parent_info : dict
        Output of ``identify_parent_grains()``.
    prop : dict
        Per-grain property dict (grain_id → dict) as from ``prop_ebsd``.
        Expected keys: ``'area'``, ``'aspect_ratio'``, ``'perimeter'``,
        ``'solidity'``.
    neigh_gid : dict
        Neighbour-grain dict (grain_id → list of neighbour IDs) as from
        ``neigh_gid_ebsd``.  Used to derive ``n_neighbours``.
    selected_props : list of str, optional
        Subset of ``['area', 'aspect_ratio', 'perimeter', 'solidity',
        'n_neighbours']``.  Defaults to all five.
    selected_groups : list of str, optional
        Subset of ``['pure_parents', 'pure_twins', 'intermediates',
        'non_role']``.  Defaults to all four.
    step_size : float
        EBSD step size (µm).  Multiplied into area (px² → µm²) and
        perimeter (px → µm) values.
    bins : int
        Number of histogram bins.
    bw_method : str
        Bandwidth selector passed to ``scipy.stats.gaussian_kde``.
    peak_prominence : float
        Fraction of KDE max used as minimum prominence for peak annotation.
    figsize_per : tuple
        ``(width, height)`` in inches for each subplot.
    dpi : int
        Figure resolution.
    suptitle : str
        Figure-level title.
    ncols : int or None
        Number of columns in the subplot grid.  ``None`` (default) places all
        subplots in a single row.  E.g. ``ncols=2`` gives a 2-column grid;
        ``ncols=1`` gives a single column.
    fontsize : float
        Base font size in points.  Axis labels and titles use this size;
        tick labels use ``fontsize - 2``; legend text uses ``fontsize - 2``;
        peak annotations use ``fontsize - 3``; the figure suptitle uses
        ``fontsize + 1``.  Default ``9.0``.

    Returns
    -------
    fig, axes
    """
    from upxo.viz.vizDistr import plot_grouped_distributions

    ALL_PROPS  = ['area', 'aspect_ratio', 'perimeter', 'solidity', 'n_neighbours']
    ALL_GROUPS = ['pure_parents', 'pure_twins', 'intermediates', 'non_role']
    PROP_LABELS = {
        'area':         'Area (µm²)',
        'aspect_ratio': 'Aspect ratio',
        'perimeter':    'Perimeter (µm)',
        'solidity':     'Solidity',
        'n_neighbours': 'Number of neighbours',
    }
    GROUP_COLORS = {
        'pure_parents':  '#4878CF',
        'pure_twins':    '#D65F5F',
        'intermediates': '#59A14F',
        'non_role':      '#888888',
    }
    GROUP_LABELS_DISPLAY = {
        'pure_parents':  'Pure parents',
        'pure_twins':    'Pure twins',
        'intermediates': 'Intermediates',
        'non_role':      'Non-role',
    }

    if selected_props  is None:
        selected_props  = ALL_PROPS
    if selected_groups is None:
        selected_groups = ALL_GROUPS

    # Build grain-role sets (merged across all CSL types)
    all_pp: set[int] = set()
    all_pt: set[int] = set()
    all_im: set[int] = set()
    for info in parent_info.values():
        all_pp.update(info['pure_parents'].tolist())
        all_pt.update(info['pure_twins'].tolist())
        all_im.update(info['intermediates'].tolist())

    only_pp_or_pt    = (all_pp | all_pt) - all_im
    pure_parent_set  = only_pp_or_pt & all_pp
    pure_twin_set    = (only_pp_or_pt & all_pt) - pure_parent_set
    intermediate_set = all_im
    all_role         = pure_parent_set | pure_twin_set | intermediate_set
    indexed_gids     = set(int(g) for g in np.unique(lfi[lfi > 0]))
    non_role_set     = indexed_gids - all_role

    grain_sets = {
        'pure_parents':  pure_parent_set,
        'pure_twins':    pure_twin_set,
        'intermediates': intermediate_set,
        'non_role':      non_role_set,
    }

    # Extract property values per group into plain arrays
    def _get_vals(gids: set, pname: str) -> np.ndarray:
        vals = []
        for gid in gids:
            if pname == 'n_neighbours':
                nb = neigh_gid.get(gid)
                if nb is not None:
                    vals.append(len(nb))
            else:
                p = prop.get(gid)
                if p is not None and pname in p:
                    v = float(p[pname])
                    if pname == 'area':
                        v *= step_size ** 2
                    elif pname == 'perimeter':
                        v *= step_size
                    vals.append(v)
        arr = np.array(vals, dtype=float)
        return arr[np.isfinite(arr)]

    data = {
        pname: {grp: _get_vals(grain_sets[grp], pname) for grp in selected_groups}
        for pname in selected_props
    }

    return plot_grouped_distributions(
        data            = data,
        prop_labels     = PROP_LABELS,
        group_colors    = {g: GROUP_COLORS[g] for g in selected_groups},
        group_labels    = {g: GROUP_LABELS_DISPLAY[g] for g in selected_groups},
        bins            = bins,
        bw_method       = bw_method,
        peak_prominence = peak_prominence,
        figsize_per     = figsize_per,
        dpi             = dpi,
        suptitle        = suptitle,
        ncols           = ncols,
        fontsize        = fontsize,
    )


def print_grain_role_ratios(ratios: dict) -> None:
    """
    Print a formatted table of grain-role counts and ratios produced by
    ``crystal_orientation.compute_grain_role_ratios()``.

    Parameters
    ----------
    ratios : dict
        Output of :func:`~upxo.xtalphy.crystal_orientation.compute_grain_role_ratios`.
    """
    N = ratios['total']['count']
    print(f'Total indexed grains : {N}\n')
    print(f"{'Role':<18}  {'Count':>7}  {'Ratio':>8}  {'% of total':>11}")
    print('\u2500' * 50)
    for key, label in [
        ('pure_parents',  'Pure parents'),
        ('pure_twins',    'Pure twins'),
        ('intermediates', 'Intermediates'),
        ('non_role',      'Non-role'),
    ]:
        n = ratios[key]['count']
        r = ratios[key]['ratio']
        print(f'{label:<18}  {n:>7}  {r:>8.4f}  {r * 100:>10.2f}%')
    print('\u2500' * 50)
    n_ar = ratios['all_role']['count']
    r_ar = ratios['all_role']['ratio']
    print(f"{'All role grains':<18}  {n_ar:>7}  {r_ar:>8.4f}  {r_ar * 100:>10.2f}%")


def plot_grain_role_map(
        lfi: 'np.ndarray',
        role_sets: dict,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 130,
        alpha_bg: float = 0.10,
        color_parent: str = '#4878CF',
        color_twin: str = '#D65F5F',
        color_intermediate: str = '#59A14F',
        suptitle: str | None = None,
) -> tuple:
    """
    Plot a grain role map from pre-computed role sets (output of
    ``crystal_orientation.assign_grain_roles_by_ratio()`` or
    ``crystal_orientation.compute_grain_role_ratios()``).

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field.
    role_sets : dict
        Dict with keys ``'pure_parents'``, ``'pure_twins'``,
        ``'intermediates'``, ``'non_role'`` — each value either a set of
        grain IDs or a sub-dict with a ``'grain_ids'`` key (as returned by
        :func:`~upxo.xtalphy.crystal_orientation.assign_grain_roles_by_ratio`).
    figsize, dpi, alpha_bg
        Figure parameters.
    color_parent, color_twin, color_intermediate : str
        Matplotlib colour specs for each role.
    suptitle : str or None
        Figure title.

    Returns
    -------
    fig, ax
    """
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    def _ids(v):
        return v['grain_ids'] if isinstance(v, dict) else v

    pp_set = _ids(role_sets['pure_parents'])
    pt_set = _ids(role_sets['pure_twins'])
    im_set = _ids(role_sets['intermediates'])
    nr_set = _ids(role_sets.get('non_role', set()))

    lfi = np.asarray(lfi, dtype=int)
    ny, nx = lfi.shape

    rgba = np.ones((ny, nx, 4), dtype=float)
    rgba[..., :3] = 0.88
    rgba[lfi <= 0, 3] = 0.0

    gid_flat  = lfi.ravel()
    rgba_flat = rgba.reshape(-1, 4)

    all_role_gids = pp_set | pt_set | im_set

    for gid in np.unique(lfi[lfi > 0]):
        if int(gid) not in all_role_gids:
            mask = gid_flat == gid
            rgba_flat[mask, :3] = 0.85
            rgba_flat[mask,  3] = alpha_bg

    for gid_set, colour_str in [
        (pt_set, color_twin),
        (pp_set, color_parent),
        (im_set, color_intermediate),
    ]:
        c = np.array(mcolors.to_rgba(colour_str), dtype=float)
        for gid in gid_set:
            mask = gid_flat == gid
            if mask.any():
                rgba_flat[mask] = c

    rgba = rgba_flat.reshape(ny, nx, 4)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgba, interpolation='nearest', aspect='equal')
    ax.set_axis_off()

    patches = [
        mpatches.Patch(color=color_parent,      label=f'Pure parents ({len(pp_set)})'),
        mpatches.Patch(color=color_twin,         label=f'Pure twins ({len(pt_set)})'),
        mpatches.Patch(color=color_intermediate, label=f'Intermediates ({len(im_set)})'),
        mpatches.Patch(color=(0.85, 0.85, 0.85, max(alpha_bg, 0.4)),
                       label=f'Non-role ({len(nr_set)})'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)

    if suptitle:
        ax.set_title(suptitle, fontsize=10)
    plt.tight_layout()
    return fig, ax


def plot_twin_introduction_map(
        lfi_before: 'np.ndarray',
        twin_result: dict,
        figsize: tuple = (14, 5),
        dpi: int = 130,
        color_parent: str = '#4878CF',
        color_twin: str = '#D65F5F',
        color_other: str = '#CCCCCC',
        alpha_other: float = 0.30,
        show_cut_lines: bool = True,
        suptitle=None,
) -> tuple:
    """
    Side-by-side comparison of the grain label field before and after twin
    introduction, with cut-lines overlaid on the "after" panel.

    Parameters
    ----------
    lfi_before : ndarray, shape (ny, nx)
        Grain label field before twin introduction.
    twin_result : dict
        Output of ``crystal_orientation.introduce_twins_by_csl()``.
    figsize : tuple
        Total figure size (width, height).
    dpi : int
        Figure resolution.
    color_parent : str
        Colour for parent grains.
    color_twin : str
        Colour for newly introduced twin grains.
    color_other : str
        Colour for all other grains.
    alpha_other : float
        Opacity for other grains.
    show_cut_lines : bool
        If True, overlay the Sline2d cut-lines on the "after" panel.
    suptitle : str or None
        Figure-level title.

    Returns
    -------
    fig, axes
    """
    import matplotlib.patches as mpatches

    lfi_after  = twin_result['lfi']
    twin_lines = twin_result['twin_lines']
    new_twin_gids = twin_result['new_twin_gids']
    csl_label  = twin_result['csl_label']

    parent_gids   = set(twin_lines.keys())
    all_twin_gids = set()
    for gids in new_twin_gids.values():
        all_twin_gids.update(gids)

    _c_p = np.array(plt.matplotlib.colors.to_rgba(color_parent))
    _c_t = np.array(plt.matplotlib.colors.to_rgba(color_twin))
    _c_o = list(plt.matplotlib.colors.to_rgb(color_other)) + [alpha_other]

    def _make_rgba(lfi):
        ny, nx = lfi.shape
        rgba = np.ones((ny, nx, 4), dtype=float)
        rgba[..., :3] = 0.90
        rgba[lfi <= 0, 3] = 0.0
        for gid in np.unique(lfi[lfi > 0]):
            gid = int(gid)
            mask = lfi == gid
            if gid in parent_gids:
                rgba[mask] = _c_p
            elif gid in all_twin_gids:
                rgba[mask] = _c_t
            else:
                rgba[mask] = _c_o
        return rgba

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    axes[0].imshow(_make_rgba(lfi_before), interpolation='nearest', aspect='equal')
    axes[0].set_axis_off()
    axes[0].set_title('Before twin introduction', fontsize=10)

    axes[1].imshow(_make_rgba(lfi_after), interpolation='nearest', aspect='equal')
    axes[1].set_axis_off()
    axes[1].set_title(f'After twin introduction — {csl_label}', fontsize=10)

    if show_cut_lines:
        for lines in twin_lines.values():
            for line in lines:
                axes[1].plot([line.x0, line.x1], [line.y0, line.y1],
                             color='yellow', linewidth=0.8, alpha=0.85,
                             linestyle='--')

    patches = [
        mpatches.Patch(color=color_parent, label=f'Parent grains ({len(parent_gids)})'),
        mpatches.Patch(color=color_twin,   label=f'New twin grains ({len(all_twin_gids)})'),
        mpatches.Patch(facecolor=color_other, alpha=alpha_other, label='Other grains'),
    ]
    axes[1].legend(handles=patches, loc='upper right', fontsize=7, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_twin_thickness_stats(
    stats: dict,
    bins: int = 30,
    figsize: tuple = (7, 4),
    dpi: int = 120,
    color: str = 'steelblue',
    title: str | None = None,
) -> tuple:
    """
    Plot a histogram of twin lamella thickness from the dict returned by
    ``compute_twin_thickness_stats``.

    Parameters
    ----------
    stats : dict
        Output of ``compute_twin_thickness_stats``.
    bins : int
        Number of histogram bins.
    figsize : (width, height)
    dpi : int
    color : str
        Bar fill colour.
    title : str or None
        Custom figure title; auto-generated when *None*.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import numpy as np

    thick_um  = stats['thick_um']
    col       = stats['col']
    step_um   = stats['step_um']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(thick_um, bins=bins, color=color, edgecolor='white', density=True)
    ax.axvline(stats['mean'],   color='red',    lw=1.8, ls='--',
               label=f"Mean {stats['mean']:.2f} µm")
    ax.axvline(stats['median'], color='orange', lw=1.8, ls=':',
               label=f"Median {stats['median']:.2f} µm")
    ax.set_xlabel('Twin thickness (µm)', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    _title = title or (
        f'EBSD – twin grain thickness distribution\n'
        f'(proxy: {col},  step = {step_um} µm/px,  N = {len(thick_um)})'
    )
    ax.set_title(_title, fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_grain_area_comparison(
    scale_result: dict,
    bins: int = 50,
    pct_clip: float = 97.0,
    figsize: tuple = (8, 4),
    dpi: int = 120,
    color_ebsd: str = 'steelblue',
    color_sim: str = 'tomato',
) -> tuple:
    """
    Plot overlapping grain-area histograms for the EBSD target and the
    scaled synthetic structure.

    Parameters
    ----------
    scale_result : dict
        Output of ``compute_scale_factor_grain_size``.  Must contain
        ``'areas_ebsd_um2'``, ``'areas_sim_um2'``, ``'mean_area_ebsd_um2'``,
        ``'scale_factor'``, and ``'mean_area_sim_um2'``.
    bins : int
        Number of histogram bins.
    pct_clip : float
        Upper percentile used to clip the x-axis range (avoids giant outlier
        grains dominating the axis).
    figsize, dpi : figure size and resolution.
    color_ebsd, color_sim : bar fill colours.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import numpy as np

    areas_ebsd = scale_result['areas_ebsd_um2']
    areas_sim  = scale_result['areas_sim_um2']
    sf         = scale_result['scale_factor']
    mean_ebsd  = scale_result['mean_area_ebsd_um2']
    mean_sim   = scale_result['mean_area_sim_um2']

    if areas_ebsd is None:
        raise ValueError(
            "scale_result['areas_ebsd_um2'] is None — pass prop_ebsd to "
            "compute_scale_factor_grain_size() to populate it."
        )

    combined   = np.concatenate([areas_ebsd, areas_sim])
    bin_edges  = np.linspace(0, np.percentile(combined, pct_clip), bins + 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(areas_ebsd, bins=bin_edges, density=True, alpha=0.6,
            color=color_ebsd, edgecolor='white',
            label=f'EBSD  (mean = {mean_ebsd:.1f} µm²)')
    ax.hist(areas_sim,  bins=bin_edges, density=True, alpha=0.6,
            color=color_sim,  edgecolor='white',
            label=f'Sim scaled  (mean = {mean_sim:.1f} µm²)')
    ax.axvline(mean_ebsd, color=color_ebsd, lw=2, ls='--')
    ax.axvline(mean_sim,  color=color_sim,  lw=2, ls='--')
    ax.set_xlabel('Grain area (µm²)', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(
        f'Grain area distribution: EBSD vs scaled synthetic\n'
        f'Scale factor = {sf:.4f} µm/px',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax
