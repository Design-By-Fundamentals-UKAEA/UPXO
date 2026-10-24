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
    'non_host':      'lightgray',   # only used when nonhost_cmap=None
    'host':          'steelblue',
    'primary_twin':  'darkorange',
    'secondary_twin': 'crimson',
}

# Default colormap for non-host grains (per-grain scalar colouring).
# Chosen to avoid the solid role colours (steelblue, darkorange, crimson).
_DEFAULT_NONHOST_CMAP = 'Greens'


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


def _ipf_key_image(n: int = 400):
    """The mtex IPF key on the fundamental sector [001]-[101]-[111], in the equal-area
    projection (radius 1 on the equator) that mtex draws it in.

    Directions are sampled on a regular grid of the projection plane, kept where they fall
    inside the sector, and coloured by :func:`upxo.xtalphy.ipf.cubic_ipf_rgb`.

    Returns
    -------
    extent : (x0, x1, y0, y1)
        Plane extent of the sector.
    rgba : ndarray (n, n, 4)
        Colours; alpha 0 outside the sector.
    outline : ndarray (m, 2)
        The three boundary arcs of the sector in the projection plane, closed.
    corners : ndarray (3, 2)
        Projected positions of [001], [101], [111].
    """
    from upxo.xtalphy.ipf import (
        CUBIC_SECTOR_VERTICES, cubic_ipf_rgb, equal_area_xy, sector_mask)

    V = CUBIC_SECTOR_VERTICES
    arcs = []
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]
        ang = np.arccos(np.clip(a @ b, -1.0, 1.0))
        t = np.linspace(0.0, 1.0, 120)[:, None]
        arc = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang)
        arcs.append(np.stack(equal_area_xy(arc), axis=-1))
    outline = np.concatenate(arcs + [arcs[0][:1]])
    x0, y0 = outline.min(axis=0)
    x1, y1 = outline.max(axis=0)

    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    r2 = X ** 2 + Y ** 2
    s = np.sqrt(np.clip(2.0 - r2, 0.0, None))
    dirs = np.stack([X * s, Y * s, 1.0 - r2], axis=-1)
    inside = (r2 <= 1.0) & sector_mask(dirs, tol=1e-9)
    rgba = np.zeros((n, n, 4), dtype=np.float64)
    rgba[inside, :3] = cubic_ipf_rgb(dirs[inside])
    rgba[inside, 3] = 1.0
    corners = np.stack(equal_area_xy(V), axis=-1)
    return (x0, x1, y0, y1), rgba, outline, corners


def plot_ipf_triangle_key(
        ax=None,
        n: int = 400,
        corner_labels: Tuple[str, str, str] = ('[001]', '[101]', '[111]'),
        figsize: Tuple[float, float] = (4., 4.),
):
    """
    Plot the IPF colour key: mtex's default key for cubic crystals, drawn on the
    equal-area projection of the fundamental sector [001]-[101]-[111]. This is the key
    :func:`plot_ipf_slice` and :func:`upxo.viz.ebsdviz.build_ipf_rgb` colour with.

    Parameters
    ----------
    ax : matplotlib Axes or None
        Plots into an existing axes (e.g. one embedded in a larger figure) if given;
        otherwise creates its own standalone figure.
    n : int
        Grid resolution (n x n before masking to the sector).
    corner_labels : (str, str, str)
        Labels for the [001]/[101]/[111] corners.
    figsize : (float, float)
        Only used when ``ax`` is None.
    """
    import matplotlib.pyplot as plt

    (x0, x1, y0, y1), rgba, outline, corners = _ipf_key_image(n)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.imshow(rgba, origin='lower', extent=(x0, x1, y0, y1), interpolation='bilinear')
    ax.plot(outline[:, 0], outline[:, 1], color='black', linewidth=1.0)
    centre = corners.mean(axis=0)
    pad = 0.04 * max(x1 - x0, y1 - y0)
    for (cx, cy), label in zip(corners, corner_labels):
        out = np.array([cx, cy]) - centre
        out = out / (np.linalg.norm(out) + 1e-12)
        ax.text(cx + pad * out[0], cy + pad * out[1], label, fontsize=9, fontweight='bold',
                ha='left' if out[0] > 0.2 else ('right' if out[0] < -0.2 else 'center'),
                va='bottom' if out[1] > 0.2 else ('top' if out[1] < -0.2 else 'center'))
    ax.set_xlim(x0 - 0.2 * (x1 - x0), x1 + 0.3 * (x1 - x0))
    ax.set_ylim(y0 - 0.2 * (y1 - y0), y1 + 0.2 * (y1 - y0))
    ax.set_aspect('equal')
    ax.axis('off')
    if standalone:
        plt.tight_layout()
    return fig, ax


def _prep_lgi_pv_mesh(lgi: np.ndarray, voxel_size=1.0, rng_seed: int = 0):
    """
    Shared grid-preparation for :func:`render_lgi_3d` and
    :func:`render_lgi_3d_compare`: builds the per-grain-shuffled-colour
    ``pv.ImageData`` mesh, thresholded down to only the labelled
    (``grain_id_norm > 0``) region.

    Returns
    -------
    (grid_thresh, n_grains)
    """
    import pyvista as pv

    unique_gids = np.unique(lgi[lgi > 0])
    n_grains = int(unique_gids.size)
    vs = tuple(voxel_size) if hasattr(voxel_size, '__len__') else (voxel_size,) * 3

    max_gid = int(unique_gids.max()) if n_grains > 0 else 0
    lut = np.zeros(max_gid + 1, dtype=np.float32)
    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(n_grains)
    denom = max(n_grains - 1, 1)
    lut[unique_gids] = 0.05 + 0.9 * shuffled / denom

    grid = pv.ImageData()
    grid.dimensions = np.array(lgi.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = vs
    grid.cell_data['grain_id_norm'] = lut[lgi].flatten(order='F')
    grid_thresh = grid.threshold(1e-6, scalars='grain_id_norm')
    return grid_thresh, n_grains


def render_lgi_3d(
        lgi: np.ndarray,
        voxel_size=1.0,
        title: Optional[str] = None,
        cmap: str = 'tab20',
        rng_seed: int = 0,
        show_edges: bool = False,
):
    """
    Render a 3D grain-label field in a separate PyVista window, each grain
    shown in a distinct colour via a qualitative colormap.

    Shared by ``TwinnedSimple3DBase.plot_temporal_slice_3d`` (which builds
    ``lgi`` from a raw MC state array and must transpose it first -- see
    that method) and callers that pass ``TwinnedSimple3DBase.lgi`` directly.

    Parameters
    ----------
    lgi : ndarray (nx, ny, nz), int
        Grain label field, already in PyVista's (nx, ny, nz) axis
        convention -- the same convention :func:`render_3d` expects for
        ``lgi_twinned``.  NOT the raw ``pxt.gs[...].s`` (nz, ny, nx) order.
    voxel_size : float or (float, float, float)
        Physical voxel edge length(s).  Isotropic spacing if a bare float.
    title : str or None
        Window title text; defaults to a grain-count summary.
    cmap : str
        Qualitative PyVista/matplotlib colormap name.
    rng_seed : int
        Seed for the ID -> colour-position shuffle.  cc3d/regionprops-style
        labelling assigns numerically close IDs to spatially close grains;
        without shuffling, tab20's discrete cycle repeats faster than IDs
        vary, so neighbouring grains land on near-identical colours and
        visually merge. Scattering IDs across the colour range keeps real
        grain boundaries distinct.
    show_edges : bool
        Draw voxel-cell edges as a wireframe overlay on the mesh.
    """
    import pyvista as pv

    grid_thresh, n_grains = _prep_lgi_pv_mesh(lgi, voxel_size, rng_seed)

    pvp = pv.Plotter()
    pvp.add_mesh(grid_thresh, scalars='grain_id_norm', cmap=cmap,
                 show_edges=show_edges, show_scalar_bar=False, clim=[0.0, 1.0])
    pvp.add_text(title or f"{n_grains} grains", font_size=10)
    pvp.show()


def render_lgi_3d_compare(
        lgi_old: np.ndarray,
        lgi_new: np.ndarray,
        voxel_size_old=1.0,
        voxel_size_new=1.0,
        title_old: Optional[str] = None,
        title_new: Optional[str] = None,
        cmap: str = 'tab20',
        rng_seed: int = 0,
        show_edges: bool = True,
):
    """
    Render two grain-label fields side by side in a single PyVista window
    (two linked-camera subplots) -- e.g. an original vs. a stretched/
    rescaled grain structure on Transformations' "Introduce
    non-equiaxiality to SGS" page, so the two are visually comparable at
    a glance instead of two separate, independently-sized windows.

    Parameters
    ----------
    lgi_old, lgi_new : ndarray (nx, ny, nz), int
        Grain label fields, already in PyVista's (nx, ny, nz) axis
        convention -- see :func:`render_lgi_3d`.
    voxel_size_old, voxel_size_new : float or (float, float, float)
        Physical voxel edge length(s) for each structure.
    title_old, title_new : str or None
        Per-panel title text; default to a grain-count summary.
    cmap : str
        Qualitative PyVista/matplotlib colormap name.
    rng_seed : int
        Seed for the ID -> colour-position shuffle (see
        :func:`render_lgi_3d`); the same seed is used for both panels.
    show_edges : bool
        Draw voxel-cell edges as a wireframe overlay on both meshes.
    """
    import pyvista as pv

    grid_old, n_old = _prep_lgi_pv_mesh(lgi_old, voxel_size_old, rng_seed)
    grid_new, n_new = _prep_lgi_pv_mesh(lgi_new, voxel_size_new, rng_seed)

    pvp = pv.Plotter(shape=(1, 2))

    pvp.subplot(0, 0)
    pvp.add_mesh(grid_old, scalars='grain_id_norm', cmap=cmap,
                 show_edges=show_edges, show_scalar_bar=False, clim=[0.0, 1.0])
    pvp.add_text(title_old or f"Original ({n_old} grains)", font_size=10)

    pvp.subplot(0, 1)
    pvp.add_mesh(grid_new, scalars='grain_id_norm', cmap=cmap,
                 show_edges=show_edges, show_scalar_bar=False, clim=[0.0, 1.0])
    pvp.add_text(title_new or f"Transformed ({n_new} grains)", font_size=10)

    pvp.link_views()
    pvp.show()


_DEFAULT_SIZE_BAND_COLORS = ('steelblue', 'seagreen', 'firebrick')
_DEFAULT_SIZE_BAND_OPACITIES = (1.0, 0.60, 0.30)


def render_lgi_3d_by_size(
        lgi: np.ndarray,
        size_thresholds: Tuple[float, float],
        voxel_size=1.0,
        title: Optional[str] = None,
        band_colors: Optional[Tuple[str, str, str]] = None,
        band_opacities: Tuple[float, float, float] = _DEFAULT_SIZE_BAND_OPACITIES,
):
    """
    Render a 3D grain-label field in a separate PyVista window, grains
    coloured and opacity-banded by their own voxel count into 3 size
    domains -- e.g. to compare a grain structure before vs. after
    cleaning (small-grain merging measurably shifts the size
    distribution, which a uniform per-grain colouring can't show).

    Parameters
    ----------
    lgi : ndarray (nx, ny, nz), int
        Grain label field, already in PyVista's (nx, ny, nz) axis
        convention (matching :func:`render_3d`'s convention -- transpose
        first if coming from the pipeline's native (nz, ny, nx) array;
        see :func:`render_lgi_3d`'s docstring).
    size_thresholds : (float, float)
        ``(t1, t2)`` voxel-count cut points, t1 < t2. Domain 1:
        grains with ``voxel_count < t1``. Domain 2: ``t1 <= voxel_count
        < t2``. Domain 3: ``voxel_count >= t2``.
    voxel_size : float or (float, float, float)
        Physical voxel edge length(s). Isotropic if a bare float.
    title : str or None
        Window title text; defaults to a per-domain grain-count summary.
    band_colors : (str, str, str) or None
        PyVista colour names for domains 1/2/3. Defaults to
        ``_DEFAULT_SIZE_BAND_COLORS`` (steelblue, seagreen, firebrick).
    band_opacities : (float, float, float)
        Opacity per domain. Defaults to (1.0, 0.60, 0.30) -- smallest
        grains fully opaque, largest grains most transparent, so small
        grains (the ones cleaning actually acts on) stay visible instead
        of being hidden behind the bulk of the structure.
    """
    import pyvista as pv

    t1, t2 = size_thresholds
    colors = band_colors if band_colors is not None else _DEFAULT_SIZE_BAND_COLORS

    gids, counts = np.unique(lgi[lgi > 0], return_counts=True)
    gid_to_count = dict(zip(gids.tolist(), counts.tolist()))

    domain_gids = [
        [g for g, c in gid_to_count.items() if c < t1],
        [g for g, c in gid_to_count.items() if t1 <= c < t2],
        [g for g, c in gid_to_count.items() if c >= t2],
    ]
    domain_labels = [
        f"< {t1:g} vox", f"{t1:g}-{t2:g} vox", f">= {t2:g} vox",
    ]

    pvp = pv.Plotter()
    legend_entries = []

    for gids_d, color, opacity, label in zip(domain_gids, colors, band_opacities, domain_labels):
        if not gids_d or opacity <= 0:
            continue
        masked = np.where(np.isin(lgi, gids_d), lgi, 0)
        grid = pv.ImageData()
        grid.dimensions = np.array(masked.shape) + 1
        grid.origin = (0, 0, 0)
        grid.spacing = tuple(voxel_size) if hasattr(voxel_size, '__len__') else (voxel_size,) * 3
        grid.cell_data['lgi'] = masked.flatten(order='F')
        grid_thresh = grid.threshold(0.5, scalars='lgi')
        pvp.add_mesh(grid_thresh, opacity=opacity, color=color, show_edges=False)
        legend_entries.append([f'{label}  ({len(gids_d)} grains)', color])

    if legend_entries:
        pvp.add_legend(labels=legend_entries, bcolor='w', border=True, size=(0.3, 0.14))
    pvp.add_text(
        title or f"Grain size domains  ({len(gids)} grains total)", font_size=10)
    pvp.show()


def render_3d(
        lgi_twinned: np.ndarray,
        twin_role: Dict[int, str],
        role_opacity: Optional[Dict[str, float]] = None,
        role_color: Optional[Dict[str, str]] = None,
        nonhost_cmap: Optional[str] = _DEFAULT_NONHOST_CMAP,
):
    """
    3D PyVista voxel render with per-role solid colour and opacity.

    Host, primary_twin and secondary_twin roles are rendered with solid
    colours so ``add_legend()`` shows correct swatches.  Non-host grains
    are rendered with per-grain scalar colouring using ``nonhost_cmap``
    so individual matrix grains are visible rather than a uniform grey.

    Parameters
    ----------
    lgi_twinned : ndarray (nx, ny, nz), int
    twin_role : dict {int: str}
        Role map: 'host' | 'primary_twin' | 'secondary_twin' | 'non_host'.
    role_opacity : dict {str: float} or None
        Per-role opacity.  Defaults to ``_DEFAULT_ROLE_OPACITY``.
    role_color : dict {str: str} or None
        Per-role PyVista colour name for solid-coloured roles.
        Defaults to ``_DEFAULT_ROLE_COLOR``.
    nonhost_cmap : str or None
        Matplotlib / PyVista colormap name for non-host grains.
        Each grain is mapped to a unique hue so individual matrix grains
        are distinguishable.  The colormap should not overlap with the
        solid role colours (steelblue, darkorange, crimson).
        Defaults to ``'Greens'``.  Pass ``None`` to fall back to the
        solid ``role_color['non_host']`` colour instead.
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
    legend_entries = []   # [label, colour] pairs for manual legend

    for role, alpha in opacity.items():
        if alpha <= 0:
            continue

        gids_role = [g for g, r in twin_role.items() if r == role]
        if not gids_role:
            continue

        masked = np.where(np.isin(lgi_twinned, gids_role), lgi_twinned, 0)

        grid = pv.ImageData()
        grid.dimensions = np.array(masked.shape) + 1
        grid.origin     = (0, 0, 0)
        grid.spacing    = (1, 1, 1)

        if role == 'non_host' and nonhost_cmap is not None:
            # Per-grain scalar colouring: map each grain ID to a normalised
            # position in [0.05, 0.95] so extreme ends of the colormap are
            # avoided and the colour range stays within the chosen cmap.
            unique_gids  = sorted(set(gids_role))
            n_gids       = len(unique_gids)
            lo, hi       = 0.05, 0.95
            gid_to_norm  = {
                gid: lo + (hi - lo) * i / max(n_gids - 1, 1)
                for i, gid in enumerate(unique_gids)
            }
            scalar_field = np.zeros(masked.size, dtype=np.float32)
            flat         = masked.flatten(order='F')
            for gid, nv in gid_to_norm.items():
                scalar_field[flat == gid] = nv

            grid.cell_data['nonhost_norm'] = scalar_field
            grid_thresh = grid.threshold(1e-6, scalars='nonhost_norm')
            pvp.add_mesh(
                grid_thresh,  # type: ignore[arg-type]
                scalars='nonhost_norm',
                cmap=nonhost_cmap,
                opacity=alpha,
                show_edges=False,
                show_scalar_bar=False,
                clim=[0.0, 1.0],
            )
            # Represent the whole non-host group with the cmap's midpoint colour
            import matplotlib as _mpl
            _mid_rgba = _mpl.colormaps[nonhost_cmap](0.5)
            _mid_hex  = '#%02x%02x%02x' % tuple(int(c * 255) for c in _mid_rgba[:3])
            legend_entries.append([f'non_host  (cmap: {nonhost_cmap})', _mid_hex])

        else:
            # Solid colour for this role (host, primary_twin, secondary_twin,
            # or non_host when nonhost_cmap=None)
            grid.cell_data['lgi'] = masked.flatten(order='F')
            grid_thresh = grid.threshold(0.5, scalars='lgi')
            role_col    = color.get(role, 'white')
            pvp.add_mesh(
                grid_thresh,  # type: ignore[arg-type]
                opacity=alpha,
                color=role_col,
                show_edges=False,
            )
            legend_entries.append([role, role_col])

    if legend_entries:
        pvp.add_legend(labels=legend_entries, bcolor='w', border=True, size=(0.28, 0.14))  # type: ignore[call-arg]
    pvp.show()


_VOXEL_SET_COLOURS = ('crimson', 'steelblue', 'darkorange', 'seagreen', 'purple', 'goldenrod',
                      'deeppink', 'teal', 'sienna', 'slategray')


def build_feature_ids(lgi, sets, mode='feature'):
    """Integer volume (shape of ``lgi``) giving every voxel of ``sets`` a feature id, 0 elsewhere.

    ``mode='grain'``: the id is the voxel's grain id in the global grain structure ``lgi``.
    ``mode='feature'``: each connected cluster (face-connected, inside one grain) of the
    union of the sets is a feature and gets its own id, 1 to the number of features.
    ``sets`` is ``{name: 1-based C-order voxel ids}``. Returns ``(ids, n_features)``."""
    lgi = np.asarray(lgi)
    mask = np.zeros(lgi.size, dtype=bool)
    for ids in sets.values():
        mask[np.asarray(ids, dtype=np.int64) - 1] = True
    mask = mask.reshape(lgi.shape)
    if mode == 'grain':
        out = np.where(mask, lgi, 0)
        return out, int(len(np.unique(out[out > 0])))
    if mode != 'feature':
        raise ValueError("mode must be 'grain' or 'feature'.")
    import cc3d
    labels = cc3d.connected_components(np.where(mask, lgi, 0).astype(np.uint32), connectivity=6)
    return labels.astype(np.int64), int(labels.max())


def render_voxel_sets(shape, sets, colors=None, opacity=1.0, off_screen=False, lgi=None, color_by=None,
                      cmap='nipy_spectral'):
    """3D render of sets of voxels, each set in its own colour.

    Parameters
    ----------
    shape : (nx, ny, nz)
        Shape of the voxel grid.
    sets : dict {name: ids}
        1-based voxel ids, the flat index in C order of an (nx, ny, nz) array
        plus one (the element ids of one element per voxel, as in the Abaqus
        element sets). Later sets are drawn over earlier ones where they overlap.
    colors : dict {name: colour} or None
        PyVista colour per set; defaults to a cycle of distinct colours.
    off_screen : bool
        Return the plotter without showing it.
    """
    import pyvista as pv

    nx, ny, nz = (int(v) for v in shape)
    plotter = pv.Plotter(off_screen=off_screen)
    outline = pv.ImageData()
    outline.dimensions = (nx + 1, ny + 1, nz + 1)
    plotter.add_mesh(outline.outline(), color='black', line_width=1)
    if color_by is not None:
        if lgi is None:
            raise ValueError("color_by needs the grain label field lgi.")
        ids, n_feat = build_feature_ids(lgi, sets, mode=color_by)
        if n_feat == 0:
            return plotter if off_screen else None
        scalars = ids.astype(np.float64)
        if color_by == 'feature':                     # neighbouring features should not get neighbouring colours
            perm = np.random.default_rng(0).permutation(n_feat) + 1
            scalars = np.where(ids > 0, perm[np.clip(ids - 1, 0, n_feat - 1)], 0).astype(np.float64)
            clim, title = (1.0, float(n_feat)), f'feature id ({n_feat} features)'
        else:
            top = float(np.max(lgi))
            clim, title = (1.0, top), 'grain id'
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.cell_data['fid'] = scalars.ravel(order='F')            # pyvista cell order: x fastest
        plotter.add_mesh(grid.threshold(0.5, scalars='fid'), scalars='fid', cmap=cmap, clim=clim,
                         opacity=opacity, show_edges=False, scalar_bar_args={'title': title})
        plotter.add_text('  |  '.join(f'{k} ({len(v)} voxels)' for k, v in sets.items())[:240], font_size=8)
        if off_screen:
            return plotter
        plotter.show()
        return None
    entries = []
    for k, (name, ids) in enumerate(sets.items()):
        ids = np.asarray(ids, dtype=np.int64) - 1
        if ids.size == 0:
            continue
        i, j, kk = np.unravel_index(ids, (nx, ny, nz))
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        values = np.zeros(nx * ny * nz, dtype=np.uint8)
        values[i + nx * (j + ny * kk)] = 1                 # pyvista cell order: x fastest
        grid.cell_data['in_set'] = values
        colour = (colors or {}).get(name) or _VOXEL_SET_COLOURS[k % len(_VOXEL_SET_COLOURS)]
        plotter.add_mesh(grid.threshold(0.5, scalars='in_set'), color=colour, opacity=opacity, show_edges=False)
        entries.append([f'{name}  ({ids.size} voxels)', colour])
    if entries:
        plotter.add_legend(labels=entries, bcolor='w', border=True, size=(0.3, 0.14))  # type: ignore[call-arg]
    if off_screen:
        return plotter
    plotter.show()
    return None
