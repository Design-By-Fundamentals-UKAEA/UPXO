"""Visualization -- Part H of the FM Steel 3D walkthrough (Viz Settings +
Pole Figures), extended per hierarchy level: an interactive 3D render, a
scatter pole figure, a 1st-order morphological property grid, and a
misorientation distribution -- for whichever of PAG/Packet/Block/Sub-block
have been reached.

Bulk property queries via `feature_props_3d.query_level`, an interactive
3D render via `GrainStructureViz3D`, pole figures via
`upxo.viz.xphy.pole_figure.PoleFigure`, and misorientation-angle
distributions via `upxo.xtalphy.crystal_orientation.
grain_boundary_misorientation_distribution` -- the same classes/functions
the GUI's own Viz Settings / Pole Figures pages already drive, plus one
genuinely new core function this needed (geom_metrics_3d.
feature_aspect_ratio_bbox -- no general 3D aspect-ratio function existed
outside one instance-bound method on a different pipeline's class).
"""

MORPHOLOGY_PROPS = [
    ('vol_physical_array', 'Volume'),
    ('surf_physical_array', 'Surface Area'),
    ('sa_vol_ratio_array', 'Surface Area / Volume'),
    ('jl_physical_array', 'Junction Line Length'),
    ('jp_nverts_array', 'Triple Junction Points'),
    ('n_neighbours_array', 'Coordination Number'),
]


def distribution_stats(fm, level):
    """Bulk geometric property query for every feature at `level`
    ('grain' | 'pag' | 'packet' | 'block' | 'subblock').

    Returns
    -------
    FeaturePropsCollection : vectorised property arrays (.vol_physical_array,
    .vol_equiv_diameter_array, .sphericity_array, ...) plus .summary() and
    .distribution(prop, bins=None).
    """
    from upxo.pxtal.fm_steel_3d.feature_props_3d import query_level
    return query_level(fm, level)


def compute_aspect_ratios(fm, level):
    """Per-feature 3D bounding-box aspect ratio, in the SAME order as
    `distribution_stats(fm, level)`'s own FeaturePropsCollection (ascending
    active integer label order) -- so it lines up element-wise with any of
    that collection's own arrays, e.g. for `plot_morphology_grid`.
    FeaturePropsCollection has no aspect-ratio field of its own; this is
    computed separately via `geom_metrics_3d.feature_aspect_ratio_bbox`.

    Returns
    -------
    np.ndarray, one value per feature.
    """
    import numpy as np
    from upxo.pxtal.fm_steel_3d.feature_props_3d import _LGI_BUILDERS
    from upxo.pxtal.fm_steel_3d.geom_metrics_3d import feature_aspect_ratio_bbox

    lgi, _s2i, _i2s = _LGI_BUILDERS[level](fm)
    ar_by_label = feature_aspect_ratio_bbox(lgi)
    active_labels = np.unique(lgi)
    active_labels = active_labels[active_labels > 0]
    return ar_by_label[active_labels]


def _hist_kde(ax, values, bins=20, color='#2B6CB0'):
    """Density-normalized histogram + Gaussian-KDE overlay of `values` on
    `ax` -- the "add a KDE to every distribution" building block shared by
    `plot_distribution` and `plot_morphology_grid`. Falls back to a plain
    histogram (no KDE line) if scipy is unavailable or there are too few/
    degenerate values for a meaningful density estimate.
    """
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    ax.hist(values, bins=bins, density=True, alpha=0.55, color=color, edgecolor='none')
    if len(values) >= 2 and len(np.unique(values)) > 3:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            xs = np.linspace(values.min(), values.max(), 200)
            ax.plot(xs, kde(xs), color=color, linewidth=1.8)
        except Exception:
            pass


def plot_distribution(collection, prop='vol_equiv_diameter_array', bins=30, ax=None):
    """Density histogram + KDE overlay of one property array from a
    FeaturePropsCollection (see `distribution_stats`).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    values = getattr(collection, prop)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    _hist_kde(ax, values, bins=bins)
    ax.set_xlabel(prop)
    ax.set_ylabel('Density')
    return ax


def plot_morphology_grid(collection, aspect_ratios, bins=20, figsize=(14, 7), suptitle=None):
    """2x4 grid of density-histogram + KDE-overlay plots for the 7 standard
    1st-order morphological properties: mean volume, surface area,
    surface-area-to-volume ratio, aspect ratio, junction-line segment
    length, triple-junction-point count, and coordination number.
    `aspect_ratios` from `compute_aspect_ratios` (not part of
    FeaturePropsCollection itself).

    Returns
    -------
    (matplotlib.figure.Figure, np.ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    specs = [(getattr(collection, prop), label) for prop, label in MORPHOLOGY_PROPS]
    specs.insert(3, (aspect_ratios, 'Aspect Ratio'))  # keep the requested display order

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    for ax, (values, label) in zip(axes.flat, specs):
        _hist_kde(ax, values, bins=bins)
        ax.set_title(label, fontsize=10)
    for ax in axes.flat[len(specs):]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def _randomize_grid_ids(lgi, seed=None):
    """Relabels `lgi`'s feature IDs to a random permutation of the same
    value set, via `gsdataops.gid_ops.shuffleLFIIDs` -- the data-side half
    of avoiding "adjacent labels get visually-similar colors": the GUI's
    own Visualization Settings page builds its `block_idx_map`/
    `block_fid_map` this same way (a random permutation of `1..n_blocks`)
    specifically because `feature_props_3d._build_block_lgi`/
    `_build_subblock_lgi` (what this module's own LGI builders use) label
    features in plain dict-insertion order, and spatially-clustered
    features (many blocks within the same grain/PAG, say) often end up
    numerically close together too under that scheme -- so even a
    colormap built to decorrelate ADJACENT labels (see
    `build_distinguishable_cmap`) is fighting labels that were never
    spatially decorrelated in the first place. Doing both (this function
    for the labels, `build_distinguishable_cmap` for the colormap) is the
    closest notebook-side match to the GUI's own approach.

    `seed`: if given, seeds numpy's GLOBAL random state before shuffling
    (matches `shuffleLFIIDs`'s own use of `np.random.shuffle`, which reads
    that global state rather than taking a local Generator) -- None uses
    whatever state numpy is already in, exactly like every other
    `shuffleLFIIDs` call already made throughout this codebase (e.g. the
    Monte Carlo base-grain-structure step).
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    from upxo.gsdataops.gid_ops import shuffleLFIIDs
    return shuffleLFIIDs(None, lgi)


def render_3d(fm, level='grain', jupyter_backend='trame', cmap=None, cmap_seed=42,
              randomize_ids=True, id_seed=None):
    """Opens an interactive PyVista render of `fm` at `level`, coloured by
    feature ID.

    `randomize_ids=True` (default) relabels the feature IDs themselves
    before rendering (see `_randomize_grid_ids`, seeded by `id_seed`) --
    the data-side half of the fix; `cmap`/`cmap_seed` (below) are the
    colormap-side half. Matches the GUI's own Visualization Settings page
    convention of doing both together.

    `cmap`: None (default) auto-builds a colormap that stays
    distinguishable even for hundreds of features (see
    `GrainStructureViz3D.build_distinguishable_cmap`, using `cmap_seed` for
    its shuffle) -- a fixed 20-color qualitative map ('tab20') otherwise
    forces many features to share an identical color once the feature
    count exceeds 20, which is what made a genuinely fine block hierarchy
    render as a handful of indistinguishable blobs instead of many thin
    plates. Pass any matplotlib colormap name/object to use it directly
    instead (bypassing the auto-build and `cmap_seed` both).
    `jupyter_backend='trame'` (default) embeds a rotatable widget directly
    in the notebook cell instead of a static screenshot or a separate
    blocking window; pass None to restore that original behaviour.

    Explicitly pins the color range to the full `[0, max(lgi)]` ID space
    (`clim`), matching the GUI's own 2D preview convention
    (`pages_base.py`'s `_on_slice_update`, `Normalize(vmin=0, vmax=n_ids)`)
    -- without this, PyVista auto-scales the color range to whichever
    subset of IDs happens to be exposed on the rendered surface (this
    render only ever shows the domain's outer shell -- most features
    don't reach it), silently remapping which colormap entry every ID
    actually gets away from what `cmap` was built assuming.
    """
    from upxo.pxtal.fm_steel_3d.feature_props_3d import _LGI_BUILDERS
    from upxo.pxtal.fm_steel_3d.viz.grain_structure_viz_3d import GrainStructureViz3D

    lgi, _s2i, _i2s = _LGI_BUILDERS[level](fm)
    if randomize_ids:
        lgi = _randomize_grid_ids(lgi, seed=id_seed)
    n_ids = int(lgi.max())
    if cmap is None:
        cmap = GrainStructureViz3D.build_distinguishable_cmap(n_ids, seed=cmap_seed)
    viz = GrainStructureViz3D()
    viz.lfi_to_polydata(lgi, lgi, voxel_size=fm.voxel_size, cmap=cmap, clim=(0, n_ids),
                        title=f'FM Steel 3D -- {level}', jupyter_backend=jupyter_backend)


def render_3d_exploded(stages, level='block', explosion_factor=0.5, jupyter_backend='trame',
                       cmap=None, cmap_seed=42, randomize_ids=True, id_seed=None):
    """Interactive PyVista render of `level`'s features ('block' or
    'subblock' only -- PAG/packet have no meaningful "parent" to explode
    away from), each rigidly translated away from its own parent's
    centroid by `explosion_factor` (0 = true, unexploded positions) --
    mirrors the GUI's own "BLK/SBLK Exploded View" (Visualization
    Settings page, windows W11/W12, via
    `GrainStructureViz3D.build_exploded_block_grid`/
    `build_exploded_subblock_grid`).

    Pulls apart features that would otherwise be hidden inside a solid
    volume: `render_3d`'s voxel-surface technique can only ever show
    whichever features happen to be exposed on the domain's outer shell
    (most aren't, for a fine-grained hierarchy) -- exploding relocates
    each feature away from its siblings so previously-buried ones become
    visible without needing a clip plane.

    `stages`: a `steps_raw_export.PipelineStages` (or the real GUI App)
    exposing whichever of `_fm_base`/`_fm_with_pags`/`_fm_with_blocks`/
    `_fm_with_subblocks` this level needs. `randomize_ids`/`id_seed`/
    `cmap`/`cmap_seed` -- see `render_3d`; the exploded grid's own
    labelling is plain sequential dict-insertion order (same as
    `feature_props_3d`'s LGI builders), so the same data-side/colormap-side
    randomization applies here too.

    Returns
    -------
    int : number of features exploded (n_blocks or n_subblocks).
    """
    from upxo.pxtal.fm_steel_3d.viz.grain_structure_viz_3d import GrainStructureViz3D

    if level == 'block':
        fm_base, fm_pag, fm_blk = stages._fm_base, stages._fm_with_pags, stages._fm_with_blocks
        grid, n_features = GrainStructureViz3D.build_exploded_block_grid(
            fm_blk.all_blocks, fm_pag.clusters_dict, fm_base.grain_locs,
            fm_pag.grain_to_pag_id, fm_blk.grain_to_blocks_map, explosion_factor)
        voxel_size = fm_blk.voxel_size
    elif level == 'subblock':
        fm_blk, fm_sub = stages._fm_with_blocks, stages._fm_with_subblocks
        grid, n_features = GrainStructureViz3D.build_exploded_subblock_grid(
            fm_sub.all_subblocks, fm_blk.all_blocks, fm_sub.block_to_subblocks_map, explosion_factor)
        voxel_size = fm_sub.voxel_size
    else:
        raise ValueError(f"render_3d_exploded only supports level='block' or 'subblock', got {level!r}")

    if randomize_ids:
        grid = _randomize_grid_ids(grid, seed=id_seed)
    n_ids = int(grid.max())
    if cmap is None:
        cmap = GrainStructureViz3D.build_distinguishable_cmap(n_ids, seed=cmap_seed)
    viz = GrainStructureViz3D()
    viz.lfi_to_polydata(
        grid, grid, voxel_size=voxel_size, cmap=cmap, clim=(0, n_ids),
        title=f'FM Steel 3D -- Exploded {level} View (factor={explosion_factor:g})',
        jupyter_backend=jupyter_backend)
    return n_features


_LEVEL_TAG = {'pag': 'pag', 'packet': 'pck', 'block': 'blk', 'subblock': 'sblk'}


def get_level_orientations(stages, level):
    """The {id: (phi1, Phi, phi2)} orientation dict for one hierarchy
    level ('pag' | 'packet' | 'block' | 'subblock' -- NOT 'grain', which
    has no orientation of its own). Reuses
    `raw_export.collect_level_data`, which already derives Packet's
    orientation as the mean of its constituent blocks'
    (`orientation_mean_3d.compute_packet_mean_orientations`) -- the exact
    same derivation for both raw export and visualization, not a second
    copy of it.

    `stages`: a `steps_raw_export.PipelineStages` (or the real GUI App --
    anything duck-typed the same way, exposing `_fm_with_pags`/
    `_fm_with_blocks`/`_fm_with_orientations`/`_fm_with_subblocks`).

    Returns
    -------
    dict, or None if this level has no orientation data yet (not reached,
    or -- for 'packet' -- the block orientations it depends on aren't set).
    """
    from upxo.pxtal.fm_steel_3d.raw_export import collect_level_data
    orientations, _fields = collect_level_data(stages, _LEVEL_TAG[level])
    return orientations


def plot_pole_figure(orientations, pole_family='100', ax=None, crystal_symmetry='cubic',
                     apply_sample_symmetries=True, use_rd=True, use_td=True, use_nd=True):
    """Scatter pole figure of `orientations` (a {id: (phi1, Phi, phi2)}
    dict, e.g. from `get_level_orientations`), in Bunge-Euler degrees.

    Includes BOTH crystal and sample symmetry:
    - Crystal symmetry (`crystal_symmetry`, default 'cubic'): handled by
      `PoleFigure` itself -- every crystallographically-equivalent pole of
      `pole_family` is plotted for each orientation, not just the one
      literal pole direction.
    - Sample symmetry: NOT handled by `PoleFigure` internally (it only
      ever applies crystal symmetry) -- before construction, every
      orientation is first replicated through the closed macroscopic
      sample-symmetry group (180-degree rotations about the sample's RD/
      TD/ND axes, `use_rd`/`use_td`/`use_nd`, via
      `pole_figure.compute_sample_symmetry_group` +
      `pole_figure.apply_sample_symmetry` -- the standard way already-
      fixed orientation data is symmetrized for a sample-symmetric pole
      figure). `gids` are tiled to match the replicated orientation count.
      Set `apply_sample_symmetries=False` to skip this and plot only the
      orientations as given (crystal symmetry alone).

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    import numpy as np
    from upxo.viz.xphy.pole_figure import (
        PoleFigure, compute_sample_symmetry_group, apply_sample_symmetry)

    eulers = np.array(list(orientations.values()), dtype=float)
    gids = np.array(list(orientations.keys()))

    group_ops = compute_sample_symmetry_group(
        apply_sample_symmetries=apply_sample_symmetries, use_rd=use_rd, use_td=use_td, use_nd=use_nd)
    if len(group_ops) > 1:
        eulers = apply_sample_symmetry(eulers, 'euler_deg', group_ops)
        gids = np.tile(gids, len(group_ops))

    pf = PoleFigure(eulers, convention='euler_deg', gids=gids, symmetry=crystal_symmetry)
    return pf.plot_scatter(pole_family=pole_family, ax=ax, color_by='hemisphere')


def misorientation_distribution(orientations, collection, n_bins=36, angle_range=(0.0, 65.0)):
    """Neighbour-pair (grain-boundary) misorientation-angle distribution
    for one hierarchy level, using each feature's real adjacency
    (`collection`'s own FeatureProps.neighbour_ids, from
    `distribution_stats(fm, level)`) so only physically adjacent pairs are
    compared -- via `xtalphy.crystal_orientation.
    grain_boundary_misorientation_distribution`. A pair is skipped if
    either feature has no entry in `orientations` (e.g. an isolated grain
    with no PAG orientation).

    Returns
    -------
    dict : {'misorientation_angles', 'misorientation_axes', 'pairs',
    'hist_counts', 'hist_bin_edges', 'hist_bin_centers', 'mean_angle',
    'median_angle', 'std_angle', 'n_pairs'}, or None if fewer than 2
    oriented features are adjacent to each other.
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import grain_boundary_misorientation_distribution

    ids = list(orientations.keys())
    if len(ids) < 2:
        return None
    id_to_idx = {gid: i for i, gid in enumerate(ids)}
    euler_array = np.array([orientations[gid] for gid in ids], dtype=float)

    pairs = []
    seen = set()
    for fp in collection:
        gid_a = fp.feature_id
        if gid_a not in id_to_idx:
            continue
        for gid_b in fp.neighbour_ids:
            if gid_b not in id_to_idx:
                continue
            key = tuple(sorted((str(gid_a), str(gid_b))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((id_to_idx[gid_a], id_to_idx[gid_b]))
    if not pairs:
        return None

    # grain_ids intentionally omitted: grain_boundary_misorientation_distribution
    # requires it int-castable, but block/sub-block IDs are strings (e.g.
    # "B_1_4_1") -- `pairs` already indexes `euler_array` directly, which is
    # all the histogram/summary stats below actually need.
    return grain_boundary_misorientation_distribution(
        euler_array, np.asarray(pairs, dtype=int), n_bins=n_bins, angle_range=angle_range)


def plot_misorientation_distribution(mdf_result, ax=None, title=None, bins=None):
    """Density histogram + KDE overlay of a `misorientation_distribution(...)`
    result's raw per-pair angles (`mdf_result['misorientation_angles']`) --
    same `_hist_kde` building block as every other distribution plot here,
    rather than the pre-binned `hist_counts`/`hist_bin_edges` bar plot
    (still available on `mdf_result` for callers who want the raw bin
    counts instead). `bins` defaults to the same bin count
    `grain_boundary_misorientation_distribution` used for its own
    `hist_counts` (`len(mdf_result['hist_bin_centers'])`), for a directly
    comparable view.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    if bins is None:
        bins = len(mdf_result['hist_bin_centers'])
    _hist_kde(ax, mdf_result['misorientation_angles'], bins=bins)
    ax.set_xlabel('Misorientation angle (deg)')
    ax.set_ylabel('Density')
    if title:
        ax.set_title(title)
    return ax
