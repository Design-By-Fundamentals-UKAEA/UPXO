"""
feature_props_3d.py
===================
Grain and twin morphological property extraction for the twinned
simple 3D pipeline.
"""

import numpy as np
from typing import Dict, Optional, List

from upxo._sup import progress


def compute_grain_volumes(
        lgi: np.ndarray,
        grain_ids: Optional[List[int]] = None,
) -> Dict[int, int]:
    """Return voxel count per grain using a single np.bincount pass (O(N_vox))."""
    counts = np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1)
    if grain_ids is None:
        grain_ids = [int(g) for g in np.unique(lgi) if g > 0]
    return {gid: int(counts[gid]) for gid in grain_ids}


def compute_twin_volume_fraction(
        lgi: np.ndarray,
        twin_gids: List[int],
) -> float:
    """Compute twin VF using a single np.bincount pass (O(N_vox))."""
    if not twin_gids:
        return 0.0
    counts   = np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1)
    total_vox = int(counts[1:].sum())   # exclude label 0 (background)
    if total_vox == 0:
        return 0.0
    twin_vox = int(sum(counts[g] for g in twin_gids if g < len(counts)))
    return twin_vox / total_vox


def compute_equivalent_diameters(
        volumes: Dict[int, int],
        voxel_size: float,
) -> Dict[int, float]:
    factor = voxel_size ** 3
    pi = np.pi
    return {
        gid: float((6.0 / pi * vox * factor) ** (1.0 / 3.0))
        for gid, vox in volumes.items()
    }


def grain_role_statistics(
        volumes: Dict[int, int],
        twin_role: Dict[int, str],
        voxel_size: float,
) -> Dict[str, Dict]:
    phys = voxel_size ** 3
    by_role: Dict[str, List[int]] = {}
    for gid, vox in volumes.items():
        role = twin_role.get(int(gid), 'non_host')
        by_role.setdefault(role, []).append(vox)

    stats = {}
    for role, vox_list in by_role.items():
        arr = np.array(vox_list, dtype=float) * phys
        stats[role] = {
            'count':         len(vox_list),
            'mean_vol_um3':  float(np.mean(arr)),
            'std_vol_um3':   float(np.std(arr)),
            'min_vol_um3':   float(np.min(arr)),
            'max_vol_um3':   float(np.max(arr)),
            'total_vox':     int(sum(vox_list)),
        }
    return stats


# ---------------------------------------------------------------------------
# EBSD vs synthetic comparison
# ---------------------------------------------------------------------------

def _descriptive_stats(arr: np.ndarray) -> dict:
    """Return mean, std, Q1, Q2, Q3 for a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan,
                    q1=np.nan, q2=np.nan, q3=np.nan)
    return dict(
        n   = int(arr.size),
        mean= float(np.mean(arr)),
        std = float(np.std(arr)),
        q1  = float(np.percentile(arr, 25)),
        q2  = float(np.percentile(arr, 50)),
        q3  = float(np.percentile(arr, 75)),
    )


def compute_ebsd_mc_comparison_stats(
        ebsd_ref: dict,
        sgc_ref: dict,
) -> dict:
    """
    Compute side-by-side descriptive statistics comparing the EBSD
    reference microstructure against the synthetic SGC structure.

    Parameters
    ----------
    ebsd_ref : dict
        ``miso_deg_full``    - 1-D ndarray  EBSD MDF angles (full, with twins)
        ``miso_deg_merged``  - 1-D ndarray  EBSD MDF angles (parent-state, merged)
        ``twin_thick_um``    - 1-D ndarray  EBSD twin lamella thicknesses (um)
        ``tvf_2d``           - scalar       EBSD 2D twin area fraction

    sgc_ref : dict
        ``miso_deg_posttwin``- 1-D ndarray  SGC post-twin MDF angles
        ``twin_thick_3d_um`` - 1-D ndarray  SGC actual 3D twin thicknesses (um)
        ``tvf_2d_slices``    - 1-D ndarray  per-slice 2D twin VF values
        ``tvf_3d``           - scalar       SGC 3D twin volume fraction

    Returns
    -------
    dict
        Nested dict with keys 'mdf', 'twin_thickness',
        'twin_volume_fraction', each containing 'ebsd' and 'mc' sub-dicts
        of descriptive statistics. Grain size (host/parent/twin/complete)
        is NOT part of this -- see compute_category_grain_size_stats,
        which replaced it with a properly 2D-vs-2D, role-split comparison.
    """
    return {
        'mdf': {
            'ebsd_full':   _descriptive_stats(ebsd_ref.get('miso_deg_full',   [])),
            'ebsd_merged': _descriptive_stats(ebsd_ref.get('miso_deg_merged', [])),
            'mc_posttwin': _descriptive_stats(sgc_ref.get('miso_deg_posttwin', [])),
        },
        'twin_thickness': {
            'ebsd':            _descriptive_stats(ebsd_ref.get('twin_thick_um',             [])),
            'mc_apparent_2d':  _descriptive_stats(sgc_ref.get('twin_thick_apparent_2d_um',  [])),
            'mc_actual_3d':    _descriptive_stats(sgc_ref.get('twin_thick_3d_um',            [])),
        },
        'twin_volume_fraction': {
            'ebsd_2d':     float(ebsd_ref.get('tvf_2d', np.nan)),
            'mc_2d_mean':  float(np.mean(sgc_ref['tvf_2d_slices']))
                           if len(sgc_ref.get('tvf_2d_slices', [])) > 0 else np.nan,
            'mc_2d_std':   float(np.std(sgc_ref['tvf_2d_slices']))
                           if len(sgc_ref.get('tvf_2d_slices', [])) > 0 else np.nan,
            'mc_3d':       float(sgc_ref.get('tvf_3d', np.nan)),
        },
    }


def print_ebsd_mc_comparison(stats: dict) -> None:
    """
    Print a formatted side-by-side comparison table from the output of
    :func:`compute_ebsd_mc_comparison_stats`.
    """
    sep  = '=' * 72
    sep2 = '-' * 72
    fmt  = '{:<34}  {:>10}  {:>10}  {:>10}'

    def row(label, ebsd_v, mc_v, unit=''):
        def _f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '     —'
            return f'{v:>10.3f}'
        return f'  {label:<32}  {_f(ebsd_v)}  {_f(mc_v)}  {unit}'

    print(sep)
    print('EBSD vs Synthetic MC  --  Statistical Comparison')
    print(sep)

    # ── MDF ────────────────────────────────────────────────────────────────
    mdf = stats['mdf']
    e   = mdf['ebsd_full']
    em  = mdf['ebsd_merged']
    mc  = mdf['mc_posttwin']
    print('\nMISORIENTATION DISTRIBUTION (MDF)')
    print(fmt.format('Property', 'EBSD (full)', 'EBSD (merged)', 'SGC post-twin'))
    print(sep2)
    for key, lbl in [('mean','Mean'),('std','Std dev'),
                      ('q1','Q1'),('q2','Median'),('q3','Q3')]:
        print(f'  {lbl:<32}  {e[key]:>10.2f}  {em[key]:>10.2f}  {mc[key]:>10.2f}  deg')
    print(f'  {"n (boundary pairs)":<32}  {e["n"]:>10d}  {em["n"]:>10d}  {mc["n"]:>10d}')

    # ── Twin thickness ──────────────────────────────────────────────────────
    print('\nTWIN LAMELLA THICKNESS  (EBSD=2D apparent; MC apparent=regionprops on slices; MC actual=2xhw from introduction)')
    print(fmt.format('Property', 'EBSD apparent 2D', 'SGC apparent 2D', 'SGC actual 3D'))
    print(sep2)
    tt_s = stats['twin_thickness']
    e_t   = tt_s['ebsd']
    mc2_t = tt_s.get('mc_apparent_2d', {})
    mc3_t = tt_s.get('mc_actual_3d', {})
    def _fv(v):
        import math
        return f'{v:>10.2f}' if (v is not None and not math.isnan(v)) else '         -'
    for key, lbl in [('mean','Mean'),('std','Std dev'),
                      ('q1','Q1'),('q2','Median'),('q3','Q3')]:
        print(f'  {lbl:<32}  {_fv(e_t.get(key,float("nan")))}  '
              f'{_fv(mc2_t.get(key,float("nan")))}  '
              f'{_fv(mc3_t.get(key,float("nan")))}  um')
    print(f'  {"n":<32}  {e_t.get("n",0):>10d}  '
          f'{mc2_t.get("n",0):>10d}  {mc3_t.get("n",0):>10d}')

    # ── Twin volume fraction ────────────────────────────────────────────────
    tvf = stats['twin_volume_fraction']
    print('\nTWIN VOLUME / AREA FRACTION')
    print(sep2)
    print(f'  {"EBSD 2D area fraction":<32}  {tvf["ebsd_2d"]:>10.4f}')
    if not np.isnan(tvf['mc_2d_mean']):
        print(f'  {"MC 2D slice mean +/- std":<32}  '
              f'{tvf["mc_2d_mean"]:>10.4f}  '
              f'(+/-{tvf["mc_2d_std"]:.4f})')
    print(f'  {"MC 3D volume fraction":<32}  {tvf["mc_3d"]:>10.4f}')
    print(sep)


# ---------------------------------------------------------------------------
# EBSD-vs-SGS raw data assembly -- shared by Summary Report (which reduces
# it to descriptive stats via compute_ebsd_mc_comparison_stats) and
# Distribution Viewer (which plots the raw arrays directly).
# ---------------------------------------------------------------------------

def assemble_ebsd_sgs_comparison_data(
        cleaner, tg, rg, parent_info, mdf, twin_thickness,
        n_slices_per_axis: int = 5,
        axes=('x', 'y', 'z'),
) -> tuple:
    """
    Assemble the ``(ebsd_ref, sgc_ref)`` raw-array input pair consumed by
    :func:`compute_ebsd_mc_comparison_stats`. Pulled out of
    ``SummaryReportPage.on_compute`` so Distribution Viewer can reuse the
    exact same assembly for its raw-array overlay plots instead of
    duplicating it.

    Parameters
    ----------
    cleaner : StructureCleaner3D
        Post-twin cleaned structure (``lgi_clean``, ``twin_role_clean``,
        ``all_quats_clean``).
    tg : TwinGenerator3D
        Post-introduction twin generator (``twin_halfwidths_vox``,
        ``summary()``, ``compute_achieved_2d_tvf``, ``base``).
    rg : repgen2d
        EBSD reference (``lfi_ebsd_merged``, ``quat_ebsd``, ``prop_ebsd``).
    parent_info : dict
        Output of ``rg.identify_parent_grains``.
    mdf : dict
        Full EBSD MDF, output of ``rg.compute_mdf_ebsd`` (must carry
        ``'miso_deg'``).
    twin_thickness : dict
        Output of ``rg.compute_mc_twin_thickness`` (must carry
        ``'thick_um'``).
    n_slices_per_axis, axes
        Forwarded to ``tg.compute_achieved_2d_tvf`` for the SGS 2D-slice
        twin-area-fraction distribution.

    Returns
    -------
    (ebsd_ref, sgc_ref) : tuple of dict
        See :func:`compute_ebsd_mc_comparison_stats`'s parameter docs.
    """
    from upxo.gsdataops.gid_ops import find_neighs2d, find_neighs3d
    from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats, expand_grain_quats_to_voxels

    vs = tg.base.voxel_size

    if getattr(rg, 'lfi_ebsd_merged', None) is None:
        rg.build_merged_ebsd_lfi(parent_info, plot=False)
    neigh_merged = find_neighs2d(rg.lfi_ebsd_merged.astype(np.int32), conn=4)
    ebsd_mdf_merged = compute_mdf_from_quats(
        rg.lfi_ebsd_merged, rg.quat_ebsd, neigh_merged,
        n_bins=65, angle_range=(0.0, 65.0))

    ebsd_ref = {
        'miso_deg_full':   np.asarray(mdf['miso_deg']),
        'miso_deg_merged': ebsd_mdf_merged['miso_deg'],
        'twin_thick_um':   np.asarray(twin_thickness['thick_um'], dtype=float),
        'tvf_2d':          tg.summary()['tvf_2d_ebsd'],
    }

    quat_3d_clean = expand_grain_quats_to_voxels(cleaner.lgi_clean, cleaner.all_quats_clean)
    neigh_post = find_neighs3d(cleaner.lgi_clean.astype(np.int32), conn=6)
    neigh_post_list = {int(g): list(ns) for g, ns in neigh_post.items()}
    mc_mdf_post = compute_mdf_from_quats(
        cleaner.lgi_clean, quat_3d_clean, neigh_post_list,
        n_bins=65, angle_range=(0.0, 65.0))

    twin_thick_3d_um = np.array(
        [2.0 * hw * vs for hw in tg.twin_halfwidths_vox.values()], dtype=float)

    twin_gids_clean = [gid for gid, role in cleaner.twin_role_clean.items()
                        if role in ('primary_twin', 'secondary_twin')]
    tvf_final_3d = compute_twin_volume_fraction(cleaner.lgi_clean, twin_gids_clean)

    achieved_2d = tg.compute_achieved_2d_tvf(
        n_slices_per_axis=n_slices_per_axis, axes=axes, return_raw=True)

    sgc_ref = {
        'miso_deg_posttwin': mc_mdf_post['miso_deg'],
        'twin_thick_3d_um':  twin_thick_3d_um,
        'tvf_2d_slices':     np.asarray(achieved_2d.get('ratios', []), dtype=float),
        'tvf_3d':            tvf_final_3d,
    }
    return ebsd_ref, sgc_ref


# ---------------------------------------------------------------------------
# SGS per-role 2D morphological/topological property distributions --
# Distribution Viewer's SGS side for the 5 EBSD-comparable properties
# (area, aspect_ratio, perimeter, solidity, n_neighbours).
# ---------------------------------------------------------------------------

#: twin_role_clean role name -> Distribution Viewer's DIST_LEVELS key.
#: 'secondary_twin' resolves to 'seca' (outward, parent is a host) or
#: 'secb' (inward, parent is a primary twin) -- the same 2a/2b split
#: AbaqusExporter3D uses for its ELSET naming.
def _sgs_role_to_level(gid, twin_role_clean, twin_parent_of_clean) -> str:
    role = twin_role_clean.get(gid, 'non_host')
    if role == 'non_host':
        return 'nonhost'
    if role == 'host':
        return 'host'
    if role == 'primary_twin':
        return 'primary'
    if role == 'secondary_twin':
        parent = twin_parent_of_clean.get(gid)
        parent_role = twin_role_clean.get(parent)
        return 'secb' if parent_role == 'primary_twin' else 'seca'
    return 'nonhost'


def compute_sgs_role_property_distributions(
        lgi_clean: np.ndarray,
        twin_role_clean: Dict[int, str],
        twin_parent_of_clean: Dict[int, int],
        selected_props: List[str],
        selected_levels: List[str],
        n_slices_per_axis: int = 5,
        axes=('x', 'y', 'z'),
        voxel_size: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Per-twin-role-level distributions of area / aspect_ratio / perimeter /
    solidity / n_neighbours, sampled from 2D cross-sections of the cleaned
    3D structure -- the SGS-side equivalent of EBSD's inherently-2D grain
    morphology, so the two are directly comparable. Mirrors
    ``TwinGenerator3D.compute_achieved_2d_tvf``'s slice-sampling
    convention (evenly-spaced slices per axis, native axis-label mapping,
    no re-labelling -- a grain's 2D cross-section keeps its 3D grain ID,
    so ``twin_role_clean``/``twin_parent_of_clean`` apply directly).

    Parameters
    ----------
    lgi_clean : ndarray (nz, ny, nx)
        ``cleaner.lgi_clean`` -- pipeline-native axis order.
    twin_role_clean, twin_parent_of_clean : dict
        ``cleaner.twin_role_clean`` / ``cleaner.twin_parent_of_clean``.
    selected_props : list of str
        Subset of ``('area', 'aspect_ratio', 'perimeter', 'solidity',
        'n_neighbours', 'eqdia')``.
    selected_levels : list of str
        Subset of ``('nonhost', 'host', 'primary', 'seca', 'secb',
        'complete')`` -- 'complete' is not mutually exclusive with the
        other five: every region counts toward it in addition to its own
        specific level, when both are selected.
    n_slices_per_axis, axes
        Evenly-spaced 2D cross-sections sampled per axis in *axes*.
    voxel_size : float
        Physical voxel edge length (um); scales area (um^2) and
        perimeter (um).

    Returns
    -------
    dict
        ``{prop_name: {level_key: ndarray}}``.
    """
    from skimage.measure import regionprops
    from upxo.gsdataops.grid_ops import section_from_3d
    from upxo.gsdataops.gid_ops import find_neighs2d

    axis_map = {'x': 2, 'y': 1, 'z': 0}
    axes_int = [axis_map[a.lower()] for a in axes if a.lower() in axis_map]

    morph_props = [p for p in selected_props
                    if p in ('area', 'aspect_ratio', 'perimeter', 'solidity', 'eqdia')]
    needs_neigh = 'n_neighbours' in selected_props
    want_complete = 'complete' in selected_levels

    values: Dict[str, Dict[str, list]] = {
        p: {lv: [] for lv in selected_levels} for p in selected_props}

    for ax in axes_int:
        domain_size = lgi_clean.shape[ax]
        positions = np.linspace(0, domain_size - 1, n_slices_per_axis, dtype=int)
        for pos in positions:
            lgi_2d = section_from_3d(lgi_clean, axis=ax, location=int(pos))
            if lgi_2d.max() <= 0:
                continue

            if morph_props:
                for region in regionprops(lgi_2d.astype(np.int32)):
                    lv = _sgs_role_to_level(region.label, twin_role_clean, twin_parent_of_clean)
                    # 'complete' is not a mutually-exclusive role -- every
                    # region counts toward it in addition to its own
                    # specific level (when that level was also selected).
                    levels_hit = []
                    if lv in selected_levels:
                        levels_hit.append(lv)
                    if want_complete:
                        levels_hit.append('complete')
                    if not levels_hit:
                        continue
                    if 'area' in morph_props:
                        for l in levels_hit:
                            values['area'][l].append(region.area * voxel_size ** 2)
                    if 'perimeter' in morph_props:
                        for l in levels_hit:
                            values['perimeter'][l].append(region.perimeter * voxel_size)
                    if 'aspect_ratio' in morph_props and region.minor_axis_length > 0:
                        for l in levels_hit:
                            values['aspect_ratio'][l].append(
                                region.major_axis_length / region.minor_axis_length)
                    if 'solidity' in morph_props:
                        for l in levels_hit:
                            values['solidity'][l].append(region.solidity)
                    if 'eqdia' in morph_props:
                        for l in levels_hit:
                            values['eqdia'][l].append(region.equivalent_diameter * voxel_size)

            if needs_neigh:
                neigh = find_neighs2d(lgi_2d.astype(np.int32), conn=4)
                for gid, nbrs in neigh.items():
                    lv = _sgs_role_to_level(gid, twin_role_clean, twin_parent_of_clean)
                    levels_hit = []
                    if lv in selected_levels:
                        levels_hit.append(lv)
                    if want_complete:
                        levels_hit.append('complete')
                    for l in levels_hit:
                        values['n_neighbours'][l].append(len(nbrs))

    return {
        p: {lv: np.array(vals, dtype=float) for lv, vals in lv_dict.items()}
        for p, lv_dict in values.items()
    }


# ── 4-category EBSD-vs-synthetic grain-size comparison (CED + IGS) ───────
# Categories mirror ebsdviz.compute_four_category_grain_sets on the EBSD
# side: 'parent' (genuinely twin-free), 'host' (hosts a twin, remaining
# matrix region), 'twin' (primary or secondary lamella, pooled), and
# 'complete' (every grain, no role filtering at all).

_SGS_CATEGORY_FROM_ROLE = {
    'non_host': 'parent',
    'host': 'host',
    'primary_twin': 'twin',
    'secondary_twin': 'twin',
}

SGS_GRAIN_SIZE_CATEGORIES = ('parent', 'host', 'twin', 'complete')


def sgs_gid_to_category(gid, twin_role_clean: Dict[int, str]) -> str:
    """Maps a cleaned synthetic structure's per-grain twin_role_clean
    label onto one of SGS_GRAIN_SIZE_CATEGORIES -- the synthetic-side
    counterpart of ebsdviz.compute_four_category_grain_sets's EBSD
    classification. An 'intermediate' EBSD grain (twin of its host AND
    itself hosting a secondary twin) has no separate synthetic label --
    the region left after carving a secondary twin out of a primary twin
    simply stays role='primary_twin' -- so both sides agree to count
    that configuration as 'twin', not 'host' (matches
    compute_four_category_grain_sets pooling EBSD's intermediates into
    its 'twin' category)."""
    role = twin_role_clean.get(gid, 'non_host')
    return _SGS_CATEGORY_FROM_ROLE.get(role, 'parent')


def compute_sgs_category_ced_by_slice(
        lgi_clean: np.ndarray,
        twin_role_clean: Dict[int, str],
        category: str,
        axis: str,
        n_slices_per_axis: int,
        voxel_size: float = 1.0,
) -> List[np.ndarray]:
    """Per-slice circle-equivalent-diameter (CED) arrays for one
    4-category grain population, sampled from 2D cross-sections of the
    FINAL cleaned structure along one axis -- the per-slice granularity
    the KDE curve-set plot needs (one thin curve per slice, see
    viz_distribution_kde.py), as distinct from
    compute_sgs_role_property_distributions's already-pooled-across-
    slices output.

    category : one of SGS_GRAIN_SIZE_CATEGORIES ('complete' = no filter).

    Returns
    -------
    list of ndarray
        One array per sampled slice position, each holding that slice's
        matching grains' CED values (um). A slice with none has an empty
        array -- callers decide whether to skip it for KDE fitting.
    """
    from skimage.measure import regionprops
    from upxo.gsdataops.grid_ops import section_from_3d

    axis_map = {'x': 2, 'y': 1, 'z': 0}
    ax = axis_map[axis.lower()]
    domain_size = lgi_clean.shape[ax]
    positions = np.linspace(0, domain_size - 1, n_slices_per_axis, dtype=int)

    per_slice = []
    for pos in positions:
        lgi_2d = section_from_3d(lgi_clean, axis=ax, location=int(pos))
        vals = []
        for region in regionprops(lgi_2d.astype(np.int32)):
            if category != 'complete' and sgs_gid_to_category(
                    region.label, twin_role_clean) != category:
                continue
            vals.append(region.equivalent_diameter * voxel_size)
        per_slice.append(np.array(vals, dtype=float))
    return per_slice


def compute_host_ced_by_axis(
        lgi: np.ndarray,
        host_grain_ids,
        axis: str,
        n_slices_per_axis: int,
        voxel_size: float = 1.0,
) -> np.ndarray:
    """Pooled circle-equivalent-diameter (CED, um) of the grains in
    `host_grain_ids`, from 2D cross-sections of `lgi` sampled along one
    axis -- the pre-twin-generation counterpart of
    compute_sgs_category_ced_by_slice, keyed on direct grain-id-set
    membership (host_grain_ids, from Twin-Host Allocation) instead of a
    post-twin twin_role_clean label, since no twin exists yet at this
    pipeline stage (the context for Twin Generation's minimum lamella
    thickness). Pooled across every sampled slice on this one axis into a
    single flat array -- feeds one mean/min/max/Q1 summary, not a
    per-slice KDE curve set.
    """
    from skimage.measure import regionprops
    from upxo.gsdataops.grid_ops import section_from_3d

    host_ids = set(int(g) for g in host_grain_ids)
    axis_map = {'x': 2, 'y': 1, 'z': 0}
    ax = axis_map[axis.lower()]
    domain_size = lgi.shape[ax]
    positions = np.linspace(0, domain_size - 1, n_slices_per_axis, dtype=int)

    vals = []
    for pos in positions:
        lgi_2d = section_from_3d(lgi, axis=ax, location=int(pos))
        for region in regionprops(lgi_2d.astype(np.int32)):
            if region.label in host_ids:
                vals.append(region.equivalent_diameter * voxel_size)
    return np.array(vals, dtype=float)


def ced_summary_stats(arr: np.ndarray) -> dict:
    """{'mean', 'min', 'max', 'q1', 'n'} for one CED array -- the
    mean/min/max/lower-quartile reference stats shown alongside a
    thickness-setting control, as distinct from _stat_block's
    mean/median/std/n (built for the Summary Report's comparison
    tables). All 0.0/0 if empty."""
    if arr.size == 0:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'q1': 0.0, 'n': 0}
    return {'mean': float(arr.mean()), 'min': float(arr.min()),
           'max': float(arr.max()), 'q1': float(np.percentile(arr, 25)),
           'n': int(arr.size)}


def compute_sgs_category_igs(
        lgi_clean: np.ndarray,
        twin_role_clean: Dict[int, str],
        category: str,
        axes,
        n_slices_per_axis,
        voxel_size: float,
        mean_ced_um: float,
) -> dict:
    """Pooled 2D intercept grain size (IGS) for one 4-category grain
    population, across every selected axis's 2D cross-sections of the
    final cleaned structure, via planar_intercept_grain_size with a
    category mask (out-of-category voxels treated as background so a
    line never counts an intercept against the "wrong" population).

    For every sampled 2D slice, BOTH of that slice's own in-plane
    directions are sampled (its rows and its columns), each sized
    independently from that direction's own physical length divided by
    `mean_ced_um` -- not the 3D domain's axis length, and not the other
    in-plane direction's length.

    n_slices_per_axis : int or dict
        Scalar (same count for every axis) or ``{axis: count}`` -- same
        dict-or-scalar convention as compute_slice_properties_by_axis.

    mean_ced_um : this SAME category's own measured mean CED (from
    compute_sgs_category_ced_by_slice, pooled across whatever
    slices/axes the caller used) -- sizes the line count for this
    category only; never the EBSD side's CED, and never another
    category's CED (small, twin-sized features would otherwise go
    under-sampled -- see the design discussion this mirrors).

    Returns
    -------
    dict
        {'mean', 'median', 'std', 'n', 'raw_segments'} pooled across every
        sampled line on every sampled slice on every requested axis.
    """
    from upxo.gsdataops.grid_ops import section_from_3d
    from upxo.pxtal.gridops import planar_intercept_grain_size

    n_slices_map = n_slices_per_axis if isinstance(n_slices_per_axis, dict) \
        else {a: n_slices_per_axis for a in axes}

    axis_map = {'x': 2, 'y': 1, 'z': 0}
    pooled = []
    for axis in axes:
        ax = axis_map[axis.lower()]
        domain_size = lgi_clean.shape[ax]
        positions = np.linspace(0, domain_size - 1, n_slices_map[axis], dtype=int)
        for pos in positions:
            lgi_2d = section_from_3d(lgi_clean, axis=ax, location=int(pos))
            if category == 'complete':
                mask = (lgi_2d > 0).astype(np.int32)
            else:
                mask = np.zeros(lgi_2d.shape, dtype=np.int32)
                for gid in np.unique(lgi_2d[lgi_2d > 0]):
                    if sgs_gid_to_category(int(gid), twin_role_clean) == category:
                        mask[lgi_2d == gid] = 1

            h, w = lgi_2d.shape
            n_lines_rows = max(1, int(round((w * voxel_size) / mean_ced_um))) \
                if mean_ced_um > 0 else 1
            n_lines_cols = max(1, int(round((h * voxel_size) / mean_ced_um))) \
                if mean_ced_um > 0 else 1

            res_rows = planar_intercept_grain_size(
                lgi_2d, voxel_size, 'x', n_lines_rows,
                phase_array=mask, valid_phase_ids=[1])
            res_cols = planar_intercept_grain_size(
                lgi_2d, voxel_size, 'y', n_lines_cols,
                phase_array=mask, valid_phase_ids=[1])
            if res_rows['raw_segments'].size:
                pooled.append(res_rows['raw_segments'])
            if res_cols['raw_segments'].size:
                pooled.append(res_cols['raw_segments'])

    raw = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    if raw.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'n': 0,
               'raw_segments': raw}
    return {
        'mean': float(raw.mean()), 'median': float(np.median(raw)),
        'std': float(raw.std()), 'n': int(raw.size),
        'raw_segments': raw,
    }


def compute_ebsd_category_ced(prop_ebsd: dict, category_gids) -> np.ndarray:
    """EBSD-side CED values (um) for one 4-category grain set -- the
    EBSD-side counterpart of compute_sgs_category_ced_by_slice. No
    slicing needed: the EBSD map is already 2D, and `prop_ebsd[gid]
    ['eq_diameter']` is already in physical units (see
    ebsd_reader._char_lfi)."""
    return np.array(
        [prop_ebsd[g]['eq_diameter'] for g in category_gids if g in prop_ebsd],
        dtype=float)


def compute_ebsd_category_igs(
        lfi_ebsd: np.ndarray,
        category_gids,
        voxel_size: float,
        mean_ced_um: float,
) -> dict:
    """EBSD-side pooled 2D intercept grain size for one 4-category grain
    set, sampled directly on the real EBSD map (both in-plane
    directions) via planar_intercept_grain_size, masked to this category
    only -- the EBSD-side counterpart of compute_sgs_category_igs.

    voxel_size : EBSD step size (um/pixel).
    mean_ced_um : this SAME category's own EBSD-measured mean CED (from
    compute_ebsd_category_ced) -- never the synthetic side's CED.
    """
    from upxo.pxtal.gridops import planar_intercept_grain_size

    mask = np.isin(lfi_ebsd, np.asarray(list(category_gids))).astype(np.int32)
    h, w = lfi_ebsd.shape
    n_lines_rows = max(1, int(round((w * voxel_size) / mean_ced_um))) \
        if mean_ced_um > 0 else 1
    n_lines_cols = max(1, int(round((h * voxel_size) / mean_ced_um))) \
        if mean_ced_um > 0 else 1

    res_rows = planar_intercept_grain_size(
        lfi_ebsd, voxel_size, 'x', n_lines_rows, phase_array=mask, valid_phase_ids=[1])
    res_cols = planar_intercept_grain_size(
        lfi_ebsd, voxel_size, 'y', n_lines_cols, phase_array=mask, valid_phase_ids=[1])
    raw = np.concatenate([res_rows['raw_segments'], res_cols['raw_segments']])
    if raw.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'n': 0, 'raw_segments': raw}
    return {
        'mean': float(raw.mean()), 'median': float(np.median(raw)),
        'std': float(raw.std()), 'n': int(raw.size), 'raw_segments': raw,
    }


def _stat_block(arr: np.ndarray) -> dict:
    """{'mean', 'median', 'std', 'n'} for one array, all 0.0/0 if empty."""
    if arr.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'n': 0}
    return {'mean': float(arr.mean()), 'median': float(np.median(arr)),
           'std': float(arr.std()), 'n': int(arr.size)}


def compute_category_grain_size_stats(
        lfi_ebsd: np.ndarray,
        prop_ebsd: dict,
        parent_info: dict,
        lgi_clean: np.ndarray,
        twin_role_clean: Dict[int, str],
        ebsd_voxel_size: float,
        sgs_voxel_size: float,
        axes=('x', 'y', 'z'),
        n_slices_per_axis=5,
) -> dict:
    """{category: {'ebsd_ced', 'sgs_ced', 'ebsd_igs', 'sgs_igs'}} pooled
    Mean/Median/Std/N stats for every one of SGS_GRAIN_SIZE_CATEGORIES --
    the numbers the Summary Report's 4 grain-size tables are built from
    (replaces the old pre-twin-3D-vs-EBSD-2D `host_eqdia` comparison).

    CED is pooled across every requested axis/slice on the synthetic
    side and measured directly (no slicing) on the EBSD side. IGS is
    derived the same way, each side sizing its own sample-line count
    from that SAME category's own just-computed mean CED on that same
    side -- never borrowed from the other side or another category.

    n_slices_per_axis : int or dict, same convention as
    compute_sgs_category_igs / compute_slice_properties_by_axis.
    """
    from upxo.viz.ebsdviz import compute_four_category_grain_sets

    cats = compute_four_category_grain_sets(lfi_ebsd, parent_info)
    n_slices_map = n_slices_per_axis if isinstance(n_slices_per_axis, dict) \
        else {a: n_slices_per_axis for a in axes}

    out = {}
    for category in SGS_GRAIN_SIZE_CATEGORIES:
        ebsd_ced = compute_ebsd_category_ced(prop_ebsd, cats[category])

        sgs_ced_parts = []
        for axis in axes:
            per_slice = compute_sgs_category_ced_by_slice(
                lgi_clean, twin_role_clean, category, axis,
                n_slices_map[axis], voxel_size=sgs_voxel_size)
            sgs_ced_parts.extend(per_slice)
        sgs_ced = (np.concatenate(sgs_ced_parts) if sgs_ced_parts
                  else np.empty(0, dtype=np.float64))

        ebsd_ced_stats = _stat_block(ebsd_ced)
        sgs_ced_stats = _stat_block(sgs_ced)

        ebsd_igs = compute_ebsd_category_igs(
            lfi_ebsd, cats[category], ebsd_voxel_size, ebsd_ced_stats['mean'])
        sgs_igs = compute_sgs_category_igs(
            lgi_clean, twin_role_clean, category, axes, n_slices_map,
            voxel_size=sgs_voxel_size, mean_ced_um=sgs_ced_stats['mean'])

        out[category] = {
            'ebsd_ced': ebsd_ced_stats,
            'sgs_ced': sgs_ced_stats,
            'ebsd_igs': {k: ebsd_igs[k] for k in ('mean', 'median', 'std', 'n')},
            'sgs_igs': {k: sgs_igs[k] for k in ('mean', 'median', 'std', 'n')},
        }
    return out


def compute_slice_properties_by_axis(
        lgi: np.ndarray,
        selected_props,
        voxel_size: float = 1.0,
        axes=('x', 'y', 'z'),
        n_slices_per_axis=5,
        connectivity_2d: int = 4,
        keep_ids=None,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    2D grain-morphology properties from evenly-spaced cross-sections of a
    3D grain-label field, kept separate per axis and per slice.

    Each section is re-labelled in 2D (``cc3d``) so a grain that meets the
    plane in several disjoint pieces counts as several 2D grains, the same
    convention as a map of a flat section. ``lgi`` keeps the pipeline's
    native ``(nz, ny, nx)`` order: axis ``'x'`` is array axis 2, ``'y'``
    is 1 and ``'z'`` is 0.

    Parameters
    ----------
    lgi : ndarray (nz, ny, nx), int
    selected_props : iterable of str
        Subset of ``{'area', 'eqdia', 'perimeter', 'aspect_ratio',
        'solidity', 'major_axis', 'minor_axis'}``. Lengths and areas are
        scaled by ``voxel_size``. ``minor_axis`` of a twin's section is its
        apparent 2D thickness.
    axes : iterable of str
        Subset of ``'x', 'y', 'z'``.
    n_slices_per_axis : int or dict
        Evenly-spaced slices per axis, or ``{'x': int, 'y': int, 'z': int}``.
    connectivity_2d : int
        ``cc3d`` connectivity for the 2D re-labelling (4 or 8).
    keep_ids : iterable of int or None
        Measure only these 3D grains (e.g. the twin hosts); every other
        grain is treated as background. None measures every grain.

    Returns
    -------
    dict ``{axis: {prop_name: [ndarray per slice, in slice order]}}``
    """
    import cc3d
    from skimage.measure import regionprops
    from upxo.gsdataops.grid_ops import section_from_3d

    selected_props = list(selected_props)
    if keep_ids is not None:
        lgi = np.where(np.isin(lgi, np.fromiter((int(g) for g in keep_ids), dtype=np.int64)), lgi, 0)
    axis_map = {'x': 2, 'y': 1, 'z': 0}
    by_axis: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for axis in axes:
        axis = axis.lower()
        if axis not in axis_map:
            continue
        ax = axis_map[axis]
        n_this_axis = (n_slices_per_axis[axis]
                       if isinstance(n_slices_per_axis, dict) else n_slices_per_axis)
        slice_positions = np.linspace(0, lgi.shape[ax] - 1, int(n_this_axis), dtype=int)

        per_prop_slices = {p: [] for p in selected_props}
        for slice_pos in progress.track(slice_positions, message=f"{axis.upper()}-axis slices"):
            lgi_2d = section_from_3d(lgi, axis=ax, location=int(slice_pos))
            relabelled = cc3d.connected_components(lgi_2d, connectivity=connectivity_2d)
            slice_vals = {p: [] for p in selected_props}
            for region in regionprops(relabelled):
                if 'area' in selected_props:
                    slice_vals['area'].append(region.area * voxel_size ** 2)
                if 'eqdia' in selected_props:
                    slice_vals['eqdia'].append(region.equivalent_diameter_area * voxel_size)
                if 'perimeter' in selected_props:
                    slice_vals['perimeter'].append(region.perimeter * voxel_size)
                if 'solidity' in selected_props:
                    slice_vals['solidity'].append(region.solidity)
                if 'major_axis' in selected_props:
                    slice_vals['major_axis'].append(region.major_axis_length * voxel_size)
                if 'minor_axis' in selected_props:
                    slice_vals['minor_axis'].append(region.minor_axis_length * voxel_size)
                if 'aspect_ratio' in selected_props:
                    maj, mn = region.major_axis_length, region.minor_axis_length
                    if mn > 0:
                        slice_vals['aspect_ratio'].append(maj / mn)
            for p in selected_props:
                per_prop_slices[p].append(np.asarray(slice_vals[p], dtype=float))
        by_axis[axis] = per_prop_slices

    return by_axis


def ebsd_merged_property_values(rg, prop: str, hosts_only: bool = False) -> np.ndarray:
    """
    Per-grain values of ``prop`` for the EBSD map with twins merged into
    their parent grains (``rg.prop_ebsd_merged``) -- the EBSD counterpart of
    a structure that has no twins yet.

    hosts_only : keep only the merged grains that contain at least one
    twin (the grains a twin-introduction step would host twins in).
    """
    if getattr(rg, 'prop_ebsd_merged', None) is None or getattr(rg, 'lfi_ebsd_merged', None) is None:
        raise RuntimeError("Merge twins into parents first (CSL Segregation & TVF).")
    props = rg.prop_ebsd_merged
    if not hosts_only:
        return np.array([p[prop] for p in props.values() if prop in p], dtype=float)

    merge_info = getattr(rg, 'merge_info', None)
    if not merge_info or not merge_info.get('hosting_original_gids'):
        raise RuntimeError("No twin-hosting grains are recorded for the merged EBSD.")
    original = rg.lfi_ebsd.ravel()
    merged = rg.lfi_ebsd_merged.ravel()
    ids, first = np.unique(original, return_index=True)
    first_by_id = dict(zip(ids.tolist(), first.tolist()))
    new_ids = {int(merged[first_by_id[g]]) for g in merge_info['hosting_original_gids']
               if g in first_by_id and merged[first_by_id[g]] > 0}
    return np.array([props[g][prop] for g in sorted(new_ids) if g in props and prop in props[g]],
                    dtype=float)


def format_size(n):
    """Human-readable file size (B/KB/MB/GB/TB)."""
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


_CATEGORY_TABLE_TITLES = {
    'parent':   'Parent-to-Parent Grain Size (genuinely twin-free grains)',
    'host':     'Twin Host-to-Host Grain Size (hosts a twin, remaining matrix region)',
    'twin':     'Twin-to-Twin Grain Size (primary + secondary lamellae, pooled)',
    'complete': 'Complete-to-Complete Grain Size (every grain, no role filtering)',
}


def build_summary_markdown_lines(role_stats, tvf_final_3d, tvf_ebsd_target, comparison_stats=None,
                                  category_grain_size_stats=None):
    """Formats grain-role volume statistics + twin volume fraction summary
    (and, if given, the full EBSD-vs-MC comparison, and the 4-category
    CED/IGS grain-size comparison) as Markdown lines. Pure formatting, no
    file I/O -- shared by every report writer that needs this summary, so
    all embed byte-identical content when the same data is available.

    category_grain_size_stats : dict or None
        Output of compute_category_grain_size_stats -- one table per
        SGS_GRAIN_SIZE_CATEGORIES key, each with a CED row-set and an
        IGS row-set (Mean/Median/Std/N, EBSD vs MC). Supersedes the old
        single "Host / Parent Grain Equivalent Diameter" row that used
        to live under comparison_stats (that compared a 2D EBSD
        sectional diameter against a pre-twin 3D volumetric one --
        replaced entirely by this, not kept alongside it)."""
    import datetime
    lines = [
        "# Twinned FCC 3D Pipeline — Final Summary Report",
        "",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Grain Role Volume Statistics",
        "",
        "| Role | Count | Mean vol (um3) | Std vol (um3) | Min vol (um3) | Max vol (um3) |",
        "|---|---|---|---|---|---|",
    ]
    for role, info in role_stats.items():
        lines.append(
            f"| {role} | {info['count']} | {info['mean_vol_um3']:.2f} | "
            f"{info['std_vol_um3']:.2f} | {info['min_vol_um3']:.2f} | {info['max_vol_um3']:.2f} |"
        )

    ratio = (tvf_final_3d / tvf_ebsd_target) if tvf_ebsd_target > 0 else float('nan')
    lines += [
        "",
        "## Twin Volume Fraction Summary",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| EBSD target twin area fraction | {tvf_ebsd_target:.4f} |",
        f"| Final 3D twin volume fraction (post-cleaning) | {tvf_final_3d:.4f} |",
        f"| Ratio (final 3D / EBSD target) | {ratio:.2f}x |",
        "",
    ]

    if comparison_stats is not None:
        def _v(x):
            return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"

        lines += ["## EBSD vs Synthetic MC Comparison", ""]
        for title, section, cols in [
            ("Misorientation Distribution (deg)", comparison_stats['mdf'],
             [("EBSD (full)", "ebsd_full"), ("EBSD (merged)", "ebsd_merged"), ("MC (post-twin)", "mc_posttwin")]),
            ("Twin Lamella Thickness (um)", comparison_stats['twin_thickness'],
             [("EBSD", "ebsd"), ("MC apparent 2D", "mc_apparent_2d"), ("MC actual 3D", "mc_actual_3d")]),
        ]:
            lines += [f"### {title}", "",
                      "| Stat | " + " | ".join(c[0] for c in cols) + " |",
                      "|---|" + "---|" * len(cols)]
            for stat_key, stat_label in [('mean', 'Mean'), ('std', 'Std'), ('q2', 'Median'), ('n', 'N')]:
                row_vals = [str(section.get(key, {}).get(stat_key)) if stat_key == 'n'
                            else _v(section.get(key, {}).get(stat_key)) for _, key in cols]
                lines.append(f"| {stat_label} | " + " | ".join(row_vals) + " |")
            lines.append("")

        tvf_cmp = comparison_stats['twin_volume_fraction']
        lines += [
            "### Twin Volume / Area Fraction", "",
            "| Quantity | Value |", "|---|---|",
            f"| EBSD 2D area fraction | {_v(tvf_cmp['ebsd_2d'])} |",
            f"| MC 2D slice mean (+/- std) | {_v(tvf_cmp['mc_2d_mean'])} (+/-{_v(tvf_cmp['mc_2d_std'])}) |",
            f"| MC 3D volume fraction | {_v(tvf_cmp['mc_3d'])} |",
            "",
        ]

    if category_grain_size_stats is not None:
        def _v2(x):
            return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"

        lines += ["## EBSD vs Synthetic Grain Size by Category (2D)", ""]
        for category in SGS_GRAIN_SIZE_CATEGORIES:
            cat_stats = category_grain_size_stats.get(category)
            if cat_stats is None:
                continue
            lines += [f"### {_CATEGORY_TABLE_TITLES[category]}", "",
                      "| Stat | EBSD CED (um) | MC CED (um) | EBSD IGS (um) | MC IGS (um) |",
                      "|---|---|---|---|---|"]
            for stat_key, stat_label in [('mean', 'Mean'), ('std', 'Std'),
                                         ('median', 'Median'), ('n', 'N')]:
                row = [stat_label]
                for section in ('ebsd_ced', 'sgs_ced', 'ebsd_igs', 'sgs_igs'):
                    val = cat_stats[section].get(stat_key)
                    row.append(str(val) if stat_key == 'n' else _v2(val))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    return lines
