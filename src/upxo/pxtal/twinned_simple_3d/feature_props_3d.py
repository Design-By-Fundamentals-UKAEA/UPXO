"""
feature_props_3d.py
===================
Grain and twin morphological property extraction for the twinned
simple 3D pipeline.
"""

import numpy as np
from typing import Dict, Optional, List


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
        ``host_eqdia_um``    - 1-D ndarray  EBSD pure-parent eq. diameters (um)
        ``tvf_2d``           - scalar       EBSD 2D twin area fraction

    sgc_ref : dict
        ``miso_deg_posttwin``- 1-D ndarray  SGC post-twin MDF angles
        ``twin_thick_3d_um`` - 1-D ndarray  SGC actual 3D twin thicknesses (um)
        ``host_eqdia_um``    - 1-D ndarray  SGC host grain eq. diameters (um)
        ``tvf_2d_slices``    - 1-D ndarray  per-slice 2D twin VF values
        ``tvf_3d``           - scalar       SGC 3D twin volume fraction

    Returns
    -------
    dict
        Nested dict with keys 'mdf', 'twin_thickness', 'host_grain_size',
        'twin_volume_fraction', each containing 'ebsd' and 'mc' sub-dicts
        of descriptive statistics.
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
        'host_grain_size': {
            'ebsd': _descriptive_stats(ebsd_ref.get('host_eqdia_um', [])),
            'mc':   _descriptive_stats(sgc_ref.get('host_eqdia_um',   [])),
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

    hg   = stats['host_grain_size']
    e_hg = hg['ebsd']
    mc_hg= hg['mc']
    print('\nHOST / PARENT GRAIN EQUIVALENT DIAMETER')
    print(fmt.format('Property', 'EBSD (parents)', '', 'MC (hosts)'))
    print(sep2)
    for key, lbl in [('mean','Mean'),('std','Std dev'),
                      ('q1','Q1'),('q2','Median'),('q3','Q3')]:
        print(f'  {lbl:<32}  {_fv(e_hg.get(key,float("nan"))):>10}  '
              f'{"":>10}  {_fv(mc_hg.get(key,float("nan"))):>10}  um')
    print(f'  {"n":<32}  {e_hg.get("n",0):>10d}  '
          f'{"":>10}  {mc_hg.get("n",0):>10d}')

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
        cleaner, tg, base, rg, parent_info, mdf, twin_thickness,
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
    base : TwinnedSimple3DBase
        Pre-twin host structure (``mprop['eqdia']``, ``host_grain_ids``).
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

    pure_parents = set()
    for info in parent_info.values():
        pure_parents.update(int(g) for g in info.get('pure_parents', []))
    host_eqdia_ebsd = np.array(
        [rg.prop_ebsd[g]['eq_diameter'] for g in pure_parents if g in rg.prop_ebsd],
        dtype=float)

    ebsd_ref = {
        'miso_deg_full':   np.asarray(mdf['miso_deg']),
        'miso_deg_merged': ebsd_mdf_merged['miso_deg'],
        'twin_thick_um':   np.asarray(twin_thickness['thick_um'], dtype=float),
        'host_eqdia_um':   host_eqdia_ebsd,
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

    host_eqdia_sgc = np.array(
        [base.mprop['eqdia'][gid] for gid in (base.host_grain_ids or [])
         if gid in base.mprop.get('eqdia', {})], dtype=float)

    twin_gids_clean = [gid for gid, role in cleaner.twin_role_clean.items()
                        if role in ('primary_twin', 'secondary_twin')]
    tvf_final_3d = compute_twin_volume_fraction(cleaner.lgi_clean, twin_gids_clean)

    achieved_2d = tg.compute_achieved_2d_tvf(
        n_slices_per_axis=n_slices_per_axis, axes=axes, return_raw=True)

    sgc_ref = {
        'miso_deg_posttwin': mc_mdf_post['miso_deg'],
        'twin_thick_3d_um':  twin_thick_3d_um,
        'host_eqdia_um':     host_eqdia_sgc,
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
        'n_neighbours')``.
    selected_levels : list of str
        Subset of ``('nonhost', 'host', 'primary', 'seca', 'secb')``.
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
                    if p in ('area', 'aspect_ratio', 'perimeter', 'solidity')]
    needs_neigh = 'n_neighbours' in selected_props

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
                    if lv not in selected_levels:
                        continue
                    if 'area' in morph_props:
                        values['area'][lv].append(region.area * voxel_size ** 2)
                    if 'perimeter' in morph_props:
                        values['perimeter'][lv].append(region.perimeter * voxel_size)
                    if 'aspect_ratio' in morph_props and region.minor_axis_length > 0:
                        values['aspect_ratio'][lv].append(
                            region.major_axis_length / region.minor_axis_length)
                    if 'solidity' in morph_props:
                        values['solidity'][lv].append(region.solidity)

            if needs_neigh:
                neigh = find_neighs2d(lgi_2d.astype(np.int32), conn=4)
                for gid, nbrs in neigh.items():
                    lv = _sgs_role_to_level(gid, twin_role_clean, twin_parent_of_clean)
                    if lv not in selected_levels:
                        continue
                    values['n_neighbours'][lv].append(len(nbrs))

    return {
        p: {lv: np.array(vals, dtype=float) for lv, vals in lv_dict.items()}
        for p, lv_dict in values.items()
    }


def format_size(n):
    """Human-readable file size (B/KB/MB/GB/TB)."""
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def build_summary_markdown_lines(role_stats, tvf_final_3d, tvf_ebsd_target, comparison_stats=None):
    """Formats grain-role volume statistics + twin volume fraction summary
    (and, if given, the full EBSD-vs-MC comparison) as Markdown lines.
    Pure formatting, no file I/O -- shared by every report writer that
    needs this summary, so all embed byte-identical content when the
    same data is available."""
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
            ("Host / Parent Grain Equivalent Diameter (um)", comparison_stats['host_grain_size'],
             [("EBSD", "ebsd"), ("MC", "mc")]),
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

    return lines
