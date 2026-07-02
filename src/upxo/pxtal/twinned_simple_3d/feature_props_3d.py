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
    if grain_ids is None:
        grain_ids = sorted(int(g) for g in np.unique(lgi) if g > 0)
    return {gid: int(np.sum(lgi == gid)) for gid in grain_ids}


def compute_twin_volume_fraction(
        lgi: np.ndarray,
        twin_gids: List[int],
) -> float:
    if not twin_gids:
        return 0.0
    total_vox = int(np.sum(lgi > 0))
    if total_vox == 0:
        return 0.0
    return sum(int(np.sum(lgi == g)) for g in twin_gids) / total_vox


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
