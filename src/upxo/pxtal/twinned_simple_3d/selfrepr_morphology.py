"""
selfrepr_morphology.py
=======================
Per-grain morphological parameter computation for EBSD self-
representativeness studies (twinned_simple_3d's Self Repr.-1 page):
area, perimeter, circle-equivalent diameter, aspect ratio, minor/major
axis length, circularity, coordination number, grain-boundary segment
length, and triple-junction-point count, computed identically for a
parent EBSD domain and any cropped sub-domain.
"""

import numpy as np

from upxo.pxtal.fm_steel_3d.slice_metrics_2d import compute_slice_metrics

MORPH_PARAM_KEYS = (
    'area', 'perimeter', 'circle_eq_dia', 'aspect_ratio',
    'minor_axis_length', 'major_axis_length', 'circularity',
    'coord_number', 'gb_segment_length', 'n_tjp',
)
MORPH_PARAM_LABELS = {
    'area': 'Area',
    'perimeter': 'Perimeter',
    'circle_eq_dia': 'Circle-Equivalent Diameter',
    'aspect_ratio': 'Aspect Ratio',
    'minor_axis_length': 'Minor Axis Length',
    'major_axis_length': 'Major Axis Length',
    'circularity': 'Circularity',
    'coord_number': 'Coordination Number',
    'gb_segment_length': 'Grain Boundary Segment Length',
    'n_tjp': 'Triple Junction Points per Grain',
}
# Short, meaningful abbreviations for table headers/row-names -- NOT
# sequential A/B/C (which reads as arbitrary and is easy to lose track
# of), but still short enough to keep a qualification table narrow.
MORPH_PARAM_ABBREV = {
    'area': 'A',
    'perimeter': 'P',
    'circle_eq_dia': 'CED',
    'aspect_ratio': 'AR',
    'minor_axis_length': 'MinAL',
    'major_axis_length': 'MajAL',
    'circularity': 'Circ',
    'coord_number': 'CN',
    'gb_segment_length': 'GBSL',
    'n_tjp': 'nTJP',
}


def compute_morphological_parameters(lgi, prop_ebsd=None, voxel_size=1.0):
    """
    Compute the full morphological-parameter bundle for one 2D grain-
    label field -- the parent EBSD domain or a cropped subset.

    Parameters
    ----------
    lgi : np.ndarray, int, shape (ny, nx)
        Grain label field.
    prop_ebsd : dict or None
        The EBSD reader's own per-grain property dict (``rg.prop_ebsd``),
        if already available (the PARENT domain always has one) --
        reused directly for minor_axis_length/major_axis_length so the
        parent side isn't recomputed. When None (a cropped subset,
        which has no prop_ebsd of its own), these two are computed
        fresh via the same underlying regionprops path
        (``ebsd_reader._char_lfi``) so parent and subset values come
        from identical logic.
    voxel_size : float
        Physical pixel size (microns), for area/perimeter/length scaling.

    Returns
    -------
    dict[str, dict[int, float]]
        One entry per key in MORPH_PARAM_KEYS, each a {grain_id: value}
        dict covering every grain actually present in `lgi`.

        'gb_segment_length' -- total boundary length between this
        grain's own triple-junction points; a grain with zero triple
        junctions on its boundary (e.g. a true island grain, fully
        surrounded by one neighbour) falls back to its full perimeter,
        since its entire boundary is then a single "segment".
    """
    metrics = compute_slice_metrics(lgi, voxel_size=voxel_size)
    gids = sorted(int(g) for g in np.unique(lgi) if g > 0)

    result = {key: {} for key in MORPH_PARAM_KEYS}
    for gid in gids:
        result['area'][gid] = float(metrics['area'][gid])
        result['perimeter'][gid] = float(metrics['perimeter'][gid])
        result['circle_eq_dia'][gid] = float(metrics['circle_eq_dia'][gid])
        result['aspect_ratio'][gid] = float(metrics['aspect_ratio'][gid])
        result['circularity'][gid] = float(metrics['circularity'][gid])
        result['coord_number'][gid] = float(metrics['n_neighbours'][gid])
        n_tjp_gid = float(metrics['n_tjp'][gid])
        result['n_tjp'][gid] = n_tjp_gid
        result['gb_segment_length'][gid] = (
            float(metrics['tjp_boundary_length'][gid]) if n_tjp_gid > 0
            else float(metrics['perimeter'][gid]))

    if prop_ebsd is not None:
        source = prop_ebsd
    else:
        from upxo.interfaces.defdap.ebsd_reader import _char_lfi
        source = _char_lfi(lgi, px_size=voxel_size, min_grain_size=0)
    for gid in gids:
        g = source.get(gid)
        if g is not None:
            result['minor_axis_length'][gid] = float(g['minor_axis_length'])
            result['major_axis_length'][gid] = float(g['major_axis_length'])

    return result


def boundary_grain_ratio_parent(rg):
    """Block [2] "Basics" glue: Ng(boundary)/Ng(total) for the parent
    EBSD domain currently loaded on the pipeline.

    Parameters
    ----------
    rg : repgen2d
        Must have ``.lfi_ebsd`` populated (post Clean & Characterise).

    Returns
    -------
    dict : {'n_boundary', 'n_internal', 'n_total', 'ratio'}
    """
    from upxo.charops.mchar import boundary_grain_fraction
    return boundary_grain_fraction(rg.lfi_ebsd)
