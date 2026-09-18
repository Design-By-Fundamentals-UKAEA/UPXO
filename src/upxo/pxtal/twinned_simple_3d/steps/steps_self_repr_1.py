"""Self Repr.-1 -- Part C of the Twinned FCC walkthrough (OPTIONAL).

"Ng(boundary)/Ng(total)" for the parent domain, moving-rectangle sub-domain
extraction, morphological-parameter distributions for parent vs. subsets,
and representativeness scoring between them.

Disabled by default -- this is a deeper, optional representativeness study,
not required for the main pipeline to proceed. Kept short here; see
selfrepr_morphology.py / representativeness_metrics.py directly for the
full feature set (per-(metric,parameter) qualification thresholds,
publication-style plots, etc.) if you want more than this notebook's
teaching-level pass.
"""
from upxo.charops.mchar import boundary_grain_fraction
from upxo.pxtal.twinned_simple_3d.selfrepr_morphology import compute_morphological_parameters
from upxo.pxtal.twinned_simple_3d.subsetting_2d import (
    generate_subset_tiles_2d, crop_lgi_2d, subset_centroid_um)
from upxo.pxtal.twinned_simple_3d.representativeness_metrics import compute_all_selected


def boundary_grain_ratio(rg):
    """Ng(boundary)/Ng(total) for the parent EBSD domain.

    Returns
    -------
    dict : {'n_boundary', 'n_internal', 'n_total', 'ratio'}
    """
    return boundary_grain_fraction(rg.lfi_ebsd)


def extract_subsets(rg, start_pct=0.0, length_pct=20.0, stride_pct=20.0):
    """Moving-rectangle sub-domain extraction -- a tile of `length_pct`%
    of the domain, stepped every `stride_pct`%, starting at `start_pct`%,
    independently along X and Y. Returns a list of dicts, each with
    'index' (i, j), 'lgi' (the cropped label field), and 'centroid_um'.
    """
    pct = {'x': start_pct, 'y': start_pct}
    length = {'x': length_pct, 'y': length_pct}
    stride = {'x': stride_pct, 'y': stride_pct}
    tiles = generate_subset_tiles_2d(rg.lfi_ebsd.shape, pct, length, stride, overflow_policy='clip')

    step_um = getattr(rg, 'ebsd_step', 1.0) or 1.0
    subsets = []
    for tile in tiles:
        lgi_sub = crop_lgi_2d(rg.lfi_ebsd, tile)
        cx, cy = subset_centroid_um(tile, rg.lfi_ebsd.shape, step_um)
        subsets.append({**tile, 'lgi': lgi_sub, 'centroid_um': (cx, cy)})
    return subsets


def compute_morphology(rg, subsets, params=('area', 'perimeter')):
    """Morphological parameters for the parent domain and every subset
    (same set of parameters, same units, for a fair comparison).

    Returns
    -------
    dict : {'parent': {param: {gid: value}}, 'subsets': {(i, j): {param: {gid: value}}}}
    """
    step_um = getattr(rg, 'ebsd_step', 1.0) or 1.0
    parent = compute_morphological_parameters(rg.lfi_ebsd, prop_ebsd=rg.prop_ebsd, voxel_size=step_um)
    subset_results = {
        s['index']: compute_morphological_parameters(s['lgi'], prop_ebsd=None, voxel_size=step_um)
        for s in subsets
    }
    return {'parent': {p: parent[p] for p in params},
            'subsets': {idx: {p: r[p] for p in params} for idx, r in subset_results.items()}}


def assess_representativeness(morph_results, metrics=('ks_sim', 'wasserstein_sim')):
    """For every subset and every parameter, scores how representative
    that subset's distribution is of the parent's, using the requested
    metrics (all "higher = more representative", bounded to [0, 1]).

    Returns
    -------
    dict : {(subset_index, param): {metric: score}}
    """
    assessment = {}
    for idx, subset_params in morph_results['subsets'].items():
        for param, subset_vals in subset_params.items():
            ref_vals = list(morph_results['parent'][param].values())
            cand_vals = list(subset_vals.values())
            if not ref_vals or not cand_vals:
                continue
            scores = compute_all_selected(metrics, ref_vals, cand_vals)
            if scores:
                assessment[(idx, param)] = scores
    return assessment
