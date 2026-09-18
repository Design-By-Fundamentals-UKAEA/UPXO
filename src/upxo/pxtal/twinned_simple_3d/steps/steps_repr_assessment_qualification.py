"""Repr. assess. & qual. -- Part F of the Twinned FCC walkthrough.

In order: Synthetic GS Assessment (rank every saved temporal slice),
Candidate Selection (re-assess under different tolerances), Temporal
Slice Shortlist (final, authoritative choice). Ranks every saved MC
temporal slice against the EBSD parent-grain reference, then picks the
best-matching one and calibrates its physical scale factor for everything
downstream.
"""

# Default tolerance/scoring config passed to
# mc_qualification.compute_shortlist_rows, which expects a dict-like
# object with these keys (a plain dict works fine).
DEFAULT_SHORTLIST_CONFIG = {
    "MC_NG_TOLERANCE_PCT": 5.0,
    "MC_PROPERTY_TOLERANCE_PCT_PER_PROP": {},
    "MC_SCALE_CHECK_TOLERANCE_PCT": 5.0,
}


def rank_slices(pxt, parent_info=None, role_props=None, start=0, step=1,
                 n_comparison_slices=5, comparison_axes=('x', 'y', 'z'),
                 outlier_trim_sides=None, scale_check_tolerance_pct=5.0,
                 score_function='exp'):
    """Synthetic GS Assessment -- ranks every saved temporal slice
    (pxt.m[start::step]) by how closely its 2D cross-sectional grain
    count/properties match the EBSD parent-grain reference, pooling
    whichever properties were computed in Grain-Role Properties
    (EBSD Analysis-2) directly from the same cross-sections.

    Returns
    -------
    list of dict : one per ranked candidate (keys include 'tslice_key',
    'ratio', 'prop_compare', 'prop_stats', 'scale_calibration', ...).
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    from upxo.pxtal.twinned_simple_3d.mc_qualification import recompute_candidate_derived

    ebsd_n_parents = None
    if parent_info:
        s3 = parent_info.get('S3  (twin)')
        if s3:
            ebsd_n_parents = s3.get('n_pure_parents')

    prop_names = list(role_props.keys()) if role_props else []
    candidates = TwinnedSimple3DBase.rank_temporal_slices_by_n(
        pxt, start=start, step=step, ebsd_n_parents=ebsd_n_parents,
        n_comparison_slices=n_comparison_slices, comparison_axes=list(comparison_axes),
        selected_props=prop_names or None)

    if prop_names:
        recompute_candidate_derived(
            pxt, candidates, role_props, prop_names, outlier_trim_sides or {},
            scale_check_tolerance_pct, score_function=score_function)
    return candidates


def reassess_candidates(pxt, candidates, role_props, outlier_trim_sides=None,
                         scale_check_tolerance_pct=5.0, score_function='exp'):
    """Candidate Selection's re-assessment step -- re-derives the
    trim/tolerance-dependent parts (scale calibration, EBSD comparisons)
    from each candidate's already-cached raw property stats, without
    re-ranking from scratch. Call this again after changing outlier
    trimming or the scale-check tolerance.

    Mutates `candidates` in place and returns it too, for convenient
    chaining.
    """
    from upxo.pxtal.twinned_simple_3d.mc_qualification import recompute_candidate_derived
    prop_names = list(role_props.keys()) if role_props else []
    if prop_names:
        recompute_candidate_derived(
            pxt, candidates, role_props, prop_names, outlier_trim_sides or {},
            scale_check_tolerance_pct, score_function=score_function)
    return candidates


def shortlist_and_select(candidates, role_props, config=None, selected_criteria=None):
    """Temporal Slice Shortlist's final selection -- ranks candidates by
    Total Stars (descending) then Aggregate Score (tie-breaker) and picks
    the top one.

    Returns
    -------
    (rows, best_row) : `rows` is every candidate's shortlist row (see
    mc_qualification.compute_shortlist_rows), `best_row` is rows[0] (the
    coupled-rank #1 pick) or None if there were no candidates.
    """
    from upxo.pxtal.twinned_simple_3d.mc_qualification import compute_shortlist_rows
    prop_names = list(role_props.keys()) if role_props else []
    rows = compute_shortlist_rows(
        candidates, prop_names, config or DEFAULT_SHORTLIST_CONFIG,
        selected_criteria=selected_criteria)
    best_row = rows[0] if rows else None
    return rows, best_row


def apply_selected_scale_factor(pxt, best_row):
    """Applies the chosen candidate's calibrated physical scale factor to
    the live simulation object -- every downstream consumer of
    pxt.vox_size (host allocation, property computation, mesh export, ...)
    picks it up automatically. No-op (returns None) if the selected row
    has no scale calibration (e.g. 'area' wasn't among the compared
    properties).

    Returns
    -------
    float or None : the applied scale factor (um/voxel).
    """
    if best_row is None or best_row.get('scale_factor') is None:
        return None
    sf = best_row['scale_factor']
    pxt.vox_size = (sf, sf, sf)
    return sf
