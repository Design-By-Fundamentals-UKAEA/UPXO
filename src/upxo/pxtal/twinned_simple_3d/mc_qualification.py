"""
mc_qualification.py
====================
Candidate representativeness scoring and shortlist ranking for the
Monte-Carlo temporal-slice qualification stage of the twinned simple 3D
pipeline: derived per-candidate statistics (scale calibration, rescaled
property stats, EBSD-distribution comparisons) and the star-count /
aggregate-score / coupled shortlist ranking built from them.

Ported from ``pxtal/twinned_simple_3d/gui/pages_mc.py`` (previously only
reachable through MCCandidateSelectionPage/MCQualificationPage -- none of
the logic here has any GUI/Tkinter dependency).
"""

import numpy as np


def ng_qualifies(ratio, tol_pct):
    """Whether a candidate's Ng ratio falls within +/- tol_pct% of 1.0."""
    return ratio is not None and abs(ratio - 1.0) <= tol_pct / 100.0


def prop_qualifies(cmp_result, tol_pct):
    """Whether a candidate's property ratio (from
    TwinnedSimple3DBase.compare_property_distributions) falls within
    +/- tol_pct% of 1.0."""
    return cmp_result is not None and abs(cmp_result['ratio'] - 1.0) <= tol_pct / 100.0


def property_tolerance_pct(shared_state, prop_name):
    """Per-property Property Tolerance, falling back to the single legacy
    MC_PROPERTY_TOLERANCE_PCT for any property that hasn't been given its
    own entry in MC_PROPERTY_TOLERANCE_PCT_PER_PROP yet."""
    per_prop = shared_state.get("MC_PROPERTY_TOLERANCE_PCT_PER_PROP") or {}
    return per_prop.get(prop_name, shared_state.get("MC_PROPERTY_TOLERANCE_PCT", 5.0))


def recompute_candidate_derived(pxt, candidates, prop_data, prop_names,
                                 outlier_trim_sides, scale_check_tolerance_pct,
                                 score_function='exp'):
    """(Re)computes, for every candidate, the scale-factor calibration, the
    display-unit property stats, and the EBSD-distribution comparisons --
    everything that depends on outlier-trim sides / the Scale Check
    tolerance -- from each candidate's already-cached raw (voxel-unit)
    property stats (candidate['prop_stats_raw']), without re-running the
    expensive 3D cross-sectional labelling
    (rank_temporal_slices_by_n / compute_slice_grain_properties).

    outlier_trim_sides: {prop_name: {'left': bool, 'right': bool}}.
    score_function: 'exp' (default) or 'reciprocal' -- see
    TwinnedSimple3DBase._squash_distance; controls how the
    representativeness_score in each property's prop_compare entry is
    derived from its (dimensionless) Wasserstein/energy distances.
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase

    def _sides(prop):
        sides = outlier_trim_sides.get(prop, {})
        return sides.get('left', True), sides.get('right', True)

    for c in candidates:
        raw_stats = c.get('prop_stats_raw', {})
        display_stats = dict(raw_stats)
        c['scale_calibration'] = None

        if 'area' in prop_names:
            ebsd_area_vals = prop_data.get('area', {}).get('pure_parents')
            synth_area_vals = raw_stats.get('area')
            if ebsd_area_vals is not None and synth_area_vals is not None:
                area_trim_left, area_trim_right = _sides('area')
                ebsd_per_vals = prop_data.get('perimeter', {}).get('pure_parents')
                synth_per_vals = raw_stats.get('perimeter')
                try:
                    c['scale_calibration'] = TwinnedSimple3DBase.calibrate_scale_factor(
                        pxt, c['tslice_key'], ebsd_area_vals, synth_area_vals,
                        ebsd_perimeter_vals=ebsd_per_vals, synth_perimeter_vals=synth_per_vals,
                        trim_left=area_trim_left, trim_right=area_trim_right,
                        cross_check_tolerance_pct=scale_check_tolerance_pct,
                    )
                except Exception as e:
                    print(f"    Warning: scale factor calibration failed: {e}")

                if c['scale_calibration'] is not None:
                    sf = c['scale_calibration']['scale_factor']
                    for length_prop in ('area', 'perimeter'):
                        if length_prop in raw_stats:
                            display_stats[length_prop] = TwinnedSimple3DBase.rescale_property_values(
                                length_prop, raw_stats[length_prop], sf)

        c['prop_stats'] = display_stats

        c['prop_compare'] = {}
        for p in prop_names:
            ebsd_vals = prop_data.get(p, {}).get('pure_parents')
            synth_vals = display_stats.get(p)
            if ebsd_vals is None or synth_vals is None:
                continue
            trim_left, trim_right = _sides(p)
            cmp_result = TwinnedSimple3DBase.compare_property_distributions(
                ebsd_vals, synth_vals, trim_left=trim_left, trim_right=trim_right,
                score_function=score_function)
            if cmp_result is not None:
                c['prop_compare'][p] = cmp_result


def compute_shortlist_rows(candidates, prop_names, shared_state, selected_criteria=None):
    """Two ranking estimates plus a default 'coupled' one: star count as
    the primary sort key, aggregate score as the tie-breaker for the
    default view, with both individual rankings also reported so a
    candidate can be judged by either criterion alone.

    aggregate_score: mean of per-metric "goodness" terms in [0, 1] --
    (1 - |Ng ratio - 1|, clipped at 0) for Ng, and KS similarity (already
    bounded [0, 1]) for each property. Kept dimensionless so Ng and every
    property contribute comparably to one blended score despite being in
    different units.

    selected_criteria: optional set of criterion keys ('ng', property
    names, 'scale_check') to restrict n_stars/aggregate_score to -- None
    (the default) means every available criterion.
    """
    ng_tol = shared_state.get("MC_NG_TOLERANCE_PCT", 5.0)

    use_ng = selected_criteria is None or 'ng' in selected_criteria
    use_scale = selected_criteria is None or 'scale_check' in selected_criteria

    rows = []
    for c in candidates:
        ratio = c.get('ratio')
        n_stars = 0
        goodness_terms = []

        if use_ng:
            if ng_qualifies(ratio, ng_tol):
                n_stars += 1
            if ratio is not None:
                goodness_terms.append(max(0.0, 1.0 - abs(ratio - 1.0)))

        for p in prop_names:
            if selected_criteria is not None and p not in selected_criteria:
                continue
            cmp_result = c.get('prop_compare', {}).get(p)
            if cmp_result is None:
                continue
            if prop_qualifies(cmp_result, property_tolerance_pct(shared_state, p)):
                n_stars += 1
            goodness_terms.append(cmp_result['ks_similarity'])

        # Scale Check (Area/Perimeter cross-check) contributes like any
        # other qualifying criterion -- 1.0 if the candidate's grains are
        # dimensionally self-consistent with EBSD, 0.0 if not. Skipped
        # (neither star nor goodness term) when Perimeter wasn't
        # available to check against, rather than penalising a candidate
        # for missing data.
        if use_scale:
            cross_check_ok = (c.get('scale_calibration') or {}).get('cross_check_ok')
            if cross_check_ok is not None:
                if cross_check_ok:
                    n_stars += 1
                goodness_terms.append(1.0 if cross_check_ok else 0.0)

        aggregate_score = float(np.mean(goodness_terms)) if goodness_terms else 0.0
        calib = c.get('scale_calibration')
        rows.append({
            'tslice_key': c['tslice_key'],
            'n_stars': n_stars,
            'aggregate_score': aggregate_score,
            'scale_factor': calib['scale_factor'] if calib else None,
            'implied_rve_size_um': calib['implied_rve_size_um'] if calib else None,
        })

    star_rank = {r['tslice_key']: i + 1 for i, r in enumerate(
        sorted(rows, key=lambda r: -r['n_stars']))}
    score_rank = {r['tslice_key']: i + 1 for i, r in enumerate(
        sorted(rows, key=lambda r: -r['aggregate_score']))}
    for r in rows:
        r['star_rank'] = star_rank[r['tslice_key']]
        r['score_rank'] = score_rank[r['tslice_key']]

    rows.sort(key=lambda r: (-r['n_stars'], -r['aggregate_score']))
    for i, r in enumerate(rows, start=1):
        r['coupled_rank'] = i
    return rows
