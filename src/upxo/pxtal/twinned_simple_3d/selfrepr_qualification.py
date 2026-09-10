"""
selfrepr_qualification.py
==========================
Threshold-based qualification of representativeness-assessment results
(twinned_simple_3d's Self Repr.-1 page): a subset's morphological
parameter is "qualified" under a given representativeness metric when
that metric's score is at or above a user-set threshold.
"""


def qualifies(value, threshold):
    """A representativeness score qualifies when it's >= threshold
    (both are "higher = more representative" scores, so this is a
    plain one-sided comparison)."""
    return value is not None and value >= threshold


def build_param_legend(selected_params, param_labels, param_abbrevs):
    """Maps each selected morphological parameter to its fixed
    abbreviation (e.g. "A" for Area, "AR" for Aspect Ratio -- from
    `selfrepr_morphology.MORPH_PARAM_ABBREV`, NOT a sequential A/B/C
    tied to selection order, which read as arbitrary/confusing) and
    returns the ordered [(abbrev, param_key, param_label), ...] list
    plus a formatted legend string ("A = Area, AR = Aspect Ratio,
    ...") -- both the table-building code and the on-screen legend
    line use the SAME ordering, so the abbreviations always agree."""
    entries = [
        (param_abbrevs.get(key, key.upper()), key, param_labels.get(key, key))
        for key in selected_params
    ]
    legend = ", ".join(f"{abbrev} = {label}" for abbrev, _key, label in entries)
    return entries, legend


def build_metric_legend(selected_metrics, metric_labels):
    """Maps each selected representativeness metric to a positional
    "RM1", "RM2", ... abbreviation (metric names like "Wasserstein
    Similarity" are too long for a table header) and returns the
    ordered [(abbrev, metric_key, metric_label), ...] list plus a
    formatted legend string ("RM1 = Wasserstein Similarity, ...")."""
    entries = [
        (f"RM{i + 1}", key, metric_labels.get(key, key))
        for i, key in enumerate(selected_metrics)
    ]
    legend = ", ".join(f"{abbrev} = {label}" for abbrev, _key, label in entries)
    return entries, legend


def build_qualification_table(assessment, selected_params, selected_metrics,
                               thresholds, subsets, param_labels, param_abbrevs,
                               metric_labels):
    """
    Builds the Block [8] qualification table: one row per
    (subset, parameter) pair, one column per selected metric.

    Parameters
    ----------
    assessment : dict[(subset_index, param_key), dict[metric_key, float]]
        Block [6]'s "Assess" output -- ``pipeline['selfrepr1_assessment']``.
    selected_params : list of str
        Morphological parameter keys, in the order rows are grouped.
    selected_metrics : list of str
        Representativeness metric keys, in column order ("RM1", "RM2", ...).
    thresholds : dict[(metric_key, param_key), float]
        Per-(metric, parameter) qualification threshold.
    subsets : list of dict
        Each with 'index' (tuple) and 'centroid_um' (tuple) -- e.g. the
        tiles from ``subsetting_2d.generate_subset_tiles_2d`` plus a
        precomputed 'centroid_um'.
    param_labels : dict[str, str]
        Display label per parameter key (e.g. selfrepr_morphology.MORPH_PARAM_LABELS).
    param_abbrevs : dict[str, str]
        Fixed abbreviation per parameter key (e.g. selfrepr_morphology.MORPH_PARAM_ABBREV).
    metric_labels : dict[str, str]
        Display label per metric key (e.g. {k: label for k, (label, _fn)
        in representativeness_metrics.METRIC_REGISTRY.items()}).

    Returns
    -------
    (rows, legend) : (list of dict, str)
        rows: one dict per (subset, parameter), keys 'subset', 'centroid',
        'row_name' (the parameter's abbreviation), plus one bool (or
        None if not computable) per metric key in `selected_metrics`.
        legend: "MP = Morphological Parameter (A = Area, ...). RM =
        Representativeness Metric (RM1 = Wasserstein Similarity, ...)."
    """
    param_entries, param_legend = build_param_legend(selected_params, param_labels, param_abbrevs)
    _metric_entries, metric_legend = build_metric_legend(selected_metrics, metric_labels)
    legend = (f"MP = Morphological Parameter ({param_legend}).  "
              f"RM = Representativeness Metric ({metric_legend}).")

    rows = []
    for subset in subsets:
        subset_idx = subset['index']
        centroid = subset.get('centroid_um')
        for abbrev, param_key, _label in param_entries:
            row = {'subset': subset_idx, 'centroid': centroid, 'row_name': abbrev}
            scores = assessment.get((subset_idx, param_key), {})
            for metric_key in selected_metrics:
                value = scores.get(metric_key)
                threshold = thresholds.get((metric_key, param_key))
                if value is None or threshold is None:
                    row[metric_key] = None
                else:
                    row[metric_key] = qualifies(value, threshold)
            rows.append(row)
    return rows, legend


def rank_subsets_by_qualification_count(assessment, metric_key, selected_params,
                                         thresholds, subsets, default_threshold=0.6):
    """
    Ranks every sub-set by how many of `selected_params` it QUALIFIES
    on, for ONE representativeness metric -- "which sub-set looks most
    like the parent domain across the most properties, under this one
    metric" (as opposed to [8]'s table, which shows every metric x
    parameter pair without collapsing them into a single ranking).

    Parameters
    ----------
    assessment : dict[(subset_index, param_key), dict[metric_key, float]]
        Block [6]'s "Assess" output -- ``pipeline['selfrepr1_assessment']``.
    metric_key : str
        The single representativeness metric to rank by.
    selected_params : list of str
        Morphological parameter keys to consider.
    thresholds : dict[(metric_key, param_key), float]
        Per-(metric, parameter) qualification threshold -- typically
        [8]'s ``qual_threshold_vars``. A (metric_key, param_key) pair
        with no configured threshold falls back to `default_threshold`,
        so ranking works even for a metric/parameter combination [8]
        was never configured with.
    subsets : list of dict
        Each with 'index' (tuple) and 'centroid_um' (tuple).
    default_threshold : float
        Fallback threshold for any (metric_key, param_key) pair not
        present in `thresholds`.

    Returns
    -------
    list of dict, sorted by 'score' descending (ties broken by 'subset'
    index for a stable, reproducible order):
        {'subset': (i, j), 'centroid': (x, y), 'score': int,
         'total': int, 'qualifying_params': [param_key, ...]}
    """
    rows = []
    for subset in subsets:
        subset_idx = subset['index']
        centroid = subset.get('centroid_um')
        qualifying = []
        for param_key in selected_params:
            value = assessment.get((subset_idx, param_key), {}).get(metric_key)
            threshold = thresholds.get((metric_key, param_key), default_threshold)
            if qualifies(value, threshold):
                qualifying.append(param_key)
        rows.append({
            'subset': subset_idx, 'centroid': centroid,
            'score': len(qualifying), 'total': len(selected_params),
            'qualifying_params': qualifying,
        })
    rows.sort(key=lambda r: (-r['score'], r['subset']))
    return rows
