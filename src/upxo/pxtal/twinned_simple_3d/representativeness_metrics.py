"""
representativeness_metrics.py
==============================
A registry of two-sample distribution-comparison statistics, each
wrapped as a "higher score = higher representativeness" similarity --
for comparing a candidate (subset) property distribution against a
reference (parent) distribution. Decomposes
``TwinnedSimple3DBase.compare_property_distributions``'s bundled
3-metric score into individually selectable entries, and adds more
standard two-sample statistics on top.
"""

import warnings

import numpy as np

from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase as _T


def _std_normalized_squash(d, ref, score_function):
    ref_std = float(np.std(ref))
    d_norm = (d / ref_std) if ref_std > 0 else (0.0 if d == 0 else float('inf'))
    return _T._squash_distance(d_norm, score_function)


def _wasserstein_sim(ref, cand, score_function='exp'):
    from scipy.stats import wasserstein_distance
    return _std_normalized_squash(float(wasserstein_distance(ref, cand)), ref, score_function)


def _energy_sim(ref, cand, score_function='exp'):
    from scipy.stats import energy_distance
    return _std_normalized_squash(float(energy_distance(ref, cand)), ref, score_function)


def _ks_sim(ref, cand, score_function='exp'):
    from scipy.stats import ks_2samp
    stat, _p = ks_2samp(ref, cand)
    return float(1.0 - stat)


def _ks_pvalue(ref, cand, score_function='exp'):
    from scipy.stats import ks_2samp
    _stat, p = ks_2samp(ref, cand)
    return float(p)


def _cvm_sim(ref, cand, score_function='exp'):
    from scipy.stats import cramervonmises_2samp
    r = cramervonmises_2samp(ref, cand)
    return _T._squash_distance(float(r.statistic), score_function)


def _anderson_sim(ref, cand, score_function='exp'):
    # Uses the statistic (squashed), NOT .significance_level/.pvalue --
    # both are interpolated from a fixed table and effectively capped
    # well below 1.0 even for identical distributions (confirmed: tops
    # out around 0.2-0.25), so they'd never clear a representativeness
    # threshold set anywhere near the middle of [0, 1].
    #
    # Unlike KS/CVM/Wasserstein/energy, the Anderson-Darling k-sample
    # statistic is NOT guaranteed non-negative -- for near-identical
    # samples it can go slightly negative, which would push
    # exp(-x) above 1.0 and break every other metric's [0,1] bound.
    # Clamp to 0 first (a negative statistic means "at least as similar
    # as an exact match", the same ceiling as x=0).
    from scipy.stats import anderson_ksamp
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = anderson_ksamp([ref, cand])
    return _T._squash_distance(max(0.0, float(r.statistic)), score_function)


def _mannwhitney_pvalue(ref, cand, score_function='exp'):
    from scipy.stats import mannwhitneyu
    _stat, p = mannwhitneyu(ref, cand, alternative='two-sided')
    return float(p)


def _kruskal_pvalue(ref, cand, score_function='exp'):
    from scipy.stats import kruskal
    _stat, p = kruskal(ref, cand)
    return float(p)


def _js_sim(ref, cand, score_function='exp', bins=40):
    from scipy.spatial.distance import jensenshannon
    combined = np.concatenate([ref, cand])
    vmin, vmax = combined.min(), combined.max()
    if vmin == vmax:
        return 1.0
    edges = np.linspace(vmin, vmax, bins + 1)
    h_ref, _ = np.histogram(ref, bins=edges)
    h_cand, _ = np.histogram(cand, bins=edges)
    h_ref = h_ref.astype(float)
    h_cand = h_cand.astype(float)
    if h_ref.sum() == 0 or h_cand.sum() == 0:
        return float('nan')
    h_ref /= h_ref.sum()
    h_cand /= h_cand.sum()
    d = float(jensenshannon(h_ref, h_cand, base=2))
    if not np.isfinite(d):
        d = 1.0
    return float(1.0 - d)


# {key: (display_label, compute_fn(ref, cand, score_function='exp') -> float)}
# Every entry: higher = more representative (the user's explicit rule).
METRIC_REGISTRY = {
    'wasserstein_sim': ('Wasserstein Similarity', _wasserstein_sim),
    'energy_sim': ('Energy Distance Similarity', _energy_sim),
    'ks_sim': ('KS2 Similarity', _ks_sim),
    'ks_pvalue': ('KS2 P-Value', _ks_pvalue),
    'cvm_sim': ('Cramer-von Mises Similarity', _cvm_sim),
    'anderson_sim': ('Anderson-Darling Similarity', _anderson_sim),
    'mannwhitney_pvalue': ('Mann-Whitney U P-Value', _mannwhitney_pvalue),
    'kruskal_pvalue': ('Kruskal-Wallis P-Value', _kruskal_pvalue),
    'js_sim': ('Jensen-Shannon Similarity', _js_sim),
}


def compute_all_selected(selected_keys, ref_vals, cand_vals, low_pct=0.0, high_pct=100.0, score_function='exp'):
    """
    Compute every selected representativeness metric comparing
    `cand_vals` (a subset's property distribution) against `ref_vals`
    (the parent/reference distribution), after independently trimming
    each side to [low_pct, high_pct] percentiles of its own range
    (``TwinnedSimple3DBase.percentile_trim``).

    Parameters
    ----------
    selected_keys : iterable of str
        Keys into METRIC_REGISTRY.
    ref_vals, cand_vals : array-like
        Raw per-grain property values.
    low_pct, high_pct : float
        Percentile-of-range outlier trim, applied identically to both
        sides. (0, 100) is a no-op.
    score_function : str
        'exp' or 'reciprocal' -- see ``TwinnedSimple3DBase._squash_distance``.

    Returns
    -------
    dict[str, float]
        {metric_key: score}, one entry per requested key that both
        exists in METRIC_REGISTRY and computed successfully to a finite
        value. A metric that raises (e.g. too few points after
        trimming) OR returns NaN/inf (some statistics -- e.g.
        Kruskal-Wallis on two literally-identical samples -- degenerate
        to NaN rather than raising) is silently omitted rather than
        aborting the whole batch -- callers should treat a missing key
        as "not computable for this pair", not as an error. Empty dict
        if either side has fewer than 2 points after trimming.
    """
    ref = _T.percentile_trim(np.asarray(ref_vals, dtype=float), low_pct, high_pct)
    cand = _T.percentile_trim(np.asarray(cand_vals, dtype=float), low_pct, high_pct)

    scores = {}
    if ref.size < 2 or cand.size < 2:
        return scores
    for key in selected_keys:
        entry = METRIC_REGISTRY.get(key)
        if entry is None:
            continue
        _label, fn = entry
        try:
            value = fn(ref, cand, score_function=score_function)
        except Exception:
            continue
        if np.isfinite(value):
            scores[key] = value
    return scores
