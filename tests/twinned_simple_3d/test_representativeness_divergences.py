"""KL divergence and JS distance in the representativeness metrics and the multi-axis slice validator."""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.repr_validator_3d import RepresentativenessValidator3D
from upxo.pxtal.twinned_simple_3d.representativeness_metrics import (
    METRIC_REGISTRY, compute_all_selected, distribution_divergences)


def test_identical_values_have_zero_divergence_and_distance():
    data = np.random.default_rng(0).normal(size=300)
    kl, js = distribution_divergences(data, data)
    assert kl == pytest.approx(0, abs=1e-9) and js == pytest.approx(0, abs=1e-9)
    assert distribution_divergences([2.0, 2.0], [2.0, 2.0, 2.0]) == (0.0, 0.0)


def test_disjoint_values_have_distance_one_and_a_finite_divergence():
    kl, js = distribution_divergences(np.linspace(0, 1, 50), np.linspace(10, 11, 50))
    assert js == pytest.approx(1.0)
    assert np.isfinite(kl) and kl > 10


def test_values_without_a_finite_entry_give_nan():
    kl, js = distribution_divergences([np.nan], [1.0, 2.0])
    assert np.isnan(kl) and np.isnan(js)
    kl, js = distribution_divergences([], [1.0, 2.0])
    assert np.isnan(kl) and np.isnan(js)


def test_the_registry_has_the_kl_similarity_between_zero_and_one():
    assert 'kl_sim' in METRIC_REGISTRY and 'js_sim' in METRIC_REGISTRY
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 500)
    near, far = rng.normal(0, 1, 500), rng.normal(6, 1, 500)
    scores = compute_all_selected(['kl_sim', 'js_sim'], ref, near)
    assert set(scores) == {'kl_sim', 'js_sim'}
    assert all(0 < v <= 1 for v in scores.values())
    far_scores = compute_all_selected(['kl_sim', 'js_sim'], ref, far)
    assert far_scores['kl_sim'] < scores['kl_sim'] and far_scores['js_sim'] < scores['js_sim']
    assert compute_all_selected(['kl_sim'], ref, ref)['kl_sim'] == pytest.approx(1.0)


def _structure(n=24, cell=6):
    labels = np.arange(1, (n // cell) ** 3 + 1).reshape((n // cell,) * 3)
    return np.kron(labels, np.ones((cell,) * 3, dtype=int)).astype(np.int32)


def test_the_validator_records_kl_and_js_per_slice_and_the_mean_per_axis():
    lgi = _structure()
    equal_areas = np.full(40, 36.0)                                  # the same cell area as every slice grain
    validator = RepresentativenessValidator3D(n_slices_x=3, n_slices_y=3, n_slices_z=3)
    validator.validate_morphological(lgi, equal_areas)
    for axis, results in validator.slice_results.items():
        assert all(r['kl'] == pytest.approx(0, abs=1e-9) and r['js'] == pytest.approx(0, abs=1e-9) for r in results)
        info = validator.axis_acceptance[axis]
        assert info['mean_kl'] == pytest.approx(0, abs=1e-9) and info['mean_js'] == pytest.approx(0, abs=1e-9)
    assert validator.overall_accepted is True
    assert 'mean KL' in validator.report() and 'mean JS' in validator.report()

    different = RepresentativenessValidator3D(n_slices_x=2, n_slices_y=2, n_slices_z=2)
    different.validate_morphological(lgi, np.concatenate([np.full(20, 5.0), np.full(20, 500.0)]))
    assert all(info['mean_js'] > 0 and info['mean_kl'] > 0 for info in different.axis_acceptance.values())


def test_the_divergences_do_not_decide_acceptance():
    validator = RepresentativenessValidator3D(p_percentage=50.0)
    validator.slice_results = {'X': [{'passes': True, 'kl': 9.0, 'js': 1.0}, {'passes': False, 'kl': np.nan, 'js': np.nan}]}
    validator.aggregate()
    info = validator.axis_acceptance['X']
    assert info['accepted'] is True and info['pct_pass'] == 50.0
    assert info['mean_kl'] == 9.0 and info['mean_js'] == 1.0         # slices without a value are left out of the mean
