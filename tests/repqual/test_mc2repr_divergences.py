"""KL divergence and JS distance of the mc2repr tests: values on shared histogram bins."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from upxo.repqual.mcgs2d_representativeness_assesser import (
    js_distance, kl_divergence, mc2repr, shared_bin_probabilities)


def test_probabilities_use_the_same_bins_and_sum_to_one():
    p, q = shared_bin_probabilities([0, 1, 2, 3], [2, 3, 4, 5], bins=5)
    assert len(p) == len(q) == 5
    assert p.sum() == pytest.approx(1) and q.sum() == pytest.approx(1)
    np.testing.assert_allclose(p, [0.25, 0.25, 0.25, 0.25, 0.0])


def test_identical_data_has_zero_divergence_and_distance():
    data = np.random.default_rng(0).normal(size=300)
    assert kl_divergence(data, data) == pytest.approx(0, abs=1e-9)
    assert js_distance(data, data) == pytest.approx(0, abs=1e-9)


def test_disjoint_data_has_distance_one_and_a_large_finite_divergence():
    a, b = np.linspace(0, 1, 50), np.linspace(10, 11, 50)
    assert js_distance(a, b, bins=20) == pytest.approx(1.0)
    kl = kl_divergence(a, b, bins=20)
    assert np.isfinite(kl) and kl > 10


def test_known_two_bin_values():
    target = [0] * 3 + [1] * 1                                     # p = (0.75, 0.25)
    sample = [0] * 1 + [1] * 3                                     # q = (0.25, 0.75)
    expected_kl = 0.75 * np.log(3) + 0.25 * np.log(1 / 3)
    assert kl_divergence(target, sample, bins=2) == pytest.approx(expected_kl, rel=1e-6)
    m = np.array([0.5, 0.5])
    p, q = np.array([0.75, 0.25]), np.array([0.25, 0.75])
    div = 0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum()
    assert js_distance(target, sample, bins=2) == pytest.approx(np.sqrt(div), rel=1e-6)


def test_missing_values_are_dropped_and_empty_data_gives_nan():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    assert kl_divergence(np.append(base, np.nan), base) == pytest.approx(kl_divergence(base, base))
    assert np.isnan(kl_divergence([], base)) and np.isnan(js_distance(base, [np.nan]))


def test_distance_is_symmetric_and_divergence_is_not():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 400), rng.normal(0.5, 2, 400)
    assert js_distance(a, b) == pytest.approx(js_distance(b, a))
    assert kl_divergence(a, b) != pytest.approx(kl_divergence(b, a))


def test_the_tests_fill_the_performance_record():
    rng = np.random.default_rng(3)
    target = pd.DataFrame({'area': rng.normal(10, 2, 300)})
    close = pd.DataFrame({'area': rng.normal(10, 2, 300)})
    far = pd.DataFrame({'area': rng.normal(25, 2, 300)})
    fake = SimpleNamespace(stest={'tests': ['kldiv', 'jsdiv']}, parameters=['area'],
                           target=SimpleNamespace(prop=target), performance={},
                           samples={'close': SimpleNamespace(prop=close), 'far': SimpleNamespace(prop=far)})
    mc2repr.test(fake)
    near, distant = fake.performance['close']['area'], fake.performance['far']['area']
    assert 0 <= near['jsdiv'] < distant['jsdiv'] <= 1
    assert 0 <= near['kldiv'] < distant['kldiv']
