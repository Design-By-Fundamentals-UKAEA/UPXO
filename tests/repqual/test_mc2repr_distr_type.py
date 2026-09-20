"""Regression tests for mc2repr.determine_distr_type per-sample values."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from scipy.stats import kurtosis, skew

from upxo.repqual import mcgs2d_representativeness_assesser as mod
from upxo.repqual.mcgs2d_representativeness_assesser import mc2repr


def _fake(target, sample):
    return SimpleNamespace(
        parameters=['area'],
        target=SimpleNamespace(prop={'area': target}),
        samples={'s1': SimpleNamespace(prop={'area': sample})},
        distr_type=defaultdict(lambda: defaultdict(dict)),
        build_distribution_dataset=lambda: None,
    )


def test_sample_records_use_sample_moments():
    """Sample skewness/kurtosis used to be copied from the target."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=200)
    sample = rng.exponential(size=200)
    fake = _fake(target, sample)
    mc2repr.determine_distr_type(fake)
    rec = fake.distr_type['s1']['area']
    assert rec['skewness'] == skew(sample)
    assert rec['kurtosis'] == kurtosis(sample)
    assert rec['skewness'] != fake.distr_type['target']['area']['skewness']


def test_sample_normality_uses_sample_shapiro_p(monkeypatch):
    """The sample 'normal' flag used to reuse the target's Shapiro p-value."""
    p_by_call = iter([(0.9, 0.9), (0.9, 0.001)])  # target normal, sample not
    monkeypatch.setattr(mod, 'shapiro', lambda x: next(p_by_call))
    rng = np.random.default_rng(1)
    data = rng.normal(size=200)  # skew/kurtosis both within the normal band
    fake = _fake(data, data.copy())
    mc2repr.determine_distr_type(fake)
    assert fake.distr_type['target']['area']['normal'] is True
    assert fake.distr_type['s1']['area']['normal'] is False
