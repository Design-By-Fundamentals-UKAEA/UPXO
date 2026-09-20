"""Regression tests for mc2repr.set_*_thresh."""
from types import SimpleNamespace

import pytest

from upxo.repqual.mcgs2d_representativeness_assesser import mc2repr


def _fake():
    return SimpleNamespace(stest={}, _unit_interval=mc2repr._unit_interval)


def _no_input(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('input() must not be called for a valid value')
    monkeypatch.setattr('builtins.input', boom)


@pytest.mark.parametrize('method, key, args, expected', [
    ('set_cor_thresh', 'cor_threshold', (0.8,), 0.8),
    ('set_kldiv_thresh', 'kldiv_thresh', (0.3,), 0.3),
    ('set_jsdiv_thresh', 'jsdiv_thresh', (0.0,), 0.0),
])
def test_valid_threshold_is_stored_without_prompt(monkeypatch, method, key,
                                                   args, expected):
    """A valid value used to be silently discarded."""
    _no_input(monkeypatch)
    fake = _fake()
    getattr(mc2repr, method)(fake, *args)
    assert fake.stest[key] == expected


def test_valid_ks_thresholds_are_stored(monkeypatch):
    """Both KS thresholds are stored when valid."""
    _no_input(monkeypatch)
    fake = _fake()
    mc2repr.set_ks_thresh(fake, 0.05, 1.0)
    assert fake.stest['ks_thresh_D'] == 0.05
    assert fake.stest['ks_thresh_P'] == 1.0


def test_out_of_range_prompts_until_valid(monkeypatch):
    """An invalid value prompts and terminates once a valid one is given."""
    answers = iter(['5', '-1', '0.4'])
    monkeypatch.setattr('builtins.input', lambda *a: next(answers))
    fake = _fake()
    mc2repr.set_cor_thresh(fake, 2.0)
    assert fake.stest['cor_threshold'] == 0.4
    mc2repr.set_kldiv_thresh(fake, 0.5)  # valid: must not consume input
    assert fake.stest['kldiv_thresh'] == 0.5
