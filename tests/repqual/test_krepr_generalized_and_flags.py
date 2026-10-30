"""Regression tests: the generalized RKF measures return their matrix; the 3D/3D flag setup does not call a missing method."""
from types import SimpleNamespace

import numpy as np
import pytest

from upxo.repqual.grain_network_repr_assesser import KREPR as KREPR2D
from upxo.repqual.grain_network_repr_assesser_3D import KREPR as KREPR3D

MEASURES = [('calculate_rkf_wd_on_generalized', 'calculate_rkf_wd_pairwise'),
            ('calculate_rkf_ksp_on_generalized', 'calculate_rkf_ksp_pairwise'),
            ('calculate_rkf_ed_on_generalized', 'calculate_rkf_ed_pairwise'),
            ('calculate_rkf_ed_nlsd_generalized', 'calculate_rkf_nlsd_pairwise')]


def _fake(pairwise_name):
    """Two target networks of order 1 and three sample networks of order 2; the pairwise value encodes the pair."""
    def pairwise(ktgt, ksmp, **kwargs):
        return 10.0 * ktgt + ksmp
    return SimpleNamespace(tkset={1: {0: 1, 1: 2}}, skset={2: {0: 3, 1: 4, 2: 5}}, ntid=2, nsid=3,
                           **{pairwise_name: pairwise})


@pytest.mark.parametrize('krepr', [KREPR2D, KREPR3D], ids=['2d', '3d'])
@pytest.mark.parametrize('method, pairwise', MEASURES)
def test_generalized_measure_returns_the_sample_by_target_matrix(krepr, method, pairwise):
    fake = _fake(pairwise)
    result = getattr(krepr, method)(fake, neigh_order_tgt=1, neigh_order_smp=2)
    assert result.shape == (3, 2)
    expected = np.array([[10 * t + s for t in (1, 2)] for s in (3, 4, 5)], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_three_dimensional_pair_sets_only_the_morphological_flags():
    calls = []
    fake = SimpleNamespace(dim=SimpleNamespace(tgt=3, smp=3), set_mprop3d_flags=lambda: calls.append('m3'),
                           set_mprop2d_flags=lambda: calls.append('m2'))
    KREPR2D.init_subdef_set_prop_flags(fake)
    assert calls == ['m3']
