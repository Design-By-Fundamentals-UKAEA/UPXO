"""Regression tests: NLSD results must land in rkf['nlsd'], not rkf['ed']."""
from types import SimpleNamespace

import numpy as np
import pytest

from upxo.repqual.grain_network_repr_assesser import KREPR as KREPR2D
from upxo.repqual.grain_network_repr_assesser_3D import KREPR as KREPR3D


def _fake_krepr(nlsd_value=7.0, ed_value=-1.0):
    """Minimal stand-in exposing what calculate_rkf_nlsd_on touches."""
    shape = (2, 3)  # (n samples, n targets)
    return SimpleNamespace(
        tkset={1: {i: f't{i}' for i in range(shape[1])}},
        skset={1: {i: f's{i}' for i in range(shape[0])}},
        rkf={'ed': {1: np.full(shape, ed_value)},
             'nlsd': {1: np.zeros(shape)}},
        calculate_rkf_nlsd_pairwise=lambda *a, **k: nlsd_value,
    )


@pytest.mark.parametrize('krepr', [KREPR2D, KREPR3D], ids=['2d', '3d'])
def test_nlsd_written_to_nlsd_field_only(krepr):
    """NLSD values fill rkf['nlsd'] and leave rkf['ed'] untouched."""
    fake = _fake_krepr()
    krepr.calculate_rkf_nlsd_on(fake, neigh_order=1)
    assert np.all(fake.rkf['nlsd'][1] == 7.0)
    assert np.all(fake.rkf['ed'][1] == -1.0)
