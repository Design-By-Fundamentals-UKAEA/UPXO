"""numba and numpy backends of the grain-element index agree exactly."""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d import fast_cleaning
from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D

needs_numba = pytest.mark.skipif(not fast_cleaning.numba_available(), reason="numba is not installed")
rng = np.random.default_rng(5)
SHAPES = [(1, 1, 7), (2, 2, 2), (3, 5, 4), (1, 6, 6), (9, 9, 9), (14, 10, 12)]


def _lgi(shape, n_labels, dtype, gaps=False, zeros=False):
    a = rng.integers(1, n_labels + 1, shape)
    if gaps:
        a = a * 3 + 2
    if zeros:
        a = np.where(rng.random(shape) < 0.05, 0, a)
    return np.ascontiguousarray(a.astype(dtype))


def _rotations(n):
    q = np.linalg.qr(rng.normal(size=(n, 3, 3)))[0]
    return q * np.sign(np.linalg.det(q))[:, None, None]


@needs_numba
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_grain_element_index_is_identical(dtype):
    for shape in SHAPES:
        for opts in ({}, {'gaps': True}, {'zeros': True}):
            lgi = _lgi(shape, int(rng.integers(1, 30)), dtype, **opts)
            roles = {int(g): 'host' for g in np.unique(lgi) if g > 0}
            a = AbaqusExporter3D.build_index(lgi, roles, {}, verbose=False, backend='numpy')
            b = AbaqusExporter3D.build_index(lgi, roles, {}, verbose=False, backend='numba')
            assert set(a.grain_elems) == set(b.grain_elems)
            for g in a.grain_elems:
                assert np.array_equal(a.grain_elems[g], b.grain_elems[g])
                assert a.grain_elems[g].dtype == b.grain_elems[g].dtype


def _structure():
    lgi = np.kron(rng.integers(1, 7, (4, 4, 4)), np.ones((4, 4, 4), int)).astype(np.int32)
    lgi = (np.unique(lgi, return_inverse=True)[1].reshape(lgi.shape) + 1).astype(np.int32)
    names = ['non_host', 'host', 'primary_twin', 'secondary_twin']
    role = {int(g): names[g % 4] for g in np.unique(lgi)}
    quats = {int(g): (lambda v: v / np.linalg.norm(v))(rng.normal(size=4)) for g in role}
    return lgi, role, {int(g): 1 for g in role}, quats


def test_without_numba_auto_runs_the_numpy_code_and_numba_is_refused(monkeypatch):
    monkeypatch.setattr(fast_cleaning, 'NUMBA_AVAILABLE', False)
    lgi = _lgi((6, 6, 6), 9, np.int32)
    roles = {int(g): 'host' for g in np.unique(lgi)}
    ref = AbaqusExporter3D.build_index(lgi, roles, {}, verbose=False, backend='numpy')
    auto = AbaqusExporter3D.build_index(lgi, roles, {}, verbose=False, backend='auto')
    assert all(np.array_equal(ref.grain_elems[g], auto.grain_elems[g]) for g in ref.grain_elems)
    with pytest.raises(ImportError):
        AbaqusExporter3D.build_index(lgi, roles, {}, verbose=False, backend='numba')
