"""numba and numpy backends of the MC small-grain and spike removal agree exactly."""
import types

import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d import fast_cleaning
from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase

cc3d = pytest.importorskip("cc3d")
gaussian_filter = pytest.importorskip("scipy.ndimage").gaussian_filter
needs_numba = pytest.mark.skipif(not fast_cleaning.numba_available(), reason="numba is not installed")


def _structure(n=28, q=12, sigma=1.4, speckle=0.01, seed=0):
    rng = np.random.default_rng(seed)
    best = np.full((n, n, n), -1.0, dtype=np.float32)
    state = np.zeros((n, n, n), dtype=np.int16)
    for k in range(q):
        f = gaussian_filter(rng.standard_normal((n, n, n)).astype(np.float32), sigma)
        m = f > best
        best[m], state[m] = f[m], k
    noise = rng.random((n, n, n)) < speckle
    state[noise] = rng.integers(0, q, int(noise.sum()))
    return state, cc3d.connected_components(state, connectivity=6).astype(np.int32)


def _pxt(state, lgi):
    ns = types.SimpleNamespace(s=state.copy(), lgi=lgi.copy())
    return types.SimpleNamespace(m=[0], gs={0: ns}), ns


def _run(state, lgi, backend, **kw):
    pxt, ns = _pxt(state, lgi)
    summary = TwinnedSimple3DBase.clean_temporal_slices(pxt, verbose=False, backend=backend, **kw)
    return ns, summary[0]


@needs_numba
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("merge, spike", [(True, False), (False, True), (True, True)])
def test_numba_gives_the_numpy_labels_and_counts(seed, merge, spike):
    state, lgi = _structure(sigma=0.9 + 0.4 * seed, speckle=[0.0, 0.003, 0.02][seed % 3], seed=seed)
    size = [1, 2, 4, 8, 16, 40][seed]
    a, sa = _run(state, lgi, 'numpy', min_grain_size=size, do_merge_small=merge, do_spike_removal=spike)
    b, sb = _run(state, lgi, 'numba', min_grain_size=size, do_merge_small=merge, do_spike_removal=spike)
    assert sa == sb
    assert np.array_equal(a.lgi, b.lgi) and np.array_equal(a.s, b.s)


@needs_numba
def test_non_contiguous_labels_and_unlabelled_voxels_agree():
    state, lgi = _structure(seed=3)
    lgi = np.where(lgi % 7 == 0, 0, lgi * 3 + 5).astype(np.int32)
    a, sa = _run(state, lgi, 'numpy', min_grain_size=6)
    b, sb = _run(state, lgi, 'numba', min_grain_size=6)
    assert sa == sb and np.array_equal(a.lgi, b.lgi)


@needs_numba
def test_recursive_cleaning_passes_the_backend_through():
    state, lgi = _structure(seed=2)
    out = {}
    for backend in ('numpy', 'numba'):
        pxt, ns = _pxt(state, lgi)
        out[backend] = (TwinnedSimple3DBase.clean_temporal_slices_recursive(
            pxt, min_grain_size=4, n_passes=4, verbose=False, backend=backend), ns.lgi)
    assert out['numpy'][0] == out['numba'][0] and np.array_equal(out['numpy'][1], out['numba'][1])


def test_the_numpy_backend_does_not_need_numba(monkeypatch):
    monkeypatch.setattr(fast_cleaning, 'NUMBA_AVAILABLE', False)
    state, lgi = _structure(n=16, seed=1)
    ns, summary = _run(state, lgi, 'auto', min_grain_size=3)
    ref, ref_summary = _run(state, lgi, 'numpy', min_grain_size=3)
    assert summary == ref_summary and np.array_equal(ns.lgi, ref.lgi)


def test_resolve_backend(monkeypatch):
    monkeypatch.setattr(fast_cleaning, 'NUMBA_AVAILABLE', False)
    assert fast_cleaning.resolve_backend('auto') == 'numpy'
    assert fast_cleaning.resolve_backend('numpy') == 'numpy'
    with pytest.raises(ImportError):
        fast_cleaning.resolve_backend('numba')
    monkeypatch.setattr(fast_cleaning, 'NUMBA_AVAILABLE', True)
    assert fast_cleaning.resolve_backend('auto') == 'numba'
    assert fast_cleaning.resolve_backend('numba') == 'numba'
    assert fast_cleaning.resolve_backend('numpy') == 'numpy'
    with pytest.raises(ValueError):
        fast_cleaning.resolve_backend('cuda')
