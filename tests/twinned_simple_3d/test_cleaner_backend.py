"""StructureCleaner3D: the numba and numpy backends give the same cleaned structure."""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d import fast_cleaning
from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D

gaussian_filter = pytest.importorskip("scipy.ndimage").gaussian_filter
needs_numba = pytest.mark.skipif(not fast_cleaning.numba_available(), reason="numba is not installed")


def _structure(n, q, seed, sigma=1.6, speckle=0.003):
    """Labels reused across space, so grains have several lobes, plus speckle spikes."""
    rng = np.random.default_rng(seed)
    best = np.full((n, n, n), -1.0, dtype=np.float32)
    lgi = np.zeros((n, n, n), dtype=np.int32)
    for k in range(q):
        f = gaussian_filter(rng.standard_normal((n, n, n)).astype(np.float32), sigma)
        m = f > best
        best[m], lgi[m] = f[m], k + 1
    noise = rng.random((n, n, n)) < speckle
    lgi[noise] = rng.integers(1, q + 1, int(noise.sum()))
    names = ['non_host', 'host', 'primary_twin', 'secondary_twin']
    gids = [int(g) for g in np.unique(lgi)]
    role = {g: names[g % 4] for g in gids}
    quats = {g: (lambda v: v / np.linalg.norm(v))(rng.normal(size=4)) for g in gids}
    return lgi, quats, role, {g: 1 for g in gids}


def _clean(backend, lgi, quats, role, parent, **kw):
    c = StructureCleaner3D(rng_seed=4, backend=backend, **kw)
    c.clean(lgi, dict(quats), dict(role), dict(parent))
    return c


def _same(a, b):
    return (np.array_equal(a.lgi_clean, b.lgi_clean) and a.spike_count == b.spike_count
            and a.n_splits == b.n_splits and a.remaining_spikes == b.remaining_spikes
            and a.remaining_multi == b.remaining_multi and a.twin_role_clean == b.twin_role_clean
            and a.twin_parent_of_clean == b.twin_parent_of_clean and a.split_events == b.split_events
            and set(a.all_quats_clean) == set(b.all_quats_clean)
            and all(np.array_equal(a.all_quats_clean[k], b.all_quats_clean[k]) for k in a.all_quats_clean))


@needs_numba
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("opts", [
    {},
    {'split_jitter_deg': 2.0},
    {'min_clean_voxels': 6},
    {'upscale_fallback': True},
    {'do_spike_removal': False},
    {'do_lobe_split': False},
])
def test_both_backends_clean_to_the_same_structure(seed, opts, capsys):
    lgi, quats, role, parent = _structure(n=22 + 2 * seed, q=[8, 20, 40, 12, 60][seed], seed=seed)
    a = _clean('numpy', lgi, quats, role, parent, **opts)
    b = _clean('numba', lgi, quats, role, parent, **opts)
    assert _same(a, b)


@needs_numba
def test_recursive_cleaning_passes_the_backend_through(capsys):
    lgi, quats, role, parent = _structure(n=24, q=30, seed=9)
    out = {}
    for backend in ('numpy', 'numba'):
        cleaner, n = StructureCleaner3D.clean_recursive(
            lgi, dict(quats), dict(role), dict(parent), n_passes=4, rng_seed=2, verbose=False, backend=backend)
        out[backend] = (cleaner, n)
    assert out['numpy'][1] == out['numba'][1]
    assert np.array_equal(out['numpy'][0].lgi_clean, out['numba'][0].lgi_clean)


def test_the_input_arrays_are_not_modified(capsys):
    lgi, quats, role, parent = _structure(n=18, q=20, seed=1)
    before = lgi.copy()
    _clean('auto', lgi, quats, role, parent)
    assert np.array_equal(lgi, before)


def test_without_numba_auto_uses_numpy_and_numba_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(fast_cleaning, 'NUMBA_AVAILABLE', False)
    lgi, quats, role, parent = _structure(n=16, q=15, seed=2)
    assert StructureCleaner3D(backend='auto').backend == 'numpy'
    a = _clean('numpy', lgi, quats, role, parent)
    b = _clean('auto', lgi, quats, role, parent)
    assert _same(a, b)
    with pytest.raises(ImportError):
        StructureCleaner3D(backend='numba')
