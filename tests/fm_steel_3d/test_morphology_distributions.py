"""Per-feature and pooled-slice morphology values in physical units."""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import geom_metrics_3d as gm
from upxo.pxtal.fm_steel_3d import slice_metrics_2d as sm


def _cube(side=3, voxel_grid=7, offset=2):
    lgi = np.zeros((voxel_grid,) * 3, dtype=np.int32)
    lgi[offset:offset + side, offset:offset + side, offset:offset + side] = 1
    return lgi


# ── 3-D ───────────────────────────────────────────────────────────────────────

def test_a_cube_in_void():
    vs, a = 2.0, 3
    lgi = _cube(a)
    volume, surface = (a * vs) ** 3, 6 * (a * vs) ** 2
    expected = {
        'vol': volume,
        'surf': surface,
        'eqdia': (6 * volume / np.pi) ** (1 / 3),
        'sphericity': np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface,
        'nnb': 0,                      # void is not a neighbour
        'jl': 0,                       # only two labels meet anywhere
        'jp': 0,
        'aspect_ratio': 1.0,
    }
    for key, value in expected.items():
        got = gm.feature_distribution(lgi, vs, key)
        assert got.shape == (1,), key
        assert got[0] == pytest.approx(value), key
    assert expected['sphericity'] == pytest.approx((np.pi / 6) ** (1 / 3))


def test_two_slabs_see_each_other():
    lgi = np.ones((4, 3, 5), dtype=np.int32)
    lgi[2:] = 2
    assert list(gm.feature_distribution(lgi, 1.0, 'nnb')) == [1, 1]
    assert list(gm.feature_distribution(lgi, 0.5, 'vol')) == [2 * 3 * 5 / 8, 2 * 3 * 5 / 8]
    assert list(gm.feature_distribution(lgi, 1.0, 'aspect_ratio')) == [5 / 2, 5 / 2]


def test_labels_that_do_not_occur_are_left_out_and_order_follows_the_label():
    lgi = np.zeros((6, 4, 4), dtype=np.int32)
    lgi[0:1] = 1                       # 16 voxels
    lgi[2:5] = 3                       # 48 voxels; label 2 never occurs
    assert list(gm.feature_distribution(lgi, 1.0, 'vol')) == [16.0, 48.0]
    assert gm.feature_distribution(lgi, 1.0, 'nnb').shape == (2,)
    assert gm.feature_distribution(lgi, 1.0, 'aspect_ratio').shape == (2,)


def test_junction_values_scale_with_the_voxel_size():
    x, y = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    lgi = np.broadcast_to((1 + (x >= 3) + 2 * (y >= 3))[:, :, None], (6, 6, 6)).astype(np.int32)
    metrics = gm.all_metrics(lgi)
    jl = gm.feature_distribution(lgi, 0.25, 'jl')
    jp = gm.feature_distribution(lgi, 0.25, 'jp')
    assert jl.sum() > 0 and jp.sum() > 0
    np.testing.assert_array_equal(jl, metrics['jl'][1:] * 0.25)
    np.testing.assert_array_equal(jp, metrics['jp'][1:])
    assert list(gm.feature_distribution(lgi, 1.0, 'nnb')) == [2, 2, 2, 2]


@pytest.mark.parametrize('key', gm.FEATURE_DISTRIBUTION_KEYS)
def test_precomputed_metrics_give_the_same_values(key):
    rng = np.random.default_rng(4)
    lgi = rng.integers(0, 6, size=(8, 8, 8)).astype(np.int32)
    metrics = gm.all_metrics(lgi)
    np.testing.assert_array_equal(gm.feature_distribution(lgi, 1.3, key),
                                  gm.feature_distribution(lgi, 1.3, key, metrics=metrics))


def test_unknown_key():
    with pytest.raises(ValueError, match='key must be one of'):
        gm.feature_distribution(_cube(), 1.0, 'colour')


# ── 2-D ───────────────────────────────────────────────────────────────────────

def test_slice_values_skip_void():
    slice2d = np.array([[0, 1, 1], [0, 2, 2], [0, 2, 2]])
    np.testing.assert_array_equal(sm.slice_metric_values(slice2d, 0.5, 'area'), [2 * 0.25, 4 * 0.25])
    np.testing.assert_array_equal(sm.slice_metric_values(slice2d, 1.0, 'n_neighbours'), [1, 1])


def _layers(n=5, ny=4, nz=3):
    lgi = np.zeros((n, ny, nz), dtype=np.int32)
    for i in range(n):
        lgi[i] = i + 1
    return lgi


def test_pooled_slices_along_each_axis():
    lgi = _layers()
    vs = 2.0
    x = sm.pooled_slice_metric(lgi, vs, 'area', ('X',), 5)
    assert list(x) == [4 * 3 * vs ** 2] * 5                   # one label per slice
    y = sm.pooled_slice_metric(lgi, vs, 'area', ('Y',), 4)
    assert list(y) == [3 * vs ** 2] * (4 * 5)                 # five labels on each of four slices
    z = sm.pooled_slice_metric(lgi, vs, 'area', ('Z',), 2)
    assert list(z) == [4 * vs ** 2] * (2 * 5)


def test_pooled_slices_follow_the_order_of_the_axes():
    lgi = _layers()
    xz = sm.pooled_slice_metric(lgi, 1.0, 'area', ('X', 'Z'), 2)
    zx = sm.pooled_slice_metric(lgi, 1.0, 'area', ('Z', 'X'), 2)
    assert len(xz) == len(zx) == 2 + 10
    assert list(xz[:2]) == list(zx[10:])
    assert list(xz[2:]) == list(zx[:10])


def test_requesting_more_slices_than_voxels_takes_each_once():
    lgi = _layers(n=3)
    assert len(sm.pooled_slice_metric(lgi, 1.0, 'area', ('X',), 50)) == 3
    assert len(sm.pooled_slice_metric(lgi, 1.0, 'area', ('Z',), 50)) == 3 * 3


def test_no_slices():
    lgi = _layers()
    assert sm.pooled_slice_metric(lgi, 1.0, 'area', (), 3).shape == (0,)
    assert sm.pooled_slice_metric(lgi, 1.0, 'area', ('X',), 0).shape == (0,)
