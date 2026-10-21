"""
Unit tests for upxo.pxtal.gridops.axis_intercept_grain_size.

Run from repo root:
    pytest tests/pxtal/test_gridops.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.gridops import axis_intercept_grain_size, planar_intercept_grain_size


def _slab_lgi(nx=40, ny=40, nz=40, slab_thickness=4):
    """Grains stacked as slabs perpendicular to array-axis-2 ("z" in the
    (NX, NY, NZ) convention): label depends only on z, each slab spans the
    full x,y extent. Lines along x or y (fixed z) never cross a boundary
    within one slab; lines along z cross one every slab_thickness voxels.
    """
    z_idx = np.arange(nz)
    labels_1d = (z_idx // slab_thickness) + 1
    return np.broadcast_to(labels_1d[None, None, :], (nx, ny, nz)).astype(np.int32).copy()


def test_axis_mapping_x_y_much_larger_than_z():
    nx, ny, nz, slab = 40, 40, 40, 4
    voxel_size = 2.5
    lgi = _slab_lgi(nx, ny, nz, slab)

    sx = axis_intercept_grain_size(lgi, voxel_size, 'x', n_lines_target=100)
    sy = axis_intercept_grain_size(lgi, voxel_size, 'y', n_lines_target=100)
    sz = axis_intercept_grain_size(lgi, voxel_size, 'z', n_lines_target=100)

    assert sx['mean'] > sz['mean'] * 5
    assert sy['mean'] > sz['mean'] * 5


def test_physical_unit_scaling_exact_for_slab_case():
    nx, ny, nz, slab = 40, 40, 40, 4
    voxel_size = 2.5
    lgi = _slab_lgi(nx, ny, nz, slab)

    sx = axis_intercept_grain_size(lgi, voxel_size, 'x', n_lines_target=100)
    sy = axis_intercept_grain_size(lgi, voxel_size, 'y', n_lines_target=100)
    sz = axis_intercept_grain_size(lgi, voxel_size, 'z', n_lines_target=100)

    assert sx['mean'] == pytest.approx(nx * voxel_size)
    assert sy['mean'] == pytest.approx(ny * voxel_size)
    assert sz['mean'] == pytest.approx(slab * voxel_size)


def test_voxel_size_one_matches_raw_voxel_counts():
    lgi = _slab_lgi(20, 20, 20, 5)
    stats = axis_intercept_grain_size(lgi, 1.0, 'z', n_lines_target=50)
    assert stats['mean'] == pytest.approx(5.0)


def test_invalid_axis_raises():
    lgi = _slab_lgi(10, 10, 10, 2)
    with pytest.raises(ValueError):
        axis_intercept_grain_size(lgi, 1.0, 'w', n_lines_target=20)


def test_background_label_excluded():
    # Zero out the first two whole slabs (label 1: z=0-3, label 2: z=4-7),
    # cutting exactly on a slab boundary so every *remaining* slab (3,4,5)
    # stays full-thickness -- the intercept stats should only reflect real
    # grain (nonzero) segments, all still exactly slab_thickness long.
    lgi = _slab_lgi(20, 20, 20, 4)
    lgi[:, :, :8] = 0
    stats = axis_intercept_grain_size(lgi, 1.0, 'z', n_lines_target=50)
    assert stats['mean'] == pytest.approx(4.0)
    assert stats['n_segments'] > 0


def test_n_lines_respects_target_ballpark():
    lgi = _slab_lgi(100, 100, 20, 4)
    stats = axis_intercept_grain_size(lgi, 1.0, 'z', n_lines_target=200)
    # incr is chosen so n_lines is roughly on the order of n_lines_target
    # (exact count depends on integer rounding of the grid increment).
    assert 20 <= stats['n_lines'] <= 2000


# ── planar_intercept_grain_size (2D, axis-aligned only) ──────────────────

def _stripe_lgi_2d(h=10, w=20, stripe_width=5):
    """Grains as vertical stripes (label depends only on column): a line
    along x (a row) crosses one boundary every stripe_width columns; a
    line along y (a column) never crosses a boundary at all."""
    col_idx = np.arange(w)
    labels_1d = (col_idx // stripe_width) + 1
    return np.broadcast_to(labels_1d[None, :], (h, w)).astype(np.int32).copy()


def test_planar_axis_x_matches_hand_computed_segments():
    lgi = np.array([
        [1, 1, 1, 2, 2, 3, 3, 3, 3],
        [1, 1, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 4, 4, 5, 5, 5, 5, 5],
    ], dtype=np.int32)
    res = planar_intercept_grain_size(lgi, voxel_size=2.0, axis='x', n_lines=3)
    expected = np.array([3, 2, 4, 2, 3, 4, 4, 5], dtype=float) * 2.0
    assert res['n_segments'] == 8
    assert res['n_lines'] == 3
    assert np.isclose(res['mean'], expected.mean())
    assert np.allclose(np.sort(res['raw_segments']), np.sort(expected))


def test_planar_axis_y_never_crosses_vertical_stripes():
    lgi = _stripe_lgi_2d(h=10, w=20, stripe_width=5)
    res = planar_intercept_grain_size(lgi, voxel_size=1.0, axis='y', n_lines=20)
    # every column is a single uninformative label end-to-end -> one segment
    # per sampled column, each exactly h voxels long.
    assert res['n_segments'] == 20
    assert res['mean'] == pytest.approx(10.0)


def test_planar_axis_x_crosses_every_stripe_boundary():
    lgi = _stripe_lgi_2d(h=4, w=20, stripe_width=5)
    res = planar_intercept_grain_size(lgi, voxel_size=1.0, axis='x', n_lines=4)
    assert res['mean'] == pytest.approx(5.0)
    assert res['n_segments'] == 4 * 4  # 4 stripes x 4 sampled rows


def test_planar_invalid_axis_raises():
    lgi = np.ones((5, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        planar_intercept_grain_size(lgi, 1.0, 'z', n_lines=5)


def test_planar_requires_2d_input():
    lgi_3d = np.ones((5, 5, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        planar_intercept_grain_size(lgi_3d, 1.0, 'x', n_lines=5)


def test_planar_all_background_gives_zero_stats():
    bg = np.zeros((5, 5), dtype=np.int32)
    res = planar_intercept_grain_size(bg, 1.0, 'x', n_lines=5)
    assert res['n_segments'] == 0
    assert res['mean'] == 0.0
    assert res['raw_segments'].size == 0


def test_planar_background_excluded_but_grains_counted():
    lgi = np.array([[0, 0, 1, 1, 1, 0, 0, 2, 2, 2]], dtype=np.int32)
    res = planar_intercept_grain_size(lgi, 1.0, 'x', n_lines=1)
    assert res['n_segments'] == 2
    assert sorted(res['raw_segments'].tolist()) == [3.0, 3.0]


def test_planar_n_lines_clips_to_available_rows():
    lgi = np.ones((3, 10), dtype=np.int32)
    res = planar_intercept_grain_size(lgi, 1.0, 'x', n_lines=1000)
    assert res['n_lines'] == 3


def test_planar_phase_filter_masks_non_target_phase():
    lgi = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.int32)
    phase = np.where(lgi == 2, 1, 0)
    res = planar_intercept_grain_size(lgi, 1.0, 'x', n_lines=1,
                                      phase_array=phase, valid_phase_ids=[1])
    assert res['n_segments'] == 1
    assert res['raw_segments'][0] == pytest.approx(3.0)


def test_planar_voxel_size_scales_physical_length():
    lgi = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
    res = planar_intercept_grain_size(lgi, voxel_size=0.5, axis='x', n_lines=1)
    assert res['mean'] == pytest.approx(2.5)
