"""
Unit tests for upxo.pxtal.gridops.axis_intercept_grain_size.

Run from repo root:
    pytest tests/pxtal/test_gridops.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.gridops import axis_intercept_grain_size


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
