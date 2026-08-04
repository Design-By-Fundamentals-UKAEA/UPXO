"""Tests for upxo.propOps.morphops."""
import numpy as np
import pytest
from upxo.propOps.morphops import volumes_with_bincount, volume_dict


def test_volumes_with_bincount_single_grain():
    """Count voxels for a single grain."""
    lgi = np.array([[1, 1], [1, 1]], dtype=int)
    vols = volumes_with_bincount(lgi)
    assert vols[1] == 4


def test_volumes_with_bincount_multiple_grains():
    """Count voxels for multiple grains."""
    lgi = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 3]
    ], dtype=int)
    vols = volumes_with_bincount(lgi)
    assert vols[1] == 3
    assert vols[2] == 3
    assert vols[3] == 3


def test_volumes_with_bincount_includes_zero():
    """Bincount includes background/zero label."""
    lgi = np.array([[0, 1], [1, 1]], dtype=int)
    vols = volumes_with_bincount(lgi)
    assert vols[0] == 1
    assert vols[1] == 3


def test_volumes_with_bincount_3d():
    """Count voxels in a 3D array."""
    lgi = np.ones((3, 3, 3), dtype=int)
    lgi[0, 0, 0] = 2
    vols = volumes_with_bincount(lgi)
    assert vols[1] == 26
    assert vols[2] == 1


def test_volumes_with_bincount_custom_nlabels():
    """Use custom nlabels to set output size."""
    lgi = np.array([[1, 1], [2, 2]], dtype=int)
    vols = volumes_with_bincount(lgi, nlabels=10)
    assert len(vols) >= 10
    assert vols[1] == 2
    assert vols[2] == 2


def test_volume_dict_single_grain():
    """volume_dict returns dict keyed by grain ID."""
    lgi = np.array([[1, 1], [1, 1]], dtype=int)
    vol_d = volume_dict(lgi, [1])
    assert isinstance(vol_d, dict)
    assert vol_d[1] == 4


def test_volume_dict_multiple_grains():
    """volume_dict for specified grain list."""
    lgi = np.array([
        [1, 1, 2],
        [1, 2, 2]
    ], dtype=int)
    vol_d = volume_dict(lgi, [1, 2])
    assert vol_d[1] == 3
    assert vol_d[2] == 3


def test_volume_dict_subset():
    """volume_dict returns only requested grains."""
    lgi = np.array([
        [1, 1, 2, 3],
        [1, 2, 2, 3]
    ], dtype=int)
    vol_d = volume_dict(lgi, [1, 3])
    assert 1 in vol_d
    assert 3 in vol_d
    assert 2 not in vol_d


def test_volumes_with_bincount_zero_label():
    """Background (0) can be included in counts."""
    lgi = np.array([[0, 0], [1, 1]], dtype=int)
    vols = volumes_with_bincount(lgi)
    assert vols[0] == 2
    assert vols[1] == 2
