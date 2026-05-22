import numpy as np
import pytest
from upxo.gsdataops.grid_ops import section_from_3d, resample_grid_2d, rescale_grid_2d


def test_section_from_3d_axis0():
    db = np.arange(64).reshape(4, 4, 4)
    section = section_from_3d(db, axis=0, location=0)
    assert section.shape == (4, 4)
    np.testing.assert_array_equal(section, db[0, :, :])


def test_section_from_3d_axis1():
    db = np.arange(64).reshape(4, 4, 4)
    section = section_from_3d(db, axis=1, location=2)
    np.testing.assert_array_equal(section, db[:, 2, :])


def test_resample_grid_2d_uigrid_none():
    data = np.ones((10, 10), dtype=np.int32)
    resampled, x_new, y_new, _, _ = resample_grid_2d(data, uigrid=None, sf=0.5)
    assert resampled.shape[0] <= 6 and resampled.shape[1] <= 6


def test_rescale_grid_2d_doubles_size():
    data = np.ones((5, 5), dtype=np.int32)
    scaled = rescale_grid_2d(data, scale_factor=2.0)
    assert scaled.shape == (10, 10)
