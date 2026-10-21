"""compute_host_ced_by_axis / ced_summary_stats -- the pre-twin-generation
synthetic-side counterpart of the Summary Report's category CED machinery,
feeding the reference values for Twin Generation's Min/Max Lamella Thickness.
"""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.feature_props_3d import (
    compute_host_ced_by_axis, ced_summary_stats)


def _fixture():
    """(3, 4, 4): grain 1 (area 8) and grain 2 (area 8) repeated identically
    in every z-layer, so slicing along 'z' gives the same CED every slice."""
    lgi = np.zeros((3, 4, 4), dtype=np.int32)
    lgi[:, 0:2, :] = 1
    lgi[:, 2:4, :] = 2
    return lgi


def test_pools_matching_host_grains_across_every_sampled_slice():
    lgi = _fixture()
    ced = compute_host_ced_by_axis(lgi, {1}, 'z', n_slices_per_axis=3, voxel_size=1.0)
    expected = 2.0 * np.sqrt(8.0 / np.pi)
    assert ced.shape == (3,)
    assert ced == pytest.approx(expected, rel=1e-6)


def test_excludes_non_host_grains_even_when_same_size():
    lgi = _fixture()
    ced = compute_host_ced_by_axis(lgi, {1}, 'z', n_slices_per_axis=3, voxel_size=1.0)
    # grain 2 has the identical area/shape -- must not sneak in (would be 6, not 3)
    assert len(ced) == 3


def test_scales_linearly_with_voxel_size():
    lgi = _fixture()
    ced_1 = compute_host_ced_by_axis(lgi, {1}, 'z', n_slices_per_axis=1, voxel_size=1.0)
    ced_2 = compute_host_ced_by_axis(lgi, {1}, 'z', n_slices_per_axis=1, voxel_size=2.0)
    assert ced_2[0] == pytest.approx(ced_1[0] * 2.0)


def test_empty_when_the_host_id_is_not_present_in_any_slice():
    lgi = _fixture()
    ced = compute_host_ced_by_axis(lgi, {99}, 'z', n_slices_per_axis=3, voxel_size=1.0)
    assert ced.shape == (0,)


def test_other_axis_letters_do_not_error():
    lgi = _fixture()
    for axis in ('x', 'y'):
        ced = compute_host_ced_by_axis(lgi, {1}, axis, n_slices_per_axis=2, voxel_size=1.0)
        assert isinstance(ced, np.ndarray)


def test_ced_summary_stats_basic():
    stats = ced_summary_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert stats['mean'] == pytest.approx(2.5)
    assert stats['min'] == pytest.approx(1.0)
    assert stats['max'] == pytest.approx(4.0)
    assert stats['q1'] == pytest.approx(1.75)
    assert stats['n'] == 4


def test_ced_summary_stats_empty():
    stats = ced_summary_stats(np.array([]))
    assert stats == {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'q1': 0.0, 'n': 0}
