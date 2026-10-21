"""Voxel masks of grain boundaries, junction lines and junction points."""
import itertools

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.voxel_interface_masks import (
    grain_boundary_voxels, junction_line_voxels, junction_point_voxels, shift_3d)


def _label_at(labels, i, j, k):
    """Label at a position; positions beyond the grid read as 0."""
    if min(i, j, k) < 0 or i >= labels.shape[0] or j >= labels.shape[1] or k >= labels.shape[2]:
        return 0
    return int(labels[i, j, k])


def _inside(labels, i, j, k):
    return 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1] and 0 <= k < labels.shape[2]


def _reference_grain_boundary(labels):
    out = np.zeros(labels.shape, dtype=bool)
    for i, j, k in np.ndindex(labels.shape):
        own = labels[i, j, k]
        if own <= 0:
            continue
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if _inside(labels, ni, nj, nk) and labels[ni, nj, nk] > 0 and labels[ni, nj, nk] != own:
                out[i, j, k] = True
    return out


def _reference_junction_lines(labels):
    """Edge by edge: the edge at the low corner of each voxel, for the three directions."""
    out = np.zeros(labels.shape, dtype=bool)
    for i, j, k in np.ndindex(labels.shape):
        for axis in range(3):
            a, b = [ax for ax in range(3) if ax != axis]
            around = []
            for da, db in itertools.product((0, 1), repeat=2):
                position = [i, j, k]
                position[a] -= da
                position[b] -= db
                around.append(tuple(position))
            if len({_label_at(labels, *p) for p in around}) >= 3:
                for p in around:
                    if _inside(labels, *p):
                        out[p] = True
    return out


def _reference_junction_points(labels):
    out = np.zeros(labels.shape, dtype=bool)
    for i, j, k in np.ndindex(labels.shape):
        around = [(i - da, j - db, k - dc)
                  for da, db, dc in itertools.product((0, 1), repeat=3)]
        if len({_label_at(labels, *p) for p in around}) >= 3:
            for p in around:
                if _inside(labels, *p):
                    out[p] = True
    return out


def _grids():
    rng = np.random.default_rng(7)
    for shape in ((4, 5, 3), (3, 3, 3), (1, 4, 4), (5, 1, 2)):
        yield rng.integers(0, 5, size=shape)                                  # noisy, with void
    blocky = rng.integers(0, 4, size=(3, 3, 3))
    yield np.kron(blocky, np.ones((2, 2, 2), dtype=int))                      # sparse features
    yield np.full((4, 4, 4), 3)                                               # one grain


@pytest.mark.parametrize('labels', list(_grids()))
def test_masks_agree_with_voxel_by_voxel_definitions(labels):
    np.testing.assert_array_equal(grain_boundary_voxels(labels), _reference_grain_boundary(labels))
    np.testing.assert_array_equal(junction_line_voxels(labels), _reference_junction_lines(labels))
    np.testing.assert_array_equal(junction_point_voxels(labels), _reference_junction_points(labels))


def _four_columns():
    labels = np.zeros((6, 6, 6), dtype=int)
    x, y = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    labels[:] = (1 + (x >= 3) + 2 * (y >= 3))[:, :, None]
    return labels


def test_one_grain_has_no_features():
    labels = np.full((4, 4, 4), 2)
    assert not grain_boundary_voxels(labels).any()
    assert not junction_line_voxels(labels).any()
    assert not junction_point_voxels(labels).any()


def test_void_voxels_are_not_boundaries():
    labels = np.array([[[1, 0, 2]]])
    assert not grain_boundary_voxels(labels).any()


def test_grain_boundary_marks_both_sides_of_a_flat_interface():
    labels = np.ones((4, 3, 3), dtype=int)
    labels[2:] = 2
    marked = grain_boundary_voxels(labels)
    assert marked[1:3].all()
    assert not marked[0].any() and not marked[3].any()


def test_line_where_four_grains_meet_is_a_junction_line():
    marked = junction_line_voxels(_four_columns())
    # the four columns around x=3, y=3 along the whole of z
    assert marked[2:4, 2:4, :].all()
    # a voxel in the middle of a column, away from the line
    assert not marked[1, 1, 3]


def test_vertex_where_four_grains_meet_marks_eight_voxels():
    marked = junction_point_voxels(_four_columns())
    assert marked[2:4, 2:4, 2:4].all()
    assert not marked[1, 1, 3]


def test_grid_faces_count_as_a_further_label():
    # two grains meet on the x = 0 face of a one-voxel-thick grid
    labels = np.array([[[1], [2]]])
    assert junction_line_voxels(labels).all()


def test_shift_3d_moves_data_and_pads_with_zero():
    arr = np.arange(1, 9).reshape(2, 2, 2)
    shifted = shift_3d(arr, 1, 0, 0)
    np.testing.assert_array_equal(shifted[0], 0)
    np.testing.assert_array_equal(shifted[1], arr[0])
    np.testing.assert_array_equal(shift_3d(arr, 0, -1, 0)[:, 0], arr[:, 1])
    np.testing.assert_array_equal(shift_3d(arr, 0, 0, 0), arr)
    np.testing.assert_array_equal(shift_3d(arr, 5, 0, 0), np.zeros_like(arr))
