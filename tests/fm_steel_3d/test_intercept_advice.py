"""Intercept grain sizes along the directions blocks are sliced in."""
from types import SimpleNamespace

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import intercept_advice as ia
from upxo.pxtal.fm_steel_3d.block_generator_3d import BlockGenerator3D


def _slabs(thickness=3, n=4, cross=8):
    lgi = np.zeros((thickness * n, cross, cross), dtype=np.int32)
    for k in range(n):
        lgi[k * thickness:(k + 1) * thickness] = k + 1
    return lgi


# ── slicing directions of a PAG ──────────────────────────────────────────────

def _pag(grain_ids, present=None):
    present = grain_ids if present is None else present
    return SimpleNamespace(
        clusters_dict={1: list(grain_ids)}, pag_orientations={1: (10.0, 20.0, 30.0)},
        grain_locs={g: np.array([[0, 0, 0]]) for g in present})


def test_grains_cycle_through_the_candidate_planes():
    pag = _pag([11, 12, 13, 14, 15, 16, 17, 18, 19])
    planes = BlockGenerator3D().get_slicing_planes_for_packet(pag_id=1, pag_orientations=pag.pag_orientations)
    normals = ia.prospective_slicing_normals(pag)
    assert list(normals) == [11, 12, 13, 14, 15, 16, 17, 18, 19]
    for index, grain in enumerate(normals):
        np.testing.assert_allclose(normals[grain], planes[index % len(planes)].unit_normal)
        assert np.linalg.norm(normals[grain]) == pytest.approx(1.0)


def test_grains_without_voxels_are_skipped_but_still_count_in_the_cycle():
    pag = _pag([11, 12, 13], present=[11, 13])
    planes = BlockGenerator3D().get_slicing_planes_for_packet(pag_id=1, pag_orientations=pag.pag_orientations)
    normals = ia.prospective_slicing_normals(pag)
    assert list(normals) == [11, 13]
    np.testing.assert_allclose(normals[13], planes[2 % len(planes)].unit_normal)


def test_each_pag_cycles_on_its_own():
    pag = SimpleNamespace(clusters_dict={1: [1, 2], 2: [3, 4]},
                          pag_orientations={1: (0., 0., 0.), 2: (40., 50., 60.)},
                          grain_locs={g: np.array([[0, 0, 0]]) for g in (1, 2, 3, 4)})
    normals = ia.prospective_slicing_normals(pag)
    plane_1 = BlockGenerator3D().get_slicing_planes_for_packet(pag_id=1, pag_orientations=pag.pag_orientations)
    plane_2 = BlockGenerator3D().get_slicing_planes_for_packet(pag_id=2, pag_orientations=pag.pag_orientations)
    np.testing.assert_allclose(normals[1], plane_1[0].unit_normal)
    np.testing.assert_allclose(normals[3], plane_2[0].unit_normal)


# ── intercepts along directions ──────────────────────────────────────────────

def test_intercepts_across_slabs():
    lgi = _slabs(thickness=3, n=4)
    out = ia.intercept_sizes_by_direction(lgi, 0.5, {1: [1.0, 0.0, 0.0]})
    assert out['n_directions'] == 1 and out['total_lines'] > 0
    assert out['combined'].size > 0
    assert out['direction_means'] == [pytest.approx(1.5)]                 # 3 voxels of 0.5
    assert out['combined'].mean() == pytest.approx(1.5)


def test_ids_sharing_a_direction_are_measured_once():
    lgi = _slabs()
    same = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0], 2: [-1.0, 0.0, 0.0], 3: [2.0, 0.0, 0.0]})
    one = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]})
    assert same['n_directions'] == 1
    np.testing.assert_array_equal(same['combined'], one['combined'])
    assert same['total_lines'] == one['total_lines']


def test_the_line_budget_is_shared_between_directions():
    lgi = _slabs(thickness=4, n=4, cross=12)
    one = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]}, line_budget=300)
    three = ia.intercept_sizes_by_direction(
        lgi, 1.0, {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0]}, line_budget=300)
    assert three['n_directions'] == 3 and len(three['direction_means']) == 3
    assert three['total_lines'] <= 2 * one['total_lines']                  # not three times the lines
    small = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]}, line_budget=1)
    assert small['total_lines'] > 0                                       # at least 20 lines are tried


def test_a_direction_along_the_slabs_measures_their_extent_not_their_thickness():
    out = ia.intercept_sizes_by_direction(_slabs(thickness=3, n=4, cross=8), 1.0, {1: [0.0, 1.0, 0.0]})
    assert out['direction_means'] == [pytest.approx(8.0)]
    assert set(np.round(out['combined'], 6).tolist()) == {8.0}


def test_no_directions():
    out = ia.intercept_sizes_by_direction(_slabs(), 1.0, {})
    assert out['combined'].size == 0 and out['direction_means'] == []
    assert out['total_lines'] == 0 and out['n_directions'] == 0


def test_only_the_valid_phases_are_measured():
    lgi = _slabs()
    phase = np.full(lgi.shape, 2, dtype=np.int32)
    plain = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]})
    allowed = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]}, phase_array=phase, valid_phase_ids=(2,))
    excluded = ia.intercept_sizes_by_direction(lgi, 1.0, {1: [1.0, 0.0, 0.0]}, phase_array=phase, valid_phase_ids=(1,))
    np.testing.assert_array_equal(allowed['combined'], plain['combined'])
    assert excluded['combined'].size == 0 and excluded['direction_means'] == []


# ── thickness range ───────────────────────────────────────────────────────────

def test_thickness_range_is_the_interquartile_range_over_all_axes():
    stats = {'x': {'raw_segments': np.array([1.0, 2.0, 3.0])},
             'y': {'raw_segments': np.array([4.0, 5.0])},
             'z': {'raw_segments': np.array([6.0, 7.0, 8.0])}}
    q1, q3 = ia.thickness_range_from_intercepts(stats)
    assert (q1, q3) == pytest.approx(tuple(np.percentile(np.arange(1.0, 9.0), [25, 75])))
    assert isinstance(q1, float) and isinstance(q3, float)


def test_thickness_range_without_measurements():
    empty = {a: {'raw_segments': np.empty(0)} for a in 'xyz'}
    assert ia.thickness_range_from_intercepts(empty) is None
