"""Summary numbers of a labelled grain structure."""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import geom_metrics_3d as gm
from upxo.pxtal.fm_steel_3d.grain_structure_summary import grain_structure_summary, voxel_spacing


# ── voxel spacing ─────────────────────────────────────────────────────────────

def test_cubic_voxels():
    assert voxel_spacing((100., 100., 100.), (50, 50, 50)) == (2.0, 2.0, 2.0, 0.0)


def test_deviation_from_cubic_voxels():
    vx, vy, vz, deviation = voxel_spacing((100., 80., 60.), (50, 50, 50))
    assert (vx, vy, vz) == (2.0, 1.6, 1.2)
    assert deviation == pytest.approx(max(0.4 / 2.0, 0.8 / 2.0, 0.4 / 1.6))


def test_deviation_uses_all_three_pairs():
    # y is twice x and z: |vx - vy| / vx = 1 and |vy - vz| / vy = 0.5
    assert voxel_spacing((10., 20., 10.), (10, 10, 10))[3] == pytest.approx(1.0)
    # z is twice x and y: |vx - vz| / vx = 1
    assert voxel_spacing((10., 10., 20.), (10, 10, 10))[3] == pytest.approx(1.0)
    # y and z are half of x: |vx - vy| / vx = 0.5, |vy - vz| / vy = 0
    assert voxel_spacing((10., 5., 5.), (10, 10, 10))[3] == pytest.approx(0.5)


def test_an_axis_without_voxels_has_unit_spacing():
    assert voxel_spacing((10., 10., 10.), (0, 5, 5))[0] == 1.0


def test_no_deviation_is_reported_for_a_zero_first_spacing():
    assert voxel_spacing((0., 1., 1.), (4, 4, 4))[3] == 0.0


# ── grain statistics ──────────────────────────────────────────────────────────

def _two_cubes():
    lgi = np.zeros((10, 10, 10), dtype=np.int32)
    lgi[0:2, 0:2, 0:2] = 1            # 8 voxels
    lgi[4:7, 4:7, 4:7] = 2            # 27 voxels
    return lgi


def test_grain_size_statistics():
    vs = 0.5
    s = grain_structure_summary(_two_cubes(), vs)
    assert s['n_grains'] == 2
    assert (s['mean_vox'], s['min_vox'], s['max_vox'], s['std_vox']) == (17.5, 8.0, 27.0, 9.5)
    assert s['mean_volume'] == pytest.approx(17.5 * vs ** 3)
    assert s['min_volume'] == pytest.approx(8 * vs ** 3)
    assert s['max_volume'] == pytest.approx(27 * vs ** 3)
    assert s['std_volume'] == pytest.approx(9.5 * vs ** 3)


def test_mean_morphology_values():
    vs = 0.5
    s = grain_structure_summary(_two_cubes(), vs)
    diameters = [(6 * v * vs ** 3 / np.pi) ** (1 / 3) for v in (8, 27)]
    assert s['mean_eqdia'] == pytest.approx(np.mean(diameters))
    assert s['mean_sphericity'] == pytest.approx((np.pi / 6) ** (1 / 3))      # both are cubes
    assert s['mean_nnb'] == 0.0


def test_neighbour_mean():
    lgi = np.ones((4, 3, 3), dtype=np.int32)
    lgi[2:] = 2
    assert grain_structure_summary(lgi, 1.0)['mean_nnb'] == 1.0


def test_precomputed_metrics_do_not_change_the_result():
    lgi = _two_cubes()
    assert grain_structure_summary(lgi, 1.7) == grain_structure_summary(lgi, 1.7, metrics=gm.all_metrics(lgi))


def test_a_grid_without_grains_is_all_zero():
    summary = grain_structure_summary(np.zeros((4, 4, 4), dtype=np.int32), 2.0)
    assert summary['n_grains'] == 0
    assert all(v == 0.0 for k, v in summary.items() if k != 'n_grains')


def test_a_grid_without_void_counts_every_label():
    lgi = np.ones((3, 3, 3), dtype=np.int32)
    lgi[1:] = 2
    assert grain_structure_summary(lgi, 1.0)['n_grains'] == 2
