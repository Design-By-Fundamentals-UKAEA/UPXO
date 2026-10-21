"""4-category (parent/host/twin/complete) grain-size extraction on the
synthetic side -- the counterpart to ebsdviz.compute_four_category_grain_sets,
feeding the Summary Report's corrected (2D-slice, not pre-twin-3D) CED/IGS
comparison.
"""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.feature_props_3d import (
    sgs_gid_to_category, compute_sgs_category_ced_by_slice, compute_sgs_category_igs,
    SGS_GRAIN_SIZE_CATEGORIES, compute_ebsd_category_ced, compute_ebsd_category_igs,
    compute_category_grain_size_stats)


def _structure():
    """shape (nz, ny, nx) = (1, 10, 10): grain 1 (non_host, left half, 5x10),
    grain 2 (host, top-right 5x5), grain 3 (primary_twin, bottom-right 5x5)."""
    lgi = np.zeros((1, 10, 10), dtype=np.int32)
    lgi[0, :, :5] = 1
    lgi[0, :5, 5:] = 2
    lgi[0, 5:, 5:] = 3
    twin_role_clean = {1: 'non_host', 2: 'host', 3: 'primary_twin'}
    return lgi, twin_role_clean


def test_sgs_gid_to_category_mapping():
    _, roles = _structure()
    assert sgs_gid_to_category(1, roles) == 'parent'
    assert sgs_gid_to_category(2, roles) == 'host'
    assert sgs_gid_to_category(3, roles) == 'twin'
    assert sgs_gid_to_category(999, roles) == 'parent'  # unknown gid defaults like non_host


def test_sgs_gid_to_category_secondary_twin_is_also_twin():
    roles = {4: 'secondary_twin'}
    assert sgs_gid_to_category(4, roles) == 'twin'


def test_categories_constant_matches_ebsd_side_names():
    assert set(SGS_GRAIN_SIZE_CATEGORIES) == {'parent', 'host', 'twin', 'complete'}


def test_ced_by_slice_filters_to_requested_category():
    lgi, roles = _structure()
    ced = compute_sgs_category_ced_by_slice(lgi, roles, 'parent', 'z', 1, voxel_size=1.0)
    assert len(ced) == 1
    assert ced[0].size == 1
    assert ced[0][0] == pytest.approx(np.sqrt(4 * 50 / np.pi))  # 5x10 rectangle


def test_ced_by_slice_complete_includes_every_grain():
    lgi, roles = _structure()
    ced = compute_sgs_category_ced_by_slice(lgi, roles, 'complete', 'z', 1, voxel_size=1.0)
    assert ced[0].size == 3


def test_ced_by_slice_voxel_size_scales_physical_units():
    lgi, roles = _structure()
    ced1 = compute_sgs_category_ced_by_slice(lgi, roles, 'host', 'z', 1, voxel_size=1.0)
    ced2 = compute_sgs_category_ced_by_slice(lgi, roles, 'host', 'z', 1, voxel_size=2.0)
    assert ced2[0][0] == pytest.approx(ced1[0][0] * 2.0)


def test_ced_by_slice_empty_category_gives_empty_array():
    lgi, roles = _structure()
    # category 'host' with a far larger n_slices still only ever sees grain 2
    ced = compute_sgs_category_ced_by_slice(lgi, roles, 'host', 'z', 1, voxel_size=1.0)
    assert ced[0].size == 1
    # a category that matches nothing in this tiny structure
    lgi2 = np.zeros((1, 4, 4), dtype=np.int32)
    lgi2[0, :, :] = 1
    roles2 = {1: 'non_host'}
    ced_twin = compute_sgs_category_ced_by_slice(lgi2, roles2, 'twin', 'z', 1, voxel_size=1.0)
    assert ced_twin[0].size == 0


def test_igs_pools_rows_and_columns_per_slice():
    lgi, roles = _structure()
    ced = compute_sgs_category_ced_by_slice(lgi, roles, 'complete', 'z', 1, voxel_size=1.0)
    mean_ced = float(np.mean(ced[0]))
    igs = compute_sgs_category_igs(lgi, roles, 'complete', ['z'], 1,
                                   voxel_size=1.0, mean_ced_um=mean_ced)
    assert igs['n'] > 0
    assert igs['mean'] > 0
    assert igs['raw_segments'].size == igs['n']


def test_igs_category_mask_excludes_other_categories():
    lgi, roles = _structure()
    igs_parent = compute_sgs_category_igs(lgi, roles, 'parent', ['z'], 1,
                                          voxel_size=1.0, mean_ced_um=5.0)
    # 'parent' is grain 1 only, a 10 (rows) x 5 (cols) rectangle: a
    # row-direction intercept can be at most 5 (it never reaches the
    # host/twin half), a column-direction one at most 10 (the full
    # height) -- either way, never more than the full 10x10 domain.
    assert igs_parent['n'] > 0
    assert igs_parent['raw_segments'].max() <= 10.0 + 1e-9
    assert igs_parent['raw_segments'].min() > 0


def test_igs_zero_mean_ced_falls_back_to_one_line():
    lgi, roles = _structure()
    igs = compute_sgs_category_igs(lgi, roles, 'complete', ['z'], 1,
                                   voxel_size=1.0, mean_ced_um=0.0)
    assert igs['n'] > 0  # does not raise ZeroDivisionError, still samples >=1 line


def test_igs_empty_category_gives_zero_stats():
    lgi2 = np.zeros((1, 4, 4), dtype=np.int32)
    lgi2[0, :, :] = 1
    roles2 = {1: 'non_host'}
    igs = compute_sgs_category_igs(lgi2, roles2, 'twin', ['z'], 1,
                                   voxel_size=1.0, mean_ced_um=4.0)
    assert igs['n'] == 0
    assert igs['mean'] == 0.0
    assert igs['raw_segments'].size == 0


# ── EBSD-side counterparts (no slicing -- the map is already 2D) ────────

def _ebsd_fixture():
    lfi_ebsd = np.array([[1, 1, 1, 2, 2],
                         [1, 1, 1, 2, 2],
                         [3, 3, 3, 3, 3]], dtype=np.int32)
    prop_ebsd = {1: {'eq_diameter': 5.0}, 2: {'eq_diameter': 3.0}, 3: {'eq_diameter': 4.0}}
    return lfi_ebsd, prop_ebsd


def test_ebsd_category_ced_selects_requested_gids_only():
    _, prop_ebsd = _ebsd_fixture()
    ced = compute_ebsd_category_ced(prop_ebsd, [1, 3])
    assert sorted(ced.tolist()) == [4.0, 5.0]


def test_ebsd_category_ced_skips_gids_missing_from_prop():
    _, prop_ebsd = _ebsd_fixture()
    ced = compute_ebsd_category_ced(prop_ebsd, [1, 999])
    assert ced.tolist() == [5.0]


def test_ebsd_category_ced_empty_set_gives_empty_array():
    _, prop_ebsd = _ebsd_fixture()
    ced = compute_ebsd_category_ced(prop_ebsd, [])
    assert ced.size == 0


def test_ebsd_category_igs_masks_to_requested_gids():
    lfi_ebsd, _ = _ebsd_fixture()
    igs = compute_ebsd_category_igs(lfi_ebsd, [1], voxel_size=1.0, mean_ced_um=5.0)
    assert igs['n'] > 0
    assert igs['raw_segments'].max() <= 3.0 + 1e-9  # grain 1 is 2 rows x 3 cols


def test_ebsd_category_igs_empty_gids_gives_zero_stats():
    lfi_ebsd, _ = _ebsd_fixture()
    igs = compute_ebsd_category_igs(lfi_ebsd, [], voxel_size=1.0, mean_ced_um=4.0)
    assert igs['n'] == 0
    assert igs['mean'] == 0.0


# ── compute_category_grain_size_stats (full orchestration) ──────────────

def _orchestration_fixture():
    lfi_ebsd = np.array([[1, 1, 1, 2, 2],
                         [1, 1, 1, 2, 2],
                         [3, 3, 3, 3, 3]], dtype=np.int32)
    prop_ebsd = {1: {'eq_diameter': 5.0}, 2: {'eq_diameter': 3.0}, 3: {'eq_diameter': 4.0}}
    # grain 3 never in a pair -> twin-free ('parent'); 1 hosts 2 -> 'host'/'twin'.
    parent_info = {'S3': {
        'all_parents': np.array([1]), 'all_twins': np.array([2]),
        'pure_parents': np.array([1]), 'pure_twins': np.array([2]),
        'intermediates': np.array([]),
    }}
    lgi_clean = np.zeros((1, 10, 10), dtype=np.int32)
    lgi_clean[0, :, :5] = 1
    lgi_clean[0, :5, 5:] = 2
    lgi_clean[0, 5:, 5:] = 3
    twin_role_clean = {1: 'non_host', 2: 'host', 3: 'primary_twin'}
    return lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean


def test_orchestration_returns_all_four_categories_with_both_metrics():
    lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean = _orchestration_fixture()
    out = compute_category_grain_size_stats(
        lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean,
        ebsd_voxel_size=1.0, sgs_voxel_size=1.0, axes=('z',), n_slices_per_axis=1)
    assert set(out) == set(SGS_GRAIN_SIZE_CATEGORIES)
    for category in SGS_GRAIN_SIZE_CATEGORIES:
        for key in ('ebsd_ced', 'sgs_ced', 'ebsd_igs', 'sgs_igs'):
            assert set(out[category][key]) == {'mean', 'median', 'std', 'n'}


def test_orchestration_category_populations_match_expected_grain_counts():
    lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean = _orchestration_fixture()
    out = compute_category_grain_size_stats(
        lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean,
        ebsd_voxel_size=1.0, sgs_voxel_size=1.0, axes=('z',), n_slices_per_axis=1)
    assert out['parent']['ebsd_ced']['n'] == 1  # grain 3
    assert out['host']['ebsd_ced']['n'] == 1    # grain 1
    assert out['twin']['ebsd_ced']['n'] == 1    # grain 2
    assert out['complete']['ebsd_ced']['n'] == 3
    assert out['complete']['sgs_ced']['n'] == 3


def test_orchestration_per_axis_slice_dict_is_respected():
    lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean = _orchestration_fixture()
    out = compute_category_grain_size_stats(
        lfi_ebsd, prop_ebsd, parent_info, lgi_clean, twin_role_clean,
        ebsd_voxel_size=1.0, sgs_voxel_size=1.0, axes=('z',),
        n_slices_per_axis={'z': 1})
    assert out['complete']['sgs_ced']['n'] == 3  # same as scalar n_slices_per_axis=1
