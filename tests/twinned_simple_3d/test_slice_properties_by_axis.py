import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.feature_props_3d import compute_slice_properties_by_axis


def _eight_cubes():
    lgi = np.zeros((8, 8, 8), dtype=np.int32)
    label = 1
    for z in (0, 4):
        for y in (0, 4):
            for x in (0, 4):
                lgi[z:z + 4, y:y + 4, x:x + 4] = label
                label += 1
    return lgi


def test_each_axis_and_slice_is_reported_separately():
    out = compute_slice_properties_by_axis(
        _eight_cubes(), ["area"], voxel_size=2.0,
        axes=("x", "y", "z"), n_slices_per_axis={"x": 3, "y": 2, "z": 1})
    assert set(out) == {"x", "y", "z"}
    assert [len(out[a]["area"]) for a in "xyz"] == [3, 2, 1]
    for axis in "xyz":
        for values in out[axis]["area"]:
            assert values.tolist() == [64.0] * 4      # 4 squares of 4x4 voxels at 2 um


def test_axes_not_requested_are_absent_and_scalar_slice_count_applies_everywhere():
    out = compute_slice_properties_by_axis(_eight_cubes(), ["area", "eqdia"], axes=("z",),
                                           n_slices_per_axis=4)
    assert list(out) == ["z"] and len(out["z"]["area"]) == 4
    assert np.allclose(out["z"]["eqdia"][0], np.sqrt(4 * 16 / np.pi))


def test_a_grain_cut_into_two_pieces_counts_twice():
    lgi = np.zeros((1, 3, 7), dtype=np.int32)
    lgi[0, :, 0] = 1
    lgi[0, :, 6] = 1                                    # same label, disjoint in the plane
    out = compute_slice_properties_by_axis(lgi, ["area"], axes=("z",), n_slices_per_axis=1)
    assert sorted(out["z"]["area"][0].tolist()) == [3.0, 3.0]


def test_unknown_axis_names_are_ignored():
    out = compute_slice_properties_by_axis(_eight_cubes(), ["area"], axes=("x", "w"))
    assert list(out) == ["x"]


def test_keep_ids_measures_only_those_grains():
    out = compute_slice_properties_by_axis(
        _eight_cubes(), ["area"], axes=("z",), n_slices_per_axis=1, keep_ids=[1, 2])
    assert out["z"]["area"][0].tolist() == [16.0, 16.0]       # slice z=0 holds cubes 1-4; two kept


def test_ebsd_merged_values_all_and_hosts_only():
    import types
    from upxo.pxtal.twinned_simple_3d.feature_props_3d import ebsd_merged_property_values
    original = np.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    merged = np.array([[1, 1, 1, 1], [2, 2, 3, 3]])           # grain 2 merged into 1
    rg = types.SimpleNamespace(
        lfi_ebsd=original, lfi_ebsd_merged=merged,
        prop_ebsd_merged={1: {'area': 4.0}, 2: {'area': 2.0}, 3: {'area': 2.0}},
        merge_info={'hosting_original_gids': {1}})
    assert sorted(ebsd_merged_property_values(rg, 'area').tolist()) == [2.0, 2.0, 4.0]
    assert ebsd_merged_property_values(rg, 'area', hosts_only=True).tolist() == [4.0]
    rg.merge_info = {'hosting_original_gids': set()}
    with pytest.raises(RuntimeError):
        ebsd_merged_property_values(rg, 'area', hosts_only=True)
    with pytest.raises(RuntimeError):
        ebsd_merged_property_values(types.SimpleNamespace(), 'area')
