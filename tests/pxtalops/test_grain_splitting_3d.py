"""
Unit tests for grain_splitting_3d.split_grains_geodesic -- general-purpose
geodesic within-grain splitting for 3D labeled grain structures (see module
docstring for rationale; originally built for, and still used by, the
fm_steel_3d PAG/packet pipeline's "Technique B").

Standalone: pure numpy/stdlib, no dependency on any specific grain-structure
class.

Run from repo root:
    pytest tests/pxtalops/test_grain_splitting_3d.py -v
"""

import cc3d
import numpy as np
import pytest

from upxo.pxtalops.grain_splitting_3d import split_grains_geodesic, size_balance_metrics


def _voronoi_lfi(n=20, n_seeds=15, seed=0):
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, n, n_seeds).astype(np.float32)
    cy = rng.integers(0, n, n_seeds).astype(np.float32)
    cz = rng.integers(0, n, n_seeds).astype(np.float32)
    xi = np.arange(n, dtype=np.float32)[:, None, None, None]
    yi = np.arange(n, dtype=np.float32)[None, :, None, None]
    zi = np.arange(n, dtype=np.float32)[None, None, :, None]
    sq = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return (np.argmin(sq, axis=3) + 1).astype(np.int32)


DIST_3_4 = {"sizes": [3, 4], "probs": [0.95, 0.05]}


@pytest.fixture(scope="module")
def lfi():
    return _voronoi_lfi()


def test_nonzero_mask_preserved(lfi):
    sub_lgi, _, _ = split_grains_geodesic(lfi, DIST_3_4, min_voxels_to_split=20, random_seed=1)
    assert sub_lgi.shape == lfi.shape
    np.testing.assert_array_equal(sub_lgi > 0, lfi > 0)


def test_clusters_partition_original_grains_exactly(lfi):
    sub_lgi, sub_to_grain, clusters_dict = split_grains_geodesic(
        lfi, DIST_3_4, min_voxels_to_split=20, random_seed=1)
    for grain_id, sub_ids in clusters_dict.items():
        grain_mask = (lfi == grain_id)
        sub_mask = np.isin(sub_lgi, sub_ids)
        np.testing.assert_array_equal(grain_mask, sub_mask,
            err_msg=f"grain {grain_id}'s sub-regions don't exactly reconstruct its voxel set")


def test_sub_to_grain_consistent_with_clusters_dict(lfi):
    sub_lgi, sub_to_grain, clusters_dict = split_grains_geodesic(
        lfi, DIST_3_4, min_voxels_to_split=20, random_seed=1)
    for grain_id, sub_ids in clusters_dict.items():
        for sid in sub_ids:
            assert sub_to_grain[sid] == grain_id
    all_sub_ids_from_clusters = {sid for ids in clusters_dict.values() for sid in ids}
    assert all_sub_ids_from_clusters == set(sub_to_grain.keys())
    assert all_sub_ids_from_clusters == set(np.unique(sub_lgi)) - {0}


def test_every_sub_region_is_a_single_connected_component(lfi):
    sub_lgi, _, _ = split_grains_geodesic(lfi, DIST_3_4, min_voxels_to_split=20, random_seed=1)
    for sid in np.unique(sub_lgi):
        if sid == 0:
            continue
        mask = (sub_lgi == sid).astype(np.uint8)
        cc = cc3d.connected_components(mask, connectivity=6)
        n_components = int(cc.max())
        assert n_components == 1, f"sub-region {sid} is split across {n_components} disconnected regions"


def test_grains_below_threshold_stay_single_sub_region(lfi):
    # A very high threshold means nothing qualifies for splitting.
    sub_lgi, sub_to_grain, clusters_dict = split_grains_geodesic(
        lfi, DIST_3_4, min_voxels_to_split=10**9, random_seed=1)
    for grain_id, sub_ids in clusters_dict.items():
        assert len(sub_ids) == 1
    # Sub-region-level structure should be a pure relabelling of the grain structure.
    assert len(np.unique(sub_lgi)) - 1 == len(np.unique(lfi)) - 1


def test_no_empty_sub_regions_when_grain_smaller_than_requested_size(lfi):
    # sizes=[100] forces "100 sub-regions" on every eligible grain, but most
    # grains here have far fewer than 100 voxels -- n_subs must be capped to
    # n_vox so no sub-region ends up empty.
    dist = {"sizes": [100], "probs": [1.0]}
    sub_lgi, sub_to_grain, clusters_dict = split_grains_geodesic(
        lfi, dist, min_voxels_to_split=1, random_seed=2)
    counts = np.bincount(sub_lgi.ravel())
    for sid in sub_to_grain:
        assert counts[sid] >= 1, f"sub-region {sid} has zero voxels"


def test_reproducible_with_same_seed(lfi):
    out1 = split_grains_geodesic(lfi, DIST_3_4, min_voxels_to_split=20, random_seed=42)
    out2 = split_grains_geodesic(lfi, DIST_3_4, min_voxels_to_split=20, random_seed=42)
    np.testing.assert_array_equal(out1[0], out2[0])
    assert out1[1] == out2[1]
    assert out1[2] == out2[2]


def test_sub_region_count_matches_distribution_for_large_grains():
    # One single huge cubic grain filling the whole domain -- deterministic
    # sub-region count (single-value distribution) should be honoured exactly.
    lfi = np.ones((15, 15, 15), dtype=np.int32)
    dist = {"sizes": [4], "probs": [1.0]}
    _, _, clusters_dict = split_grains_geodesic(lfi, dist, min_voxels_to_split=10, random_seed=3)
    assert clusters_dict[1] and len(clusters_dict[1]) == 4


def test_single_grain_domain_produces_exact_voxel_count_partition():
    lfi = np.ones((10, 10, 10), dtype=np.int32)
    dist = {"sizes": [3], "probs": [1.0]}
    sub_lgi, _, clusters_dict = split_grains_geodesic(lfi, dist, min_voxels_to_split=5, random_seed=4)
    assert len(clusters_dict[1]) == 3
    assert int((sub_lgi > 0).sum()) == 1000


# ---------------------------------------------------------------------------
# size_balance_metrics
# ---------------------------------------------------------------------------

def test_perfectly_equal_sizes_score_zero_ideal():
    m = size_balance_metrics([50, 50, 50, 50])
    assert m['cv'] == pytest.approx(0.0)
    assert m['min_max_ratio'] == pytest.approx(1.0)
    assert m['gini'] == pytest.approx(0.0)


def test_single_size_is_trivially_perfectly_equal():
    m = size_balance_metrics([123])
    assert m['cv'] == 0.0
    assert m['min_max_ratio'] == 1.0
    assert m['gini'] == 0.0


def test_maximally_unequal_sizes_deviate_from_ideal():
    m = size_balance_metrics([1, 1, 1, 97])
    assert m['cv'] > 0.5
    assert m['min_max_ratio'] == pytest.approx(1.0 / 97.0)
    assert m['gini'] > 0.5


def test_more_unequal_split_gives_worse_scores_than_less_unequal():
    balanced = size_balance_metrics([25, 25, 25, 25])
    mild = size_balance_metrics([20, 25, 25, 30])
    severe = size_balance_metrics([5, 5, 5, 85])
    assert balanced['cv'] < mild['cv'] < severe['cv']
    assert balanced['min_max_ratio'] > mild['min_max_ratio'] > severe['min_max_ratio']
    assert balanced['gini'] < mild['gini'] < severe['gini']


def test_gini_bounded_in_zero_one():
    m = size_balance_metrics([1, 2, 4, 8, 16, 32])
    assert 0.0 <= m['gini'] <= 1.0


def test_metrics_independent_of_total_scale():
    # Scaling every size by a constant shouldn't change any of these
    # (relative) indicators.
    m1 = size_balance_metrics([10, 20, 30])
    m2 = size_balance_metrics([100, 200, 300])
    assert m1['cv'] == pytest.approx(m2['cv'])
    assert m1['min_max_ratio'] == pytest.approx(m2['min_max_ratio'])
    assert m1['gini'] == pytest.approx(m2['gini'])
