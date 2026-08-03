"""
Unit tests for pag_clustering_3d (Technique A "lever 2": simultaneous
multi-source MIS-seeded PAG clustering).

Run from repo root:
    pytest tests/fm_steel_3d/test_pag_clustering.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.pag_clustering_3d import (
    greedy_maximal_independent_set,
    cluster_grains_simultaneous,
    repair_singleton_pags,
)
from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase


def _grid_graph(nx, ny):
    """Simple 2D grid adjacency (4-connectivity), ids 0..nx*ny-1, for fast,
    easy-to-reason-about synthetic graph tests."""
    neigh = {}
    for x in range(nx):
        for y in range(ny):
            gid = x * ny + y
            nbrs = []
            if x > 0:
                nbrs.append((x - 1) * ny + y)
            if x < nx - 1:
                nbrs.append((x + 1) * ny + y)
            if y > 0:
                nbrs.append(x * ny + (y - 1))
            if y < ny - 1:
                nbrs.append(x * ny + (y + 1))
            neigh[gid] = nbrs
    return neigh


def _voronoi_lfi(n=20, n_seeds=25, seed=0):
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


# ---------------------------------------------------------------------------
# greedy_maximal_independent_set
# ---------------------------------------------------------------------------

def test_mis_no_two_members_adjacent():
    neigh = _grid_graph(10, 10)
    rng = np.random.default_rng(1)
    mis = greedy_maximal_independent_set(neigh, neigh.keys(), rng)
    mis_set = set(mis)
    for g in mis:
        assert not (set(neigh[g]) & mis_set), f"grain {g} is adjacent to another MIS member"


def test_mis_is_maximal():
    neigh = _grid_graph(10, 10)
    rng = np.random.default_rng(2)
    mis = greedy_maximal_independent_set(neigh, neigh.keys(), rng)
    mis_set = set(mis)
    for g in neigh:
        if g not in mis_set:
            assert set(neigh[g]) & mis_set, f"grain {g} is neither in the MIS nor adjacent to it"


def test_mis_reproducible_with_same_seed():
    neigh = _grid_graph(8, 8)
    mis1 = greedy_maximal_independent_set(neigh, neigh.keys(), np.random.default_rng(42))
    mis2 = greedy_maximal_independent_set(neigh, neigh.keys(), np.random.default_rng(42))
    assert mis1 == mis2


def test_mis_over_subset_only():
    neigh = _grid_graph(10, 10)
    subset = [g for g in neigh if g % 3 == 0]
    rng = np.random.default_rng(3)
    mis = greedy_maximal_independent_set(neigh, subset, rng)
    assert set(mis) <= set(subset)


# ---------------------------------------------------------------------------
# cluster_grains_simultaneous
# ---------------------------------------------------------------------------

def test_every_grain_clustered_exactly_once():
    neigh = _grid_graph(15, 15)
    clusters = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=10)
    all_clustered = [g for ids in clusters.values() for g in ids]
    assert sorted(all_clustered) == sorted(neigh.keys())
    assert len(all_clustered) == len(set(all_clustered))  # no duplicates


def test_no_pag_exceeds_max_cap():
    neigh = _grid_graph(15, 15)
    clusters = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4,
                                           max_packets_per_pag=4, random_seed=11)
    for pid, ids in clusters.items():
        assert len(ids) <= 4


def test_cap_enforced_even_if_distribution_exceeds_it():
    neigh = _grid_graph(15, 15)
    dist = {"sizes": [10], "probs": [1.0]}
    clusters = cluster_grains_simultaneous(neigh, neigh.keys(), dist,
                                           max_packets_per_pag=4, random_seed=12)
    for pid, ids in clusters.items():
        assert len(ids) <= 4


def test_reproducible_with_same_seed():
    neigh = _grid_graph(12, 12)
    c1 = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=99)
    c2 = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=99)
    assert c1 == c2


def test_single_isolated_grain_becomes_its_own_pag():
    neigh = {1: [], 2: [3], 3: [2]}
    clusters = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=1)
    all_clustered = [g for ids in clusters.values() for g in ids]
    assert sorted(all_clustered) == [1, 2, 3]
    grain1_pag = [ids for ids in clusters.values() if 1 in ids][0]
    assert grain1_pag == [1]


def test_on_real_voronoi_grain_adjacency():
    lfi = _voronoi_lfi()
    fm = FMSteel3DBase.from_lfi(
        lfi, physical_dimensions=(20.0, 20.0, 20.0), voxel_size=1.0,
        connectivity=6, min_grain_nvoxels=8, random_seed=42, verbosity=0)
    clusters = cluster_grains_simultaneous(fm.neigh_gid, fm.grain_locs.keys(), DIST_3_4,
                                           max_packets_per_pag=4, random_seed=5)
    all_clustered = [g for ids in clusters.values() for g in ids]
    assert sorted(all_clustered) == sorted(fm.grain_locs.keys())
    for ids in clusters.values():
        assert 1 <= len(ids) <= 4


# ---------------------------------------------------------------------------
# repair_singleton_pags ("lever 3")
# ---------------------------------------------------------------------------

def _n_singletons(clusters):
    return sum(1 for ids in clusters.values() if len(ids) == 1)


def _all_grains(clusters):
    return sorted(g for ids in clusters.values() for g in ids)


def test_repair_preserves_every_grain_exactly_once():
    neigh = _grid_graph(15, 15)
    before = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=20)
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=21)
    assert _all_grains(after) == sorted(neigh.keys())
    all_g = [g for ids in after.values() for g in ids]
    assert len(all_g) == len(set(all_g))


def test_repair_never_exceeds_cap():
    neigh = _grid_graph(15, 15)
    before = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=22)
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=23)
    for ids in after.values():
        assert len(ids) <= 4


def test_repair_never_increases_singleton_count():
    neigh = _grid_graph(15, 15)
    before = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=24)
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=25)
    assert _n_singletons(after) <= _n_singletons(before)


def test_two_adjacent_singletons_get_merged():
    # Grains 1 and 2 are each their own singleton PAG and are adjacent to
    # each other; PAG 102 (grains 3,4) is already at the cap (2), so
    # absorption into it isn't an option -- the only way to rescue 1 and 2
    # is to pair them together into a fresh 2-grain PAG.
    neigh = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}
    before = {100: [1], 101: [2], 102: [3, 4]}
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=2, random_seed=1)
    assert _all_grains(after) == [1, 2, 3, 4]
    pag_of_1 = [ids for ids in after.values() if 1 in ids][0]
    assert sorted(pag_of_1) == [1, 2], "adjacent singletons 1 and 2 should have been merged together"


def test_singleton_absorbed_into_spare_capacity_neighbor():
    # Grain 1 is a lone singleton, adjacent only to grain 2, which belongs
    # to an existing 2-grain PAG (spare capacity up to cap=4).
    neigh = {1: [2], 2: [1, 3], 3: [2]}
    before = {10: [1], 11: [2, 3]}
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=2)
    assert _all_grains(after) == [1, 2, 3]
    pag_of_1 = [ids for ids in after.values() if 1 in ids][0]
    assert sorted(pag_of_1) == [1, 2, 3]


def test_singleton_stays_singleton_if_only_neighbor_pag_is_full():
    # Grain 1 is a lone singleton, its only neighbour (2) belongs to a PAG
    # that is already at the cap -- nothing to merge into.
    neigh = {1: [2], 2: [1, 3, 4, 5], 3: [2], 4: [2], 5: [2]}
    before = {10: [1], 11: [2, 3, 4, 5]}  # PAG 11 already has 4 members == cap
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=3)
    assert _all_grains(after) == [1, 2, 3, 4, 5]
    pag_of_1 = [ids for ids in after.values() if 1 in ids][0]
    assert pag_of_1 == [1], "grain 1 has nowhere to go and should remain a singleton"


def test_truly_isolated_grain_stays_singleton():
    neigh = {1: [], 2: [3], 3: [2]}
    before = {10: [1], 11: [2, 3]}
    after = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=4)
    pag_of_1 = [ids for ids in after.values() if 1 in ids][0]
    assert pag_of_1 == [1]


def test_repair_reproducible_with_same_seed():
    neigh = _grid_graph(12, 12)
    before = cluster_grains_simultaneous(neigh, neigh.keys(), DIST_3_4, random_seed=30)
    after1 = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=31)
    after2 = repair_singleton_pags(before, neigh, max_packets_per_pag=4, random_seed=31)
    assert after1 == after2


def test_repair_reduces_singleton_rate_on_real_structure():
    lfi = _voronoi_lfi(n=30, n_seeds=80, seed=9)
    fm = FMSteel3DBase.from_lfi(
        lfi, physical_dimensions=(30.0, 30.0, 30.0), voxel_size=1.0,
        connectivity=6, min_grain_nvoxels=8, random_seed=42, verbosity=0)
    before = fm.generate_pag_clusters(
        pag_size_distribution=DIST_3_4, pag_grain_fraction=1.0,
        use_non_neigh_pag=True, isolated_grain_strategy='auto', random_seed=42,
    ).clusters_dict
    after = repair_singleton_pags(before, fm.neigh_gid, max_packets_per_pag=4, random_seed=6)
    assert _all_grains(after) == sorted(fm.grain_locs.keys())
    assert _n_singletons(after) < _n_singletons(before)
