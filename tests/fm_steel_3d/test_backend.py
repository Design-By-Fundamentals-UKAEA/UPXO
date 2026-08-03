"""
Backend regression tests for the FM Steel 3D pipeline.

Covers:
  FMSteel3DBase          — LFI ingestion, grain cleanup
  FMSteel3DWithPAGs      — PAG clustering
  FMSteel3DWithBlocks    — block generation
  FMSteel3DWithOrientations — KS orientation assignment, misorientation stats
  FMSteel3DWithSubBlocks — sub-block generation

All fixtures are session-scoped so the pipeline runs once per test session.
Domain: 20×20×20 voxels, 20 grain seeds — small enough for fast CI.

Run from repo root:
    pytest tests/fm_steel_3d/test_backend.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import (
    FMSteel3DBase,
    FMSteel3DWithPAGs,
    FMSteel3DWithBlocks,
    FMSteel3DWithOrientations,
    FMSteel3DWithSubBlocks,
)

# ---------------------------------------------------------------------------
# LFI builder (Voronoi-like, vectorised)
# ---------------------------------------------------------------------------

def _make_lfi(nx=20, ny=20, nz=20, n_seeds=20, seed=42):
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, nx, n_seeds).astype(np.float32)
    cy = rng.integers(0, ny, n_seeds).astype(np.float32)
    cz = rng.integers(0, nz, n_seeds).astype(np.float32)
    xi = np.arange(nx, dtype=np.float32)[:, None, None, None]
    yi = np.arange(ny, dtype=np.float32)[None, :, None, None]
    zi = np.arange(nz, dtype=np.float32)[None, None, :, None]
    sq = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return (np.argmin(sq, axis=3) + 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Session-scoped pipeline fixtures — pipeline runs exactly once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lfi():
    return _make_lfi()


@pytest.fixture(scope="session")
def fm_base(lfi):
    return FMSteel3DBase.from_lfi(
        lfi,
        physical_dimensions=(20.0, 20.0, 20.0),
        voxel_size=1.0,
        units="microns",
        connectivity=6,
        min_grain_nvoxels=4,
        random_seed=42,
    )


@pytest.fixture(scope="session")
def fm_pag(fm_base):
    fm = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0,
        random_seed=42,
    )
    # generate_blocks() requires PAG orientations to already be assigned
    # (assign_pag_orientations mutates fm in place; matches how the GUI's
    # PagClusteringPage requires "Assign PAG Orientations" before "Generate
    # Blocks" is enabled).
    fm.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    return fm


@pytest.fixture(scope="session")
def fm_blk(fm_pag):
    return fm_pag.generate_blocks(
        block_thickness_range=(2.0, 5.0),
        random_seed=42,
    )


@pytest.fixture(scope="session")
def fm_ori(fm_blk):
    return fm_blk.assign_orientations(
        random_seed=42,
    )


@pytest.fixture(scope="session")
def fm_sb(fm_ori):
    return fm_ori.generate_subblocks(
        subblock_thickness_range_um=(0.5, 2.0),
        intrablock_ori_spread_deg=2.0,
        thin_block_strategy="skip",
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# LFI / base grain structure
# ---------------------------------------------------------------------------

def test_lfi_shape(lfi):
    assert lfi.shape == (20, 20, 20)


def test_lfi_dtype(lfi):
    assert lfi.dtype == np.int32


def test_lfi_has_multiple_grains(lfi):
    assert len(np.unique(lfi)) >= 10


def test_base_type(fm_base):
    assert isinstance(fm_base, FMSteel3DBase)


def test_base_grains_positive(fm_base):
    assert fm_base.n_grains > 0


def test_base_grain_locs_populated(fm_base):
    assert len(fm_base.grain_locs) == fm_base.n_grains


def test_base_neigh_network_exists(fm_base):
    assert len(fm_base.neigh_gid) > 0


def test_base_lgi_shape_preserved(fm_base, lfi):
    assert fm_base.lgi.shape == lfi.shape


def test_base_grain_cleanup_removed_tiny_grains(fm_base, lfi):
    # min_grain_nvoxels=4 — cleaned structure should have ≤ raw unique count
    raw_grains = len(np.unique(lfi))
    assert fm_base.n_grains <= raw_grains


# ---------------------------------------------------------------------------
# PAG clustering
# ---------------------------------------------------------------------------

def test_pag_type(fm_pag):
    assert isinstance(fm_pag, FMSteel3DWithPAGs)


def test_pag_count_positive(fm_pag):
    assert fm_pag.n_pags > 0


def test_pag_statistics_keys(fm_pag):
    stats = fm_pag.get_pag_statistics()
    for key in ("n_pags", "min_grains_per_pag", "max_grains_per_pag",
                "mean_grains_per_pag", "n_isolated_grains", "total_clustered_grains",
                "pag_size_voxels", "pag_size_physical", "pag_size_physical_units",
                "pag_degree", "packet_degree", "retained_austenite", "packet_size_balance"):
        assert key in stats, f"Missing key: {key}"


def test_packet_size_balance_keys_and_bounds(fm_pag):
    balance = fm_pag.get_pag_statistics()["packet_size_balance"]
    for key in ("n_multi_packet_pags", "cv", "min_max_ratio", "gini"):
        assert key in balance
    if balance["n_multi_packet_pags"] > 0:
        assert 0.0 <= balance["gini"]["mean"] <= 1.0 + 1e-9
        assert 0.0 <= balance["min_max_ratio"]["mean"] <= 1.0 + 1e-9


def test_pag_count_matches_clusters_dict(fm_pag):
    stats = fm_pag.get_pag_statistics()
    assert stats["n_pags"] == len(fm_pag.clusters_dict)


def test_pag_size_physical_matches_voxel_size_scaling(fm_pag):
    stats = fm_pag.get_pag_statistics()
    voxel_vol = fm_pag.voxel_size ** 3
    assert stats["pag_size_physical"]["mean"] == pytest.approx(
        stats["pag_size_voxels"]["mean"] * voxel_vol)


def test_pag_degree_stats_nonzero_when_pags_exist(fm_pag):
    stats = fm_pag.get_pag_statistics()
    if fm_pag.n_pags > 1:
        assert stats["pag_degree"]["mean"] >= 0.0


def test_retained_austenite_stats_zero_when_no_isolated_grains(fm_pag):
    stats = fm_pag.get_pag_statistics()
    if not fm_pag.isolated_grains:
        assert stats["retained_austenite"]["n_grains"] == 0
        assert stats["retained_austenite"]["volume_fraction_achieved"] == 0.0


def test_morphology_stats_opt_in(fm_pag):
    stats_without = fm_pag.get_pag_statistics()
    assert "pag_morphology" not in stats_without
    stats_with = fm_pag.get_pag_statistics(compute_morphology=True)
    assert "pag_morphology" in stats_with
    assert "aspect_ratio" in stats_with["pag_morphology"]
    assert "solidity" in stats_with["pag_morphology"]
    for key in ("aspect_ratio", "solidity"):
        for stat in ("min", "max", "mean", "median"):
            assert stat in stats_with["pag_morphology"][key]


def test_solidity_never_exceeds_one(fm_pag):
    # Solidity = voxel volume / convex hull volume -- a bounded voxel region
    # can never occupy more volume than its own convex hull, so this must
    # never exceed 1.0 (the hull has to be built from voxel *corners*, not
    # bare centre points, or it undercounts volume and breaks this bound).
    stats = fm_pag.get_pag_statistics(compute_morphology=True)
    solidity = stats["pag_morphology"]["solidity"]
    assert solidity["max"] <= 1.0 + 1e-9
    assert solidity["mean"] <= 1.0 + 1e-9


def test_pag_grain_to_pag_reverse_lookup(fm_pag):
    # Every grain in clusters_dict should map back through grain_to_pag_id
    for pag_id, grain_ids in fm_pag.clusters_dict.items():
        for gid in grain_ids:
            assert fm_pag.grain_to_pag_id[gid] == pag_id


# ---------------------------------------------------------------------------
# Block generation
# ---------------------------------------------------------------------------

def test_block_type(fm_blk):
    assert isinstance(fm_blk, FMSteel3DWithBlocks)


def test_block_count_positive(fm_blk):
    stats = fm_blk.get_block_statistics()
    assert stats["n_blocks"] > 0


def test_block_statistics_keys(fm_blk):
    stats = fm_blk.get_block_statistics()
    for key in ("n_blocks", "min_voxels_per_block", "max_voxels_per_block",
                "mean_voxels_per_block", "n_packets"):
        assert key in stats, f"Missing key: {key}"


def test_block_count_matches_all_blocks(fm_blk):
    stats = fm_blk.get_block_statistics()
    assert stats["n_blocks"] == len(fm_blk.all_blocks)


def test_block_voxel_arrays_are_2d(fm_blk):
    for bid, vox in fm_blk.all_blocks.items():
        assert vox.ndim == 2 and vox.shape[1] == 3, f"Block {bid} has wrong voxel shape"


def test_block_ids_use_naming_convention(fm_blk):
    for bid in fm_blk.all_blocks:
        assert bid.startswith("B_"), f"Block ID '{bid}' doesn't start with 'B_'"


# ---------------------------------------------------------------------------
# Orientation assignment
# ---------------------------------------------------------------------------

def test_orientations_type(fm_ori):
    assert isinstance(fm_ori, FMSteel3DWithOrientations)


def test_block_orientations_assigned(fm_ori):
    assert len(fm_ori.block_orientations) > 0


def test_pag_orientations_assigned(fm_ori):
    assert len(fm_ori.pag_orientations) > 0


def test_orientation_euler_angles_are_3tuples(fm_ori):
    for bid, ea in list(fm_ori.block_orientations.items())[:5]:
        assert len(ea) == 3, f"Block {bid} orientation is not a 3-tuple"


def test_full_hierarchy_stats_keys(fm_ori):
    stats = fm_ori.get_full_hierarchy_statistics()
    for key in ("n_grains", "n_pags", "n_blocks",
                "n_block_orientations_assigned", "n_pag_orientations_assigned"):
        assert key in stats, f"Missing key: {key}"


def test_full_hierarchy_stats_counts_positive(fm_ori):
    stats = fm_ori.get_full_hierarchy_statistics()
    assert stats["n_grains"] > 0
    assert stats["n_pags"] > 0
    assert stats["n_blocks"] > 0
    assert stats["n_block_orientations_assigned"] > 0


def test_misorientation_stats_returns_dict(fm_ori):
    stats = fm_ori.get_misorientation_statistics()
    assert isinstance(stats, dict)


def test_misorientation_stats_nonempty_when_blocks_exist(fm_ori):
    if len(fm_ori.block_orientations) >= 2:
        stats = fm_ori.get_misorientation_statistics()
        assert len(stats) > 0


# ---------------------------------------------------------------------------
# Sub-block generation
# ---------------------------------------------------------------------------

def test_subblock_type(fm_sb):
    assert isinstance(fm_sb, FMSteel3DWithSubBlocks)


def test_subblock_count_nonnegative(fm_sb):
    stats = fm_sb.get_subblock_statistics()
    assert stats["n_subblocks"] >= 0


def test_subblock_statistics_keys(fm_sb):
    stats = fm_sb.get_subblock_statistics()
    for key in ("n_subblocks", "mean_voxels_per_subblock",
                "n_blocks_with_subblocks", "mean_subblocks_per_block"):
        assert key in stats, f"Missing key: {key}"


def test_subblock_ids_use_naming_convention(fm_sb):
    for sbid in list(fm_sb.all_subblocks.keys())[:10]:
        assert sbid.startswith("SB_"), f"Sub-block ID '{sbid}' doesn't start with 'SB_'"


def test_subblock_parent_block_map_consistent(fm_sb):
    # Every sub-block listed in block_to_subblocks_map must exist in all_subblocks
    for blk_id, sb_ids in fm_sb.block_to_subblocks_map.items():
        for sbid in sb_ids:
            assert sbid in fm_sb.all_subblocks, f"{sbid} missing from all_subblocks"


def test_subblock_orientations_count_matches_subblocks(fm_sb):
    assert len(fm_sb.subblock_orientations) == len(fm_sb.all_subblocks)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_pag_count(fm_base):
    pag_dist = {"sizes": [3, 4], "probs": [0.7, 0.3]}
    run1 = fm_base.generate_pag_clusters(pag_dist, random_seed=7).n_pags
    run2 = fm_base.generate_pag_clusters(pag_dist, random_seed=7).n_pags
    assert run1 == run2


def test_different_seeds_different_pag_count(fm_base):
    pag_dist = {"sizes": [3, 4], "probs": [0.7, 0.3]}
    counts = {
        fm_base.generate_pag_clusters(pag_dist, random_seed=s).n_pags
        for s in range(10)
    }
    # With 10 different seeds, at least 2 different PAG counts should appear
    assert len(counts) > 1, "All 10 seeds produced identical PAG counts — seeding likely broken"
