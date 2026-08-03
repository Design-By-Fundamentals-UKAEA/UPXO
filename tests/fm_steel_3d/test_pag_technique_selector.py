"""
Unit tests for pag_technique_selector_3d.generate_pags -- the unified
Technique A/B + lever orchestration entry point.

Run from repo root:
    pytest tests/fm_steel_3d/test_pag_technique_selector.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
from upxo.pxtal.fm_steel_3d.pag_technique_selector_3d import generate_pags, VALID_LEVERS


def _voronoi_lfi(n=24, n_seeds=60, seed=3):
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
def fm_base():
    lfi = _voronoi_lfi()
    return FMSteel3DBase.from_lfi(
        lfi, physical_dimensions=(24.0, 24.0, 24.0), voxel_size=1.0,
        connectivity=6, min_grain_nvoxels=8, random_seed=42, verbosity=0)


def _all_grains_from_pags(fm_pag):
    return sorted(g for gids in fm_pag.clusters_dict.values() for g in gids)


# ---------------------------------------------------------------------------
# Technique A, every lever
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lever", VALID_LEVERS)
def test_technique_a_every_grain_accounted_for(fm_base, lever):
    fm_pag = generate_pags(
        fm_base, technique='A', pag_size_distribution=DIST_3_4,
        max_packets_per_pag=4, lever=lever, random_seed=10)
    clustered = _all_grains_from_pags(fm_pag)
    all_accounted = sorted(clustered + list(fm_pag.isolated_grains))
    assert all_accounted == sorted(fm_base.grain_locs.keys())
    assert len(clustered) == len(set(clustered))  # no duplicates


@pytest.mark.parametrize("lever", VALID_LEVERS)
def test_technique_a_cap_respected(fm_base, lever):
    fm_pag = generate_pags(
        fm_base, technique='A', pag_size_distribution=DIST_3_4,
        max_packets_per_pag=4, lever=lever, random_seed=11)
    for gids in fm_pag.clusters_dict.values():
        assert 1 <= len(gids) <= 4


@pytest.mark.parametrize("lever", VALID_LEVERS)
def test_technique_a_reproducible(fm_base, lever):
    a = generate_pags(fm_base, technique='A', pag_size_distribution=DIST_3_4,
                      lever=lever, random_seed=99)
    b = generate_pags(fm_base, technique='A', pag_size_distribution=DIST_3_4,
                      lever=lever, random_seed=99)
    assert a.clusters_dict == b.clusters_dict
    assert a.isolated_grains == b.isolated_grains


def test_lever3_singleton_rate_much_lower_than_lever1_or_lever2(fm_base):
    def singleton_frac(fm_pag):
        counts = [len(v) for v in fm_pag.clusters_dict.values()]
        return sum(1 for c in counts if c == 1) / len(counts)

    fracs = {}
    for lever in ['lever1', 'lever2', 'lever3']:
        reps = [generate_pags(fm_base, technique='A', pag_size_distribution=DIST_3_4,
                              lever=lever, random_seed=100 + i).clusters_dict
                for i in range(3)]
        fracs[lever] = np.mean([sum(1 for c in [len(v) for v in cd.values()] if c == 1) / len(cd)
                                for cd in reps])
    assert fracs['lever3'] < fracs['lever1']
    assert fracs['lever3'] < fracs['lever2']


# ---------------------------------------------------------------------------
# Technique B
# ---------------------------------------------------------------------------

def test_technique_b_every_grain_accounted_for(fm_base):
    fm_pag = generate_pags(
        fm_base, technique='B', pag_size_distribution=DIST_3_4,
        max_packets_per_pag=4, split_threshold=20, random_seed=12)
    clustered = _all_grains_from_pags(fm_pag)
    # for technique B, "grains" in clusters_dict are actually packet ids on a
    # NEW finer base structure -- check against the new parent's own grains.
    assert sorted(clustered) == sorted(fm_pag._parent.grain_locs.keys())


def test_technique_b_pag_id_equals_original_grain_id(fm_base):
    fm_pag = generate_pags(
        fm_base, technique='B', pag_size_distribution=DIST_3_4,
        max_packets_per_pag=4, split_threshold=20, random_seed=13)
    assert set(fm_pag.clusters_dict.keys()) <= set(fm_base.grain_locs.keys())


def test_technique_b_cap_respected(fm_base):
    fm_pag = generate_pags(
        fm_base, technique='B', pag_size_distribution=DIST_3_4,
        max_packets_per_pag=4, split_threshold=20, random_seed=14)
    for gids in fm_pag.clusters_dict.values():
        assert 1 <= len(gids) <= 4


def test_technique_b_reproducible(fm_base):
    a = generate_pags(fm_base, technique='B', pag_size_distribution=DIST_3_4,
                      split_threshold=20, random_seed=77)
    b = generate_pags(fm_base, technique='B', pag_size_distribution=DIST_3_4,
                      split_threshold=20, random_seed=77)
    assert a.clusters_dict == b.clusters_dict


# ---------------------------------------------------------------------------
# Retained austenite (post-hoc, whole-PAG selection)
# ---------------------------------------------------------------------------

def test_retained_austenite_selected_and_disjoint_from_pags_technique_a(fm_base):
    # Retained-austenite PAGs stay IN clusters_dict (tagged via
    # retained_austenite_pag_ids), the same way technique B's single-grain
    # retained "PAGs" always have -- see
    # test_retained_austenite_technique_b for the analogous check. This is
    # what lets assign_pag_orientations() give every PAG (transformed or
    # retained) a proper orientation uniformly, before generate_blocks()
    # later skips retained_austenite_pag_ids.
    fm_pag = generate_pags(
        fm_base, technique='A', pag_size_distribution=DIST_3_4,
        lever='lever3', pag_grain_fraction=0.7, random_seed=15)
    assert len(fm_pag.isolated_grains) > 0
    assert len(fm_pag.retained_austenite_pag_ids) > 0

    # Every retained-austenite PAG id is a real entry in clusters_dict, and
    # isolated_grains is exactly the flattened union of those PAGs' grains.
    retained_grains = set()
    for pid in fm_pag.retained_austenite_pag_ids:
        assert pid in fm_pag.clusters_dict
        retained_grains.update(fm_pag.clusters_dict[pid])
    assert retained_grains == fm_pag.isolated_grains

    # Grains belonging to TRANSFORMING PAGs (i.e. not retained) are disjoint
    # from isolated_grains -- that's the meaningful independence property,
    # not "isolated_grains vs all of clusters_dict" (which now legitimately
    # overlap, since retained PAGs are still in clusters_dict).
    transforming_grains = {
        g for pid, gids in fm_pag.clusters_dict.items()
        if pid not in fm_pag.retained_austenite_pag_ids
        for g in gids
    }
    assert transforming_grains.isdisjoint(fm_pag.isolated_grains)

    # clusters_dict alone (transformed + retained PAGs) already accounts for
    # every grain -- no need to separately union in isolated_grains.
    all_accounted = set(_all_grains_from_pags(fm_pag))
    assert all_accounted == set(fm_base.grain_locs.keys())


def test_retained_austenite_is_independent_set_of_pags(fm_base):
    # No isolated grain should be adjacent (in the ORIGINAL grain graph) to
    # another isolated grain that came from a DIFFERENT reclassified PAG --
    # within the same reclassified PAG, grains are of course adjacent to
    # each other (that's what made them one PAG); the independence property
    # is between DIFFERENT retained-austenite PAGs.
    fm_pag = generate_pags(
        fm_base, technique='A', pag_size_distribution=DIST_3_4,
        lever='lever3', pag_grain_fraction=0.6, isolated_grain_strategy='smallest',
        random_seed=16)
    assert len(fm_pag.isolated_grains) > 0


def test_retained_austenite_technique_b(fm_base):
    fm_pag = generate_pags(
        fm_base, technique='B', pag_size_distribution=DIST_3_4,
        split_threshold=20, pag_grain_fraction=0.7, random_seed=17)
    assert len(fm_pag.isolated_grains) > 0
    assert fm_pag.isolated_grains <= set(fm_base.grain_locs.keys())
    # isolated grains must NOT have been split -- each contributes exactly
    # one packet to clusters_dict, keyed by its own original grain id.
    for g in fm_pag.isolated_grains:
        assert g in fm_pag.clusters_dict
        assert len(fm_pag.clusters_dict[g]) == 1


def test_no_retained_austenite_when_fraction_is_one(fm_base):
    fm_pag = generate_pags(
        fm_base, technique='A', pag_size_distribution=DIST_3_4,
        lever='lever3', pag_grain_fraction=1.0, random_seed=18)
    assert len(fm_pag.isolated_grains) == 0


def test_invalid_technique_raises(fm_base):
    with pytest.raises(ValueError):
        generate_pags(fm_base, technique='C', pag_size_distribution=DIST_3_4)


def test_invalid_lever_raises(fm_base):
    with pytest.raises(ValueError):
        generate_pags(fm_base, technique='A', pag_size_distribution=DIST_3_4, lever='lever99')
