"""IslandTwinIntroducer -- bounded, Sigma3-oriented ellipsoidal twin
inclusions placed after Structure Cleaning, into a user-selected mix of
remaining parent material / non-host grains / existing primary twins.
"""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.island_twin_3d import (
    IslandTwinIntroducer, introduce_island_twin_3d, ISLAND_CATEGORIES)


def _rot_z(deg):
    half = np.radians(deg) / 2
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _fixture_lgi():
    """30x30x30: grain 1 (host, 30x20x20), grain 2 (non_host, 30x30x10),
    grain 3 (primary_twin, 30x10x20) -- disjoint, no overlap."""
    n = 30
    lgi = np.ones((n, n, n), dtype=np.int32)
    lgi[:, :, 20:] = 2
    lgi[:, 20:, :20] = 3
    return lgi


def _fixture():
    lgi = _fixture_lgi()
    twin_role = {1: 'host', 2: 'non_host', 3: 'primary_twin'}
    twin_parent_of = {3: 1}
    all_quats = {1: _rot_z(0.0), 2: _rot_z(10.0), 3: _rot_z(5.0)}
    twin_thickness = {'thick_um': np.linspace(0.5, 1.5, 20)}
    return lgi, all_quats, twin_role, twin_parent_of, twin_thickness


def _disorientation_deg(q_a, q_b):
    from upxo.xtalphy.crystal_orientation import quat_to_R_batch
    from upxo.xtalphy.gb_disorientation_map import pair_disorientation_deg
    R = quat_to_R_batch(np.array([q_a, q_b]))
    return float(pair_disorientation_deg(R, np.array([0]), np.array([1]))[0])


# ── construction validation ──────────────────────────────────────────────

def test_category_probs_must_sum_to_one():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    with pytest.raises(ValueError, match="sum to 1"):
        IslandTwinIntroducer(
            lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
            category_probs={'remaining_parent': 0.5, 'non_host': 0.2},
            island_vf_target=0.05, twin_thickness=twin_thickness)


def test_negative_vf_target_rejected():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    with pytest.raises(ValueError, match="island_vf_target"):
        IslandTwinIntroducer(
            lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
            category_probs={'non_host': 1.0}, island_vf_target=-0.1,
            twin_thickness=twin_thickness)


def test_zero_vf_target_introduces_nothing():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'non_host': 1.0}, island_vf_target=0.0,
        twin_thickness=twin_thickness)
    introducer.introduce_islands(verbose=False)
    assert introducer.n_islands_placed == 0
    assert np.array_equal(introducer.lgi_islanded, lgi)


# ── placement correctness ────────────────────────────────────────────────

def test_islands_never_touch_any_boundary_including_the_domain_edge():
    """The defining property of an island: dilating it by 1 voxel touches
    nothing but its own label and its container's -- never a third grain,
    and never the domain's own outer edge."""
    from scipy.ndimage import binary_dilation
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'remaining_parent': 0.4, 'non_host': 0.3, 'primary_twin': 0.3},
        island_vf_target=0.05, twin_thickness=twin_thickness,
        island_aspect_ratio=2.0, rng_seed=3)
    introducer.introduce_islands(verbose=False)
    assert introducer.n_islands_placed > 0

    for gid, container_gid in introducer.twin_parent_of.items():
        if gid not in introducer.island_quats:
            continue   # not an island (e.g. the fixture's own twin_parent_of entry)
        island_mask = introducer.lgi_islanded == gid
        assert island_mask.any()
        touches_edge = (
            island_mask[0, :, :].any() or island_mask[-1, :, :].any()
            or island_mask[:, 0, :].any() or island_mask[:, -1, :].any()
            or island_mask[:, :, 0].any() or island_mask[:, :, -1].any())
        assert not touches_edge, f"island {gid} touches the domain edge"
        dilated = binary_dilation(island_mask)
        neighbour_labels = set(np.unique(introducer.lgi_islanded[dilated])) - {gid}
        assert neighbour_labels == {container_gid}, (
            f"island {gid} touches {neighbour_labels}, expected only its "
            f"container {container_gid}")


def test_achieves_the_target_volume_fraction():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'non_host': 1.0}, island_vf_target=0.05,
        twin_thickness=twin_thickness, island_aspect_ratio=2.0, rng_seed=2)
    introducer.introduce_islands(verbose=False)
    total_vox = int(np.sum(lgi > 0))
    achieved = introducer.cumulative_island_vox / total_vox
    # The +1-voxel separation margin can make the very last bit of target
    # VF unreachable (every remaining candidate too close to an existing
    # island/boundary) -- vf_stopped_early is then correctly True even
    # though the achieved VF still landed acceptably close to the target.
    assert achieved == pytest.approx(0.05, abs=0.01)


def test_category_distribution_converges_to_the_requested_probabilities():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'remaining_parent': 0.5, 'non_host': 0.3, 'primary_twin': 0.2},
        island_vf_target=0.15, twin_thickness=twin_thickness,
        island_aspect_ratio=2.0, max_total_attempts=5000, rng_seed=1)
    introducer.introduce_islands(verbose=False)
    n = introducer.n_islands_placed
    assert n > 100
    frac = {k: v / n for k, v in introducer.summary()['n_by_category'].items()}
    assert frac['remaining_parent'] == pytest.approx(0.5, abs=0.08)
    assert frac['non_host'] == pytest.approx(0.3, abs=0.08)
    assert frac['primary_twin'] == pytest.approx(0.2, abs=0.08)


def test_single_category_only_places_in_that_category():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'primary_twin': 1.0}, island_vf_target=0.02,
        twin_thickness=twin_thickness, island_aspect_ratio=2.0, rng_seed=4)
    introducer.introduce_islands(verbose=False)
    assert introducer.n_islands_placed > 0
    assert set(introducer.island_category_of.values()) == {'primary_twin'}
    assert all(introducer.twin_parent_of[gid] == 3
              for gid in introducer.island_quats)


def test_empty_category_is_flagged_and_does_not_crash(capsys):
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    # 'primary_twin' has a grain, but pretend none were ever allocated.
    twin_role_no_primary = dict(twin_role)
    twin_role_no_primary[3] = 'non_host'
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role_no_primary, {}, voxel_size=1.0,
        category_probs={'primary_twin': 0.3, 'non_host': 0.7},
        island_vf_target=0.02, twin_thickness=twin_thickness,
        island_aspect_ratio=2.0, max_total_attempts=500, rng_seed=5)
    introducer.introduce_islands(verbose=True)
    out = capsys.readouterr().out
    assert "no eligible grains" in out
    assert introducer.n_islands_placed > 0   # non_host alone still works


# ── orientation (Sigma3) ─────────────────────────────────────────────────

def test_island_orientation_is_a_sigma3_relationship_to_its_container():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'non_host': 1.0}, island_vf_target=0.02,
        twin_thickness=twin_thickness, island_aspect_ratio=2.0,
        twin_orient_scatter_deg=0.0, rng_seed=6)
    introducer.introduce_islands(verbose=False)
    assert introducer.n_islands_placed > 0
    container_q = all_quats[2]
    for gid, q_island in introducer.island_quats.items():
        angle = _disorientation_deg(container_q, q_island)
        assert angle == pytest.approx(60.0, abs=1.0)


def test_island_orientation_scatter_stays_close_to_sigma3():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'non_host': 1.0}, island_vf_target=0.03,
        twin_thickness=twin_thickness, island_aspect_ratio=2.0,
        twin_orient_scatter_deg=1.5, rng_seed=7)
    introducer.introduce_islands(verbose=False)
    assert introducer.n_islands_placed > 0
    container_q = all_quats[2]
    for gid, q_island in introducer.island_quats.items():
        angle = _disorientation_deg(container_q, q_island)
        assert angle == pytest.approx(60.0, abs=5.0)   # small scatter allowance


# ── low-level carving primitive ──────────────────────────────────────────

def test_introduce_island_twin_3d_is_bounded_within_the_container():
    lgi = _fixture_lgi()
    center = np.array([15.0, 10.0, 10.0])   # well inside grain 1 (30x20x20)
    normal = np.array([0.0, 0.0, 1.0])
    voxels = introduce_island_twin_3d(lgi, 1, center, normal,
                                      half_thickness=2.0, half_width=4.0, next_gid=99)
    assert voxels is not None and len(voxels) > 0
    assert np.all(lgi[voxels[:, 0], voxels[:, 1], voxels[:, 2]] == 99)
    # every carved voxel must have originally belonged to grain 1
    assert np.all((voxels[:, 1] < 20) & (voxels[:, 2] < 20))


def test_introduce_island_twin_3d_returns_none_outside_the_container():
    lgi = _fixture_lgi()
    center = np.array([15.0, 25.0, 25.0])   # inside grain 2, not grain 1
    normal = np.array([0.0, 0.0, 1.0])
    voxels = introduce_island_twin_3d(lgi, 1, center, normal,
                                      half_thickness=1.0, half_width=1.0, next_gid=99)
    assert voxels is None


# ── regression: realistic (fine) voxel size must not collapse success rate ─

def test_realistic_fine_voxel_size_still_places_islands_in_large_regions():
    """Regression for the bug this was designed to catch: an isotropic
    placement margin based on the island's full WIDTH (not just its
    thickness) made every direction demand width-sized clearance, which
    balloons past ordinary grain sizes the finer voxel_size gets -- at a
    realistic sub-micron voxel size, this made success near-zero
    regardless of settings. A large (80^3-ish) region at a realistic
    voxel_size must still comfortably place many islands."""
    n = 80
    lgi = np.ones((n, n, n), dtype=np.int32)
    lgi[:, :, 40:] = 2   # non_host, large
    twin_role = {1: 'host', 2: 'non_host'}
    all_quats = {1: _rot_z(0.0), 2: _rot_z(10.0)}
    twin_thickness = {'thick_um': np.linspace(0.3, 2.0, 20)}   # realistic EBSD range

    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, {}, voxel_size=0.1,   # realistic fine calibration
        category_probs={'non_host': 1.0}, island_vf_target=0.02,
        twin_thickness=twin_thickness, island_aspect_ratio=4.0,
        max_total_attempts=500, rng_seed=0)
    introducer.introduce_islands(verbose=False)

    assert introducer.n_islands_placed >= 10
    achieved = introducer.cumulative_island_vox / int(np.sum(lgi > 0))
    assert achieved == pytest.approx(0.02, abs=0.005)


def test_thin_slab_container_genuinely_cannot_fit_a_thick_island():
    """The flip side of the above: a container thinner than the sampled
    island thickness range cannot fit ANY island regardless of the
    algorithm -- this must fail honestly (tracked in
    n_attempts_by_category), not silently appear to work by placing a
    too-large island anyway."""
    n = 80
    lgi = np.ones((n, n, n), dtype=np.int32)
    lgi[:, 36:44, :] = 3   # a thin (8-voxel) slab -- grain 3
    lgi[lgi == 1] = 2      # the rest is non_host (unused here)
    twin_role = {2: 'non_host', 3: 'primary_twin'}
    all_quats = {2: _rot_z(10.0), 3: _rot_z(5.0)}
    # Thickness range whose FULL extent (2*half_thickness) exceeds the
    # slab's own 8-voxel thickness at this voxel_size: 2.0 um / 0.1 / 2
    # = 10 vox half-thickness -> 20 vox full thickness > 8.
    twin_thickness = {'thick_um': np.linspace(1.0, 2.0, 20)}

    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, {}, voxel_size=0.1,
        category_probs={'primary_twin': 1.0}, island_vf_target=0.05,
        twin_thickness=twin_thickness, island_aspect_ratio=4.0,
        max_attempts_per_island=10, max_total_attempts=100, rng_seed=0)
    introducer.introduce_islands(verbose=False)

    assert introducer.n_islands_placed == 0
    assert introducer.vf_stopped_early is True
    summary = introducer.summary()
    assert summary['n_attempts_by_category']['primary_twin'] > 0
    assert summary['n_by_category']['primary_twin'] == 0


def test_summary_reports_attempts_by_category_even_when_zero_placed():
    lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
    introducer = IslandTwinIntroducer(
        lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
        category_probs={'non_host': 1.0}, island_vf_target=0.0,
        twin_thickness=twin_thickness, rng_seed=0)
    introducer.introduce_islands(verbose=False)
    summary = introducer.summary()
    assert summary['n_attempts_by_category'] == {
        'remaining_parent': 0, 'non_host': 0, 'primary_twin': 0}


# ── numba kernel vs. numpy fallback parity ──────────────────────────────
# introduce_island_twin_3d has two independent code paths for the same
# ellipsoid-membership test (a scalar numba loop when numba is importable,
# a vectorized numpy fallback otherwise) purely for speed -- see
# island_twin_3d's module-level _HAS_NUMBA switch. They must select
# exactly the same voxels for the same inputs; this forces BOTH paths in
# the same process (regardless of which one introduce_islands would pick
# on this machine) and compares their output directly, rather than
# trusting that two differently-tuned implementations of the same formula
# happen to agree.

def test_numba_kernel_and_numpy_fallback_select_identical_voxels(monkeypatch):
    import upxo.pxtal.twinned_simple_3d.island_twin_3d as m
    if not m._HAS_NUMBA:
        pytest.skip("numba not available in this environment")

    lgi = _fixture_lgi()
    coords = np.argwhere(lgi == 1)
    # Deliberately "ungly" (non-integer, non-axis-aligned) center/normal/
    # half-axes: a tidy, symmetric choice (integer center, axis-aligned
    # normal, integer half-axes) can put lattice voxels EXACTLY on the
    # ellipsoid surface, where the numpy (BLAS dot product) and numba
    # (scalar FMA) paths can legitimately round to opposite sides of
    # <= 1.0 by a single ULP -- not a bug in either path, just floating-
    # point noise at a measure-zero boundary. These values make an exact
    # tie astronomically unlikely, so the comparison below tests real
    # agreement, not an artifact of a convenient test fixture.
    center = np.array([10.37, 9.84, 10.21])
    normal = np.array([0.5321, 0.7438, 0.4017])
    normal = normal / np.linalg.norm(normal)
    half_thickness, half_width = 3.17, 11.83

    lgi_a = lgi.copy()
    voxels_numba = introduce_island_twin_3d(
        lgi_a, 1, center, normal, half_thickness, half_width, 100,
        container_coords=coords)

    monkeypatch.setattr(m, "_HAS_NUMBA", False)
    lgi_b = lgi.copy()
    voxels_numpy = introduce_island_twin_3d(
        lgi_b, 1, center, normal, half_thickness, half_width, 100,
        container_coords=coords)

    assert voxels_numba is not None and voxels_numpy is not None
    a_set = {tuple(v) for v in voxels_numba}
    b_set = {tuple(v) for v in voxels_numpy}
    assert a_set == b_set
    assert np.array_equal(lgi_a, lgi_b)


def test_introducer_results_identical_regardless_of_numba_availability(monkeypatch):
    import upxo.pxtal.twinned_simple_3d.island_twin_3d as m
    if not m._HAS_NUMBA:
        pytest.skip("numba not available in this environment")

    def _run():
        lgi, all_quats, twin_role, twin_parent_of, twin_thickness = _fixture()
        introducer = IslandTwinIntroducer(
            lgi, all_quats, twin_role, twin_parent_of, voxel_size=1.0,
            category_probs={'remaining_parent': 0.5, 'non_host': 0.5},
            island_vf_target=0.05, twin_thickness=twin_thickness,
            island_aspect_ratio=2.0, rng_seed=3)
        introducer.introduce_islands(verbose=False)
        return introducer

    with_numba = _run()
    monkeypatch.setattr(m, "_HAS_NUMBA", False)
    without_numba = _run()

    assert with_numba.n_islands_placed == without_numba.n_islands_placed > 0
    assert with_numba.n_attempts == without_numba.n_attempts
    assert with_numba.cumulative_island_vox == without_numba.cumulative_island_vox
    assert np.array_equal(with_numba.lgi_islanded, without_numba.lgi_islanded)
