"""smooth_technique_a/b + disorientation_from_polygons -- the backend of
the pixellated/smoothed display of EBSD maps."""
import numpy as np
import pytest

from upxo.viz.ebsd_smoothing import (
    smooth_technique_a, smooth_technique_b, disorientation_from_polygons)


def _four_grain_lfi():
    lfi = np.zeros((20, 20), dtype=np.int32)
    lfi[:10, :10] = 1
    lfi[:10, 10:] = 2
    lfi[10:, :10] = 3
    lfi[10:, 10:] = 4
    return lfi


def _random_quat(shape, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=shape + (4,))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def test_technique_a_returns_one_polygon_per_grain():
    lfi = _four_grain_lfi()
    polygons, grains, gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    assert set(polygons) == {1, 2, 3, 4}
    for poly in polygons.values():
        assert poly.geom_type in ('Polygon', 'MultiPolygon')
        assert poly.area > 0
    assert grains.shape == lfi.shape
    assert set(gb_rings) == {1, 2, 3, 4}   # one ring per grain, same keys as polygons


def test_technique_a_rejects_unlabelled_pixels():
    lfi = _four_grain_lfi()
    lfi[0, 0] = 0
    with pytest.raises(ValueError, match="grain label"):
        smooth_technique_a(lfi)


def test_technique_a_rejects_invalid_pass_count():
    lfi = _four_grain_lfi()
    with pytest.raises(ValueError, match="smooth_passes"):
        smooth_technique_a(lfi, smooth_passes=3, max_smooth_levels=(3, 3))


def test_technique_b_returns_one_polygon_per_grain_and_reuses_lfi():
    lfi = _four_grain_lfi()
    polygons, grains, gb_ridges = smooth_technique_b(lfi, seeds_per_grain=15, rng_seed=0)
    assert set(polygons) == {1, 2, 3, 4}
    for poly in polygons.values():
        assert poly.geom_type in ('Polygon', 'MultiPolygon')
        assert poly.area > 0
    assert grains is lfi   # no relabelling, unlike technique A
    assert gb_ridges   # a 4-grain square map has several shared boundaries
    for (gid_a, gid_b), segs in gb_ridges.items():
        assert gid_a < gid_b
        for seg in segs:
            assert seg.ndim == 2 and seg.shape == (2, 2)


def test_technique_b_seed_count_scales_with_grains_and_density():
    lfi = _four_grain_lfi()
    # higher density -> finer approximation; just check it runs and still
    # covers every grain without crashing at a much higher density.
    polygons, _, _ = smooth_technique_b(lfi, seeds_per_grain=50, rng_seed=2)
    assert set(polygons) == {1, 2, 3, 4}


def test_technique_b_is_reproducible_with_a_fixed_seed():
    lfi = _four_grain_lfi()
    p1, _, _ = smooth_technique_b(lfi, seeds_per_grain=10, rng_seed=42)
    p2, _, _ = smooth_technique_b(lfi, seeds_per_grain=10, rng_seed=42)
    for gid in p1:
        assert p1[gid].equals(p2[gid])


def test_disorientation_from_polygons_technique_a():
    lfi = _four_grain_lfi()
    polygons, grains, gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat)
    assert result.n_grains == 4
    assert len(result.segments) > 0
    assert len(result.segments) == len(result.pairs) == len(result.angles)
    assert np.isfinite(result.angles).all()   # every grain has neighbours here -- no edge segments
    for seg in result.segments:
        assert seg.ndim == 2 and seg.shape[1] == 2


def test_disorientation_from_polygons_with_gb_rings_matches_compute_gb_disorientation():
    # The whole point of gb_rings: numerically exact, not a polygon-
    # intersection reconstruction -- must match compute_gb_disorientation's
    # own direct (known-correct) computation exactly: same pairs, same
    # total boundary length, same angles.
    from upxo.xtalphy.gb_disorientation_map import compute_gb_disorientation

    lfi = _four_grain_lfi()
    quat = _random_quat(lfi.shape)
    direct = compute_gb_disorientation(lfi, quat, smooth_passes=1, max_smooth_levels=(3,))

    polygons, grains, gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    via_rings = disorientation_from_polygons(polygons, grains, quat, gb_rings=gb_rings)

    assert len(via_rings.segments) == len(direct.segments)
    assert set(map(tuple, via_rings.pairs.tolist())) == set(map(tuple, direct.pairs.tolist()))

    def _seglen(s):
        d = np.diff(s, axis=0)
        return np.hypot(d[:, 0], d[:, 1]).sum()
    assert sum(_seglen(s) for s in via_rings.segments) == pytest.approx(
        sum(_seglen(s) for s in direct.segments))

    direct_by_pair = {tuple(p): a for p, a in zip(direct.pairs.tolist(), direct.angles.tolist())}
    for pair, angle in zip(via_rings.pairs.tolist(), via_rings.angles.tolist()):
        want = direct_by_pair[tuple(pair)]
        if np.isnan(want):
            assert np.isnan(angle)
        else:
            assert angle == pytest.approx(want)


def test_disorientation_from_polygons_edge_pairs_get_nan_angle_not_pair_disorientation_deg():
    # Regression: gb_rings produces (gid, 0) "edge of map" pairs (0 = no
    # second owner) -- these must stay NaN, not get a real disorientation
    # value computed against grain id 0 (a placeholder, not a real grain).
    # The old polygon-intersection fallback never produced a (gid, 0) pair
    # at all (it only iterates real grain-to-grain adjacency), so this was
    # latent until gb_rings started producing them.
    lfi = _four_grain_lfi()
    quat = _random_quat(lfi.shape)
    polygons, grains, gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    result = disorientation_from_polygons(polygons, grains, quat, gb_rings=gb_rings)

    edge_mask = result.pairs[:, 1] == 0
    assert edge_mask.any()   # the 4-grain square map has map-edge segments
    assert np.isnan(result.angles[edge_mask]).all()
    assert np.isfinite(result.angles[~edge_mask]).all()


def test_disorientation_from_polygons_technique_b():
    lfi = _four_grain_lfi()
    polygons, grains, _gb_ridges = smooth_technique_b(lfi, seeds_per_grain=15, rng_seed=0)
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat)
    assert result.n_grains == 4
    assert len(result.segments) > 0


def test_disorientation_from_polygons_with_gb_ridges_is_not_fragmented():
    # The whole point of gb_ridges: unlike the polygon-intersection
    # fallback (which was confirmed to collapse a true boundary down to a
    # single short fragment per grain pair), using the exact raw Voronoi
    # ridge data should recover substantially more of the true boundary
    # length -- cross-checked against the SAME manifold's own polygon-
    # intersection total, not a hand-derived "true" number (see
    # test_grain_boundary_ridges.py for why).
    lfi = _four_grain_lfi()
    polygons, grains, gb_ridges = smooth_technique_b(lfi, seeds_per_grain=15, rng_seed=0)
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat, gb_ridges=gb_ridges)

    assert len(result.segments) > 4   # several ridges per grain pair, not one each
    assert np.isfinite(result.angles).all()   # Technique B never produces an edge (gid, 0) pair
    assert np.all(result.pairs[:, 1] > 0)

    def _seglen(s):
        d = np.diff(s, axis=0)
        return np.hypot(d[:, 0], d[:, 1]).sum()
    ridges_total = sum(_seglen(s) for s in result.segments)

    fallback = disorientation_from_polygons(polygons, grains, quat)   # no gb_ridges
    fallback_total = sum(_seglen(s) for s in fallback.segments)
    assert ridges_total >= fallback_total


def test_disorientation_from_polygons_original_ids_match_the_original_lfi():
    lfi = _four_grain_lfi()
    polygons, grains, _gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat, original_lfi=lfi)
    # Technique A relabels via connected components, so grains' own ids need not
    # equal lfi's -- original_ids must still map each relabelled id back correctly.
    for relabelled_id, original_id in enumerate(result.original_ids):
        if relabelled_id == 0:
            continue
        pixel = np.flatnonzero(grains.ravel() == relabelled_id)[0]
        assert lfi.ravel()[pixel] == original_id


def test_disorientation_from_polygons_without_original_lfi_is_identity():
    lfi = _four_grain_lfi()
    polygons, grains, _gb_ridges = smooth_technique_b(lfi, seeds_per_grain=15, rng_seed=0)
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat)
    assert np.array_equal(result.original_ids, np.arange(int(grains.max()) + 1))


def test_disorientation_from_polygons_missing_polygon_is_skipped_not_errored():
    lfi = _four_grain_lfi()
    polygons, grains, _gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    del polygons[1]   # simulate an incomplete/filtered polygon set
    quat = _random_quat(grains.shape)
    result = disorientation_from_polygons(polygons, grains, quat)
    # no segment should reference grain 1 as either side of a pair
    assert all(1 not in pair for pair in result.pairs)
