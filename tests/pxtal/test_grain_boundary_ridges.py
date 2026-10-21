"""GrainManifold2D.grain_boundary_ridges() -- exact per-grain-pair boundary
segments from the raw Voronoi ridges between individual cells of different
grain labels, captured BEFORE assemble_cells unions same-label cells
together and any shared-edge precision is lost.

Built to fix upxo.viz.ebsd_smoothing.disorientation_from_polygons's
Technique B path, which previously had to reconstruct a grain-pair's shared
edge via ``boundary.intersection(boundary)`` on the already-unioned
polygons -- numerically fragile, confirmed in practice to fragment a true
boundary into many tiny pieces (see ebsd_smoothing.py's own docstrings).
grain_boundary_ridges() needs no such reconstruction: two adjacent Voronoi
cells share the same vertex coordinates by construction, so the ridge
between them is numerically exact.
"""
import numpy as np
import pytest
from shapely.geometry import box, Point

import upxo.gsdataops.grid_ops as gridOps
from upxo.gsdataops.gid_ops import find_neighs2d
from upxo.pxtal.geometrification import GrainManifold2D


def _four_quadrant_lgi():
    """8x8 label image, four square grains (quadrants). Known adjacency:
    (1,2), (1,3), (2,4), (3,4) share a straight edge of length 8 each;
    (1,4) and (2,3) only touch at the centre point, no shared edge."""
    lgi = np.zeros((8, 8), dtype=np.int32)
    lgi[:4, :4] = 1
    lgi[:4, 4:] = 2
    lgi[4:, :4] = 3
    lgi[4:, 4:] = 4
    return lgi


def _dense_manifold(lgi, spacing=0.5, **kw):
    seeds = gridOps.generate_constrained_hybrid_seeds(
        lgi, target_spacing=spacing, bulk_spacing=spacing,
        jitter_factor=0.25, margin=0.5, padding=2.0, **kw)
    return GrainManifold2D.by_tessellation(lgi, seeds)


def _seglen(s):
    d = s[1] - s[0]
    return float(np.hypot(d[0], d[1]))


def test_empty_before_tessellation():
    lgi = _four_quadrant_lgi()
    manifold = GrainManifold2D(lgi)   # no by_tessellation -- no ridge data yet
    assert manifold.grain_boundary_ridges() == {}


def test_keys_are_sorted_grain_pairs_with_no_self_pairs():
    lgi = _four_quadrant_lgi()
    manifold = _dense_manifold(lgi)
    ridges = manifold.grain_boundary_ridges()
    assert ridges   # non-empty for a real tessellation
    for gid_a, gid_b in ridges:
        assert gid_a < gid_b
        assert gid_a != gid_b


def test_segments_are_well_formed_2x2_arrays():
    lgi = _four_quadrant_lgi()
    manifold = _dense_manifold(lgi)
    ridges = manifold.grain_boundary_ridges()
    for segs in ridges.values():
        for seg in segs:
            assert seg.shape == (2, 2)
            assert seg.dtype == np.float64


def test_known_adjacent_pairs_found_diagonal_pair_absent_or_negligible():
    lgi = _four_quadrant_lgi()
    manifold = _dense_manifold(lgi)
    ridges = manifold.grain_boundary_ridges()
    found = set(ridges)
    # every direct (edge-sharing) neighbour pair must be present
    for pair in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        assert pair in found
    # the two diagonal pairs only touch at a single point (the domain
    # centre) -- if present at all (a seed landing exactly on that corner
    # is a near-zero-probability coincidence), it must be a vanishingly
    # short artifact, not a real shared edge.
    for pair in [(1, 4), (2, 3)]:
        if pair in found:
            assert sum(_seglen(s) for s in ridges[pair]) < 0.5


def test_total_length_is_consistent_with_the_cells_own_polygon_intersection():
    # Not asserting a hand-derived "true" boundary length here -- this
    # fixture's exact pixel-vs-coordinate convention (pixel-centre vs.
    # pixel-edge, and where _generate_clipped_polygons's own domain box
    # sits relative to it) is GrainManifold2D's own pre-existing
    # business, out of scope for this method. What matters is that the
    # exact ridge-based total lands in the same ballpark as the already-
    # built cell polygons' own boundary.intersection() for the SAME
    # manifold, i.e. ridges aren't wildly over- or under-counting
    # relative to the geometry actually being used elsewhere.
    lgi = _four_quadrant_lgi()
    manifold = _dense_manifold(lgi, spacing=0.4)
    ridges = manifold.grain_boundary_ridges()
    for pair in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        ridge_total = sum(_seglen(s) for s in ridges[pair])
        gid_a, gid_b = pair
        inter = manifold.cells[gid_a].boundary.intersection(manifold.cells[gid_b].boundary)
        assert ridge_total > 0
        assert ridge_total == pytest.approx(inter.length, rel=0.5)


def test_matches_pixel_level_adjacency_on_an_irregular_map():
    # Voronoi tessellation is a continuous approximation; it can pick up a
    # few extra, vanishingly-short diagonal-corner touches that 4-
    # connected pixel adjacency misses -- not a bug (the two cells
    # genuinely meet at that point), but the two should agree on
    # everything with non-trivial shared length.
    n = 40
    rng = np.random.default_rng(3)
    n_seeds = 12
    pts = rng.integers(0, n, size=(n_seeds, 2))
    coords = np.indices((n, n)).reshape(2, -1).T
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(coords, k=1)
    lgi = (idx + 1).reshape(n, n).astype(np.int32)

    seeds = rng.uniform([0, 0], [n, n], size=(n_seeds * 25, 2))
    manifold = GrainManifold2D.by_tessellation(lgi, seeds)
    ridges = manifold.grain_boundary_ridges()

    neigh = find_neighs2d(lgi.astype(np.int32), conn=4)
    for (gid_a, gid_b), segs in ridges.items():
        total = sum(_seglen(s) for s in segs)
        if total > 1.0:   # ignore negligible diagonal-corner touches
            assert gid_b in neigh.get(gid_a, []), (
                f"pair {(gid_a, gid_b)} has {total:.2f} of shared length "
                "but is not pixel-adjacent")


def test_ridges_stay_within_the_clipped_domain():
    lgi = _four_quadrant_lgi()
    height, width = lgi.shape
    manifold = _dense_manifold(lgi)
    ridges = manifold.grain_boundary_ridges()
    domain = box(0, 0, width, height)
    for segs in ridges.values():
        for seg in segs:
            for pt in seg:
                assert domain.buffer(1e-6).contains(Point(*pt))
