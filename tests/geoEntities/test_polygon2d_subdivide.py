"""
Unit tests for Polygon2d.subdivide_segment.

Run from repo root:
    pytest tests/geoEntities/test_polygon2d_subdivide.py -v
"""
import numpy as np
import pytest

from upxo.geoEntities.mulsline2d import MSline2d, ring2d
from upxo.geoEntities.sline2d import Sline2d as sl2d
from upxo.geoEntities.polygon2d import Polygon2d


def _msline_from_coords(coords):
    lines = [sl2d(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
            for i in range(len(coords) - 1)]
    return MSline2d.from_lines(lines, close=False)


def _two_grain_shared_wall():
    """Two grains sharing one straight wall from (2,0) to (2,4) -- a
    realistic wall segment (open, two distinct junction endpoints), matching
    subdivide_segment's intended use case."""
    shared = _msline_from_coords([(2, 0), (2, 4)])
    left_only = _msline_from_coords([(0, 0), (0, 4), (2, 4)])
    right_only = _msline_from_coords([(2, 4), (4, 4), (4, 0), (2, 0)])
    ring_a = ring2d([left_only, shared], segids=[0, 1], segflips=[False, False])
    ring_b = ring2d([shared, right_only], segids=[0, 1], segflips=[False, False])
    return (Polygon2d.from_ring2d(ring_a, gid=1),
           Polygon2d.from_ring2d(ring_b, gid=2), shared)


def test_subdivide_n_evenly_spaced():
    poly_a, poly_b, shared = _two_grain_shared_wall()
    poly_a.subdivide_segment(1, n=3)
    coords = poly_a.ring.segments[1].get_node_coords()
    # (2,0) -> (2,4), length 4, 3 evenly-spaced interior points at y=1,2,3
    expected = np.array([[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])
    np.testing.assert_allclose(coords, expected)


def test_subdivide_at_fractions_explicit():
    poly_a, poly_b, shared = _two_grain_shared_wall()
    poly_a.subdivide_segment(1, at_fractions=[0.25, 0.75])
    coords = poly_a.ring.segments[1].get_node_coords()
    expected = np.array([[2, 0], [2, 1], [2, 3], [2, 4]])
    np.testing.assert_allclose(coords, expected)


def test_subdivide_endpoint_identity_preserved():
    poly_a, poly_b, shared = _two_grain_shared_wall()
    start_before = shared.nodes[0]
    end_before = shared.nodes[-1]
    poly_a.subdivide_segment(1, n=2)
    assert shared.nodes[0] is start_before
    assert shared.nodes[-1] is end_before


def test_subdivide_propagates_to_neighbour():
    poly_a, poly_b, shared = _two_grain_shared_wall()
    poly_a.subdivide_segment(1, n=2)
    np.testing.assert_array_equal(poly_a.ring.segments[1].get_node_coords(),
                                  poly_b.ring.segments[0].get_node_coords())
    assert poly_a.ring.segments[1] is poly_b.ring.segments[0]


def test_subdivide_requires_exactly_one_of_n_or_at_fractions():
    poly_a, poly_b, shared = _two_grain_shared_wall()
    with pytest.raises(ValueError):
        poly_a.subdivide_segment(1)
    with pytest.raises(ValueError):
        poly_a.subdivide_segment(1, n=2, at_fractions=[0.5])


def _shapely_square_polygon():
    from shapely.geometry import Polygon as ShPolygon
    return Polygon2d.from_shapely_polygon(
        ShPolygon([(0, 0), (4, 0), (4, 4), (0, 4)]), gid=1)


def test_subdivide_closed_loop_from_shapely_covers_the_closing_edge():
    """A from_shapely_polygon ring is one closed segment whose ``nodes`` list
    omitted the return to the start (by_coords(close=False) stores one node
    per line). Subdividing then spaced points along the open chain only, so
    the closing edge got none."""
    poly = _shapely_square_polygon()
    poly.subdivide_segment(0, at_fractions=[0.125, 0.875])
    seg = poly.ring.segments[0]
    coords = seg.get_node_coords()
    np.testing.assert_allclose(
        coords,
        [(0, 0), (2, 0), (4, 0), (4, 4), (0, 4), (0, 2), (0, 0)])
    assert len(seg.nodes) == len(seg.lines) + 1
    assert (seg.nodes[-1].x, seg.nodes[-1].y) == (0.0, 0.0)
    assert poly.area == pytest.approx(16.0)


def test_edit_closed_loop_from_shapely_accepts_the_true_end_row():
    poly = _shapely_square_polygon()
    coords = poly.ring.segments[0].get_node_coords()
    assert np.allclose(coords[0], coords[-1])
    new = coords.copy()
    new[1] = (5.0, 0.0)          # square (0,0),(5,0),(4,4),(0,4): shoelace area 18
    poly.edit_segment(0, new)
    assert poly.area == pytest.approx(18.0)
