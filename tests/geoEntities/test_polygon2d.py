"""
Unit tests for upxo.geoEntities.polygon2d: Polygon2d, NestedPolygon2d, and
the props/DataFrame scalar-field bridge.

Run from repo root:
    pytest tests/geoEntities/test_polygon2d.py -v
"""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon as ShPolygon

from upxo.geoEntities.mulsline2d import MSline2d, ring2d
from upxo.geoEntities.sline2d import Sline2d as sl2d
from upxo.geoEntities.polygon2d import (
    Polygon2d, NestedPolygon2d,
    polygons_to_prop_dataframe, apply_prop_dataframe,
)


def _square_ring(x0=0.0, y0=0.0, s=2.0):
    """A single-segment ring2d for an axis-aligned square."""
    coords = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
    msl = MSline2d.by_coords(coords, close=False)
    return ring2d([msl], segids=[0], segflips=[False])


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

def test_make_shapely_matches_hand_built_square():
    poly = Polygon2d.from_ring2d(_square_ring(0, 0, 2))
    shp = poly.make_shapely()
    expected = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
    np.testing.assert_array_equal(np.array(list(shp.exterior.coords)),
                                  np.array(expected))


def test_shapely_round_trip_exact():
    poly = Polygon2d.from_ring2d(_square_ring(0, 0, 2))
    shp1 = poly.make_shapely()

    poly2 = Polygon2d.from_shapely_polygon(shp1)
    shp2 = poly2.make_shapely()

    np.testing.assert_array_equal(np.array(list(shp1.exterior.coords)),
                                  np.array(list(shp2.exterior.coords)))


def test_from_shapely_polygon_non_axis_aligned():
    shp = ShPolygon([(0, 0), (4, 1), (3, 5), (-1, 3)])
    poly = Polygon2d.from_shapely_polygon(shp)

    assert poly.area == pytest.approx(shp.area)
    assert poly.perimeter == pytest.approx(shp.length)
    # exactly 4 distinct nodes -- from_shapely_polygon must not drop the
    # last vertex (see the by_coords(close=True) bug documented in the
    # module docstring).
    assert poly.ring.segments[0].nnodes == 4


# ---------------------------------------------------------------------------
# edit_segment: shared-segment propagation and endpoint protection
# ---------------------------------------------------------------------------

def _msline_from_coords(coords):
    """MSline2d.from_lines correctly captures every coordinate, including
    the final one -- unlike MSline2d.by_coords, whose own node-building
    never appends the final line's pntb (a real, reproducible bug; see
    Polygon2d.from_shapely_polygon's docstring). from_lines is also what
    the real pipeline uses for open wall segments
    (geometrification.py's splice_grain_boundary_segments_at_junction_points)."""
    lines = [sl2d(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
            for i in range(len(coords) - 1)]
    return MSline2d.from_lines(lines, close=False)


def _shared_wall_fixture():
    """Two rings sharing one MSline2d wall, mirroring how
    consolidate_gbsegments makes two neighbouring grains' rings reference
    the literal same segment object."""
    shared = _msline_from_coords([(2, 0), (2, 1), (2, 2)])
    left_only = _msline_from_coords([(0, 0), (0, 2), (2, 2)])
    right_only = _msline_from_coords([(2, 2), (4, 2), (4, 0), (2, 0)])
    ring_a = ring2d([left_only, shared], segids=[0, 1], segflips=[False, False])
    ring_b = ring2d([shared, right_only], segids=[0, 1], segflips=[False, False])
    return ring_a, ring_b, shared


def test_edit_segment_propagates_to_neighbour_by_shared_identity():
    ring_a, ring_b, shared = _shared_wall_fixture()
    poly_a = Polygon2d.from_ring2d(ring_a)
    poly_b = Polygon2d.from_ring2d(ring_b)

    assert poly_b.ring.segments[0] is poly_a.ring.segments[1] is shared

    poly_a.edit_segment(1, [(2, 0), (2.5, 1), (2, 2)])

    # object identity is preserved -- the segment was mutated, not replaced
    assert poly_b.ring.segments[0] is poly_a.ring.segments[1] is shared
    np.testing.assert_array_equal(
        poly_a.ring.segments[1].get_node_coords(),
        poly_b.ring.segments[0].get_node_coords())
    np.testing.assert_array_equal(
        poly_a.ring.segments[1].get_node_coords(),
        np.array([[2.0, 0.0], [2.5, 1.0], [2.0, 2.0]]))


def test_edit_segment_rejects_start_endpoint_move():
    ring_a, _, _ = _shared_wall_fixture()
    poly_a = Polygon2d.from_ring2d(ring_a)
    with pytest.raises(ValueError):
        poly_a.edit_segment(1, [(2, 0.5), (2.5, 1), (2, 2)])


def test_edit_segment_rejects_end_endpoint_move():
    ring_a, _, _ = _shared_wall_fixture()
    poly_a = Polygon2d.from_ring2d(ring_a)
    with pytest.raises(ValueError):
        poly_a.edit_segment(1, [(2, 0), (2.5, 1), (2, 2.5)])


def test_edit_segment_rejects_too_few_rows():
    ring_a, _, _ = _shared_wall_fixture()
    poly_a = Polygon2d.from_ring2d(ring_a)
    with pytest.raises(ValueError):
        poly_a.edit_segment(1, [(2, 0)])


# ---------------------------------------------------------------------------
# clone(): shared in the clone, independent of the original
# ---------------------------------------------------------------------------

def test_clone_preserves_sharing_and_isolates_from_original():
    ring_a, ring_b, shared = _shared_wall_fixture()
    poly_a = Polygon2d.from_ring2d(ring_a)
    poly_b = Polygon2d.from_ring2d(ring_b)

    seg_clones = {}
    clone_a = poly_a.clone(seg_clones)
    clone_b = poly_b.clone(seg_clones)

    # shared with each other in the clone...
    assert clone_a.ring.segments[1] is clone_b.ring.segments[0]
    # ...but not with the originals
    assert clone_a.ring.segments[1] is not shared

    before = poly_a.ring.segments[1].get_node_coords().copy()
    clone_a.edit_segment(1, [(2, 0), (2.9, 1), (2, 2)])

    # original untouched by editing the clone
    np.testing.assert_array_equal(poly_a.ring.segments[1].get_node_coords(), before)
    # clone_b sees clone_a's edit (sharing preserved within the clone set)
    np.testing.assert_array_equal(clone_a.ring.segments[1].get_node_coords(),
                                  clone_b.ring.segments[0].get_node_coords())


# ---------------------------------------------------------------------------
# NestedPolygon2d
# ---------------------------------------------------------------------------

def test_nested_polygon_single_level_hole_round_trip():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (4, 2), (4, 4), (2, 4)]
    shp = ShPolygon(shell, holes=[hole])

    np2d = NestedPolygon2d.from_shapely_polygon(shp)

    assert np2d.area == pytest.approx(100.0 - 4.0)
    assert all(isinstance(h, Polygon2d) for h in np2d.holes)


def test_nested_polygon_hole_in_hole_constructed_natively():
    # Outer host: 10x10 square, area 100.
    # Hole: 6x6 square (area 36), itself containing a hole: 2x2 square (area 4).
    outer_host = Polygon2d.from_ring2d(_square_ring(0, 0, 10))
    inner_hole_host = Polygon2d.from_ring2d(_square_ring(2, 2, 6))
    innermost_hole = Polygon2d.from_ring2d(_square_ring(3, 3, 2))

    hole_with_its_own_hole = NestedPolygon2d(host=inner_hole_host,
                                             holes=[innermost_hole])
    outer = NestedPolygon2d(host=outer_host, holes=[hole_with_its_own_hole])

    # area = outer(100) - [inner_hole_host(36) - innermost_hole(4)] = 68
    expected_area = 100.0 - (36.0 - 4.0)
    assert outer.make_shapely().area == pytest.approx(expected_area)


def test_nested_polygon_gids_all_recursive():
    outer_host = Polygon2d.from_ring2d(_square_ring(0, 0, 10), gid=1)
    hole1 = Polygon2d.from_ring2d(_square_ring(2, 2, 2), gid=2)
    hole2 = Polygon2d.from_ring2d(_square_ring(5, 5, 2), gid=3)
    outer = NestedPolygon2d(host=outer_host, holes=[hole1, hole2], gid=1)

    assert outer.gids_all == [1, 2, 3]


# ---------------------------------------------------------------------------
# Props / DataFrame bridge
# ---------------------------------------------------------------------------

def test_props_dataframe_bridge_round_trip():
    p1 = Polygon2d.from_ring2d(_square_ring(0, 0, 2), gid=1,
                               props={'area': 4.0, 'phase': 1})
    p2 = Polygon2d.from_ring2d(_square_ring(2, 0, 2), gid=2,
                               props={'area': 4.0})
    polygons = {1: p1, 2: p2}

    df = polygons_to_prop_dataframe(polygons)
    assert list(df.index) == [0, 1]
    assert df.loc[0, 'area'] == 4.0
    assert pd.isna(df.loc[1, 'phase'])

    df.loc[1, 'phase'] = 2
    apply_prop_dataframe(polygons, df)

    assert p2.props['phase'] == 2
    assert p1.props == {'area': 4.0, 'phase': 1}
