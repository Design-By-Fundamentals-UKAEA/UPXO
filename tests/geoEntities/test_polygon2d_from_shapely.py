"""
Unit tests for upxo.geoEntities.polygon2d_from_shapely.polygon_collection_from_shapely:
a Shapely multi-polygon grain structure -> topologically-linked
{gid: Polygon2d | NestedPolygon2d}, with genuine cross-grain shared-segment
object identity.

Run from repo root:
    pytest tests/geoEntities/test_polygon2d_from_shapely.py -v
"""
import numpy as np
import pytest
from shapely.geometry import Polygon as ShPolygon
from shapely.ops import unary_union

from upxo.geoEntities.polygon2d import Polygon2d, NestedPolygon2d
from upxo.geoEntities.polygon2d_from_shapely import polygon_collection_from_shapely


def _two_squares():
    return {
        1: ShPolygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        2: ShPolygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
    }


def _shared_segment(p1, p2):
    for i1, s1 in enumerate(p1.ring.segments):
        for i2, s2 in enumerate(p2.ring.segments):
            if s1 is s2:
                return i1, i2, s1
    return None


def test_shared_wall_identity():
    result = polygon_collection_from_shapely(_two_squares())
    p1, p2 = result[1], result[2]
    match = _shared_segment(p1, p2)
    assert match is not None, "no shared segment object found between neighbours"


def test_edit_propagation_no_gap_and_area_conserved():
    result = polygon_collection_from_shapely(_two_squares())
    p1, p2 = result[1], result[2]
    idx1, idx2, shared = _shared_segment(p1, p2)

    p1.subdivide_segment(idx1, n=1)
    coords = p1.ring.segments[idx1].get_node_coords()
    mid = len(coords) // 2
    new_coords = coords.copy()
    new_coords[mid] = new_coords[mid] + np.array([0.3, 0.0])
    p1.edit_segment(idx1, new_coords)

    np.testing.assert_array_equal(p1.ring.segments[idx1].get_node_coords(),
                                  p2.ring.segments[idx2].get_node_coords())
    assert p1.area == pytest.approx(4.3)
    assert p2.area == pytest.approx(3.7)
    assert p1.area + p2.area == pytest.approx(8.0)


def test_area_round_trip_fidelity():
    cells = _two_squares()
    result = polygon_collection_from_shapely(cells)
    for gid, poly in result.items():
        assert poly.area == pytest.approx(cells[gid].area)
    assert sum(p.area for p in result.values()) == pytest.approx(
        sum(c.area for c in cells.values()))


def test_multipolygon_valued_cell():
    cells = {
        1: unary_union([ShPolygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        ShPolygon([(3, 0), (4, 0), (4, 1), (3, 1)])]),
    }
    result = polygon_collection_from_shapely(cells)
    p = result[1]
    assert isinstance(p, Polygon2d)
    assert p.area == pytest.approx(1.0)
    extra = p.props['extra_parts']
    assert len(extra) == 1
    assert extra[0].area == pytest.approx(1.0)


def test_hole_bearing_polygon_becomes_nested():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (4, 2), (4, 4), (2, 4)]
    cells = {1: ShPolygon(shell, holes=[hole])}
    result = polygon_collection_from_shapely(cells)
    p = result[1]
    assert isinstance(p, NestedPolygon2d)
    assert p.area == pytest.approx(96.0)


def test_triple_point_three_grains_share_walls_correctly():
    # Three unit squares arranged in an L, meeting at (1,1):
    #   grain 1: [0,1]x[0,1]   grain 2: [1,2]x[0,1]   grain 3: [0,1]x[1,2]
    cells = {
        1: ShPolygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        2: ShPolygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        3: ShPolygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
    }
    result = polygon_collection_from_shapely(cells)
    p1, p2, p3 = result[1], result[2], result[3]

    for gid, poly in result.items():
        assert poly.area == pytest.approx(cells[gid].area)

    # grain 1 and grain 2 share the wall x=1, y in [0,1]
    assert _shared_segment(p1, p2) is not None
    # grain 1 and grain 3 share the wall y=1, x in [0,1]
    assert _shared_segment(p1, p3) is not None
    # grain 2 and grain 3 do not touch (only meet at the single point (1,1))
    assert _shared_segment(p2, p3) is None


def test_gid_key_remaps_output_keys():
    cells = _two_squares()
    result = polygon_collection_from_shapely(cells, gid_key=lambda g: f"grain_{g}")
    assert set(result.keys()) == {"grain_1", "grain_2"}
