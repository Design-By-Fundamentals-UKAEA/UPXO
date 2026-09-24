"""
Regression tests: dtype='upxo' construction must not KeyError on island
grains (a grain fully enclosed by another). ``ring2d`` has no hole
representation, so a host grain's own hole ring is the translated ring2d
of its enclosed island, wrapped into a ``NestedPolygon2d`` via the new
``self.GB_holes`` bookkeeping in ``_assemble_island_results``; the island
itself gets its own ``Polygon2d`` from that same, translated ring2d object.

Before this fix, ``self.GB``/``self.GBCoords`` only covered a structure's
main/host grains (remapped from the filled sub-structure via
``self._main_map``); an island id had no entry at all, so
``construct_geometric_xtals_from_gbcoords(..., dtype='upxo')`` raised
``KeyError`` on the first island id it hit.
"""
import numpy as np
import pytest

from upxo.geoEntities.polygon2d import Polygon2d, NestedPolygon2d
from upxo.pxtal.geometrification import polygonised_grain_structure


def _single_island_lgi():
    """5x5 label image: grain 2 is a single pixel enclosed by grain 1."""
    return np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.int32)


def _nested_island_lgi():
    """9x9 label image: grain 1 -> encloses grain 2 -> encloses grain 3."""
    lgi = np.full((9, 9), 1, dtype=np.int32)
    lgi[2:7, 2:7] = 2
    lgi[4, 4] = 3
    return lgi


def _multi_grain_cluster_lgi():
    """4x5 label image: grains 2 and 3 (adjacent, sharing a wall) both
    directly enclosed by host 1. The cluster's bounding-box crop is
    non-rectangular (cell (2,2) is host material, not part of the hole),
    so the crop also needs a filler grain."""
    return np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1],
        [1, 3, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.int32)


def test_single_island_no_keyerror_and_area_conserved():
    lgi = _single_island_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)

    assert set(result.keys()) == {1, 2}
    assert isinstance(result[1], NestedPolygon2d)
    assert isinstance(result[2], Polygon2d)
    assert sum(p.area for p in result.values()) == pytest.approx(lgi.size)
    assert result[1].area == pytest.approx(gs.GRAINS[1].area)
    assert result[2].area == pytest.approx(gs.GRAINS[2].area)


def test_single_island_shares_ring_identity_with_host_hole():
    lgi = _single_island_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)
    host, island = result[1], result[2]

    assert gs.GB[2] is gs.GB_holes[1][0]
    assert host.holes[0].ring is island.ring
    assert host.holes[0].gid == 2
    assert host.gids_all == [1, 2]


def test_editing_island_boundary_propagates_to_host_hole():
    lgi = _single_island_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)
    host, island = result[1], result[2]
    total_before = sum(p.area for p in result.values())

    island.subdivide_segment(0, n=3)
    coords = island.ring.segments[0].get_node_coords()
    tangent = coords[-1] - coords[0]
    normal = np.array([-tangent[1], tangent[0]])
    normal = normal / np.linalg.norm(normal)
    new_coords = coords.copy()
    for k in range(1, len(coords) - 1):
        new_coords[k] = new_coords[k] + 0.3 * normal
    island.edit_segment(0, new_coords)

    total_after = sum(p.area for p in result.values())
    assert total_after == pytest.approx(total_before)
    assert host.area == pytest.approx(lgi.size - island.area)


def test_nested_island_within_island():
    lgi = _nested_island_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)

    assert set(result.keys()) == {1, 2, 3}
    assert isinstance(result[1], NestedPolygon2d)
    assert isinstance(result[2], NestedPolygon2d)
    assert isinstance(result[3], Polygon2d)
    assert len(gs.GB_holes[1]) == 1
    assert len(gs.GB_holes[2]) == 1
    # grain 1's hole is grain 2 itself, not grain 3 (nested one level deeper)
    assert gs.GB_holes[1][0] is gs.GB[2]
    assert gs.GB_holes[2][0] is gs.GB[3]

    for gid, poly in result.items():
        assert poly.area == pytest.approx(gs.GRAINS[gid].area)
    assert sum(p.area for p in result.values()) == pytest.approx(lgi.size)


def test_multi_grain_cluster_siblings_not_double_translated():
    """Sibling grains in the same island cluster share Point2d objects at
    their common wall (confirmed directly: 2 shared points for this
    fixture). Translating each sibling's ring independently, with its own
    fresh dedup set, moves a shared point twice -- once per ring that
    references it -- corrupting exactly the siblings' shapes while the
    cluster's total area coincidentally stays put. ring2d.translate's
    ``seen`` parameter, shared by _assemble_island_results across every
    grain in one cluster, must prevent this.
    """
    lgi = _multi_grain_cluster_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)

    assert result[2].area == pytest.approx(gs.GRAINS[2].area)
    assert result[3].area == pytest.approx(gs.GRAINS[3].area)
    assert result[1].area == pytest.approx(gs.GRAINS[1].area)
    for gid in (2, 3):
        minx, miny, maxx, maxy = result[gid].make_shapely().bounds
        assert -1 <= minx and maxx <= lgi.shape[1] + 1
        assert -1 <= miny and maxy <= lgi.shape[0] + 1


def test_no_islands_leaves_GB_holes_empty_and_all_polygon2d():
    lgi = np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
    ], dtype=np.int32)
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)

    assert all(isinstance(p, Polygon2d) and not isinstance(p, NestedPolygon2d)
              for p in result.values())
