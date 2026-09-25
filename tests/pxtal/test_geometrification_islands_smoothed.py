"""
Smoothed Technique A output for structures with island grains.

``polygonised_grain_structure._assemble_island_results`` used to translate each
island cluster's rings in place into the global frame. ``smooth_gbsegs`` then
re-smoothed those rings there, and the parent translated the smoothed grains
by the cluster offset again, so a smoothed island landed one offset away from
where it belongs. The smoothed set (``self.smoothed[name]``) also had no entry
for island grains. Now the cluster rings stay in their local frame and the
parent gets shifted copies (``ring2d.translated_copy``), for the raw and the
smoothed set alike.
"""
import contextlib
import io

import numpy as np
import pytest

from upxo.geoEntities.polygon2d import NestedPolygon2d, Polygon2d
from upxo.pxtal.geometrification import polygonised_grain_structure


def _single_island_lgi():
    lgi = np.ones((5, 5), dtype=np.int32)
    lgi[2, 2] = 2
    return lgi


def _nested_island_lgi():
    lgi = np.full((9, 9), 1, dtype=np.int32)
    lgi[2:7, 2:7] = 2
    lgi[4, 4] = 3
    return lgi


def _multi_grain_cluster_lgi():
    """Grains 2 and 3 are adjacent and both enclosed by host 1."""
    return np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1],
        [1, 3, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.int32)


def _smoothed(lgi, name='s'):
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    with contextlib.redirect_stdout(io.StringIO()):
        gs.pix_to_geom(verbose=False)
        gs.smooth_gbsegs(gs.GB, npasses=2, max_smooth_levels=[3, 3],
                         plot=False, name=name)
    return gs


def test_smoothing_does_not_translate_an_island_twice():
    gs = _smoothed(_single_island_lgi())
    assert gs.GRAINS[2].bounds == pytest.approx((1.5, 1.5, 2.5, 2.5))
    assert gs.smoothed['s']['GRAINS'][2].bounds == pytest.approx((1.5, 1.5, 2.5, 2.5))


def test_island_cluster_rings_stay_in_their_local_frame():
    lgi = _single_island_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)
    child = gs._islands[0]['geom']
    local = np.vstack([s.get_node_coords() for s in child.GB[1].segments])
    assert local.max() < 2.0                    # near the origin, not shifted by the offset
    assert gs.GB[2] is not child.GB[1]          # the parent holds a shifted copy


def test_smoothed_set_has_island_entries_and_holes():
    gs = _smoothed(_multi_grain_cluster_lgi())
    sm = gs.smoothed['s']
    assert set(sm['GB']) == {1, 2, 3} and set(sm['GBCoords']) == {1, 2, 3}
    assert len(sm['GB_holes'][1]) == 2          # both cluster grains are holes of the host
    assert sm['GB_holes'][1][0] is sm['GB'][2]
    assert sm['GB_holes'][1][1] is sm['GB'][3]


def test_smoothed_upxo_grains_conserve_area_and_share_the_island_ring():
    lgi = _multi_grain_cluster_lgi()
    gs = _smoothed(lgi)
    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True, smoothed='s')
    assert isinstance(result[1], NestedPolygon2d)
    assert isinstance(result[2], Polygon2d) and isinstance(result[3], Polygon2d)
    assert sum(p.area for p in result.values()) == pytest.approx(lgi.size)
    assert [h.gid for h in result[1].holes] == [2, 3]
    assert result[1].holes[0].ring is result[2].ring
    assert result[1].holes[1].ring is result[3].ring


def test_smoothed_nested_island_is_placed_correctly():
    lgi = _nested_island_lgi()
    gs = _smoothed(lgi)
    sm = gs.smoothed['s']
    assert len(sm['GB_holes'][1]) == 1 and len(sm['GB_holes'][2]) == 1
    assert sm['GB_holes'][1][0] is sm['GB'][2]
    assert sm['GB_holes'][2][0] is sm['GB'][3]
    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True, smoothed='s')
    assert sum(p.area for p in result.values()) == pytest.approx(lgi.size)
    for gid in (2, 3):
        assert sm['GRAINS'][gid].bounds == pytest.approx(gs.GRAINS[gid].bounds, abs=0.6)


def test_smoothed_option_leaves_a_structure_without_islands_all_polygon2d():
    lgi = np.array([[1, 1, 1, 2, 2, 2]] * 4, dtype=np.int32)
    gs = _smoothed(lgi)
    result = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True, smoothed='s')
    assert all(type(p) is Polygon2d for p in result.values())
    assert sum(p.area for p in result.values()) == pytest.approx(lgi.size)
