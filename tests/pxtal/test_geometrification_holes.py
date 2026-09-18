"""Regression tests for polygonised_grain_structure.holes_exist.

``polygons_raw_holes`` is pre-populated with one entry per grain in
``__init__`` (``{gid: [] for gid in self.gid}``) and only grains that
actually have a hole get their entry overwritten from ``[]`` to a
``Polygon``/``dict`` by ``make_polygonal_grains_raw``. ``holes_exist`` used
to check ``len(self.polygons_raw_holes.keys()) == 0``, which is always
false once any grain exists (keys are never removed) -- so the property
returned False whenever holes actually existed, and its callers (``hlpol``,
``allpol``, ``make_gsmp``) silently dropped hole geometry from the output.
"""
import numpy as np
from shapely.geometry import Polygon

from upxo.pxtal.geometrification import polygonised_grain_structure


def _make_gs(gids):
    lgi = np.zeros((2, 2), dtype=int)
    return polygonised_grain_structure(lgi, gids, neigh_gid_pxtal={})


def test_holes_exist_false_when_no_grain_has_a_hole():
    gs = _make_gs([1, 2])
    gs.polygons_raw_exteriors[1] = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gs.polygons_raw_exteriors[2] = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    # polygons_raw_holes left at its __init__ default: {1: [], 2: []}

    assert gs.holes_exist is False
    assert gs.hlpol == []
    assert gs.allpol == gs.expol


def test_holes_exist_true_and_hole_retained_when_one_grain_has_a_hole():
    gs = _make_gs([1, 2])
    gs.polygons_raw_exteriors[1] = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    gs.polygons_raw_exteriors[2] = Polygon([(3, 0), (5, 0), (5, 3), (3, 3)])
    hole = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    gs.polygons_raw_holes[1] = hole
    # gs.polygons_raw_holes[2] left at its __init__ default: []

    assert gs.holes_exist is True
    assert gs.hlpol == [hole]
    assert len(gs.allpol) == len(gs.expol) + 1
    assert hole in gs.allpol
