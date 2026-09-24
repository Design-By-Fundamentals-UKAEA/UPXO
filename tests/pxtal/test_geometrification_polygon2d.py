"""
Integration smoke test: construct_geometric_xtals_from_gbcoords(dtype='upxo')
must reproduce the existing dtype='shapely' path's geometry exactly, on a
real (small, hand-built) grain structure run through the full Technique A
pipeline.
"""
import numpy as np
import pytest

from upxo.geoEntities.polygon2d import Polygon2d
from upxo.pxtal.geometrification import polygonised_grain_structure


def _two_grain_lgi():
    """4x6 label image, two rectangular grains split by a vertical wall."""
    return np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
    ], dtype=np.int32)


def test_upxo_dtype_reproduces_shapely_dtype_areas():
    lgi = _two_grain_lgi()
    gs = polygonised_grain_structure(lgi, np.unique(lgi), None)
    gs.pix_to_geom(verbose=False)

    shapely_grains = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='shapely', saa=False, throw=True)
    upxo_grains = gs.construct_geometric_xtals_from_gbcoords(
        gs.GBCoords, dtype='upxo', saa=False, throw=True)

    assert set(upxo_grains.keys()) == set(shapely_grains.keys())
    for gid, poly in upxo_grains.items():
        assert isinstance(poly, Polygon2d)
        assert poly.gid == gid
        assert poly.area == pytest.approx(shapely_grains[gid].area)
