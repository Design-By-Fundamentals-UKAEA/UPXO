"""Regression test for mesh_mcgs2d.map_elements_grainids' element-id lookup.

Elements are generated x-outer, y-inner (see
mesh_abaqus_upxo_nonconformal_quad4), so the flat *element-id* array is in
(nx, ny) order. The old code reshaped it directly to ``self.xgrid.shape``,
which is (ny, nx) (from ``np.meshgrid(..., indexing='xy')``) -- for a
non-square grid this silently produces a wrongly-shaped array (and even for
a square grid it silently transposes x and y), so grains got mapped to the
wrong elements with no error raised.

This test builds a real (non-square, and separately a square but
non-symmetric) grain map, runs the actual meshing + mapping pipeline, and
independently verifies -- via each element's own node coordinates, not via
any assumption about internal ordering -- that every element assigned to a
grain id genuinely lies on that grain's pixel(s) in the input lfi.
"""
import numpy as np
import pytest

from upxo.meshing.mesher_2d import mesh_mcgs2d


def _build_mesher(lfi):
    ny, nx = lfi.shape
    meshInfo = dict(elementType='quad4', FESoftware='Abaqus',
                    mesher='UPXO', reducedIntegration=False)
    gridInfo = dict(xmin=0, xinc=1, xmax=nx - 1,
                    ymin=0, yinc=1, ymax=ny - 1)
    gsInfo = dict(m=1, grainAreas=np.random.random(int(lfi.max())))
    return mesh_mcgs2d(getFEControlsFromUPXO=False, getGridControlsFromUPXO=False,
                       meshInfo=meshInfo, gridInfo=gridInfo, gsInfo=gsInfo, lfi=lfi)


def _assert_grain_element_map_correct(mesher):
    """For every element assigned to a grain, its centroid must land on
    that grain's pixel in the original lfi -- checked independently of any
    assumption about generation order."""
    x_vals = np.unique(mesher.xgrid)
    y_vals = np.unique(mesher.ygrid)

    grain_element_sets = mesher.map_elements_grainids()
    assert grain_element_sets  # non-empty

    n_checked = 0
    for dict_key, eids in grain_element_sets.items():
        true_gid = dict_key + 1  # dict is 0-indexed, grain ids are 1-indexed
        for eid in eids:
            coords = mesher.coords_element_nodes[int(eid) - 1]
            cx, cy = coords[0].mean(), coords[1].mean()
            x_idx = int(np.argmin(np.abs(x_vals - cx)))
            y_idx = int(np.argmin(np.abs(y_vals - cy)))
            assert mesher.lfi[y_idx, x_idx] == true_gid, (
                f"element {eid} assigned to grain {true_gid} but its "
                f"centroid ({cx},{cy}) -> lfi[{y_idx},{x_idx}]="
                f"{mesher.lfi[y_idx, x_idx]}"
            )
            n_checked += 1
    assert n_checked == mesher.lfi.size


def test_non_square_grid_grain_element_mapping():
    # 4 columns (nx) x 3 rows (ny) -- deliberately non-square, and every
    # value distinct enough that a transpose or reshape-scramble would be
    # immediately visible as a wrong grain id.
    lfi = np.array([
        [1, 1, 2, 2],
        [1, 3, 3, 2],
        [4, 4, 3, 2],
    ], dtype=np.int32)
    mesher = _build_mesher(lfi)
    _assert_grain_element_map_correct(mesher)


def test_square_grid_grain_element_mapping_not_transposed():
    # Square (3x3) but not symmetric under transpose, so a pure
    # x<->y swap bug (which a non-square test alone wouldn't need to rule
    # out at this size) is also caught.
    lfi = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 2],
    ], dtype=np.int32)
    mesher = _build_mesher(lfi)
    _assert_grain_element_map_correct(mesher)
