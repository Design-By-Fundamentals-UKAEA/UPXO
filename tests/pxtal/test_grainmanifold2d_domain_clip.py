"""
Regression test: GrainManifold2D.smooth_interfaces must not leave cells
extending past the RVE domain.

``_generate_clipped_polygons`` clips the initial Voronoi cells to
``[0, width] x [0, height]``, but Taubin smoothing's anti-shrinkage (``mu``,
negative) step can inflate a boundary-adjacent vertex outward, and
``_laplacian_step`` only pins a vertex sitting *exactly* on the domain edge
-- one merely near it (e.g. a T-junction one step in, where a grain wall
meets the boundary) is free to move outward, with nothing afterward pulling
it back in. Before the fix, cells could bulge past the domain after
smoothing (visible as blob/teardrop shapes sticking out past the image
bounds); ``smooth_interfaces`` now re-clips every cell to the domain box at
the end, mirroring the box used at construction.
"""
import numpy as np
import pytest
from shapely.geometry import box

import upxo.gsdataops.grid_ops as gridOps
from upxo.pxtal.geometrification import GrainManifold2D


def _four_quadrant_lgi():
    """8x8 label image, four square grains (quadrants)."""
    lgi = np.zeros((8, 8), dtype=np.int32)
    lgi[:4, :4] = 1
    lgi[:4, 4:] = 2
    lgi[4:, :4] = 3
    lgi[4:, 4:] = 4
    return lgi


def test_smoothed_cells_stay_within_domain_bounds():
    lgi = _four_quadrant_lgi()
    height, width = lgi.shape
    seeds = gridOps.generate_constrained_hybrid_seeds(
        lgi, target_spacing=1.0, bulk_spacing=1.0,
        jitter_factor=0.25, margin=0.5, padding=2.0)
    manifold = GrainManifold2D.by_tessellation(lgi, seeds)
    manifold.smooth_interfaces(iterations=10, lmbda=0.5, mu=-0.53)

    domain = box(0, 0, width, height)
    for gid, cell in manifold.cells.items():
        minx, miny, maxx, maxy = cell.bounds
        assert minx >= -1e-9 and miny >= -1e-9
        assert maxx <= width + 1e-9 and maxy <= height + 1e-9
        assert domain.contains(cell.buffer(-1e-9))


def test_smoothed_total_area_still_exactly_fills_domain():
    lgi = _four_quadrant_lgi()
    height, width = lgi.shape
    seeds = gridOps.generate_constrained_hybrid_seeds(
        lgi, target_spacing=1.0, bulk_spacing=1.0,
        jitter_factor=0.25, margin=0.5, padding=2.0)
    manifold = GrainManifold2D.by_tessellation(lgi, seeds)
    manifold.smooth_interfaces(iterations=10, lmbda=0.5, mu=-0.53)

    total_area = sum(c.area for c in manifold.cells.values())
    assert total_area == pytest.approx(width * height, rel=1e-6)
