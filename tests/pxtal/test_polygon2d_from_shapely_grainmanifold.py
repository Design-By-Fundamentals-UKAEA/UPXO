"""
End-to-end test: polygon_collection_from_shapely against GrainManifold2D
(Technique B)'s real output -- the primary target input, not just
hand-built synthetic polygons.
"""
import numpy as np
import pytest

import upxo.gsdataops.grid_ops as gridOps
from upxo.pxtal.geometrification import GrainManifold2D
from upxo.geoEntities.polygon2d_from_shapely import polygon_collection_from_shapely


def _four_quadrant_lgi():
    """8x8 label image, four square grains (quadrants)."""
    lgi = np.zeros((8, 8), dtype=np.int32)
    lgi[:4, :4] = 1
    lgi[:4, 4:] = 2
    lgi[4:, :4] = 3
    lgi[4:, 4:] = 4
    return lgi


def test_grainmanifold2d_output_converts_with_area_conservation_and_shared_wall():
    lgi = _four_quadrant_lgi()
    seeds = gridOps.generate_constrained_hybrid_seeds(
        lgi, target_spacing=1.0, bulk_spacing=1.0,
        jitter_factor=0.25, margin=0.5, padding=2.0)
    manifold = GrainManifold2D.by_tessellation(lgi, seeds)
    cells = dict(manifold.cells)

    result = polygon_collection_from_shapely(cells, tol=1e-6)

    assert set(result.keys()) == set(cells.keys())
    total_cells_area = sum(c.area for c in cells.values())
    total_result_area = sum(p.area for p in result.values())
    assert total_result_area == pytest.approx(total_cells_area, rel=1e-6)

    # At least one interior wall must be identity-shared between neighbours.
    gids = list(result.keys())
    shared_found = False
    for i, gid_a in enumerate(gids):
        for gid_b in gids[i + 1:]:
            pa, pb = result[gid_a], result[gid_b]
            for sa in pa.ring.segments:
                for sb in pb.ring.segments:
                    if sa is sb:
                        shared_found = True
    assert shared_found, "no identity-shared wall found between any pair of grains"
