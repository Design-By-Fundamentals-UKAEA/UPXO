import numpy as np
import pytest

# gssmooth2d transitively imports vtk (via geometrification → mulsline2d).
# VTK is bundled with pyvista and is not available in the lightweight CI
# install, so skip this test when vtk cannot be imported.
vtk = pytest.importorskip("vtk", reason="vtk not available in this environment")

from upxo.pxtalops.gssmooth2d import _merge_small_grains


def test_merge_single_pixel_grain(lfi_with_tiny_grain):
    n_before = len(np.unique(lfi_with_tiny_grain))
    merged = _merge_small_grains(lfi_with_tiny_grain, area_threshold=1)
    n_after = len(np.unique(merged))
    assert n_after < n_before
