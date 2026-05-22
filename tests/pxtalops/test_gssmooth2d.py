import numpy as np
from upxo.pxtalops.gssmooth2d import _merge_small_grains


def test_merge_single_pixel_grain(lfi_with_tiny_grain):
    n_before = len(np.unique(lfi_with_tiny_grain))
    merged = _merge_small_grains(lfi_with_tiny_grain, area_threshold=1)
    n_after = len(np.unique(merged))
    # the single-pixel grain must have been absorbed — grain count must decrease
    assert n_after < n_before
