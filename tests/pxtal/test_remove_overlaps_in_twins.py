"""Regression tests for mcgs3_grain_structure.remove_overlaps_in_twins."""
import numpy as np

from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure


def test_zero_twins_returns_empty_list():
    """A grain with no twins used to raise UnboundLocalError."""
    assert mcgs3_grain_structure.remove_overlaps_in_twins(None, 1, []) == []


def test_single_twin_returned_unchanged():
    """One twin has nothing to overlap with."""
    twin = np.array([[0, 0, 0], [1, 1, 1]])
    out = mcgs3_grain_structure.remove_overlaps_in_twins(None, 1, [twin])
    assert len(out) == 1
    assert np.array_equal(out[0], twin)
