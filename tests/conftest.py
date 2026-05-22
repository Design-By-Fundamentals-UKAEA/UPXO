"""Shared fixtures for the UPXO test suite."""
import numpy as np
import pytest


@pytest.fixture
def simple_lfi_2d():
    """5x5 LFI with 4 grains; corner grains touch the boundary."""
    return np.array([
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 1, 3, 3, 2],
        [4, 4, 3, 3, 3],
        [4, 4, 4, 3, 3],
    ], dtype=np.int32)


@pytest.fixture
def lfi_with_tiny_grain():
    """5x5 LFI where grain 3 is a single pixel (size=1)."""
    return np.array([
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 3, 2, 2],
        [4, 4, 4, 4, 2],
        [4, 4, 4, 4, 4],
    ], dtype=np.int32)
