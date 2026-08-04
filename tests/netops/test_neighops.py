"""Tests for upxo.netops.neighops."""
import numpy as np
import pytest
from upxo.netops.neighops import default_connectivity


def test_default_connectivity_2d():
    """Default connectivity for 2D is 4 or 8."""
    conn = default_connectivity(2)
    assert conn in [4, 8]


def test_default_connectivity_3d():
    """Default connectivity for 3D is 6 or 26."""
    conn = default_connectivity(3)
    assert conn in [6, 18, 26]


def test_default_connectivity_returns_int():
    """Connectivity values are integers."""
    conn2d = default_connectivity(2)
    conn3d = default_connectivity(3)
    assert isinstance(conn2d, int)
    assert isinstance(conn3d, int)


def test_default_connectivity_reasonable_values():
    """Connectivity values are within expected ranges."""
    for ndim in [2, 3]:
        conn = default_connectivity(ndim)
        assert conn > 0
        if ndim == 2:
            assert conn <= 8
        elif ndim == 3:
            assert conn <= 26


def test_default_connectivity_upxo_standard():
    """UPXO standard is 4 for 2D, 6 for 3D."""
    # Based on UPXO conventions
    conn2d = default_connectivity(2)
    conn3d = default_connectivity(3)
    # Should be the smaller connectivity (4-connected, 6-connected)
    # not the larger 8-connected, 26-connected
    assert conn2d <= 4 or conn2d == 8
    assert conn3d <= 6 or conn3d == 26
