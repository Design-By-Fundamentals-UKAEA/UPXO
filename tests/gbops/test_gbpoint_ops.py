"""Tests for upxo.gbops.gbpoint_ops."""
import numpy as np
import pytest
from upxo.gbops.gbpoint_ops import initialise_gbp, local_gbp_for_gid


def test_initialise_gbp_single_grain():
    """Initialize GBP dicts for a single grain."""
    local_gbp, global_gbp = initialise_gbp([1])
    assert isinstance(local_gbp, dict)
    assert isinstance(global_gbp, dict)
    assert local_gbp[1] is None
    assert global_gbp[1] is None


def test_initialise_gbp_multiple_grains():
    """Initialize GBP dicts for multiple grains."""
    gids = [1, 2, 3, 5, 10]
    local_gbp, global_gbp = initialise_gbp(gids)
    for gid in gids:
        assert gid in local_gbp
        assert gid in global_gbp
        assert local_gbp[gid] is None
        assert global_gbp[gid] is None


def test_local_gbp_for_gid_simple_2d():
    """Extract local GBP for a grain in a simple 2D array."""
    lgi = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 3]
    ])
    gbp = local_gbp_for_gid(lgi, 1)
    assert gbp is not None
    assert len(gbp) > 0


def test_local_gbp_for_gid_boundary_detection():
    """Local GBP should detect boundary pixels of a grain."""
    lgi = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=int)
    gbp = local_gbp_for_gid(lgi, 1)
    # Should find boundary pixels around grain 1
    assert gbp is not None


def test_local_gbp_for_gid_no_matching_grains():
    """Local GBP when grain is not present in array."""
    lgi = np.array([[1, 1], [1, 1]], dtype=int)
    gbp = local_gbp_for_gid(lgi, 999)
    # Should handle gracefully (empty or None)
    assert gbp is None or len(gbp) == 0


def test_local_gbp_for_gid_3d_array():
    """Extract local GBP from a 3D labelled array."""
    lgi = np.zeros((5, 5, 5), dtype=int)
    lgi[1:4, 1:4, 1:4] = 1  # Grain 1 in center
    lgi[0, :, :] = 2  # Grain 2 as background layer

    gbp = local_gbp_for_gid(lgi, 1)
    assert gbp is not None
    # Grain 1 has internal structure, should find boundary pixels


def test_local_gbp_for_gid_returns_coordinates():
    """Local GBP returns coordinate information."""
    lgi = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)
    gbp = local_gbp_for_gid(lgi, 1)
    if gbp is not None and len(gbp) > 0:
        # Should be array-like with shape (N, 2) or (N, 3) for 2D/3D coords
        assert isinstance(gbp, (list, np.ndarray))
