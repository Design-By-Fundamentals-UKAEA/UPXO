import numpy as np
import pytest
from upxo.gsdataops.gid_ops import (
    find_small_fids,
    find_boundary_fids2d,
    find_O1_neigh_2d,
)


def test_find_small_fids(lfi_with_tiny_grain):
    small = find_small_fids(lfi_with_tiny_grain, threshold=1)
    assert 3 in small


def test_find_small_fids_excludes_large(lfi_with_tiny_grain):
    small = find_small_fids(lfi_with_tiny_grain, threshold=1)
    assert 1 not in small and 4 not in small


def test_find_boundary_fids2d(simple_lfi_2d):
    boundary = find_boundary_fids2d(simple_lfi_2d)
    # grains 1, 2, 4 touch the border; 3 may also
    assert 1 in boundary
    assert 2 in boundary
    assert 4 in boundary


def test_find_O1_neigh_2d_returns_dict(simple_lfi_2d):
    neigh = find_O1_neigh_2d(simple_lfi_2d)
    assert isinstance(neigh, dict)
    # every grain ID should appear as a key
    for gid in np.unique(simple_lfi_2d):
        assert gid in neigh


def test_find_O1_neigh_symmetry(simple_lfi_2d):
    neigh = find_O1_neigh_2d(simple_lfi_2d)
    for gid, nbrs in neigh.items():
        for nbr in nbrs:
            assert gid in neigh[nbr]
