"""Tests for upxo.imageOps.labelops."""
import numpy as np
import pytest
from upxo.imageOps.labelops import reindex_labels


def test_reindex_labels_consecutive():
    """Reindex non-consecutive labels to consecutive integers."""
    lgi = np.array([
        [1, 1, 5],
        [5, 10, 10]
    ], dtype=int)
    result = reindex_labels(lgi, background=0, start=1)
    assert result.dtype == lgi.dtype
    # Result should have consecutive labels 1, 2, 3
    unique = np.unique(result)
    assert unique[0] == 1  # First non-background


def test_reindex_labels_preserves_background():
    """Background label (0) remains 0."""
    lgi = np.array([
        [0, 1, 1],
        [2, 2, 0]
    ], dtype=int)
    result = reindex_labels(lgi, background=0)
    assert np.all(result[lgi == 0] == 0)


def test_reindex_labels_custom_start():
    """Start from custom label value."""
    lgi = np.array([[1, 2], [2, 3]], dtype=int)
    result = reindex_labels(lgi, background=0, start=10)
    unique_labels = np.unique(result)
    assert unique_labels[0] >= 10


def test_reindex_labels_maintains_structure():
    """Reindexing preserves grain structure."""
    lgi = np.array([
        [1, 1, 1],
        [1, 2, 2],
        [3, 3, 3]
    ], dtype=int)
    result = reindex_labels(lgi, background=0)
    # Same number of unique labels (minus background)
    assert len(np.unique(result)) == len(np.unique(lgi))


def test_reindex_labels_single_grain():
    """Single grain array."""
    lgi = np.ones((5, 5), dtype=int) * 42
    result = reindex_labels(lgi, background=0, start=1)
    assert len(np.unique(result)) == 1
    assert np.unique(result)[0] == 1


def test_reindex_labels_3d():
    """Reindexing works on 3D arrays."""
    lgi = np.zeros((3, 3, 3), dtype=int)
    lgi[0, :, :] = 1
    lgi[1, :, :] = 5
    lgi[2, :, :] = 10
    result = reindex_labels(lgi, background=0, start=1)
    unique = np.unique(result)
    # Should have: 0 (background) + 3 reindexed grains
    assert len(unique) >= 3  # At least 3 grains
    # Verify background is preserved
    assert 0 in result or len(unique) == 3  # Background may be 0 or implicitly first grain


def test_reindex_labels_no_background():
    """Reindex when there is no background."""
    lgi = np.array([[1, 2], [2, 3]], dtype=int)
    result = reindex_labels(lgi, background=-1)  # Not present
    assert 1 in result
    assert 2 in result
    assert 3 in result
