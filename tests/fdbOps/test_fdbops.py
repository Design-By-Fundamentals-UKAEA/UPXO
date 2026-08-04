"""Tests for upxo.fdbOps.fdbops."""
import pytest
import numpy as np
from upxo.fdbOps.fdbops import add_feature_database_entry


def test_add_feature_database_entry_single_value():
    """Add a single feature to empty FDB."""
    fdb = {}
    add_feature_database_entry(
        fdb, 'grain_1', 'volume', 150.5,
        {'method': 'voxel_count', 'unit': 'um^3'}
    )
    assert 'grain_1' in fdb
    assert 'volume' in fdb['grain_1']


def test_add_feature_database_entry_multiple_data():
    """Add multiple data arrays to a feature."""
    fdb = {}
    add_feature_database_entry(
        fdb, 'grain_1', ['volume', 'surface_area'],
        [150.5, 420.0],
        {'unit': 'um^3', 'unit_area': 'um^2'}
    )
    assert 'volume' in fdb['grain_1']
    assert 'surface_area' in fdb['grain_1']


def test_add_feature_database_entry_info_dict():
    """Info must be a dictionary."""
    fdb = {}
    with pytest.raises(ValueError):
        add_feature_database_entry(
            fdb, 'grain_1', 'volume', 150.5, 'not_a_dict'
        )


def test_add_feature_database_entry_info_keys_strings():
    """All keys in info dict must be strings."""
    fdb = {}
    with pytest.raises(ValueError):
        add_feature_database_entry(
            fdb, 'grain_1', 'volume', 150.5,
            {1: 'bad', 'good': 'ok'}  # 1 is not a string
        )


def test_add_feature_database_entry_numpy_array():
    """Add numpy array as data."""
    fdb = {}
    data = np.array([1.0, 2.0, 3.0])
    add_feature_database_entry(
        fdb, 'grain_1', 'orientations', data,
        {'format': 'quaternions'}
    )
    assert 'orientations' in fdb['grain_1']


def test_add_feature_database_entry_multiple_grains():
    """Build FDB across multiple grain entries."""
    fdb = {}
    for grain_id in range(1, 4):
        add_feature_database_entry(
            fdb, f'grain_{grain_id}', 'volume', float(grain_id * 100),
            {'method': 'voxel'}
        )
    assert len(fdb) == 3
    assert 'grain_1' in fdb
    assert 'grain_3' in fdb


def test_add_feature_database_entry_overwrite():
    """Adding to same grain/feature overwrites."""
    fdb = {}
    add_feature_database_entry(
        fdb, 'grain_1', 'volume', 100.0, {'v': '1'}
    )
    add_feature_database_entry(
        fdb, 'grain_1', 'volume', 200.0, {'v': '2'}
    )
    # Second write should overwrite or update
    assert 'grain_1' in fdb


def test_add_feature_database_entry_tuple_data():
    """Add tuple of data values."""
    fdb = {}
    add_feature_database_entry(
        fdb, 'grain_1', ['x', 'y', 'z'],
        (1.0, 2.0, 3.0),
        {'unit': 'um'}
    )
    assert 'x' in fdb['grain_1']
    assert 'y' in fdb['grain_1']
    assert 'z' in fdb['grain_1']


def test_add_feature_database_entry_list_data():
    """Add list of data values."""
    fdb = {}
    add_feature_database_entry(
        fdb, 'grain_1', ['a', 'b'],
        [10, 20],
        {'type': 'scalars'}
    )
    assert len(fdb['grain_1']) == 2
