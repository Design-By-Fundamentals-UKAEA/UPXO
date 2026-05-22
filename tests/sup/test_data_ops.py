import math
import numpy as np
import pytest
from upxo._sup.data_ops import (
    distance_between_two_points,
    calculate_angular_distance,
    find_outliers_iqr,
)


def test_distance_3_4_5():
    assert math.isclose(distance_between_two_points((0, 0), (3, 4)), 5.0)


def test_angular_distance_orthogonal():
    angle = calculate_angular_distance((1, 0), (0, 1))
    assert math.isclose(angle, math.pi / 2, rel_tol=1e-6)


def test_angular_distance_parallel():
    angle = calculate_angular_distance((1, 0), (2, 0))
    assert math.isclose(angle, 0.0, abs_tol=1e-9)


def test_find_outliers_iqr_detects_spike():
    data = np.array([1, 2, 3, 4, 100])
    indices = find_outliers_iqr(data, mode='both')
    assert 4 in indices  # index of 100
