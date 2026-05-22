import numpy as np
from upxo._sup.dataTypeHandlers import are_all_numbers, strip_str, DEEPCHECK_is_coord2d_list


def test_are_all_numbers_true():
    assert are_all_numbers([1, 2, 3.0]) is True


def test_are_all_numbers_false():
    assert are_all_numbers([1, 'a']) is False


def test_strip_str_removes_underscore():
    result = strip_str('pol_area')
    assert '_' not in result


def test_deepcheck_coord2d_valid():
    assert DEEPCHECK_is_coord2d_list([[1, 2], [3, 4]]) is True


def test_deepcheck_coord2d_invalid_shape():
    assert DEEPCHECK_is_coord2d_list([[1, 2, 3], [4, 5, 6]]) is False
