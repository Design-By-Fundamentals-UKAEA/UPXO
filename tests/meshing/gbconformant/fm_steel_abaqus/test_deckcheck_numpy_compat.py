"""Regression tests for deckcheck.py's text-to-integer parsing under numpy >=2.4.

np.loadtxt/np.fromstring with an integer dtype raise ValueError (not just a
DeprecationWarning) from numpy 2.4 onward if a value is written with a decimal point
(e.g. "1.0") -- the same failure mode as the CTF BC-column bug fixed in
upxo.interfaces.defdap.ebsd_reader. _read_table and _ids now always parse as float and
cast/round to the target integer dtype afterwards, so a float-formatted value degrades to
being rounded rather than raising, under any numpy version in the supported range.
"""
import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import _ids, _read_table


def test_ids_parses_clean_integers():
    result = _ids(["1, 2, 3", "4,5"])
    assert result.dtype == np.int64
    assert result.tolist() == [1, 2, 3, 4, 5]


def test_ids_tolerates_a_float_formatted_value():
    # A value written as "2.0" must not raise, and must round-trip to the integer it names --
    # this is exactly the shape of bug that would otherwise raise ValueError under numpy>=2.4.
    result = _ids(["1, 2.0, 3"])
    assert result.dtype == np.int64
    assert result.tolist() == [1, 2, 3]


def test_ids_empty_input():
    result = _ids([])
    assert result.dtype == np.int64
    assert result.tolist() == []


def test_read_table_casts_clean_integers(tmp_path):
    path = tmp_path / "elements.inp"
    path.write_text("*Element, type=C3D4\n1, 1, 2, 3, 4\n2, 2, 3, 4, 5\n", encoding="ascii")

    result = _read_table(path, 5, dtype=np.int32)

    assert result.dtype == np.int32
    assert result.tolist() == [[1, 1, 2, 3, 4], [2, 2, 3, 4, 5]]


def test_read_table_tolerates_a_float_formatted_value(tmp_path):
    path = tmp_path / "elements.inp"
    path.write_text("*Element, type=C3D4\n1, 1, 2, 3, 4.0\n", encoding="ascii")

    result = _read_table(path, 5, dtype=np.int32)

    assert result.dtype == np.int32
    assert result.tolist() == [[1, 1, 2, 3, 4]]


def test_read_table_default_dtype_is_float(tmp_path):
    path = tmp_path / "nodes.inp"
    path.write_text("*Node\n1, 0.5, 1.5, 2.5\n", encoding="ascii")

    result = _read_table(path, 4)

    assert result.dtype == np.float64
    assert result.tolist() == [[1.0, 0.5, 1.5, 2.5]]
