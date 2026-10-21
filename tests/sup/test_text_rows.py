"""write_rows writes what np.savetxt writes, character for character."""
import io
import os
import tempfile

import numpy as np
import pytest

from upxo._sup import text_rows
from upxo._sup.text_rows import write_rows


def both(rows, fmt, delimiter=' '):
    a, b = io.StringIO(), io.StringIO()
    np.savetxt(a, rows, fmt=fmt, delimiter=delimiter)
    write_rows(b, rows, fmt, delimiter)
    return a.getvalue(), b.getvalue()


def same(rows, fmt, delimiter=' '):
    want, got = both(rows, fmt, delimiter)
    assert got == want


def test_integer_tables_of_every_dtype_and_width():
    rng = np.random.default_rng(0)
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        info = np.iinfo(dtype)
        for width in (1, 2, 4, 7):
            rows = rng.integers(max(info.min, -50), min(info.max, 100000), size=(137, width)).astype(dtype)
            for fmt, delimiter in (('%d', ' '), ('%d', ', '), ('%10d', ', '), ('%d,', '')):
                same(rows, fmt, delimiter)


def test_float_tables_and_the_formats_the_writers_use():
    rng = np.random.default_rng(1)
    nodes = np.column_stack((np.arange(1, 501), rng.normal(size=(500, 3)) * 1e3))
    same(nodes, ['%10d', '%14.6e', '%14.6e', '%14.6e'], ', ')
    same(nodes, ['%d', '%.9e', '%.9e', '%.9e'], ' ')
    same(rng.uniform(0, 1, size=(300, 3)), '%.6f', ' ')
    same(np.column_stack((np.arange(1, 301), rng.uniform(0, 1, size=(300, 3)))), ['%d'] + ['%.6f'] * 3, ' ')
    same(rng.normal(size=(200, 3)) * 1e-12, '%.17g', ' ')
    same(rng.normal(size=(100, 3)).astype(np.float32), '%.9e', ' ')


def test_special_values_and_integers_formatted_from_floats():
    rows = np.array([[0., -0., 1e300], [np.inf, -np.inf, np.nan], [-1.5, 2.5, 1e-300], [5., 6., 7.]])
    same(rows, '%.6e', ' ')
    same(rows[[0, 3]], '%d', ' ')                              # %d of a float truncates, as in np.savetxt
    same(np.array([[2 ** 62, -2 ** 62], [0, 1]], dtype=np.int64), '%d', ' ')


def test_block_boundaries(monkeypatch):
    monkeypatch.setattr(text_rows, 'BLOCK_ROWS', 7)
    rng = np.random.default_rng(2)
    for n in (1, 6, 7, 8, 14, 15, 50):
        same(rng.integers(0, 1000, size=(n, 3)), '%d', ' ')


def test_everything_else_is_left_to_np_savetxt():
    rng = np.random.default_rng(3)
    for rows, fmt in ((np.empty((0, 3), int), '%d'), (np.arange(5), '%d'), (rng.integers(0, 9, size=(4, 2)), '%d %d'),
                      (np.array([[True, False]]), '%d'), (rng.integers(0, 9, size=(4, 2)).astype(object), '%d')):
        want, got = both(rows, fmt)
        assert got == want
    for bad in (['%d'], ['%d', '%d', '%d']):                     # wrong number of formats: np.savetxt's own error
        with pytest.raises(Exception) as want:
            np.savetxt(io.StringIO(), np.ones((3, 2)), fmt=bad)
        with pytest.raises(type(want.value)):
            write_rows(io.StringIO(), np.ones((3, 2)), bad)


def test_binary_handle_goes_to_np_savetxt():
    rows = np.arange(12).reshape(4, 3)
    a, b = io.BytesIO(), io.BytesIO()
    np.savetxt(a, rows, fmt='%d')
    write_rows(b, rows, '%d')
    assert a.getvalue() == b.getvalue()


def test_text_mode_files_translate_newlines_the_same_way():
    rng = np.random.default_rng(4)
    rows = np.column_stack((np.arange(1, 1001), rng.normal(size=(1000, 3))))
    with tempfile.TemporaryDirectory() as folder:
        paths = [os.path.join(folder, name) for name in ('a.inp', 'b.inp')]
        for path, writer in zip(paths, (lambda f: np.savetxt(f, rows, fmt=['%d', '%.6e', '%.6e', '%.6e'], delimiter=', '),
                                        lambda f: write_rows(f, rows, ['%d', '%.6e', '%.6e', '%.6e'], ', '))):
            with open(path, 'w', encoding='ascii') as f:
                f.write('*Node\n')
                writer(f)
                f.write('*End\n')
        with open(paths[0], 'rb') as a, open(paths[1], 'rb') as b:
            assert a.read() == b.read()


def test_input_array_is_not_modified():
    rows = np.arange(12).reshape(4, 3)
    before = rows.copy()
    write_rows(io.StringIO(), rows, '%d')
    np.testing.assert_array_equal(rows, before)
