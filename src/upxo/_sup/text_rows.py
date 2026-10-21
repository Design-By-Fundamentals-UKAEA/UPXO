"""Write rows of numbers as text exactly as ``np.savetxt`` writes them, faster.

``np.savetxt`` formats one row at a time. For the long, narrow tables of a mesh deck (node and element
lists) that per-row work dominates the writing. ``write_rows`` formats a block of rows with one ``%``
operation on a format string repeated once per row; Python then does the number formatting in C.
Each number goes through the same ``%`` conversion ``np.savetxt`` applies to it (``np.float64`` is a
``float`` and an ``np.int64`` formats as the ``int`` it converts to), so the text is the same,
character for character, and a text-mode file translates the newlines in the same way.

Only the cases the mesh writers use are handled here (a 2-D integer or float array, a text handle,
a per-column format list or one format for every column). Anything else goes to ``np.savetxt``.
"""
import io

import numpy as np

BLOCK_ROWS = 100_000          # rows formatted per ``%`` operation; bounds the size of the temporary text


def _row_format(rows, fmt, delimiter):
    """The format of one row as np.savetxt builds it, or None when np.savetxt should do the writing."""
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0 or rows.dtype.kind not in 'iuf':
        return None
    columns = rows.shape[1]
    if isinstance(fmt, (list, tuple)):
        if len(fmt) != columns or not all(isinstance(f, str) for f in fmt):
            return None
        return delimiter.join(fmt)
    if isinstance(fmt, str) and fmt.count('%') == 1:
        return delimiter.join([fmt] * columns)
    return None


def write_rows(handle, rows, fmt, delimiter=' '):
    """``np.savetxt(handle, rows, fmt=fmt, delimiter=delimiter)`` with the same output, faster for a 2-D array.

    handle: a text-mode file (or ``io.StringIO``). fmt: one format for every column, or one per column.
    """
    rows = np.asarray(rows)
    row_format = _row_format(rows, fmt, delimiter) if isinstance(handle, io.TextIOBase) else None
    if row_format is None:
        np.savetxt(handle, rows, fmt=fmt, delimiter=delimiter)
        return
    line = row_format + '\n'
    for start in range(0, len(rows), BLOCK_ROWS):
        block = rows[start:start + BLOCK_ROWS]
        handle.write(line * len(block) % tuple(block.ravel().tolist()))
