"""Integer-row and edge tables: the results of np.unique(axis=0), from one-dimensional keys.

np.unique(a, axis=0) sorts rows through a structured view, which is several times slower than
sorting one integer per row. For non-negative-shifted integer rows the two orders agree: pack each
row into one int64 with the first column most significant, sort the keys, and unpack. The unique
rows, the inverse and the counts are then equal to numpy's, element for element and in the same
order. When the columns are too wide to pack at once (a grain label in front of three node numbers
of a large mesh), the leading columns are replaced by their rank among the distinct values, which
keeps the order, and packing continues; the unique rows are then read back from the input. Rows
that still do not fit, and anything that is not integer, go to np.unique.
"""
import numpy as np

_KEY_LIMIT = 2 ** 62          # product of the column spans must stay below this to pack into int64


def _packing(rows):
    """(low, span, stride) per column when the rows pack into one int64 key, else None."""
    if rows.ndim != 2 or rows.dtype.kind not in 'iu' or rows.size == 0 or rows.shape[1] == 0:
        return None
    if rows.dtype.kind == 'u' and rows.dtype.itemsize >= 8:
        return None
    low = rows.min(axis=0)
    high = rows.max(axis=0)
    span = [int(h) - int(l) + 1 for l, h in zip(low, high)]
    total = 1
    for s in span:
        total *= s
        if total >= _KEY_LIMIT:
            return None
    stride, step = [], 1
    for s in reversed(span):
        stride.append(step)
        step *= s
    return [int(l) for l in low], span, stride[::-1]


def _ranked_key(rows):
    """int64 key per row, with the lexicographic order of the rows and equal exactly for equal rows.

    Columns are packed left to right; when the next column would overflow the key, the key so far is
    replaced by the rank of its value among the distinct values (order preserving) and packing goes
    on. None when even that does not fit, or when the rows are not integer.
    """
    if rows.ndim != 2 or rows.dtype.kind not in 'iu' or rows.size == 0 or rows.shape[1] == 0:
        return None
    if rows.dtype.kind == 'u' and rows.dtype.itemsize >= 8:
        return None
    key, span = None, 1
    for j in range(rows.shape[1]):
        column = rows[:, j]
        low = int(column.min())
        width = int(column.max()) - low + 1
        if width >= _KEY_LIMIT:                       # checked first: the shift below must not wrap around
            return None
        shifted = column.astype(np.int64) - low
        if key is None:
            key, span = shifted, width
            continue
        if span * width >= _KEY_LIMIT:
            distinct, key = np.unique(key, return_inverse=True)
            key = key.reshape(-1).astype(np.int64)
            span = len(distinct)
            if span * width >= _KEY_LIMIT:
                return None
        key = key * width + shifted
        span *= width
    return key


def _unique_from_key(rows, key, return_inverse, return_counts):
    """The unique rows of ``rows`` (and inverse, counts) in the order of ``key``."""
    perm = np.argsort(key)
    sorted_key = key[perm]
    first = np.empty(len(sorted_key), dtype=bool)
    first[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    result = [rows[perm[starts]]]
    if return_inverse:
        inverse = np.empty(len(key), dtype=np.intp)
        inverse[perm] = np.cumsum(first) - 1
        result.append(inverse)
    if return_counts:
        result.append(np.diff(np.append(starts, len(key))))
    return result[0] if len(result) == 1 else tuple(result)


def unique_rows(rows, return_inverse=False, return_counts=False):
    """np.unique(rows, axis=0, return_inverse=..., return_counts=...) for an integer array.

    Returns the unique rows (sorted lexicographically, dtype of ``rows``), then the inverse when asked
    (always one-dimensional, one entry per row) and then the counts when asked. Equal to numpy's
    results for integer rows; other arrays and rows that do not pack are handed to numpy.
    """
    rows = np.asarray(rows)
    packing = _packing(rows)
    if packing is None:
        key = _ranked_key(rows)
        if key is not None:
            return _unique_from_key(rows, key, return_inverse, return_counts)
        result = np.unique(rows, axis=0, return_inverse=return_inverse, return_counts=return_counts)
        if return_inverse:
            result = (result[0], np.asarray(result[1]).reshape(-1)) + tuple(result[2:])
        return result
    low, span, stride = packing
    key = np.zeros(len(rows), dtype=np.int64)
    for j in range(rows.shape[1]):
        key += (rows[:, j].astype(np.int64) - low[j]) * stride[j]
    found = np.unique(key, return_inverse=return_inverse, return_counts=return_counts)
    unique = found[0] if (return_inverse or return_counts) else found
    out = np.empty((len(unique), rows.shape[1]), dtype=rows.dtype)
    for j in range(rows.shape[1]):
        out[:, j] = (unique // stride[j]) % span[j] + low[j]
    if not (return_inverse or return_counts):
        return out
    return (out,) + tuple(found[1:])


def node_element_index(elements, n_points):
    """(numbers, offsets): the elements around each node, as the code of d3v2p0 builds them.

    ``numbers[offsets[n]:offsets[n + 1]]`` are the numbers of the elements (tets, triangles) that contain
    node ``n``, ascending; equal to ``np.argsort(elements.ravel(), kind='stable') // k`` for k nodes per
    element, with the matching cumulative counts. A counting sort (a compressed sparse row matrix)
    builds it in linear time. An element that repeats a node would be merged by the sparse format, and
    without scipy there is no such format: both cases use the argsort.
    """
    elements = np.asarray(elements)
    k = elements.shape[1]
    flat = elements.ravel()
    offsets = np.r_[0, np.cumsum(np.bincount(flat, minlength=n_points))]
    try:
        from scipy.sparse import coo_matrix
        owner = np.repeat(np.arange(len(flat) // k, dtype=np.int64), k)
        matrix = coo_matrix((np.ones(len(flat), dtype=np.int8), (flat, owner)),
                            shape=(n_points, len(flat) // k)).tocsr()
    except ImportError:
        matrix = None
    if matrix is None or matrix.nnz != len(flat):
        return np.argsort(flat, kind='stable') // k, offsets
    return matrix.indices.astype(np.int64), offsets


def triangle_edges(triangles):
    """The three edges of every triangle as sorted node pairs, shape (3 * n, 2), triangle by triangle."""
    f = np.asarray(triangles)
    return np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


class EdgeIncidence:
    """The edges of a triangle list and the triangles around each, as d3v2p0's facet code builds them.

    ``edges`` are the unique sorted node pairs (lexicographic order), ``inverse`` maps each of the 3 * n
    triangle edges (triangle by triangle) to its unique edge, ``counts`` is the number of triangles on
    each edge, ``order`` lists the triangle edges grouped by unique edge (stable) and ``offsets`` are the
    group starts. Triangle number of a triangle edge ``e`` is ``e // 3``.
    """

    def __init__(self, triangles):
        f = np.asarray(triangles)
        self.n_triangles = len(f)
        self.edges, self.inverse, self.counts = unique_rows(triangle_edges(f), True, True)
        self.order = np.argsort(self.inverse, kind='stable')
        self.offsets = np.r_[0, np.cumsum(self.counts)]
