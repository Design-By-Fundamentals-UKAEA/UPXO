"""Generic 2D geometry helpers."""

import numpy as np


def offset_local_to_global(local_coord, offset):
    """Apply a local-to-global coordinate offset.

    Parameters
    ----------
    local_coord : array-like, shape (..., 2)
        Local 2D coordinate or coordinate array.
    offset : array-like, shape (2,)
        2D offset to add to ``local_coord``.

    Returns
    -------
    numpy.ndarray
        Global coordinate array produced by adding ``offset``.
    """
    return np.asarray(local_coord) + np.asarray(offset)
