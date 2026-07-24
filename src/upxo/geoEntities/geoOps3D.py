"""Generic 3D geometry helpers."""

import numpy as np


def local_global_offset(bounds, gid):
    """Return z-y-x offset from bounds for a one-based grain ID.

    Parameters
    ----------
    bounds : dict
        Bounds dictionary containing ``'zmins'``, ``'ymins'``, and ``'xmins'``
        arrays indexed by zero-based grain position.
    gid : int
        One-based grain identifier.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Offset ordered as ``[zmin, ymin, xmin]``.
    """
    idx = int(gid) - 1
    return np.array([bounds['zmins'][idx],
                     bounds['ymins'][idx],
                     bounds['xmins'][idx]])


def offset_local_to_global(local_coord, offset):
    """Apply a z-y-x local-to-global coordinate offset.

    Parameters
    ----------
    local_coord : array-like, shape (..., 3)
        Local 3D coordinate or coordinate array.
    offset : array-like, shape (3,)
        Offset ordered consistently with ``local_coord``.

    Returns
    -------
    numpy.ndarray
        Global coordinate array produced by adding ``offset``.
    """
    return np.asarray(local_coord) + np.asarray(offset)
