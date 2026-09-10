"""
subsetting_2d.py
=================
Rectangular sub-domain extraction from a 2D EBSD grain-label field:
moving-rectangle tile generation and cropping. 2D sibling of
``subsetting.py`` (which does the same for 3D cuboids) -- reuses that
module's axis-interval math rather than duplicating it.
"""

import numpy as np

from upxo.pxtal.twinned_simple_3d.subsetting import _axis_intervals

_AXES = ('x', 'y')
# rg.lfi_ebsd carries the EBSD reader's native (ny, nx) axis order --
# row = y, column = x.
_AXIS_TO_SHAPE_IDX = {'x': 1, 'y': 0}


def _auto_tile_count(start_pct, stride_pct):
    """How many tiles to generate along one axis so the moving-rectangle
    sweep covers the full [start_pct, 100] extent, given only a start/
    stride percentage (the Moving Rectangle config has no explicit tile-
    count field). Deliberately generous -- any extra tile that would
    start at/past 100% is silently dropped by ``_axis_intervals`` itself,
    so overestimating here is harmless."""
    if stride_pct <= 0 or start_pct >= 100.0:
        return 1
    return max(1, int(np.ceil((100.0 - start_pct) / stride_pct)) + 1)


def generate_subset_tiles_2d(lgi_shape, start_pct, length_pct, stride_pct, overflow_policy='clip'):
    """
    Build the full set of rectangular subset tiles for a 2D EBSD grain-
    label field, per a moving-rectangle sweep.

    Parameters
    ----------
    lgi_shape : tuple (ny, nx)
    start_pct, length_pct, stride_pct : dict {'x': float, 'y': float}
        Percentages of the ORIGINAL domain extent along each axis.
    overflow_policy : 'clip' or 'skip'

    Returns
    -------
    list of dict, each: {'index': (i, j),
                          'pct': {'x': (s, e), 'y': (s, e)},
                          'px': {'x': (x0, x1), 'y': (y0, y1)}}
    'px' bounds are array-index, end-exclusive, ready for direct slicing
    of an (ny, nx)-shaped array.
    """
    axis_size = {a: lgi_shape[_AXIS_TO_SHAPE_IDX[a]] for a in _AXES}
    n_cuboids = {a: _auto_tile_count(start_pct[a], stride_pct[a]) for a in _AXES}
    per_axis = {
        a: _axis_intervals(n_cuboids[a], start_pct[a], length_pct[a], stride_pct[a], overflow_policy)
        for a in _AXES
    }

    tiles = []
    for i, sx, ex in per_axis['x']:
        for j, sy, ey in per_axis['y']:
            pct = {'x': (sx, ex), 'y': (sy, ey)}
            px = {}
            ok = True
            for a in _AXES:
                s, e = pct[a]
                axsz = axis_size[a]
                v0 = int(round(s / 100.0 * axsz))
                v1 = int(round(e / 100.0 * axsz))
                v1 = max(v1, v0 + 1)
                v1 = min(v1, axsz)
                v0 = min(v0, axsz - 1)
                if v1 <= v0:
                    ok = False
                    break
                px[a] = (v0, v1)
            if not ok:
                continue
            tiles.append({'index': (i, j), 'pct': pct, 'px': px})
    return tiles


def crop_lgi_2d(lgi, tile):
    """Crop a 2D grain-label field to `tile`'s pixel bounds."""
    y0, y1 = tile['px']['y']
    x0, x1 = tile['px']['x']
    return lgi[y0:y1, x0:x1].copy()


def subset_centroid_um(tile, lgi_shape, step_um):
    """Physical (x, y) centroid of a tile, in the same units as
    step_um (typically microns), rounded to 2 decimal places -- the
    centroid of the ORIGINAL (un-cropped) domain's coordinate system,
    not the crop's own local coordinates, so tiles from different
    subsets remain directly comparable by position.
    """
    axis_size = {a: lgi_shape[_AXIS_TO_SHAPE_IDX[a]] for a in _AXES}
    centroid = {}
    for a in _AXES:
        v0, v1 = tile['px'][a]
        centroid[a] = round((v0 + v1) / 2.0 * step_um, 2)
    return centroid['x'], centroid['y']
