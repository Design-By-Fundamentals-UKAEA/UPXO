"""Voxel masks of grain boundaries, junction lines and junction points.

Each function takes a 3-D grid of integer grain labels and returns a boolean
grid of the same shape that marks the voxels touching the feature.

Void voxels (label 0) and positions beyond the grid are both read as label 0.
Together they count as one further label wherever labels are counted, so a
boundary between two grains that lies on a face of the grid has three labels
around it and is marked as a junction line.
"""
import numpy as np

__all__ = ['shift_3d', 'grain_boundary_voxels', 'junction_line_voxels',
           'junction_point_voxels']

# The two grid axes perpendicular to an edge that runs along X, Y or Z.
_EDGE_AXES = ((1, 2), (0, 2), (0, 1))


def shift_3d(arr, da, db, dc):
    """Shift ``arr`` by ``(da, db, dc)`` voxels along its three axes.

    ``res[i, j, k] == arr[i - da, j - db, k - dc]``; positions that receive
    nothing are 0.
    """
    res = np.zeros_like(arr)
    s_dest = [slice(max(0, d), arr.shape[i] + min(0, d)) for i, d in enumerate((da, db, dc))]
    s_src = [slice(max(0, -d), arr.shape[i] + min(0, -d)) for i, d in enumerate((da, db, dc))]
    res[tuple(s_dest)] = arr[tuple(s_src)]
    return res


def _distinct_label_count(corner_grids):
    """Number of distinct labels among the given same-shaped grids, per voxel."""
    stacked = np.stack([g.ravel() for g in corner_grids], axis=1)
    ordered = np.sort(stacked, axis=1)
    return 1 + (np.diff(ordered, axis=-1) != 0).sum(axis=-1)


def _offset(axes_and_steps):
    offset = [0, 0, 0]
    for axis, step in axes_and_steps:
        offset[axis] = step
    return tuple(offset)


def grain_boundary_voxels(labels):
    """Voxels that share a face with a voxel of a different grain.

    Only pairs of positive labels count; void (label 0 or negative) voxels
    are never part of a boundary.
    """
    labels = np.asarray(labels)
    marked = np.zeros(labels.shape, dtype=bool)
    for ax in range(3):
        lower = [slice(None)] * 3
        lower[ax] = slice(None, -1)
        upper = [slice(None)] * 3
        upper[ax] = slice(1, None)
        a = labels[tuple(lower)]
        b = labels[tuple(upper)]
        differ = (a != b) & (a > 0) & (b > 0)
        marked[tuple(lower)] |= differ
        marked[tuple(upper)] |= differ
    return marked


def junction_line_voxels(labels):
    """Voxels around every grid edge where three or more labels meet.

    An edge has four voxels around it, so each qualifying edge marks four
    voxels.
    """
    labels = np.asarray(labels)
    marked = np.zeros(labels.shape, dtype=bool)
    for a, b in _EDGE_AXES:
        corners = [labels,
                   shift_3d(labels, *_offset([(a, 1)])),
                   shift_3d(labels, *_offset([(b, 1)])),
                   shift_3d(labels, *_offset([(a, 1), (b, 1)]))]
        on_edge = (_distinct_label_count(corners) >= 3).reshape(labels.shape)
        marked |= (on_edge
                   | shift_3d(on_edge, *_offset([(a, -1)]))
                   | shift_3d(on_edge, *_offset([(b, -1)]))
                   | shift_3d(on_edge, *_offset([(a, -1), (b, -1)])))
    return marked


def junction_point_voxels(labels):
    """Voxels around every grid vertex where three or more labels meet.

    A vertex has eight voxels around it, so each qualifying vertex marks
    eight voxels.
    """
    labels = np.asarray(labels)
    corners = [shift_3d(labels, da, db, dc)
               for da in (0, 1) for db in (0, 1) for dc in (0, 1)]
    at_vertex = (_distinct_label_count(corners) >= 3).reshape(labels.shape)
    marked = np.zeros(labels.shape, dtype=bool)
    for da in (0, -1):
        for db in (0, -1):
            for dc in (0, -1):
                marked |= shift_3d(at_vertex, da, db, dc)
    return marked
