import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - keeps gridops importable without numba.
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


@njit
def _count_labels(label_array_flat, label_to_index):
    """Count voxels for each requested label index."""
    counts = np.zeros(label_to_index.max()+1, dtype=np.int64)
    for t in range(label_array_flat.size):
        label = label_array_flat[t]
        if 0 <= label < label_to_index.size:
            idx = label_to_index[label]
            if idx >= 0:
                counts[idx] += 1
    return counts


@njit
def _fill_label_coords(label_array, label_to_index, counts):
    """Fill coordinate arrays for each requested label index."""
    n0, n1, n2 = label_array.shape
    out_i = [np.empty(c, dtype=np.int64) for c in counts]
    out_j = [np.empty(c, dtype=np.int64) for c in counts]
    out_k = [np.empty(c, dtype=np.int64) for c in counts]
    pos = np.zeros(len(counts), dtype=np.int64)

    for k in range(n2):
        for j in range(n1):
            for i in range(n0):
                label = label_array[i, j, k]
                if 0 <= label < label_to_index.size:
                    idx = label_to_index[label]
                    if idx >= 0:
                        w = pos[idx]
                        out_i[idx][w] = i
                        out_j[idx][w] = j
                        out_k[idx][w] = k
                        pos[idx] += 1
    return out_i, out_j, out_k


def find_label_voxel_locs(label_array, labels=None, dtype=np.int32):
    """Return a mapping from label ID to voxel coordinates."""
    label_array = np.asarray(label_array)
    if labels is None:
        labels = np.unique(label_array)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return {}

    max_label = int(max(labels.max(), label_array.max()))
    label_to_index = -np.ones(max_label+1, dtype=np.int64)
    for idx, label in enumerate(labels):
        if label >= 0:
            label_to_index[label] = idx

    counts = _count_labels(label_array.ravel(order="C"), label_to_index)
    out_i, out_j, out_k = _fill_label_coords(label_array, label_to_index,
                                             counts)

    label_locs = {}
    for idx, label in enumerate(labels):
        coords = np.column_stack((
            out_i[idx].astype(dtype, copy=False),
            out_j[idx].astype(dtype, copy=False),
            out_k[idx].astype(dtype, copy=False)
        ))
        label_locs[int(label)] = coords
    return label_locs


def compute_label_bounds(label_locs, array_shape):
    """Return tight and one-voxel-extended bounds for label coordinates."""
    zmins = np.array([loc[:, 0].min() for loc in label_locs.values()])
    zmaxs = np.array([loc[:, 0].max() for loc in label_locs.values()])
    zmins_ex = zmins - (zmins > 0)*1
    zmaxs_ex = zmaxs + (zmaxs < array_shape[0]-1)*1

    ymins = np.array([loc[:, 1].min() for loc in label_locs.values()])
    ymaxs = np.array([loc[:, 1].max() for loc in label_locs.values()])
    ymins_ex = ymins - (ymins > 0)*1
    ymaxs_ex = ymaxs + (ymaxs < array_shape[1]-1)*1

    xmins = np.array([loc[:, 2].min() for loc in label_locs.values()])
    xmaxs = np.array([loc[:, 2].max() for loc in label_locs.values()])
    xmins_ex = xmins - (xmins > 0)*1
    xmaxs_ex = xmaxs + (xmaxs < array_shape[2]-1)*1

    bounds = {'xmins': xmins, 'xmaxs': xmaxs,
              'ymins': ymins, 'ymaxs': ymaxs,
              'zmins': zmins, 'zmaxs': zmaxs}
    bounds_ex = {'xmins': xmins_ex, 'xmaxs': xmaxs_ex,
                 'ymins': ymins_ex, 'ymaxs': ymaxs_ex,
                 'zmins': zmins_ex, 'zmaxs': zmaxs_ex}
    return bounds, bounds_ex


def extract_bounded_subarray(array, bounds, label_id):
    """Return a bounded subarray using one-based label-id indexing."""
    idx = int(label_id) - 1
    xsl = slice(bounds['xmins'][idx], bounds['xmaxs'][idx]+1)
    ysl = slice(bounds['ymins'][idx], bounds['ymaxs'][idx]+1)
    zsl = slice(bounds['zmins'][idx], bounds['zmaxs'][idx]+1)
    return np.asarray(array)[zsl, ysl, xsl]


def map_values_to_labels(label_array, labels, values_by_label,
                         default_value=-1):
    """Map per-label values onto a labelled array."""
    mapped = np.array(label_array, copy=True)
    for label in labels:
        if label in values_by_label:
            mapped[mapped == label] = values_by_label[label]
        else:
            mapped[mapped == label] = default_value
    return mapped


def slice_scalar_field_3d(array, normal='x', index=0):
    """Return an axis-normal slice from a 3D scalar field."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError('Expected a 3D scalar field.')
    normal = normal.lower()
    index = int(index)
    axis_by_normal = {'z': 0, 'y': 1, 'x': 2}
    if normal not in axis_by_normal:
        raise ValueError("normal must be one of 'x', 'y' or 'z'.")
    axis = axis_by_normal[normal]
    if index < 0 or index >= array.shape[axis]:
        raise ValueError('Invalid slice index specified.')
    if normal == 'x':
        return array[:, :, index]
    if normal == 'y':
        return array[:, index, :]
    return array[index, :, :]


def mask_labels(array, labels, mask_value=-10, valid_labels=None):
    """Mask selected labels in an array."""
    masked = np.array(array, copy=True).astype(int)
    labels = np.atleast_1d(labels)
    valid = None if valid_labels is None else set(valid_labels)
    skipped = []
    for label in labels:
        if valid is None or label in valid:
            masked[masked == label] = mask_value
        else:
            skipped.append(label)
    return masked, mask_value, skipped


def bresenham_line_3d(start, end):
    """Generate integer locations on a 3D Bresenham line."""
    x1, y1, z1 = start
    x2, y2, z2 = end
    points = [(x1, y1, z1)]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x1, y1, z1))
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x1, y1, z1))
    return points


def values_along_line_3d(array, start, end):
    """Return array values along a 3D Bresenham line."""
    array = np.asarray(array)
    locs = np.asarray(bresenham_line_3d(start, end), dtype=int)
    if np.any(locs < 0) or np.any(locs >= np.asarray(array.shape)):
        raise IndexError('Line contains locations outside array bounds.')
    return array[locs[:, 0], locs[:, 1], locs[:, 2]]


def intercept_properties_from_values(values):
    """Return intercept grain-size properties from sampled labels."""
    values = np.asarray(values)
    values_unique = np.unique(values)
    counts = np.array([np.argwhere(values == value).squeeze().size
                       for value in values_unique])
    return {'ng': values_unique.size,
            'nv': counts,
            'igs': counts.mean(),
            'igs_median': np.median(counts),
            'igs_range': np.ptp(counts),
            'igs_std': counts.std(),
            'igs_var': counts.var(),
            'sv': values,
            'sv_unique': values_unique}


def intercept_summary_from_values(values, metric='mean', minimum=True,
                                  maximum=True, std=True, variance=True):
    """Return summary intercept grain-size statistics from sampled labels."""
    values = np.asarray(values)
    values_unique = np.unique(values)
    counts = np.array([np.argwhere(values == value).squeeze().size
                       for value in values_unique])
    igs = counts.mean() if metric in ('mean', 'average', 'avg') else None
    if metric in ('med', 'median'):
        igs = np.median(counts)
    return {'igs': igs,
            'metric': metric,
            'min': counts.min() if minimum else None,
            'max': counts.max() if maximum else None,
            'std': counts.std() if std else None,
            'var': counts.var() if variance else None}


def opposing_boundary_points(shape, plane='z', start_skip1=0, start_skip2=0,
                             incr1=2, incr2=2, inclination='none',
                             inclination_extent=0, shift_seperately=False,
                             shift_starts=False, shift_ends=True,
                             start_shift=0, end_shift=0):
    """Return start/end points on opposing boundary planes of a 3D grid."""
    plane = plane.lower()
    start_skip1 = int(start_skip1)
    start_skip2 = int(start_skip2)
    if start_skip1 < 0 or start_skip2 < 0:
        raise ValueError('start_skip1 and start_skip2 must be non-negative integers.')
    nz, ny, nx = shape
    incr1 = 1 if incr1 in (None, 0) else int(incr1)
    incr2 = 1 if incr2 in (None, 0) else int(incr2)
    if incr1 < 1 or incr2 < 1:
        raise ValueError('incr1 and incr2 must be positive integers.')

    if plane == 'z':
        yloc, xloc = np.meshgrid(np.arange(start_skip1, ny, incr1),
                                 np.arange(start_skip2, nx, incr2),
                                 indexing='ij')
        start_points = np.column_stack((np.zeros(yloc.size, dtype=int),
                                        yloc.ravel(), xloc.ravel()))
        end_points = np.column_stack((np.full(yloc.size, nz-1, dtype=int),
                                      yloc.ravel(), xloc.ravel()))
    elif plane == 'y':
        zloc, xloc = np.meshgrid(np.arange(start_skip1, nz, incr1),
                                 np.arange(start_skip2, nx, incr2),
                                 indexing='ij')
        start_points = np.column_stack((zloc.ravel(),
                                        np.zeros(zloc.size, dtype=int),
                                        xloc.ravel()))
        end_points = np.column_stack((zloc.ravel(),
                                      np.full(zloc.size, ny-1, dtype=int),
                                      xloc.ravel()))
    elif plane == 'x':
        zloc, yloc = np.meshgrid(np.arange(start_skip1, nz, incr1),
                                 np.arange(start_skip2, ny, incr2),
                                 indexing='ij')
        start_points = np.column_stack((zloc.ravel(), yloc.ravel(),
                                        np.zeros(zloc.size, dtype=int)))
        end_points = np.column_stack((zloc.ravel(), yloc.ravel(),
                                      np.full(zloc.size, nx-1, dtype=int)))
    else:
        raise ValueError("plane must be one of 'x', 'y' or 'z'.")

    if inclination == 'none':
        return start_points, end_points
    if inclination == 'constant' and inclination_extent == 0:
        return start_points, end_points
    if inclination == 'constant' and inclination_extent != 0:
        if shift_seperately:
            start_points = np.roll(start_points, start_shift, axis=0)
            end_points = np.roll(end_points, end_shift, axis=0)
        else:
            start_points = np.roll(start_points, inclination_extent, axis=0)
            end_points = np.roll(end_points, -inclination_extent, axis=0)
        return start_points, end_points
    if inclination == 'random':
        np.random.shuffle(start_points)
        np.random.shuffle(end_points)
        return start_points, end_points
    raise ValueError("inclination must be 'none', 'constant' or 'random'.")


_AXIS_TO_PLANE = {'x': 'z', 'y': 'y', 'z': 'x'}
_AXIS_TO_INDEX = {'x': 0, 'y': 1, 'z': 2}


def axis_intercept_grain_size(lgi, voxel_size, axis, n_lines_target=300,
                              phase_array=None, valid_phase_ids=None):
    """Mean linear-intercept grain size along one axis of a 3D labeled grid,
    in physical units.

    Reuses this module's existing line-sampling/intercept primitives
    (opposing_boundary_points, values_along_line_3d,
    intercept_properties_from_values) rather than re-implementing the
    intercept-counting algorithm -- see their docstrings for the underlying
    method (sample parallel lines spanning the full extent of the grid along
    the chosen axis; for each line, measure how many voxels belong to each
    grain it passes through; pool those counts across every sampled line).
    Label 0 (background/void) is excluded from the pooled statistics.

    Axis-order note: opposing_boundary_points' `plane` argument assumes a
    (z, y, x) axis order (its own historical convention: plane='z' sweeps
    array axis 0, plane='x' sweeps array axis 2), whereas `lgi` here follows
    UPXO's fm_steel_3d convention of (x, y, z) axis order, i.e. shape
    (NX, NY, NZ) with axis 0 = X and axis 2 = Z. Concretely this means axis
    'x' and axis 'z' are *not* the same as the like-named `plane` value --
    'y' happens to coincide (axis 1 either way), but 'x'/'z' are swapped.
    This function takes `axis` in the natural ('x', 'y', 'z') sense and
    translates it to the correct `plane` value internally so callers never
    need to reason about the mismatch themselves.

    Parameters
    ----------
    lgi : np.ndarray
        3D labeled grid, shape (NX, NY, NZ). 0 = background/void, excluded.
    voxel_size : float
        Physical size of one (assumed isotropic) voxel edge. Every returned
        length statistic is voxel_count * voxel_size.
    axis : str
        'x', 'y', or 'z' -- which axis to sample lines along.
    n_lines_target : int, optional
        Approximate number of sample lines to use, spread evenly over the
        cross-section perpendicular to `axis`. Default 300 -- enough for a
        stable estimate without scanning every possible line on large grids.
    phase_array : np.ndarray, optional
        Per-voxel phase ID, same shape as `lgi`. When given together with
        `valid_phase_ids`, any voxel whose phase is not in `valid_phase_ids`
        is excluded from the pooled statistics -- exactly like label 0 is --
        regardless of what label `lgi` happens to carry there. This lets a
        caller reject intercepts made with a non-target-phase grain (e.g.
        retained austenite domains in a `lgi` that otherwise labels blocks)
        by phase identity rather than relying on those domains happening to
        already be label 0 in `lgi`.
    valid_phase_ids : sequence, optional
        Phase IDs considered valid/countable; only meaningful together with
        `phase_array`. Both must be given together to have any effect.

    Returns
    -------
    dict
        {'mean', 'median', 'std', 'min', 'max'} (all in physical units),
        plus 'n_segments' (grain crossings pooled) and 'n_lines' (lines
        sampled) as plain counts, plus 'raw_segments' -- the full pooled
        per-crossing length array (physical units, shape (n_segments,)) the
        summary statistics above were computed from, for callers that need
        e.g. IQR/percentiles rather than just mean/median/min/max (matches
        directional_intercept_grain_size's own 'raw_segments' key/purpose).
    """
    axis = axis.lower()
    if axis not in _AXIS_TO_PLANE:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    plane = _AXIS_TO_PLANE[axis]
    axis_idx = _AXIS_TO_INDEX[axis]

    shape = lgi.shape
    other_dims = [shape[i] for i in range(3) if i != axis_idx]
    n_possible_lines = other_dims[0] * other_dims[1]
    incr = max(1, int(round((n_possible_lines / max(1, n_lines_target)) ** 0.5)))

    starts, ends = opposing_boundary_points(shape, plane=plane, incr1=incr, incr2=incr)

    use_phase_filter = phase_array is not None and valid_phase_ids is not None
    pooled_counts = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        values = values_along_line_3d(lgi, start, end)
        if use_phase_filter:
            phase_values = values_along_line_3d(phase_array, start, end)
            values = values.copy()
            values[~np.isin(phase_values, valid_phase_ids)] = 0
        props = intercept_properties_from_values(values)
        nonzero = props['sv_unique'] != 0
        if nonzero.any():
            pooled_counts.extend(props['nv'][nonzero].tolist())

    if not pooled_counts:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
               'n_segments': 0, 'n_lines': int(len(starts)),
               'raw_segments': np.empty(0, dtype=np.float64)}

    seg = np.asarray(pooled_counts, dtype=np.float64) * float(voxel_size)
    return {
        'mean': float(seg.mean()), 'median': float(np.median(seg)),
        'std': float(seg.std()), 'min': float(seg.min()), 'max': float(seg.max()),
        'n_segments': int(seg.size), 'n_lines': int(len(starts)),
        'raw_segments': seg,
    }


def directional_intercept_grain_size(lgi, voxel_size, direction, n_lines_target=300,
                                     phase_array=None, valid_phase_ids=None):
    """Mean linear-intercept grain size along an arbitrary 3D direction, in
    physical units. Same statistics/output shape as axis_intercept_grain_size
    (and reuses its underlying primitives -- values_along_line_3d,
    intercept_properties_from_values, the same optional phase_array/
    valid_phase_ids masking), but not restricted to the X/Y/Z axes: `direction`
    can be any 3-vector (need not be unit -- it is normalized internally).

    This is a *separate* function from axis_intercept_grain_size (which is
    left completely untouched) specifically so a caller measuring along a
    block's/packet's own real slicing-plane normal -- generally not aligned
    with any coordinate axis -- gets a geometrically correct answer instead
    of one derived from the axis-aligned statistics.

    Sampling method: opposing_boundary_points only supports axis-aligned
    sweeps, so sample lines here are generated by intersecting candidate
    lines (parallel to `direction`, spread over a regular grid in the plane
    perpendicular to it, centered on the grid's centroid) against the array's
    bounding box via the standard ray/box "slab" method. Candidates that miss
    the box entirely are dropped -- this is what naturally limits sampling to
    the box's true (generally non-rectangular, direction-dependent) footprint
    without needing to compute that polygon explicitly. `n_lines_target` is
    therefore an approximate target, same as in axis_intercept_grain_size.

    Parameters
    ----------
    lgi : np.ndarray
        3D labeled grid, shape (NX, NY, NZ). 0 = background/void, excluded.
    voxel_size : float
        Physical size of one (assumed isotropic) voxel edge.
    direction : array-like, shape (3,)
        Direction to sample lines along, in the same (x, y, z) voxel-index
        axis order as `lgi` (not the (z, y, x) `plane` convention used
        elsewhere in this module for axis-aligned sweeps).
    n_lines_target : int, optional
        Approximate number of sample lines. Default 300.
    phase_array, valid_phase_ids : optional
        See axis_intercept_grain_size -- identical semantics.

    Returns
    -------
    dict
        Same shape as axis_intercept_grain_size's return value.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-12:
        raise ValueError("direction must be a nonzero vector.")
    d = d / norm_d

    shape = np.asarray(lgi.shape, dtype=np.float64)

    # Orthonormal in-plane basis (u, v) perpendicular to d.
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, d) * d
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)

    center = (shape - 1.0) / 2.0
    radius = float(np.linalg.norm(center))  # half-diagonal: generous cross-section bound

    n_side = max(1, int(np.ceil(np.sqrt(max(1, n_lines_target) * 1.3))))
    offsets = np.linspace(-radius, radius, n_side)

    use_phase_filter = phase_array is not None and valid_phase_ids is not None
    pooled_counts = []
    n_lines = 0

    for uo in offsets:
        for vo in offsets:
            origin = center + uo * u + vo * v

            t_enter, t_exit = -np.inf, np.inf
            degenerate_miss = False
            for i in range(3):
                if abs(d[i]) > 1e-12:
                    t1 = (0.0 - origin[i]) / d[i]
                    t2 = (shape[i] - 1.0 - origin[i]) / d[i]
                    t_enter = max(t_enter, min(t1, t2))
                    t_exit = min(t_exit, max(t1, t2))
                elif origin[i] < -0.5 or origin[i] > shape[i] - 0.5:
                    degenerate_miss = True
                    break
            if degenerate_miss or t_enter > t_exit:
                continue

            start = origin + t_enter * d
            end = origin + t_exit * d
            start = np.clip(np.round(start), 0, shape - 1).astype(int)
            end = np.clip(np.round(end), 0, shape - 1).astype(int)
            if np.array_equal(start, end):
                continue

            n_lines += 1
            values = values_along_line_3d(lgi, start, end)
            if use_phase_filter:
                phase_values = values_along_line_3d(phase_array, start, end)
                values = values.copy()
                values[~np.isin(phase_values, valid_phase_ids)] = 0
            props = intercept_properties_from_values(values)
            nonzero = props['sv_unique'] != 0
            if nonzero.any():
                pooled_counts.extend(props['nv'][nonzero].tolist())

    if not pooled_counts:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
               'n_segments': 0, 'n_lines': int(n_lines),
               'raw_segments': np.empty(0, dtype=np.float64)}

    # Bresenham advances by exactly 1 voxel_size per sampled point only along an
    # axis-aligned line. Along a tilted `d`, the dominant-axis stepping used by
    # bresenham_line_3d takes max(|d_x|,|d_y|,|d_z|)-normalized steps, so the true
    # physical distance between consecutive sampled points is 1/max(|d_i|) voxel
    # units, not 1 -- e.g. sqrt(2) for an exact 45 degree diagonal in a plane,
    # sqrt(3) for a full 3D diagonal. Without this factor, raw voxel counts along
    # a tilted direction would understate the true physical intercept length.
    step_length_factor = 1.0 / np.max(np.abs(d))
    seg = np.asarray(pooled_counts, dtype=np.float64) * float(voxel_size) * step_length_factor
    return {
        'mean': float(seg.mean()), 'median': float(np.median(seg)),
        'std': float(seg.std()), 'min': float(seg.min()), 'max': float(seg.max()),
        'n_segments': int(seg.size), 'n_lines': int(n_lines),
        # Extra key beyond axis_intercept_grain_size's shape -- lets a caller
        # measuring multiple distinct directions (e.g. one per PAG's slicing
        # normal) combine them into one statistically correct pooled result
        # (concatenate raw_segments across calls) instead of averaging
        # per-direction summary stats, which would not be mathematically valid
        # for median/std and only valid for mean if weighted by n_segments.
        'raw_segments': seg,
    }


def _group_ids_by_direction(id_to_vector: dict, decimals: int = 4) -> dict:
    """{canonical-direction-tuple: [ids sharing it]} -- collapses near-identical
    directions (e.g. every block in one packet already shares one exact
    vector) so a caller can run one expensive directional sampling pass per
    distinct direction instead of one per id.

    +n and -n are canonicalized to the same key (they describe the same
    slicing plane / sampling line, just opposite normal signs), by flipping
    sign so the largest-magnitude component is always positive.
    """
    groups: dict = {}
    for item_id, vec in id_to_vector.items():
        v = np.asarray(vec, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        v = v / norm
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        key = tuple(np.round(v, decimals).tolist())
        groups.setdefault(key, []).append(item_id)
    return groups


def local_neighborhood(array, loc, radius=1):
    """Return a bounded cubic neighborhood around a location."""
    array = np.asarray(array)
    loc = np.asarray(loc, dtype=int)
    if loc.size != array.ndim:
        raise ValueError('Location dimensionality must match array.')
    if np.any(loc < 0) or np.any(loc >= np.asarray(array.shape)):
        raise ValueError('Invalid location specification.')
    slices = tuple(slice(max(0, i-radius), min(size, i+radius+1))
                   for i, size in zip(loc, array.shape))
    return array[slices]


def neighbor_labels_at_location(array, loc, radius=1):
    """Return labels in a local neighborhood except the center label."""
    loc = tuple(np.asarray(loc, dtype=int))
    neigh = local_neighborhood(array, loc, radius=radius)
    return set(np.unique(neigh)) - {np.asarray(array)[loc]}


def plane_slice(array, plane='xy', index=0):
    """Return a slice along one of the three fundamental planes."""
    array = np.asarray(array)
    plane = plane.lower()
    if plane not in ('xy', 'yx', 'yz', 'zy', 'xz', 'zx'):
        raise ValueError('Invalid axis specification.')
    if plane in ('xy', 'yx'):
        return array[index, :, :]
    if plane in ('yz', 'zy'):
        return array[:, :, index]
    return array[:, index, :]


def relabel_multistate_slice_2d(scalar_slice, connectivity=2):
    """Relabel connected regions independently for each value in a 2D slice."""
    from skimage.measure import label as skim_label
    scalar_slice = np.asarray(scalar_slice)
    if scalar_slice.ndim != 2:
        raise ValueError('Expected a 2D scalar slice.')
    if connectivity in (4, 8):
        connectivity = int(connectivity/4)
    if connectivity not in (1, 2):
        raise ValueError(f'Input must be in (1, 2, 4, 8). Recieved {connectivity}')

    lgi = None
    for i, value in enumerate(np.unique(scalar_slice)):
        binary = (scalar_slice == value).astype(np.uint8)
        labels, _ = skim_label(binary, return_num=True,
                               connectivity=connectivity)
        if i == 0:
            lgi = labels
        else:
            labels[labels > 0] += lgi.max()
            lgi = lgi + labels
    return lgi


def grid_axis(vmin, vmax, vinc):
    """Return one regularly-spaced grid axis."""
    return np.arange(vmin, vmax, vinc)


def domain_volume_from_axes(xaxis, yaxis, zaxis):
    """Return domain volume from grid axes."""
    return np.asarray(xaxis).size*np.asarray(yaxis).size*np.asarray(zaxis).size


def extract_random_subdomains(array, subdomain_shape, n=1, rng=None):
    """Extract random subdomains from a 3D array."""
    array = np.asarray(array)
    subdomain_shape = tuple(int(v) for v in subdomain_shape)
    if array.ndim != len(subdomain_shape):
        raise ValueError('Subdomain dimensionality must match array.')
    limits = tuple(size-sub+1 for sub, size in zip(subdomain_shape,
                                                   array.shape))
    if any(limit < 1 for limit in limits):
        raise ValueError('Subdomain shape cannot exceed array shape.')
    rng = np.random if rng is None else rng
    subdomains = []
    starts = []
    for _ in range(int(n)):
        start = np.asarray(rng.randint(0, limits), dtype=int)
        slices = tuple(slice(s, s+sub)
                       for s, sub in zip(start, subdomain_shape))
        subdomains.append(array[slices])
        starts.append(tuple(start.tolist()))
    return subdomains, starts


def make_grid_pxtal(distribution_type, **kwargs):
    """Build a grid-based polycrystal seed layout for the given distribution type."""
    if distribution_type == 'rectgrid':
        method_bounds = kwargs['method_bounds']
        usefactor     = kwargs['usefactor']
        xlimits       = kwargs['xlimits']
        ylimits       = kwargs['ylimits']
        if method_bounds == 'frombounds':
            xmin, xmax = xlimits
            ymin, ymax = ylimits
            if usefactor == True:
                grid_mul_factor = kwargs['grid_mul_factor']
                x = np.linspace(xmin, xmax, int((xmax - xmin)*grid_mul_factor))
                y = np.linspace(ymin, ymax, int((ymax - ymin)*grid_mul_factor))
            else:
                x = np.linspace(xmin, xmax, int(xmax - xmin))
                y = np.linspace(ymin, ymax, int(ymax - ymin))
            grid_spacing_x = min([x[1] - x[0], y[1] - y[0]])
            grid_spacing_y = grid_spacing_x
            x, y = np.meshgrid(x, y)
        return x, y, grid_spacing_x
    elif distribution_type == 'random':
        # arguments::: method_bounds: 'bounded_by_pxtal', 'user_data'
        if method_bounds == 'bounded_by_shmulpol':
            # arguments::: shapely object of type "MultiPolygon", like pxtal
            # arg method_bounds MUST be followed by keyword args mulpol having the 
            # shapely multi-polygon object
            import shapely
            xmin, xmax = min(mulpol.envelope.boundary.xy[0]), max(mulpol.envelope.boundary.xy[0])
            ymin, ymax = min(mulpol.envelope.boundary.xy[1]), max(mulpol.envelope.boundary.xy[1])
        elif method_bounds == 'user_data':
            # arguments: xlimits = [xmin, xmax], ylimits = [ymin, ymax]
            # arg method_bounds MUST be followed by keyword args xlimits and ylimits
            xmin, xmax = xlimits
            ylim, ymax = ylimits
            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if usefactor == True:
            pass
        elif usefactor == False:
            if distribution_subtype == 'uniform':
                # Access: make_grid_pxtal('random', **kwargs)
                # RANDOM RANDOM
                x = np.random.random((domain_size_count, domain_size_count))
                y = np.random.random((domain_size_count, domain_size_count))
            elif distribution_subtype == 'power':
                # RANDOM POWER
                exponent = 3
                x = np.reshape(np.random.power(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
                y = np.reshape(np.random.power(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
            elif distribution_subtype == 'exponential':
                # RANDOM EXPONENTIAL
                exponent = 1
                x = np.reshape(np.random.exponential(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
                y = np.reshape(np.random.exponential(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
        x = x/x.max()
        y = y/y.max()
        x = x*(xmax-xmin) + xmin
        y = y*(ymax-ymin) + ymin
