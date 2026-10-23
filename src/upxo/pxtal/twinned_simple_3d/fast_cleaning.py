"""Optional numba kernels for the small-grain and spike removal of the MC
temporal slices (``TwinnedSimple3DBase.clean_temporal_slices``).

They give the same labels and counts as the numpy code, in a fraction of the
time. numba is optional: :func:`numba_available` says whether it is installed
and :func:`resolve_backend` turns a requested backend into the one used.
"""
from __future__ import annotations

import numpy as np

try:                                     # numba is optional
    import numba as _numba
    NUMBA_AVAILABLE = True
except ImportError:                      # pragma: no cover - depends on the environment
    _numba = None
    NUMBA_AVAILABLE = False

BACKENDS = ('auto', 'numba', 'numpy')


def numba_available() -> bool:
    return NUMBA_AVAILABLE


def resolve_backend(backend: str = 'auto') -> str:
    """'numba' or 'numpy' for a requested backend.

    ``'auto'`` is numba when it is installed, else numpy; ``'numpy'`` is the
    numpy code; ``'numba'`` raises ``ImportError`` if numba is not installed.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == 'numpy':
        return 'numpy'
    if NUMBA_AVAILABLE:
        return 'numba'
    if backend == 'numba':
        raise ImportError("backend='numba' needs numba, which is not installed.")
    return 'numpy'


if NUMBA_AVAILABLE:
    @_numba.njit(cache=False)
    def _best_neighbours(lgi, small, areas):
        """Largest face neighbour (lowest id on a tie) of every small grain."""
        nx, ny, nz = lgi.shape
        best = np.zeros(small.shape[0], dtype=np.int64)
        best_area = np.full(small.shape[0], -1, dtype=np.int64)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a = lgi[i, j, k]
                    if a <= 0:
                        continue
                    for d in range(3):
                        ii, jj, kk = i, j, k
                        if d == 0:
                            ii += 1
                        elif d == 1:
                            jj += 1
                        else:
                            kk += 1
                        if ii >= nx or jj >= ny or kk >= nz:
                            continue
                        b = lgi[ii, jj, kk]
                        if b <= 0 or b == a:
                            continue
                        if small[a]:
                            if areas[b] > best_area[a] or (areas[b] == best_area[a] and b < best[a]):
                                best_area[a] = areas[b]
                                best[a] = b
                        if small[b]:
                            if areas[a] > best_area[b] or (areas[a] == best_area[b] and a < best[b]):
                                best_area[b] = areas[a]
                                best[b] = a
        return best

    @_numba.njit(cache=False)
    def _spike_kernel(lgi, threshold):
        """Reassign spike voxels in place (C order, live array); returns how many."""
        nx, ny, nz = lgi.shape
        mask = np.zeros(lgi.shape, dtype=np.bool_)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    g = lgi[i, j, k]
                    if g <= 0:
                        continue
                    c = 0
                    if i > 0 and lgi[i - 1, j, k] == g:
                        c += 1
                    if i < nx - 1 and lgi[i + 1, j, k] == g:
                        c += 1
                    if j > 0 and lgi[i, j - 1, k] == g:
                        c += 1
                    if j < ny - 1 and lgi[i, j + 1, k] == g:
                        c += 1
                    if k > 0 and lgi[i, j, k - 1] == g:
                        c += 1
                    if k < nz - 1 and lgi[i, j, k + 1] == g:
                        c += 1
                    if c <= threshold:
                        mask[i, j, k] = True
        counts = np.zeros(lgi.max() + 1, dtype=np.int64)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    counts[lgi[i, j, k]] += 1
        n_spike = 0
        labels = np.zeros(6, dtype=np.int64)
        votes = np.zeros(6, dtype=np.int64)
        di = (1, -1, 0, 0, 0, 0)
        dj = (0, 0, 1, -1, 0, 0)
        dk = (0, 0, 0, 0, 1, -1)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not mask[i, j, k]:
                        continue
                    gid = lgi[i, j, k]
                    if counts[gid] <= 1:
                        continue
                    m = 0
                    for d in range(6):
                        ii, jj, kk = i + di[d], j + dj[d], k + dk[d]
                        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                            ng = lgi[ii, jj, kk]
                            if ng > 0 and ng != gid:
                                found = -1
                                for t in range(m):
                                    if labels[t] == ng:
                                        found = t
                                if found >= 0:
                                    votes[found] += 1
                                else:
                                    labels[m] = ng
                                    votes[m] = 1
                                    m += 1
                    if m > 0:
                        best = 0
                        for t in range(1, m):
                            if votes[t] > votes[best]:
                                best = t
                        lgi[i, j, k] = labels[best]
                        n_spike += 1
                        counts[gid] -= 1
        return n_spike


def merge_small_grains(lgi, min_grain_size, max_passes=50):
    """Merge every grain of at most ``min_grain_size`` voxels into its largest
    face neighbour (lowest id on a tie), repeated until none is left to merge.

    Same result as the numpy loop of ``clean_temporal_slices``. Returns
    ``(lgi, n_merged)`` with a new label array. Needs numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("merge_small_grains needs numba, which is not installed.")
    lgi = np.asarray(lgi)
    n_merged = 0
    for _ in range(max_passes):
        areas = np.bincount(lgi.ravel())
        small = np.zeros(areas.shape[0], dtype=np.bool_)
        small[1:] = (areas[1:] <= min_grain_size) & (areas[1:] > 0)
        small_gids = np.flatnonzero(small)
        if small_gids.size == 0:
            break
        best = _best_neighbours(lgi, small, areas.astype(np.int64))
        merges = {int(g): int(best[g]) for g in small_gids if best[g] > 0}
        if not merges:
            break
        lut = np.arange(int(lgi.max()) + 1, dtype=lgi.dtype)
        for gid, dominant in merges.items():
            root, seen = dominant, {gid}
            while root in merges and root not in seen:
                seen.add(root)
                root = merges[root]
            lut[gid] = root
        lgi = lut[lgi]
        n_merged += len(merges)
    return lgi, n_merged


def remove_spikes(lgi, same_count_threshold=1):
    """Reassign spike voxels (at most ``same_count_threshold`` face neighbours
    of their own grain) to the neighbouring grain with most faces, as
    ``StructureCleaner3D._fast_clean_spikes`` does. Returns ``(lgi, n_removed)``
    with a new label array. Needs numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("remove_spikes needs numba, which is not installed.")
    out = np.array(lgi, copy=True)
    return out, int(_spike_kernel(out, int(same_count_threshold)))


# ---------------------------------------------------------------------------
# The exporter's grain-to-element index
# ---------------------------------------------------------------------------
def group_voxels_by_label(lgi_flat):
    """``{label: 1-based voxel ids (int32, ascending)}`` for every positive
    label, from one stable sort instead of one scan per label. Same result as the
    per-label scan of ``AbaqusExporter3D.build_index``."""
    flat = np.asarray(lgi_flat).ravel()
    order = np.argsort(flat, kind='stable')
    cuts = np.flatnonzero(np.diff(flat[order])) + 1
    keys = flat[order[np.r_[0, cuts]]]
    return {int(k): (g + 1).astype(np.int32) for k, g in zip(keys, np.split(order, cuts)) if k > 0}
