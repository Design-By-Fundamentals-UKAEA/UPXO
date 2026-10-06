"""Parallel triangle-intersection search with the same result as d3v2p0.

The candidate search and the pair test are d3v2p0's. Query triangles are
split into chunks that worker processes search independently; every pair is
decided by the same test, so the set of reported pairs does not depend on
the chunking or the worker count.
"""
import numpy as np
from scipy.spatial import cKDTree
from ..d3v2p0.surface_intersections import _intersecting_pairs
from .backend import plan, numba_threads
from .parallel import run_chunks


def _search(task, pair_test=None):
    """d3v2p0's batched search for the query ids of one chunk (in a worker,
    or in this process with the numba pair test)."""
    points, triangles, tolerance, batch_size, query_ids, full = task
    if pair_test is None:
        pair_test = _intersecting_pairs
    xyz = points[triangles]
    centres = xyz.mean(axis=1)
    radii = np.linalg.norm(xyz - centres[:, None], axis=2).max(axis=1)
    lo, hi = xyz.min(axis=1), xyz.max(axis=1)
    bands = np.floor(np.log2(np.maximum(radii, max(tolerance, 1e-30)))).astype(int)
    trees = []
    for band in np.unique(bands):
        members = np.flatnonzero(bands == band)
        trees.append((members, cKDTree(centres[members]), float(radii[members].max())))
    found = []
    for start in range(0, len(query_ids), batch_size):
        ids = query_ids[start:start + batch_size]
        for members, tree, maximum_radius in trees:
            neighbours = tree.query_ball_point(centres[ids], radii[ids] + maximum_radius + tolerance)
            sizes = np.array([len(n) for n in neighbours])
            if not sizes.sum():
                continue
            a = np.repeat(ids, sizes)
            b = members[np.concatenate(neighbours).astype(int)]
            keep = ((a < b) if full else (a != b)) & np.all(lo[a] <= hi[b] + tolerance, axis=1) \
                & np.all(lo[b] <= hi[a] + tolerance, axis=1)
            candidates = np.unique(np.sort(np.column_stack((a[keep], b[keep])), axis=1), axis=0)
            found.append(candidates[pair_test(points, triangles, candidates, tolerance)])
    return np.vstack(found) if found else np.empty((0, 2), int)


def find_surface_intersections(points, triangles, tolerance=None, batch_size=1024, triangle_ids=None,
                               n_workers=None, backend='auto'):
    """d3v2p0.surface_intersections.find_surface_intersections searched in
    parallel. backend / n_workers: see d3v2p1.backend (None = automatic;
    searches under 4096 query triangles per worker run serially). The numba
    tier runs the pair test as a compiled kernel on numba threads. Returns
    the same intersecting pairs as d3v2p0 for every tier and worker count."""
    points = np.asarray(points)
    triangles = np.asarray(triangles)
    if not len(triangles):
        return np.empty((0, 2), int)
    tolerance = 1e-9 * max(1., float(np.ptp(points, axis=0).max())) if tolerance is None else tolerance
    full = triangle_ids is None
    query_ids = np.arange(len(triangles)) if full else np.unique(triangle_ids)
    # small searches are not worth starting processes (or many threads) for
    chosen = plan(backend, n_workers, items=len(query_ids), min_items_per_worker=4096, numba_kernels=True)
    if chosen.numba:
        from .numba_intersections import pairs_intersect
        p64 = np.ascontiguousarray(points, dtype=np.float64)
        t64 = np.ascontiguousarray(triangles, dtype=np.int64)

        def pair_test(p_, t_, candidates, tol):
            return pairs_intersect(p64, t64, np.ascontiguousarray(candidates, dtype=np.int64), float(tol))
        with numba_threads(chosen.workers):
            found = _search((points, triangles, tolerance, 16 * batch_size, query_ids, full), pair_test)
        found = [found] if len(found) else []
    else:
        chunks = [query_ids[k::chosen.workers] for k in range(chosen.workers)]
        tasks = [(points, triangles, tolerance, batch_size, c, full) for c in chunks if len(c)]
        found = [r for r in run_chunks(_search, tasks, chosen) if len(r)]
    return np.unique(np.vstack(found), axis=0) if found else np.empty((0, 2), int)
