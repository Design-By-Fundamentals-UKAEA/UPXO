"""
Polygon/coordinate-generic grain-boundary topology helpers.

Extracted from ``polygonised_grain_structure`` (Technique A,
``pxtal/geometrification.py``) so the same junction-detection and
ring-assembly logic is usable from a Shapely-only input source
(``geoEntities/polygon2d_from_shapely.py``), not just from Technique A's
raster-derived pipeline. Both extractions are behaviour-preserving: the
originating methods on ``polygonised_grain_structure`` now delegate here.
"""
import numpy as np


def junction_points_from_polygons(polygons, labels, xyoffset=0.0):
    """
    Pairwise-intersection junction points for a set of neighbouring polygons.

    For every pair of intersecting polygons, the ends of their shared
    boundary are junction points: both ends of a line, the boundary points
    of a multi-line, or a single touching point. Mixed intersections (a
    line plus an isolated point) contribute nothing. Candidate pairs come
    from an R-tree query; all intersections are computed in one vectorised
    call.

    Parameters
    ----------
    polygons : sequence of shapely.geometry.Polygon
        One polygon per entry; index-aligned with `labels`.
    labels : sequence
        Orderable, unique-per-entry labels (e.g. grain ids or loop ids),
        index-aligned with `polygons`. Used only to keep one direction of
        each intersecting pair (`labels[j] > labels[i]`), not as an
        identity for the returned points.
    xyoffset : float, optional
        Subtracted from every returned coordinate (raster pixel-centre
        correction for Technique A; 0.0, the default, is a no-op for
        Shapely-sourced input).

    Returns
    -------
    numpy.ndarray of shape (n, 2)
        Unique junction-point coordinates.
    """
    import shapely
    from shapely.strtree import STRtree

    pols = np.asarray(polygons, dtype=object)
    labels = np.asarray(labels)
    if len(pols) == 0:
        return np.empty((0, 2))

    ii, jj = STRtree(pols).query(pols, predicate='intersects')
    keep = labels[jj] > labels[ii]
    ii, jj = ii[keep], jj[keep]
    if len(ii) == 0:
        return np.empty((0, 2))

    geoms = shapely.intersection(pols[ii], pols[jj])
    kind = shapely.get_type_id(geoms)
    parts = []
    lines = geoms[kind == 1]
    if len(lines):
        parts.append(shapely.get_coordinates(shapely.get_point(lines, 0)))
        parts.append(shapely.get_coordinates(shapely.get_point(lines, -1)))
    multilines = geoms[kind == 5]
    if len(multilines):
        bnd = shapely.boundary(multilines)
        # Only a MultiPoint boundary is used (as before).
        parts.append(shapely.get_coordinates(
            bnd[shapely.get_type_id(bnd) == 4]))
    points = geoms[kind == 0]
    if len(points):
        parts.append(shapely.get_coordinates(points))

    if not parts:
        return np.empty((0, 2))
    jnp = np.vstack(parts)
    return np.unique(jnp, axis=0) - xyoffset


def assemble_ring_from_wall_segments(segments, verbose=True):
    """
    Build a ``ring2d`` from a list of ``MSline2d`` wall segments in
    arbitrary order, reordering/flipping as needed for spatial continuity.

    If the segments are not already in ring order, a greedy search chains
    each next segment onto the current ring end via
    ``MSline2d.do_i_precede``/``do_i_proceed``, tracking the per-segment
    flip each chaining step requires.

    Parameters
    ----------
    segments : sequence of MSline2d
        The wall segments bounding one loop, in arbitrary order.
    verbose : bool, optional
        Print continuity/reordering progress.

    Returns
    -------
    ring2d
        The assembled ring. If assembly did not fully succeed (see
        `success`), it still holds however many segments the greedy search
        managed to chain.
    bool
        `success` -- True if every segment was chained into one continuous,
        closed ring.
    int
        Number of reordering iterations used (0 if the input was already
        continuous).
    """
    from upxo.geoEntities.mulsline2d import ring2d

    segments = list(segments)
    n = len(segments)
    ring = ring2d(segments=segments, segids=list(range(n)),
                  segflips=[False for _ in segments])
    continuity, flip_needed, i_precede_chain = ring.assess_spatial_continuity()

    if continuity:
        # assess_spatial_continuity's own contract: "continuity" only means
        # every consecutive pair CAN be chained, with per-pair flip needs
        # recorded in i_precede_chain -- it does not mean segflips=[False]*n
        # is already correct. Technique A's original pipeline never hits
        # this gap (it never substitutes a foreign-oriented segment object
        # into a loop), but this module's callers do (canonicalizing a
        # shared wall to one object reused, in its own fixed orientation,
        # by every loop that borders it) -- so the flips must be applied
        # explicitly here, not assumed away.
        segflips = [False] + [pair[1] for pair in i_precede_chain]
        last = segments[-1]
        last_end = last.nodes[0] if segflips[-1] else last.nodes[-1]
        if last_end.eq_fast(segments[0].nodes[0])[0]:
            if verbose:
                print('gbsegs continuous')
            ring.segflips = segflips
            return ring, True, 0
        # The per-pair flips satisfy every interior join but still don't
        # close the loop (assess_spatial_continuity never checks the
        # last-to-first wraparound) -- fall through to the general search
        # below rather than return a ring that silently doesn't close.
        if verbose:
            print('gbsegs continuous pairwise but did not close; reordering.')
    elif verbose:
        print('gbsegs not continuous. Attempting reorder.')

    segids = list(range(n))
    result = ring2d([segments[0]], [0], [False])
    used_segids = [0]
    search_segids = set(segids)
    max_iterations = 10 * len(segments)
    itcount = 1
    while len(search_segids) > 0:
        current_seg = result.segments[-1]
        flip_req_previous = result.segflips[-1]
        search_segids = set(segids) - set(used_segids)
        for candidate_segid in search_segids:
            candidate_seg = segments[candidate_segid]
            adjacency = (current_seg.do_i_proceed(candidate_seg) if flip_req_previous
                        else current_seg.do_i_precede(candidate_seg))
            if adjacency[0]:
                used_segids.append(candidate_segid)
                result.add_segment_unsafe(candidate_seg)
                result.add_segid(candidate_segid)
                result.add_segflip(adjacency[1])
                break
        if itcount >= max_iterations:
            break
        itcount += 1

    success = len(search_segids) == 0
    if success and not result.segments[0].nodes[0].eq_fast(result.segments[-1].nodes[-1]):
        result.segflips[-1] = True
    if verbose and success:
        print(f'Re-ordering success. gbsegs are continuous. '
             f'N.Segs={len(segments)}. N.Iterations={itcount}')
    return result, success, itcount
