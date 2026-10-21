"""Smoothed grain-polygon construction for the pixellated/smoothed display of
EBSD maps -- two independently-selectable
techniques. Both return ``{gid: shapely Polygon/MultiPolygon}`` and
``grains``, the integer label array the returned gids index into (NOT
necessarily the caller's original ``lfi`` -- Technique A relabels
disconnected pieces of the same crystallographic grain via
connected-components first, same as gb_disorientation_map.
compute_gb_disorientation already does internally; Technique B does not
relabel, so its ``grains`` is the input ``lfi`` unchanged). A caller that
needs a per-grain summary value (e.g. mean Euler angle for a smoothed
map) always computes it from this SAME returned ``grains`` array, so the
label spaces never need reconciling by hand.

- Technique A: UPXO's existing segment-based geometrification + boundary
  smoothing (``polygonised_grain_structure.smooth_gbsegs``) -- the exact
  computation ``gb_disorientation_map.compute_gb_disorientation`` already
  does internally for its own boundary lines. Also returns the smoothed
  GB-segment data itself (``gb_rings``, a 3rd return value) so a caller
  building a GBDisorientation from these polygons can reuse the exact
  segment geometry instead of reconstructing it from the polygons.
- Technique B: Voronoi-tessellation-based smoothing
  (``geometrification.GrainManifold2D``) -- scatter seed points,
  Voronoi-tessellate, sample the label field at each seed, union
  same-label cells into one polygon per grain. Also returns the raw
  Voronoi ridge data between individual cells of different grain labels
  (``gb_ridges``, a 3rd return value, from
  ``GrainManifold2D.grain_boundary_ridges()``), captured BEFORE same-
  label cells are unioned together and any shared-edge precision is
  lost -- numerically exact for the same reason Technique A's gb_rings
  is, just one level earlier (at the raw Voronoi cell, not the smoothed
  boundary segment).

``disorientation_from_polygons`` builds a ``GBDisorientation``-compatible
result from either technique's polygons, instead of
``compute_gb_disorientation``'s own from-scratch pixel-grid smoothing --
which lets a disorientation map reuse an already-computed smoothed
structure when the same technique was chosen for both, instead of
recomputing it. Pass
the matching technique's own exact data (``gb_rings=`` for Technique A,
``gb_ridges=`` for Technique B) when available -- otherwise it falls back
to polygon-boundary intersection, which is numerically fragile (see its
own docstring).
"""
import numpy as np

__all__ = ["smooth_technique_a", "smooth_technique_b", "disorientation_from_polygons"]


def smooth_technique_a(lfi, smooth_passes=2, max_smooth_levels=(3, 3)):
    """({gid: shapely Polygon/MultiPolygon}, grains, gb_rings) via
    segment-based geometrification + boundary smoothing (Technique A) --
    mirrors gb_disorientation_map.compute_gb_disorientation's own
    construction exactly, stopping short of the disorientation-angle step
    it doesn't need here.

    gb_rings is geom.smoothed[...]["GB"] -- the SAME smoothed-boundary-
    segment data compute_gb_disorientation itself builds its exact
    (numerically precise, shared-by-object-identity) GBDisorientation
    segments from. Returned alongside the per-grain polygons so a caller
    building a GBDisorientation from this technique's output (e.g.
    disorientation_from_polygons) can reuse that exact data too, instead
    of reconstructing each shared edge via polygon-boundary intersection
    -- see disorientation_from_polygons's own docstring for why that
    reconstruction is numerically fragile."""
    import cc3d
    from upxo.gsdataops.gid_ops import find_neighs2d
    from upxo.pxtal.geometrification import polygonised_grain_structure

    lfi = np.asarray(lfi)
    if lfi.ndim != 2:
        raise ValueError("lfi must be 2D.")
    if (lfi <= 0).any():
        raise ValueError("Every pixel needs a grain label; clean the map first.")
    passes = int(smooth_passes)
    if passes not in (1, 2) or len(max_smooth_levels) < passes:
        raise ValueError("smooth_passes must be 1 or 2, with one smoothing level per pass.")

    grains = cc3d.connected_components(lfi, connectivity=4).astype(np.int32)
    neigh = {int(g): [int(v) for v in vs] for g, vs in find_neighs2d(grains, conn=4).items()}
    geom = polygonised_grain_structure(grains, np.unique(grains), neigh)
    geom.pix_to_geom(verbose=False)
    geom.smooth_gbsegs(geom.GB, npasses=passes,
                       max_smooth_levels=list(max_smooth_levels[:passes]),
                       plot=False, name="viz")
    polygons = {int(gid): poly for gid, poly in geom.smoothed["viz"]["GRAINS"].items()}
    gb_rings = geom.smoothed["viz"]["GB"]
    return polygons, grains, gb_rings


def smooth_technique_b(lfi, seeds_per_grain=20, rng_seed=None):
    """({gid: shapely Polygon/MultiPolygon}, grains, gb_ridges) via Voronoi
    tessellation + label-sampling + same-label union (Technique B).
    ``grains`` here is ``lfi`` itself, unchanged -- no relabelling.

    seeds_per_grain : average seed density, scaled by the grain count --
    total seeds = n_grains * seeds_per_grain, scattered uniformly at
    random over the whole map. More seeds per grain gives a closer
    approximation to the true pixel boundary; fewer gives a coarser,
    smoother one.

    gb_ridges is manifold.grain_boundary_ridges() -- the exact per-grain-
    pair boundary data from the raw Voronoi ridges, BEFORE same-label
    cells are unioned together and any shared-edge precision is lost.
    Returned alongside the per-grain polygons so a caller building a
    GBDisorientation from this technique's output (e.g.
    disorientation_from_polygons) can reuse it instead of reconstructing
    each pair's shared edge via polygon-boundary intersection -- see
    disorientation_from_polygons's own docstring for why that
    reconstruction is numerically fragile for Technique B specifically
    (no equivalent shared-segment data existed for it before this)."""
    from upxo.pxtal.geometrification import GrainManifold2D

    lfi = np.asarray(lfi)
    if lfi.ndim != 2:
        raise ValueError("lfi must be 2D.")
    if (lfi <= 0).any():
        raise ValueError("Every pixel needs a grain label; clean the map first.")
    ny, nx = lfi.shape
    n_grains = int(np.unique(lfi).size)
    n_seeds = max(n_grains, n_grains * max(1, int(seeds_per_grain)))
    rng = np.random.default_rng(rng_seed)
    seeds = rng.uniform([0, 0], [nx, ny], size=(n_seeds, 2))
    manifold = GrainManifold2D.by_tessellation(lfi, seeds)
    polygons = {int(g): poly for g, poly in manifold.cells.items() if g > 0}
    gb_ridges = manifold.grain_boundary_ridges()
    return polygons, lfi, gb_ridges


def _line_coords(geom):
    """A shapely LineString's (n, 2) coords, or None if `geom` isn't one
    continuous line (MultiLineString picks the longest piece rather than
    dropping the pair entirely; Point/empty/other -> None).

    NOTE: this is the fallback path's own primitive, used only when
    neither gb_rings nor gb_ridges is available (see
    disorientation_from_polygons). Intersecting two polygons' boundaries
    that are only APPROXIMATELY coincident along their shared edge
    (independently built, not sharing the same underlying segment/ridge
    objects) is numerically fragile: GEOS routinely returns the shared
    edge fragmented into many tiny pieces rather than one clean line, and
    "pick the longest piece" then keeps only a small fraction of the true
    boundary -- confirmed in practice (synthetic repro: every returned
    segment collapsed to exactly 2 points, representing under 25% of the
    true boundary length). Prefer gb_rings/gb_ridges (exact, no
    coordinate comparison at all) whenever available."""
    if geom.is_empty:
        return None
    if geom.geom_type == 'LineString':
        return np.asarray(geom.coords, dtype=float)
    if geom.geom_type == 'MultiLineString':
        longest = max(geom.geoms, key=lambda g: g.length)
        return np.asarray(longest.coords, dtype=float)
    return None


def disorientation_from_polygons(polygons_by_gid, grains, quat, smooth_passes=0,
                                 max_smooth_levels=(), original_lfi=None,
                                 gb_rings=None, gb_ridges=None):
    """A gb_disorientation_map.GBDisorientation built from a
    ``{gid: polygon}`` dict (either smoothing technique) instead of
    compute_gb_disorientation's own from-scratch pixel-grid smoothing,
    avoiding rebuilding the smoothed structure a second time.

    gb_rings : Technique A's own smoothed GB-segment data
    (smooth_technique_a's 3rd return value), when available. Used
    directly via gb_disorientation_map.segments_and_pairs_from_rings --
    numerically exact, since it groups the ALREADY-shared segment objects
    by identity rather than comparing coordinates.

    gb_ridges : Technique B's own exact per-grain-pair boundary data
    (smooth_technique_b's 3rd return value, from
    GrainManifold2D.grain_boundary_ridges()), when available. Built from
    the raw Voronoi ridges between individual cells of different grain
    labels, BEFORE same-label cells are unioned together -- numerically
    exact for the same reason gb_rings is: the two cells sharing a ridge
    index the SAME underlying Voronoi vertex array, so no coordinate
    comparison is needed to recover it.

    When neither is supplied, falls back to reconstructing each pair's
    shared edge via polygon-boundary intersection -- one boundary segment
    per adjacent grain pair, coarser than gb_rings/gb_ridges' own multi-
    segment boundaries, and (see _line_coords's own docstring)
    numerically fragile; kept only for a caller with neither available.

    grains : the SAME label array `polygons_by_gid`'s gids index into
    (smooth_technique_a/b's second return value) -- quat must be on
    that same (ny, nx, 4) grid.

    original_lfi : the map's ORIGINAL grain labels (Technique B: identical
    to `grains`, since it doesn't relabel; Technique A: `grains` is a
    connected-components relabelling of this), needed so the result's
    ``original_ids`` -- and therefore classify_boundaries/twin-vs-grain
    matching against original label ids -- is correct for Technique A
    too, not just an identity passthrough. Omit only when the caller has
    no need to translate back to original ids.
    """
    from upxo.gsdataops.gid_ops import find_neighs2d
    from upxo.xtalphy.crystal_orientation import grain_avg_quats, quat_to_R_batch
    from upxo.xtalphy.gb_disorientation_map import (
        GBDisorientation, pair_disorientation_deg, segments_and_pairs_from_rings)

    grains = np.asarray(grains)
    gids, q_mean = grain_avg_quats(grains, quat)
    R_all = quat_to_R_batch(q_mean)
    index_of = np.zeros(int(gids.max()) + 1, dtype=int)
    index_of[gids] = np.arange(len(gids))

    if gb_rings is not None:
        segments, pairs_arr = segments_and_pairs_from_rings(gb_rings)
    elif gb_ridges is not None:
        segments, pairs = [], []
        for (gid_a, gid_b), segs in gb_ridges.items():
            for seg in segs:
                segments.append(seg)
                pairs.append((gid_a, gid_b))
        pairs_arr = np.asarray(pairs, dtype=int).reshape(-1, 2) if pairs else np.zeros((0, 2), dtype=int)
    else:
        neigh = find_neighs2d(grains.astype(np.int32), conn=4)
        seen = set()
        segments, pairs = [], []
        for gid_a, nbrs in neigh.items():
            for gid_b in nbrs:
                key = tuple(sorted((int(gid_a), int(gid_b))))
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                poly_a, poly_b = polygons_by_gid.get(key[0]), polygons_by_gid.get(key[1])
                if poly_a is None or poly_b is None:
                    continue
                boundary_a = poly_a.exterior if hasattr(poly_a, 'exterior') else poly_a.boundary
                boundary_b = poly_b.exterior if hasattr(poly_b, 'exterior') else poly_b.boundary
                coords = _line_coords(boundary_a.intersection(boundary_b))
                if coords is None:
                    continue
                segments.append(coords)
                pairs.append(key)
        pairs_arr = np.asarray(pairs, dtype=int).reshape(-1, 2) if pairs else np.zeros((0, 2), dtype=int)
    # Restricted to actually-shared (two-owner) pairs -- gid 0 is the
    # "no second owner" placeholder for an edge-of-map segment (see
    # segments_and_pairs_from_rings), not a real grain id; computing
    # pair_disorientation_deg against it would silently read R_all[0]
    # (whatever grain happens to be first in index_of) as if it were a
    # genuine neighbour instead of leaving the angle NaN. The polygon-
    # intersection fallback path never produced a (gid, 0) pair on its
    # own (it only ever iterates real grain-to-grain adjacency), so this
    # was latent there; gb_rings does produce them, by design, matching
    # compute_gb_disorientation's own pairs convention exactly.
    angles = np.full(len(pairs_arr), np.nan)
    shared = np.flatnonzero(pairs_arr[:, 1] > 0) if len(pairs_arr) else np.empty(0, dtype=int)
    if shared.size:
        unique_pairs, inverse = np.unique(pairs_arr[shared], axis=0, return_inverse=True)
        unique_angles = pair_disorientation_deg(
            R_all, index_of[unique_pairs[:, 0]], index_of[unique_pairs[:, 1]])
        angles[shared] = unique_angles[inverse.ravel()]

    if original_lfi is None:
        original_ids = np.arange(int(grains.max()) + 1)
    else:
        # Same construction as compute_gb_disorientation's own original_ids:
        # the original label at each relabelled grain's first pixel.
        original_lfi = np.asarray(original_lfi)
        first_pixel = np.unique(grains.ravel(), return_index=True)[1]
        original_ids = np.zeros(int(grains.max()) + 1, dtype=np.int64)
        original_ids[1:] = original_lfi.ravel()[first_pixel]

    return GBDisorientation(
        segments=segments, angles=angles, pairs=pairs_arr, shape=tuple(grains.shape),
        n_grains=int(len(gids)), smooth_passes=smooth_passes,
        max_smooth_levels=tuple(max_smooth_levels), original_ids=original_ids)
