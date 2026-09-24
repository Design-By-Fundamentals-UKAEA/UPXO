"""
Shapely multi-polygon grain structure -> topologically-linked
``{gid: Polygon2d | NestedPolygon2d}``.

Builds genuine UPXO objects (``Point2d``/``Sline2d``/``MSline2d``/``ring2d``)
from a Shapely-sourced multi-grain structure -- e.g. ``GrainManifold2D``
(Technique B)'s ``self.cells`` -- with the same cross-grain shared-segment
*object identity* Technique A already achieves for pixel-derived grains, so
that ``Polygon2d.edit_segment``/``subdivide_segment`` on one grain's shared
wall is visible in its neighbour with no extra step.

Junction detection and ring assembly (the parts of Technique A's pipeline
that are already polygon/coordinate-generic) are reused via
``upxo.pxtal._gb_topology``. Wall canonicalization here follows a simpler
"build once, reuse by reference" strategy instead of Technique A's
build-both-then-dedup-by-property approach -- see this module's design
notes / project plan for the reasoning.
"""
import math

import numpy as np
from shapely.geometry import Polygon as ShPolygon, Point as ShPoint

from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.mulsline2d import MSline2d
from upxo.geoEntities.polygon2d import Polygon2d, NestedPolygon2d
from upxo.pxtal._gb_topology import (junction_points_from_polygons,
                                     assemble_ring_from_wall_segments)


def _decimals(tol):
    return max(0, int(round(-math.log10(tol))))


def _iter_polygon_parts(geom):
    """Yield Polygon parts of geom, recursing into MultiPolygon/GeometryCollection."""
    gt = geom.geom_type
    if gt == 'Polygon':
        if geom.area > 0:
            yield geom
    elif gt in ('MultiPolygon', 'GeometryCollection'):
        for g in geom.geoms:
            yield from _iter_polygon_parts(g)
    # Point/LineString/other lower-dimension members: ignored.


class _Loop:
    """One boundary ring to geometrify: a grain's exterior, or one of its holes."""

    __slots__ = ('gid', 'part_index', 'kind', 'hole_index', 'coords', 'polygon')

    def __init__(self, gid, part_index, kind, hole_index, coords):
        self.gid = gid
        self.part_index = part_index
        self.kind = kind            # 'exterior' | 'hole'
        self.hole_index = hole_index
        self.coords = np.asarray(coords, dtype=float)   # (n+1, 2), Shapely-closed
        self.polygon = ShPolygon(self.coords)

    @property
    def key(self):
        return (self.gid, self.part_index, self.kind, self.hole_index)


def _insert_point_on_segment_tolerant(msl, point, tol):
    """
    Insert `point` as a new node into `msl`, in place, if it lies within
    `tol` of one of its segments, strictly between that segment's own
    endpoints.

    Mirrors ``MSline2d.add_nodes``'s mechanics (locate the containing
    ``Sline2d``, split it, insert the new line, refresh ``.nodes``) but with
    a tolerance-based location test instead of ``add_nodes``'s exact
    ``fully_contains_point`` check -- which would silently reject a
    junction coordinate computed from a *different* polygon's intersection,
    since there is no bit-exactness guarantee across independent GEOS
    float paths.

    Returns
    -------
    bool
        True if a new node was inserted; False if `point` already exists as
        a node (within `tol`) or no containing segment was found.
    """
    px, py = float(point[0]), float(point[1])
    for node in msl.nodes:
        if abs(node.x - px) < tol and abs(node.y - py) < tol:
            return False
    for i, line in enumerate(msl.lines):
        x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-30:
            continue
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
        if t <= 1e-9 or t >= 1.0 - 1e-9:
            continue
        proj_x, proj_y = x0 + t * dx, y0 + t * dy
        if math.hypot(px - proj_x, py - proj_y) < tol:
            divider = Point2d(px, py)
            new_line = line.split(method='p2d', divider=divider, saa=True,
                                  throw=True, update='pntb',
                                  perform_containment_check=False)[1]
            msl.lines.insert(i + 1, new_line)
            msl.update_nodes()
            return True
    return False


def _splice_loop(msl, junction_coords, tol):
    """Chop a closed ``MSline2d`` into wall segments at the given junction
    coordinates (already inserted as exact nodes of `msl`).

    Falls back to treating the whole loop as one segment when fewer than
    two junctions are found on it (an isolated grain with no neighbours,
    or -- for a host/island hole pair with no other neighbours -- a loop
    whose only "junction" is the entire shared ring, which
    `junction_points_from_polygons` cannot express as discrete points).
    """
    node_coords = msl.get_node_coords()
    n = len(node_coords)
    chop_indices = set()
    for jc in junction_coords:
        dists = np.linalg.norm(node_coords - jc, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < tol:
            chop_indices.add(idx)
    chop_indices = sorted(chop_indices)
    if len(chop_indices) < 2:
        return [MSline2d.from_lines(list(msl.lines), close=False)]
    segments = []
    for i in range(len(chop_indices)):
        start = chop_indices[i]
        end = chop_indices[(i + 1) % len(chop_indices)]
        lines = (msl.lines[start:end] if end > start
                else msl.lines[start:] + msl.lines[:end])
        segments.append(MSline2d.from_lines(lines, close=False))
    return segments


def _canonical_wall_key(seg, tol):
    """Direction-independent key for a wall segment, from its FULL
    coordinate path (not just endpoints). Two junction points alone are not
    always enough to identify a wall uniquely: when a grain has exactly two
    junction points, both the direct wall and the long way around its
    remaining boundary share those same two endpoints -- only the
    intermediate path tells them apart. Rounding every point to `tol` still
    lets two independently-spliced copies of the same physical wall (built
    from two different loops' own vertex data) match despite tiny
    floating-point differences.
    """
    coords = seg.get_node_coords()
    d = _decimals(tol)
    rounded = tuple(tuple(np.round(c, decimals=d)) for c in coords)
    return min(rounded, tuple(reversed(rounded)))


def polygon_collection_from_shapely(cells, gid_key=None, tol=1e-6, verbose=False):
    """
    Convert a Shapely-sourced multi-grain structure into a topologically
    linked UPXO collection.

    Parameters
    ----------
    cells : dict
        ``{gid: shapely.Polygon | MultiPolygon | GeometryCollection}``, e.g.
        ``GrainManifold2D.cells`` directly.
    gid_key : callable or None, optional
        Applied to `cells`' keys before use as the output dict's keys.
        ``None`` (default): use as-is.
    tol : float, optional
        Coordinate-match tolerance for wall canonicalization and junction-
        point insertion. Default ``1e-6`` matches ``GrainManifold2D``'s own
        ``shapely.set_precision`` grid; loosen for other Shapely sources.
    verbose : bool, optional
        Print phase progress.

    Returns
    -------
    dict
        ``{gid: Polygon2d | NestedPolygon2d}``. A ``MultiPolygon``-valued
        cell contributes its largest-area part as the dict value; smaller
        disjoint parts are appended to ``.props['extra_parts']`` (list of
        ``Polygon2d``/``NestedPolygon2d``, still wall-linked to neighbours
        through the same tolerance-keyed lookup) -- nothing is silently
        dropped.
    """
    if verbose:
        print("polygon_collection_from_shapely: normalising input parts")

    # --- 1. Normalize & flatten to parts ------------------------------
    gid_parts = {}   # gid -> [Polygon, ...] sorted by area desc
    for raw_gid, geom in cells.items():
        gid = gid_key(raw_gid) if gid_key is not None else raw_gid
        parts = sorted(_iter_polygon_parts(geom), key=lambda p: -p.area)
        if parts:
            gid_parts[gid] = parts

    # --- 2. Build the flat global loop registry -----------------------
    loops = []
    for gid, parts in gid_parts.items():
        for part_index, poly in enumerate(parts):
            loops.append(_Loop(gid, part_index, 'exterior', None,
                               list(poly.exterior.coords)))
            for hole_index, ring in enumerate(poly.interiors):
                loops.append(_Loop(gid, part_index, 'hole', hole_index,
                                   list(ring.coords)))

    if verbose:
        print(f"  {len(loops)} loop(s) across {len(gid_parts)} grain(s)")

    # --- 3. Global junction detection ---------------------------------
    loop_polygons = [loop.polygon for loop in loops]
    loop_labels = np.arange(len(loops))
    junctions = junction_points_from_polygons(loop_polygons, loop_labels, xyoffset=0.0)

    if verbose:
        print(f"  {len(junctions)} junction point(s) detected")

    def _relevant_junctions(loop):
        if len(junctions) == 0:
            return junctions
        dists = np.array([loop.polygon.exterior.distance(ShPoint(j)) for j in junctions])
        return junctions[dists < tol]

    # --- 4. Per-loop MSline2d + junction insertion ----------------------
    loop_mslines = []
    for loop in loops:
        nodes = [Point2d(c[0], c[1]) for c in loop.coords[:-1]]
        msl = MSline2d.by_nodes(nodes, close=False)
        msl.close(reclose=False)
        msl.update_nodes()   # close() does not refresh .nodes itself.

        for jc in _relevant_junctions(loop):
            _insert_point_on_segment_tolerant(msl, jc, tol)
        loop_mslines.append(msl)

    # --- 5. Splice at junction points -------------------------------------
    loop_wall_lists = [
        _splice_loop(msl, _relevant_junctions(loop), tol)
        for loop, msl in zip(loops, loop_mslines)
    ]

    # --- 6. Canonicalize walls: build once, reuse by reference -----------
    wall_lookup = {}
    order = sorted(range(len(loops)), key=lambda i: (
        str(loops[i].gid), loops[i].part_index, loops[i].kind,
        -1 if loops[i].hole_index is None else loops[i].hole_index))
    for i in order:
        canon_segs = []
        for seg in loop_wall_lists[i]:
            key = _canonical_wall_key(seg, tol)
            existing = wall_lookup.get(key)
            if existing is not None:
                canon_segs.append(existing)
            else:
                wall_lookup[key] = seg
                canon_segs.append(seg)
        loop_wall_lists[i] = canon_segs

    # --- 7. Ring assembly ---------------------------------------------------
    loop_rings = {}
    for loop, segs in zip(loops, loop_wall_lists):
        ring, success, _ = assemble_ring_from_wall_segments(segs, verbose=verbose)
        loop_rings[loop.key] = ring

    # --- 8/9. Wrap into Polygon2d / NestedPolygon2d, keyed by gid ----------
    result = {}
    for gid, parts in gid_parts.items():
        wrapped_parts = []
        for part_index in range(len(parts)):
            host = Polygon2d.from_ring2d(
                loop_rings[(gid, part_index, 'exterior', None)], gid=gid)
            holes = []
            hole_index = 0
            while (gid, part_index, 'hole', hole_index) in loop_rings:
                holes.append(Polygon2d.from_ring2d(
                    loop_rings[(gid, part_index, 'hole', hole_index)], gid=gid))
                hole_index += 1
            if holes:
                wrapped_parts.append(
                    NestedPolygon2d.from_host_and_holes(host, holes, gid=gid))
            else:
                wrapped_parts.append(host)
        result[gid] = wrapped_parts[0]
        if len(wrapped_parts) > 1:
            result[gid].props['extra_parts'] = wrapped_parts[1:]

    if verbose:
        print(f"polygon_collection_from_shapely: {len(result)} grain(s) built, "
             f"{len(wall_lookup)} unique wall segment(s)")
    return result
