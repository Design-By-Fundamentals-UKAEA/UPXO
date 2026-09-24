"""
2D polygon geometric entities for UPXO, built on the live UPXO boundary
stack (``Sline2d`` -> ``MSline2d`` -> ``ring2d``) rather than plain Shapely,
while remaining convertible to/from Shapely.

Fills a documented gap: ``mulsline2d.py``'s own "See Also" references
``upxo.geoEntities.polygon2d``, a module that did not previously exist.

Classes
-------
Polygon2d : A single grain's smoothed boundary, wrapping one ``ring2d``.
NestedPolygon2d : A grain with island/hole sub-polygons (recursive).

Usage
-----
    from upxo.geoEntities.polygon2d import Polygon2d, NestedPolygon2d

    # Zero-copy wrap of an existing ring2d (e.g. from Technique A's self.GB[gid])
    poly = Polygon2d.from_ring2d(ring, gid=7)

    # From a Shapely polygon
    poly = Polygon2d.from_shapely_polygon(shapely_poly, gid=7)

    # Back to Shapely
    shapely_poly = poly.make_shapely()
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShPolygon
from shapely.ops import unary_union

from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.sline2d import Sline2d as sl2d
from upxo.geoEntities.mulsline2d import MSline2d, ring2d


def _point_at_arclength_fraction(coords, f):
    """Point at arc-length fraction `f` (0..1) along an open polyline
    `coords` ((n, 2) array, ordered)."""
    coords = np.asarray(coords, dtype=float)
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total = seg_lengths.sum()
    target = f * total
    cum = 0.0
    for i, seg_len in enumerate(seg_lengths):
        if cum + seg_len >= target or i == len(seg_lengths) - 1:
            local_f = 0.0 if seg_len == 0 else (target - cum) / seg_len
            local_f = min(max(local_f, 0.0), 1.0)
            return coords[i] + local_f * (coords[i + 1] - coords[i])
        cum += seg_len
    return coords[-1]


def _merge_by_arclength(coords, new_entries):
    """Merge `new_entries` (list of (fraction, point) tuples) into `coords`
    ((n, 2) array), ordered by arc-length fraction along the original
    polyline."""
    coords = np.asarray(coords, dtype=float)
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    orig_positions = cum / total
    entries = [(orig_positions[i], coords[i]) for i in range(len(coords))]
    entries.extend(new_entries)
    entries.sort(key=lambda e: e[0])
    return np.array([e[1] for e in entries])


def _with_end_node(seg):
    """Return ``seg.nodes`` with the segment's final endpoint present.

    ``MSline2d.by_coords(close=False)`` stores one node per line and omits
    the last line's end point, so ``seg.nodes`` is one entry short of
    ``seg.get_node_coords()``. For a closed loop built that way (a
    ``from_shapely_polygon`` ring) the missing node is the return to the
    start, so reading ``seg.nodes`` alone leaves the closing edge out of the
    path. ``update_nodes`` restores it.
    """
    if len(seg.nodes) == len(seg.lines):
        seg.update_nodes()
    return seg.nodes


class Polygon2d():
    """
    A single grain's smoothed 2D boundary, wrapping one ``ring2d``.

    Composition, not subclassing: ``ring2d`` is shared by non-``Polygon2d``
    callers throughout ``pxtal/geometrification.py`` and is itself
    documented as "in active development", so it is kept free to evolve
    independently. Composition also makes :meth:`from_ring2d` a genuine
    zero-copy wrap.

    Attributes
    ----------
    ring : ring2d
        The wrapped boundary -- an ordered, closed loop of ``MSline2d``
        segments. A segment may be Python-object-identity-shared with a
        neighbouring grain's ``Polygon2d.ring`` (this is how UPXO's
        geometrification pipeline represents a shared grain-boundary wall);
        see :meth:`edit_segment`.
    gid : int or None
        Grain id, for bookkeeping (e.g. keying a ``{gid: Polygon2d}`` dict).
    props : dict
        Arbitrary per-grain scalar/vector data. An open dict, not
        ``__slots__``-restricted attributes -- see :func:`polygons_to_prop_dataframe`
        to bridge this to UPXO's dominant per-grain scalar convention
        (a pandas DataFrame indexed by ``gid - 1``, one column per property,
        as used by ``pxtal.mcgs2_temporal_slice``'s ``self.prop``).
    """

    __slots__ = ('ring', 'gid', 'props')

    def __init__(self, ring, gid=None, props=None):
        self.ring = ring
        self.gid = gid
        self.props = props if props is not None else {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_ring2d(cls, ring, gid=None, props=None):
        """Zero-copy wrap of an existing ``ring2d`` (e.g. ``self.GB[gid]``
        from ``polygonised_grain_structure``)."""
        return cls(ring, gid=gid, props=props)

    @classmethod
    def from_shapely_polygon(cls, poly, gid=None, props=None):
        """Build a fresh ``ring2d`` from a Shapely ``Polygon``'s exterior.

        Uses the full, already-closed coordinate ring
        (``poly.exterior.coords``, which repeats the first point at the
        end) with ``MSline2d.by_coords(..., close=False)`` -- not
        ``close=True`` on the de-duplicated ring. ``MSline2d.by_coords``'s
        own ``close=True`` path does not append the input's final point to
        ``nodes`` before closing (unlike its sibling ``from_lines``, which
        does), so closing from an already-deduplicated point list silently
        drops the last vertex and closes one edge short. Passing the
        pre-closed ring with ``close=False`` sidesteps that entirely: the
        last segment already returns to the first point, and ``nodes``
        correctly ends up with one entry per distinct vertex.
        """
        coords = list(poly.exterior.coords)
        msl = MSline2d.by_coords(coords, close=False)
        ring = ring2d([msl], segids=[0], segflips=[False])
        return cls(ring, gid=gid, props=props)

    # ------------------------------------------------------------------
    # Shapely bridge
    # ------------------------------------------------------------------

    def make_shapely(self):
        """Return a Shapely ``Polygon`` for this grain's current boundary."""
        return ShPolygon(self.ring.create_coords_from_segments(force_close=True))

    def coords(self, force_close=True):
        """Return this grain's boundary coordinates as an ``(n, 2)`` array."""
        return self.ring.create_coords_from_segments(force_close=force_close)

    # ------------------------------------------------------------------
    # Geometry accessors
    # ------------------------------------------------------------------

    @property
    def nsegments(self):
        """Number of boundary segments (``MSline2d`` walls) in this grain's ring."""
        return self.ring.nsegs

    @property
    def area(self):
        """Polygon area, via :meth:`make_shapely` (not ``ring2d.area``,
        which calls a dead-duplicate method -- see module notes)."""
        return self.make_shapely().area

    @property
    def perimeter(self):
        """Polygon perimeter, via :meth:`make_shapely`."""
        return self.make_shapely().length

    @property
    def centroid(self):
        """Mean of boundary vertex coordinates (``ring2d.centroid``'s own
        definition -- a vertex-mean, not an area centroid)."""
        return self.ring.centroid

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def edit_segment(self, seg_index, new_node_coords):
        """
        Replace one boundary segment's interior coordinates in place.

        The targeted ``MSline2d`` object's ``.lines``/``.nodes`` slots are
        rebound to freshly-built values, but the object itself keeps its
        identity -- so if this segment is shared with a neighbouring
        grain's ``Polygon2d.ring`` (the normal case for an interior wall),
        that neighbour sees the edit immediately, with no extra step. This
        mirrors exactly how ``MSline2d.smooth()`` already safely mutates a
        shared segment.

        A new ``MSline2d`` is deliberately never swapped in for the old
        one: nothing in UPXO tracks which other rings reference a given
        segment, so replacing the object (rather than mutating it) would
        silently break sharing for every other referrer.

        Parameters
        ----------
        seg_index : int
            Index into ``self.ring.segments``.
        new_node_coords : array-like of shape (n, 2), n >= 2
            New coordinates for every node of this segment, in order. The
            first and last rows are junction points -- shared by value
            (never by object identity) with whichever other segments meet
            there -- and must match the segment's existing first/last node
            coordinate exactly (within ``MSline2d.EPS_coord_coincide``).
            Moving an endpoint here would silently desync every other wall
            meeting at that junction, so it is rejected outright.

        Raises
        ------
        ValueError
            If fewer than 2 coordinate rows are given, or if the first/last
            row does not match the segment's existing endpoint coordinate.
        """
        seg = self.ring.segments[seg_index]
        _with_end_node(seg)
        new_node_coords = np.asarray(new_node_coords, dtype=float)
        if new_node_coords.shape[0] < 2:
            raise ValueError("new_node_coords needs at least 2 rows.")
        tol = seg.EPS_coord_coincide
        start_ok = np.allclose(new_node_coords[0],
                               [seg.nodes[0].x, seg.nodes[0].y], atol=tol)
        end_ok = np.allclose(new_node_coords[-1],
                             [seg.nodes[-1].x, seg.nodes[-1].y], atol=tol)
        if not start_ok:
            raise ValueError(
                "edit_segment cannot move a segment endpoint (junction): "
                "new_node_coords[0] must match the segment's existing "
                "start coordinate.")
        if not end_ok:
            raise ValueError(
                "edit_segment cannot move a segment endpoint (junction): "
                "new_node_coords[-1] must match the segment's existing "
                "end coordinate.")

        interior = [Point2d(c[0], c[1]) for c in new_node_coords[1:-1]]
        new_nodes = [seg.nodes[0]] + interior + [seg.nodes[-1]]
        new_lines = [sl2d(new_nodes[i].x, new_nodes[i].y,
                          new_nodes[i + 1].x, new_nodes[i + 1].y)
                    for i in range(len(new_nodes) - 1)]
        seg.lines = new_lines
        seg.nodes = new_nodes

    def subdivide_segment(self, seg_index, n=None, at_fractions=None):
        """
        Insert new interior points into one boundary segment, in place,
        positioned by arc-length fraction along the segment's current
        coordinate sequence.

        Composes on :meth:`edit_segment` rather than reimplementing its
        safe rebuild (fresh interior ``Point2d``s, endpoint identity
        preserved, in-place ``.lines``/``.nodes`` rebind) -- so a
        subdivided segment shared with a neighbouring grain still
        propagates the new node layout to that neighbour automatically,
        the same as any other :meth:`edit_segment` call.

        Does not use ``MSline2d.add_nodes`` (exact-collinearity only, not
        applicable to inserting points off the segment's *original*
        coordinates) or ``MSline2d.sub_divide`` (currently broken --
        raises on its very first call, see its docstring).

        Parameters
        ----------
        seg_index : int
            Index into ``self.ring.segments``.
        n : int or None
            Insert this many new points, evenly spaced at arc-length
            fractions ``1/(n+1), ..., n/(n+1)`` of the segment's current
            total length. Mutually exclusive with `at_fractions`.
        at_fractions : sequence of float or None
            Explicit arc-length fractions in ``(0, 1)`` at which to insert
            new points. Mutually exclusive with `n`.

        Raises
        ------
        ValueError
            If neither or both of `n`/`at_fractions` are given.
        """
        if (n is None) == (at_fractions is None):
            raise ValueError("exactly one of n / at_fractions is required")
        seg = self.ring.segments[seg_index]
        coords = np.array([[node.x, node.y] for node in _with_end_node(seg)])
        fractions = ([i / (n + 1) for i in range(1, n + 1)] if n is not None
                    else list(at_fractions))
        new_entries = [(f, _point_at_arclength_fraction(coords, f)) for f in fractions]
        new_full_coords = _merge_by_arclength(coords, new_entries)
        self.edit_segment(seg_index, new_full_coords)

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    def clone(self, seg_clones=None):
        """
        Return an independent copy of this grain's boundary.

        Parameters
        ----------
        seg_clones : dict or None
            ``{id(original segment): cloned segment}``, forwarded to
            ``ring2d.clone``. Cloning several ``Polygon2d`` instances that
            may share segments (e.g. all grains of one structure) must
            pass the SAME ``seg_clones`` dict to every call -- exactly as
            ``smooth_gbsegs`` does -- or the sharing will not survive the
            clone. Defaults to a fresh, private ``{}`` when omitted, which
            is only correct for cloning a single, standalone ``Polygon2d``.
        """
        if seg_clones is None:
            seg_clones = {}
        return Polygon2d(self.ring.clone(seg_clones), gid=self.gid,
                         props=dict(self.props))


class NestedPolygon2d():
    """
    A grain with island/hole sub-polygons.

    Recursive: a hole may itself be a ``NestedPolygon2d`` (a hole with its
    own hole), not only a flat ``Polygon2d`` -- this is what lets UPXO
    represent genuine hole-in-hole-in-hole structures natively, superseding
    ``polygonised_grain_structure``'s older, abandoned ``polygons_raw_holes``
    / ``hlpol`` nested-dict pathway (never wired into ``make_gsmp()``, and
    with a confirmed bug in its own level-3 branch). This class does not
    bridge to or read that legacy structure.

    Attributes
    ----------
    host : Polygon2d
        The outer boundary.
    holes : list of Polygon2d or NestedPolygon2d
        Sub-polygons cut out of ``host``.
    gid : int or None
    props : dict
    """

    __slots__ = ('host', 'holes', 'gid', 'props')

    def __init__(self, host, holes=None, gid=None, props=None):
        self.host = host
        self.holes = holes if holes is not None else []
        self.gid = gid
        self.props = props if props is not None else {}

    @classmethod
    def from_host_and_holes(cls, host, holes, gid=None, props=None):
        """Construct directly from a host ``Polygon2d`` and a list of
        holes (``Polygon2d`` or ``NestedPolygon2d``, may nest arbitrarily)."""
        return cls(host, holes=list(holes), gid=gid, props=props)

    @classmethod
    def from_shapely_polygon(cls, poly, gid=None, props=None):
        """
        Build from a Shapely ``Polygon``'s exterior and interior rings.

        Every produced hole is a flat ``Polygon2d`` -- Shapely interior
        rings are always simple (``LinearRing``, no holes of their own), so
        a Shapely-sourced ``NestedPolygon2d`` can only ever be one level
        deep. This is a permanent property of the Shapely polygon format,
        not a limitation to fix later: genuine multi-level nesting can only
        be built natively, by nesting ``NestedPolygon2d`` instances
        directly (see :meth:`from_host_and_holes`).
        """
        host = Polygon2d.from_shapely_polygon(ShPolygon(poly.exterior.coords), gid=gid)
        holes = [Polygon2d.from_shapely_polygon(ShPolygon(ring.coords))
                for ring in poly.interiors]
        return cls(host, holes=holes, gid=gid, props=props)

    def make_shapely(self):
        """
        Return a Shapely geometry for this grain, holes included.

        Takes Shapely's native shell+holes constructor when every hole is a
        flat ``Polygon2d`` (the common, cheap case). Falls back to boolean
        composition -- ``host.difference(union(holes))``, mirroring
        ``polygonised_grain_structure._collect_grains``'s own proven
        pattern -- when any hole is itself a ``NestedPolygon2d`` (a
        hole-in-hole), since Shapely's ``Polygon`` has no native
        representation for a hole ring that itself has holes.
        """
        if all(isinstance(h, Polygon2d) for h in self.holes):
            return ShPolygon(
                self.host.coords(force_close=True),
                holes=[h.coords(force_close=True) for h in self.holes])
        host_poly = ShPolygon(self.host.coords(force_close=True))
        if not self.holes:
            return host_poly
        hole_polys = [h.make_shapely() for h in self.holes]
        return host_poly.difference(unary_union(hole_polys))

    def coords(self, force_close=True):
        """Host boundary coordinates (holes are not representable as a
        single coordinate array)."""
        return self.host.coords(force_close=force_close)

    @property
    def area(self):
        """Net area (host minus holes), via :meth:`make_shapely`."""
        return self.make_shapely().area

    @property
    def gids_all(self):
        """This grain's id followed by every hole's id, recursively."""
        out = [self.gid]
        for h in self.holes:
            if isinstance(h, NestedPolygon2d):
                out.extend(h.gids_all)
            else:
                out.append(h.gid)
        return out

    def clone(self, seg_clones=None):
        """Independent copy; ``seg_clones`` is threaded through the host
        and every hole (recursively) so a segment shared between the host,
        a hole, or a sibling grain elsewhere stays shared in the clone --
        see :meth:`Polygon2d.clone`."""
        if seg_clones is None:
            seg_clones = {}
        cloned_holes = [h.clone(seg_clones) for h in self.holes]
        return NestedPolygon2d(self.host.clone(seg_clones), holes=cloned_holes,
                               gid=self.gid, props=dict(self.props))


# ---------------------------------------------------------------------------
# Scalar-field bridge: Polygon2d/NestedPolygon2d.props <-> the DataFrame
# convention used elsewhere in UPXO (pxtal.mcgs2_temporal_slice's self.prop,
# gid-1 indexed, one column per property).
# ---------------------------------------------------------------------------

def polygons_to_prop_dataframe(polygons):
    """
    Gather ``{gid: Polygon2d | NestedPolygon2d}.props`` into one DataFrame.

    Parameters
    ----------
    polygons : dict
        ``{gid: Polygon2d | NestedPolygon2d}``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``gid - 1`` (matching ``mcgs2_temporal_slice.py``'s
        ``self.prop`` convention), one column per key observed across any
        polygon's ``.props``. A polygon missing a given key gets ``NaN`` in
        that column, not a ``KeyError``.
    """
    rows = {gid - 1: dict(poly.props) for gid, poly in polygons.items()}
    return pd.DataFrame.from_dict(rows, orient='index').sort_index()


def apply_prop_dataframe(polygons, df):
    """
    Scatter a ``gid - 1``-indexed DataFrame's values back onto ``.props``.

    Parameters
    ----------
    polygons : dict
        ``{gid: Polygon2d | NestedPolygon2d}``.
    df : pandas.DataFrame
        Indexed by ``gid - 1``, as returned by
        :func:`polygons_to_prop_dataframe`.

    Notes
    -----
    Updates each polygon's ``.props`` dict in place (existing keys not
    present as columns in ``df`` are left untouched); does not replace it.
    Rows containing ``NaN`` for a given column leave that key unset on that
    polygon rather than writing a ``NaN`` value.
    """
    for gid, poly in polygons.items():
        idx = gid - 1
        if idx not in df.index:
            continue
        row = df.loc[idx]
        for col, val in row.items():
            if pd.isna(val):
                continue
            poly.props[col] = val
