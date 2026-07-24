"""
3D straight line geometric entity module for UPXO.

Provides two 3-D straight-line classes — a lean variant for minimal memory
footprint and a full-featured class for geometric operations.

Applications
------------
- Non-conformal to conformal geometry conversion.
- Hierarchical grain structure feature generation.
- General 3-D geometry operations.

Classes
-------
Sline3d_leanest
    Minimal 3-D line stored as six scalar endpoint coordinates.
Sline3d
    Full-featured 3-D line with geometric properties and spatial operations.

Coordinate system
-----------------
::

                     Y+
                     |           Z-
                     |         /
                     |       /
                     |     /
                     |   /
    X-               | /               X+
    -----------------O------------------
                    /|
                  /  |
                /    |
              /      |
            /        |
          /          |
        Z+           Y-

Usage
-----
::

    from upxo.geoEntities.sline3d import Sline3d as sl3d
    from upxo.geoEntities.sline3d import Sline3d_leanest as sl3dl

@author: Dr. Sunil Anandatheertha
"""

import math
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree
import vtk
from shapely.geometry import Point as ShPnt, Polygon as ShPol
from functools import wraps
import matplotlib.pyplot as plt
import upxo._sup.dataTypeHandlers as dth
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
np.seterr(divide='ignore')
from upxo._sup.validation_values import isinstance_many
from upxo.geoEntities.point3d import Point3d


class Sline3d_leanest():
    """Lean 3-D straight line storing only six scalar endpoint coordinates.

    Provides a minimal footprint for high-frequency operations where only
    coordinate access, length, and iteration are required.

    Usage
    -----
    ::

        from upxo.geoEntities.sline3d import Sline3d_leanest as sl3dl

    Examples
    --------
    .. code-block:: python

        from upxo.geoEntities.sline3d import Sline3d_leanest as sl3dl
        e = sl3dl(-2, 3, 4, 5, 1, 2)
        for coord in e:
            print(coord)
        print(e[1])
    """

    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1')

    def __init__(self, x0=0, y0=0, z0=0, x1=1, y1=0, z1=0):
        """Initialise with six scalar endpoint coordinates.

        Parameters
        ----------
        x0, y0, z0 : float, optional
            Start-point coordinates.
        x1, y1, z1 : float, optional
            End-point coordinates.
        """
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1

    def __repr__(self):
        """Return a compact line summary string.

        Returns
        -------
        str
            ``UPXO-sl3d-lean (x0,y0,z0)-(x1,y1,z1)`` summary.
        """
        return f'UPXO-sl3d-lean ({self.x0},{self.y0},{self.z0})-({self.x1},{self.y1},{self.z1})'

    def __iter__(self):
        """Iterate over the two endpoint coordinate tuples.

        Returns
        -------
        iterator
            Iterator over ``(x0, y0, z0)`` and ``(x1, y1, z1)``.
        """
        return (i for i in ((self.x0, self.y0, self.z0),
                            (self.x1, self.y1, self.z1)))

    def __getitem__(self, index):
        """Return an endpoint by index.

        Parameters
        ----------
        index : int
            Endpoint index. ``0`` returns the start point and ``1`` returns
            the end point.

        Returns
        -------
        tuple
            Endpoint coordinate tuple.
        """
        return ((self.x0, self.y0, self.z0),
                (self.x1, self.y1, self.z1))[index]

    @property
    def length(self):
        """Euclidean length of the line segment.

        Returns
        -------
        float
            Segment length.
        """
        return math.sqrt((self.x1-self.x0)**2+(self.y1-self.y0)**2+(self.z1-self.z0)**2)


class Sline3d():
    """Full-featured 3-D straight line entity for UPXO geometric operations.

    Stores a straight line segment by its two endpoint coordinates
    ``(x0, y0, z0)`` and ``(x1, y1, z1)``, and exposes a rich API for
    geometric queries (length, midpoint, direction cosines, angles),
    distance calculations, splitting, and iterative extension.

    Usage
    -----
    ::

        from upxo.geoEntities.sline3d import Sline3d as sl3d

    Examples
    --------
    .. code-block:: python

        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0, 0, 0, 1, 1, 1)
        print(a.length, a.mid)
    """
    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'f', 'pnta', 'pntb')

    def __init__(self, x0=0, y0=0, z0=0, x1=1, y1=0, z1=0, pnta=None, pntb=None):
        """Initialise from six scalar coordinates or two ``Point3d`` endpoints.

        Parameters
        ----------
        x0, y0, z0 : float, optional
            Start-point coordinates.  Used when ``pnta`` is ``None``.
            Default is (0, 0, 0).
        x1, y1, z1 : float, optional
            End-point coordinates.  Used when ``pntb`` is ``None``.
            Default is (1, 0, 0).
        pnta : Point3d, optional
            Start ``Point3d`` object.  When provided, ``x0/y0/z0`` are
            derived from it.
        pntb : Point3d, optional
            End ``Point3d`` object.  When provided, ``x1/y1/z1`` are
            derived from it.
        """
        if pnta is None and pntb is None:
            self.x0, self.y0, self.z0 = x0, y0, z0
            self.x1, self.y1, self.z1 = x1, y1, z1
            self.pnta, self.pntb = Point3d(x0, y0, z0), Point3d(x1, y1, z1)
        if pnta is not None and pntb is not None:
            self.pnta, self.pntb = pnta, pntb
            self.x0, self.y0, self.z0 = pnta.x, pnta.y, pnta.z
            self.x1, self.y1, self.z1 = pntb.x, pntb.y, pnta.z

    def __repr__(self):
        """Return a concise line summary string.

        Returns
        -------
        str
            ``UPXO-sl3d (x0,y0,z0)-(x1,y1,z1). <id>`` summary.
        """
        return f'UPXO-sl3d ({self.x0},{self.y0},{self.z0})-({self.x1},{self.y1},{self.z1}). {id(self)}'

    def __iter__(self):
        """Iterate over the two endpoint coordinate tuples.

        Returns
        -------
        iterator
            Iterator over ``(x0, y0, z0)`` and ``(x1, y1, z1)``.
        """
        return (i for i in ((self.x0, self.y0, self.z0),
                            (self.x1, self.y1, self.z1)))

    def __getitem__(self, index):
        """Return an endpoint by index.

        Parameters
        ----------
        index : int
            Endpoint index. ``0`` returns the start point and ``1`` returns
            the end point.

        Returns
        -------
        tuple
            Endpoint coordinate tuple.
        """
        return ((self.x0, self.y0, self.z0),
                (self.x1, self.y1, self.z1))[index]

    def __eq__(self, lines):
        """Return per-line equality with ``self`` based on length comparison.

        Parameters
        ----------
        lines : iterable of Sline3d
            Lines to compare against ``self``.

        Returns
        -------
        list of bool
            ``True`` where the other line has the same length as ``self``.
        """
        length = self.length
        return [length == e.length for e in lines]

    def __ne__(self, lines):
        """Return per-line inequality with ``self``.

        Parameters
        ----------
        lines : iterable of Sline3d
            Lines to compare against ``self``.

        Returns
        -------
        list of bool
            ``True`` where the corresponding equality comparison is false.
        """
        return [not eeq for eeq in self == lines]

    def __lt__(self, lines):
        """Return per-line ``self.length < other.length`` comparisons.

        Parameters
        ----------
        lines : iterable of float
            Length values or comparable objects used by the existing
            implementation.

        Returns
        -------
        list of bool
            Result of comparing ``self.length`` with each supplied item.
        """
        length = self.length
        return [length < l for l in lines]

    @classmethod
    def by_coord(cls, start, end):
        """Construct a ``Sline3d`` from two ``[x, y, z]`` coordinate lists.

        Parameters
        ----------
        start : list of float, length 3
            Start-point coordinates ``[x0, y0, z0]``.
        end : list of float, length 3
            End-point coordinates ``[x1, y1, z1]``.

        Returns
        -------
        Sline3d
            New line from ``start`` to ``end``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            A = sl3d.by_coord([-1, 2, 0], [3, 4, 1])
        """
        return cls(start[0], start[1], start[2], end[0], end[1], end[2])

    @classmethod
    def by_p3d(cls, start, end):
        """Construct a ``Sline3d`` from two ``Point3d`` endpoint objects.

        Parameters
        ----------
        start : Point3d
            Start-point UPXO point object.
        end : Point3d
            End-point UPXO point object.

        Returns
        -------
        Sline3d
            New line from ``start`` to ``end``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d
            from upxo.geoEntities.sline3d import Sline3d
            Sline3d.by_p3d(Point3d(-1, 2, 3), Point3d(3, 4, 1))
        """
        return cls(pnta=start, pntb=end)

    @classmethod
    def by_vector(cls, point, xyproj):
        """Construct from an endpoint and a direction vector.

        Parameters
        ----------
        point : Point3d or array-like
            Endpoint through which the line should pass.
        xyproj : array-like
            Direction or projected vector specification.

        Raises
        ------
        NotImplementedError
            This constructor is not yet implemented.
        """
        raise NotImplementedError("by_vector is not yet implemented.")

    @property
    def mid(self):
        """Midpoint of the line as ``(xmid, ymid, zmid)`` tuple.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).mid
        """
        return (self.xmid, self.ymid, self.zmid)

    @property
    def xmid(self):
        """x-coordinate of the midpoint.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).xmid
        """
        return (self.x0 + self.x1)/2

    @property
    def ymid(self):
        """y-coordinate of the midpoint.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).ymid
        """
        return (self.y0 + self.y1)/2

    @property
    def zmid(self):
        """z-coordinate of the midpoint.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).zmid
        """
        return (self.z0 + self.z1)/2

    @property
    def gradient(self):
        """Gradient of the line as ``[dx/dz, dy/dz]``.

        Returns ``math.inf`` in the corresponding component when the line is
        horizontal (``z0 == z1``).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).gradient
        """
        grad = [math.inf, math.inf]
        dx, dy, dz = self.delxyz
        if self.z0 != self.z1:
            grad[0] = (dx)/(dz)
        if self.z0 != self.z1:
            grad[1] = (dy)/(dz)
        return grad

    @property
    def delxyz(self):
        """Length increments ``(dx, dy, dz)`` along each axis.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).delxyz
        """
        return self.dx, self.dy, self.dz

    @property
    def dx(self):
        """Signed length increment along x (``x1 - x0``).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).dx
        """
        return self.x1-self.x0

    @property
    def dy(self):
        """Signed length increment along y (``y1 - y0``).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).dy
        """
        return self.y1-self.y0

    @property
    def dz(self):
        """Signed length increment along z (``z1 - z0``).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).dz
        """
        return self.z1-self.z0

    @property
    def ang(self):
        """Orientation angles in radians as ``[angle_x, angle_y, angle_z]``.

        Computed as:
        ``angle_x = atan2(dy, dz)``,
        ``angle_y = atan2(dx, dz)``,
        ``angle_z = atan2(dy, dx)``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).ang
        """
        dx, dy, dz = self.delxyz
        angle_x = math.atan2(dy, dz)
        angle_y = math.atan2(dx, dz)
        angle_z = math.atan2(dy, dx)
        return [angle_x, angle_y, angle_z]

    @property
    def angd(self):
        """Orientation angles in degrees as ``[angle_x, angle_y, angle_z]``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).angd
        """
        return [math.degrees(ang) for ang in self.ang]

    @property
    def length(self):
        """Euclidean length of the line segment.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).length
        """
        return math.sqrt((self.x1-self.x0)**2+(self.y1-self.y0)**2+(self.z1-self.z0)**2)

    @property
    def dc(self):
        """Direction cosines ``(dx/L, dy/L, dz/L)`` where L is the length.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).dc
        """
        dx, dy, dz = self.delxyz
        length = self.length
        return dx/length, dy/length, dz/length

    @property
    def coords(self):
        """Flat coordinate list ``[x0, y0, z0, x1, y1, z1]``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).coords
        """
        return [self.x0, self.y0, self.z0, self.x1, self.y1, self.z1]

    @property
    def coord_list(self):
        """Endpoint coordinates as ``[[x0, y0, z0], [x1, y1, z1]]``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            sl3d(0, 0, 0, 1, 1, 1).coord_list
        """
        return [[self.x0, self.y0, self.z0], [self.x1, self.y1, self.z1]]

    @property
    def coord_i(self):
        """Start coordinate.

        Returns
        -------
        list of float
            ``[x0, y0, z0]``.
        """
        return [self.x0, self.y0, self.z0]

    @property
    def coord_j(self):
        """End coordinate.

        Returns
        -------
        list of float
            ``[x1, y1, z1]``.
        """
        return [self.x1, self.y1, self.z1]

    @property
    def points(self):
        """Return endpoint and midpoint objects.

        Returns
        -------
        list of Point3d
            ``[Point3d(i), Point3d(mid), Point3d(j)]``.
        """
        from upxo.geoEntities.point3d import Point3d
        mp = self.mid
        return [Point3d(self.x0, self.y0, self.z0),
                Point3d(mp[0], mp[1], mp[2]),
                Point3d(self.x1, self.y1, self.z1)]

    def is_point_endpoint(self, point):
        """Return ``True`` if ``point`` coincides with either endpoint.

        Parameters
        ----------
        point : array-like, length 3
            Candidate ``[x, y, z]`` coordinate.

        Returns
        -------
        bool
            ``True`` if ``point`` equals ``(x0, y0, z0)`` or
            ``(x1, y1, z1)``; ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            line = sl3d(0, 0, 0, 1, 1, 1)
            print(line.is_point_endpoint((0, 0, 0)))  # True
            print(line.is_point_endpoint([1, 2, 3]))  # False
        """
        is_endpoint = False
        if np.any(np.all(np.array(self.coord_list) == point, axis=1)):
            is_endpoint = True
        return is_endpoint

    def invert(self):
        """Swap start and end endpoints in place.

        Returns
        -------
        None
            The line is modified in place.
        """
        ends = self.coord_list
        self.x0, self.y0, self.z0 = ends[1]
        self.x1, self.y1, self.z1 = ends[0]

    def move_i(self, point):
        """Move the start point to ``point`` in place.

        Parameters
        ----------
        point : array-like, length 3
            New start coordinate.

        Returns
        -------
        None
            The line is modified in place.
        """
        self.x0, self.y0, self.z0 = point

    def move_j(self, point):
        """Move the end point to ``point`` in place.

        Parameters
        ----------
        point : array-like, length 3
            New end coordinate.

        Returns
        -------
        None
            The line is modified in place.
        """
        self.x1, self.y1, self.z1 = point

    def distance_to_points(self, points=None, *, ref='all'):
        """Calculate the Euclidean distance from a reference location on the line to a set of points.

        Parameters
        ----------
        points : list of Point3d
            Target points to measure distance to.
        ref : {'all', 'i', 'start', 'mid', 'j', 'end'}, optional
            Reference location on the line:

            * ``'all'`` — compute from start, mid, and end (returns list of 3 arrays).
            * ``'i'`` / ``'start'`` — from the start endpoint.
            * ``'mid'`` — from the midpoint.
            * ``'j'`` / ``'end'`` — from the end endpoint.

            Default is ``'all'``.

        Returns
        -------
        list or numpy.ndarray
            When ``ref='all'``, a list of three distance arrays
            ``[dist_from_i, dist_from_mid, dist_from_j]``; otherwise a
            single distance array.

        Examples
        --------
        **Example 1** — distances from all three reference locations:

        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np

            line = sl3d(0, 0, 0, 1, 1, 1)
            points = [p3d(xy[0], xy[1], xy[2]) for xy in np.random.random((10, 3))]
            line.distance_to_points(points, ref='all')
            line.distance_to_points(points, ref='i')
            line.distance_to_points(points, ref='mid')
            line.distance_to_points(points, ref='j')
        """
        if ref == 'all':
            pnti, pntmid, pntj = self.points
            distances = [pnti.distance(points),
                         pntmid.distance(points),
                         pntj.distance(points)]
        elif ref in ('i', 'start'):
            distances = self.points[0].distance(points)
        elif ref in ('mid'):
            distances = self.points[1].distance(points)
        elif ref in ('j', 'end'):
            distances = self.points[2].distance(points)
        return distances

    def perp_distance(self, point, ptype='coord_list'):
        """Compute the perpendicular distance from a point to the infinite line.

        Parameters
        ----------
        point : array-like or list of Point3d
            Target point(s).  Interpretation depends on ``ptype``.
        ptype : {'coord_list', '[point3d]', 'point3d', 'p3d'}, optional
            Specifies the format of ``point``.  Default is ``'coord_list'``.

        Returns
        -------
        float
            Perpendicular distance from ``point`` to the line.

        Examples
        --------
        **Example 1** — single coordinate triple:

        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            line = sl3d(0, 0, 0, 1, 0, 0)
            line.perp_distance((1, 1, 1))

        **Example 2** — vectorised version for many points (see :meth:`perp_distance_vectorized`):

        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            line = sl3d(0, 0, 0, 1, 0, 0)
            plist = np.random.random((10000, 3))
            line.perp_distance_vectorized(plist, ptype='coord_list')
        """
        if ptype == 'coord_list':
            x, y, z = np.array(point)
        elif ptype in ('[point3d]', 'point3d', 'p3d'):
            if type(point) not in dth.dt.ITERABLES:
                plist = [point]
            x, y, z = np.array([[p.x, p.y, p.z] for p in plist]).T
        point = np.array([x, y, z])
        line_start = np.array(self.coord_list[0])
        line_end = np.array(self.coord_list[1])
        line_vec = line_end - line_start
        point_vec = point - line_start
        projection = np.dot(point_vec,
                            line_vec) / np.dot(line_vec, line_vec) * line_vec
        perpendicular_vec = point_vec - projection
        distance = np.linalg.norm(perpendicular_vec)
        return distance

    def perp_distance_vectorized(self, points, ptype='coord_list'):
        """Compute perpendicular distances from many points to the infinite line (vectorised).

        Parameters
        ----------
        points : numpy.ndarray, shape (M, 3), or list of Point3d
            Target points.
        ptype : {'coord_list', '[point3d]', 'point3d', 'p3d'}, optional
            Format of ``points``.  Default is ``'coord_list'``.

        Returns
        -------
        numpy.ndarray, shape (M,)
            Perpendicular distance from each point to the line.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            line = sl3d(0, 0, 0, 0.5, 0.5, 0.5)
            x = np.linspace(0, 1, 4)
            points = np.stack(np.meshgrid(x, x, x), axis=-1).reshape(-1, 3)
            line.perp_distance_vectorized(points, ptype='coord_list')
        """
        if ptype == 'coord_list':
            points = np.array(points)
        elif ptype in ('[point3d]', 'point3d', 'p3d'):
            points = np.array([[p.x, p.y, p.z] for p in points])
        line_start = np.array(self.coord_list[0])
        line_end = np.array(self.coord_list[1])
        line_vec = line_end - line_start
        point_vec = points - line_start
        line_vec_norm_sq = np.dot(line_vec, line_vec)
        projection = (np.dot(point_vec, line_vec)[:, None] / line_vec_norm_sq) * line_vec
        perpendicular_vec = point_vec - projection
        distances = np.linalg.norm(perpendicular_vec, axis=1)
        return distances

    def extend(self, dincr, direction='both', saa=True, throw=False):
        """Extend the line by ``dincr`` at the start, end, or both ends.

        Parameters
        ----------
        dincr : float
            Extension increment (in the same units as the line coordinates).
        direction : {'both', 'start', 'end'}, optional
            Which end(s) to extend.  Default is ``'both'``.
        saa : bool, optional
            If ``True``, update ``self`` in place.  Default is ``True``.
        throw : bool, optional
            If ``True``, return the new endpoint arrays.  Default is ``False``.

        Returns
        -------
        tuple of numpy.ndarray or None
            When ``throw=True``, returns ``(P1_extended, P2_extended)``;
            otherwise ``None``.
        """
        P1 = np.array([self.x0, self.y0, self.z0])
        P2 = np.array([self.x1, self.y1, self.z1])
        line_vec = P2 - P1
        unit_vec = line_vec / np.linalg.norm(line_vec)
        if direction == 'start':
            P1_extended = P1 - dincr * unit_vec
            P2_extended = P2
        elif direction == 'end':
            P1_extended = P1
            P2_extended = P2 + dincr * unit_vec
        elif direction == 'both':
            P1_extended = P1 - dincr * unit_vec
            P2_extended = P2 + dincr * unit_vec
        else:
            raise ValueError("Invalid direction. Choose 'start', 'end', or 'both'.")
        if saa:
            self.x0, self.y0, self.z0 = P1_extended
            self.x1, self.y1, self.z1 = P2_extended
        if throw:
            return P1_extended, P2_extended

    def extend_until_exhaustion(self,
                                points,
                                voxel_size=1.0,
                                dincr_factor=0.1,
                                direction='end',
                                max_iterations=1000,
                                saa_final=True,
                                throw_final=False):
        """Iteratively extend the line until no new points fall within the threshold distance.

        At each iteration the line is extended by ``dincr_factor * length`` in
        the requested direction.  Iteration stops when the count of nearby
        points ceases to increase.

        Parameters
        ----------
        points : array-like, shape (M, 3)
            3-D point cloud the line is extended towards.
        voxel_size : float, optional
            Voxel edge length; the proximity threshold is
            ``sqrt(3) * voxel_size`` (body diagonal).  Default is 1.0.
        dincr_factor : float, optional
            Extension per iteration as a fraction of the current line length.
            Default is 0.1.
        direction : {'end', 'start', 'both'}, optional
            Which end(s) to extend.  Default is ``'end'``.
        max_iterations : int, optional
            Safety cap on the number of iterations.  Default is 1000.
        saa_final : bool, optional
            If ``True``, set ``self`` endpoints to the final detected extent.
            Default is ``True``.
        throw_final : bool, optional
            If ``True``, return the final endpoint coordinates.
            Default is ``False``.

        Returns
        -------
        tuple of numpy.ndarray or None
            When ``throw_final=True`` and ``direction='end'``, returns
            ``(start_point, actual_end_point)``; for ``direction='both'``,
            returns ``(actual_start_point, actual_end_point)``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            LINE = sl3d(0, 0, 0, 0.5, 0.5, 0.5)
            x = np.linspace(0, 5, 4)
            points = np.stack(np.meshgrid(x, x, x), axis=-1).reshape(-1, 3)
            LINE.extend_until_exhaustion(points,
                                         voxel_size=x[1]-x[0],
                                         dincr_factor=0.1,
                                         direction='end')
            print(LINE)
        """
        threshold_distance = 1.7320508075688772 * voxel_size

        P1 = np.array([self.x0, self.y0, self.z0])
        P2 = np.array([self.x1, self.y1, self.z1])
        points = np.array(points)

        pdist = self.perp_distance_vectorized(points, ptype='coord_list')
        initial_count = np.sum(pdist <= threshold_distance)
        prev_count = initial_count

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            P1_extended, P2_extended = self.extend(dincr_factor*self.length,
                                                   direction=direction,
                                                   saa=False, throw=True)
            _line_ = Sline3d(P1_extended[0], P1_extended[1], P1_extended[2],
                             P2_extended[0], P2_extended[1], P2_extended[2])

            self.x0, self.y0, self.z0 = P1_extended
            self.x1, self.y1, self.z1 = P2_extended

            pdist = _line_.perp_distance_vectorized(points, ptype='coord_list')
            new_count = np.sum(pdist <= threshold_distance)

            if new_count <= prev_count:
                close_pooints = np.argwhere(pdist <= threshold_distance)
                if direction == 'both':
                    dist_i = self.distance_to_points(points, ref='start')
                    dist_j = self.distance_to_points(points, ref='end')
                    far_pnt_i, far_pnt_j = np.argmin(dist_i), np.argmin(dist_j)
                    actual_start_point = points[far_pnt_i]
                    actual_end_point = points[far_pnt_j]
                    if saa_final:
                        self.x0, self.y0, self.z0 = actual_start_point.tolist()
                        self.x1, self.y1, self.z1 = actual_end_point.tolist()
                    if throw_final:
                        print('The start and end point indices are:')
                        return actual_start_point, actual_end_point
                elif direction == 'start':
                    pass
                elif direction == 'end':
                    far_pnt = np.argmax(_line_.distance_to_points(points,
                                                                   ref='end'))
                    actual_end_point = points[far_pnt]
                    print(f'No. of iterations: {iteration}')
                    if saa_final:
                        self.x1, self.y1, self.z1 = actual_end_point
                    if throw_final:
                        print('The start and end point indices are:')
                        start_point = np.array([self.x1, self.y1, self.z1])
                        return start_point, actual_end_point
                break

            P1, P2 = P1_extended, P2_extended
            prev_count = new_count

    def split(self, f=0.5, saa=False, throw=True, retain=0):
        """Split the line at one or more fractional positions.

        Parameters
        ----------
        f : float or iterable of float
            Fractional position(s) along the line in the open interval (0, 1).
            When a scalar, the line is split into two segments.  Iterable
            support is reserved for future implementation.
        saa : bool, optional
            If ``True``, shorten ``self`` to retain the segment specified by
            ``retain`` and discard the other.  Default is ``False``.
        throw : bool, optional
            If ``True``, return the two new ``Sline3d`` objects.
            Default is ``True``.
        retain : {0, 1}, optional
            When ``saa=True``, which half to keep: 0 = start half,
            1 = end half.  Default is 0.

        Returns
        -------
        list of Sline3d or None
            When ``throw=True`` and ``saa=False``, returns
            ``[line_start_half, line_end_half]``; otherwise ``None``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.sline3d import Sline3d as sl3d
            line = sl3d(0, 0, 0, 1, 0, 0)
            halves = line.split(f=0.75, saa=False, throw=True)
            print(halves[0].length, halves[1].length)
        """
        if type(f) in dth.dt.NUMBERS:
            point = (self.x0+f*self.dx, self.y0+f*self.dy, self.z0+f*self.dz)
            if saa:
                if retain == 0:
                    self.x1, self.y1, self.z1 = point
                elif retain == 1:
                    self.x0, self.y0, self.z0 = point
                else:
                    raise ValueError('Invalid retain spec for True saa.')
            else:
                line0 = Sline3d.by_coord(self.coord_list[0], point)
                line1 = Sline3d.by_coord(point, self.coord_list[1])
                return [line0, line1]

        if type(f) in dth.dt.ITERABLES:
            # TODO
            pass
