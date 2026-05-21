"""
3D point geometry entities for UPXO.

Provides the :class:`Point3d` full-featured 3D point and the
:class:`p3d_leanest` lightweight variant for performance-critical use.

Classes
-------
Point3d
    Full-featured 3D point with coordinate operations, equality checks,
    distance calculations, and neighbour-finding methods.
p3d_leanest
    Minimal 3D point for performance-critical collections. Intended for
    internal use only.

Usage
-----
::

    from upxo.geoEntities.point3d import Point3d as p3d

@author: Dr. Sunil Anandatheertha
"""

import math
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree
import vtk
import upxo._sup.dataTypeHandlers as dth
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
from upxo.geoEntities.featmake import make_p2d, make_p3d
from upxo._sup.validation_values import find_spec_of_points
from upxo._sup.validation_values import isinstance_many

class _coord_():
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        """Store raw 3D coordinate values in ``x``, ``y``, ``z``."""
        self.x, self.y, self.z = x, y, z


class p3d_leanest():
    """
    Minimal 3D point class for performance-critical internal use.

    Notes
    -----
    Intentionally minimal — no methods beyond ``__init__`` and ``__repr__``.
    Further development should be avoided to keep memory overhead low.

    Examples
    --------
    .. code-block:: python

        from upxo.geoEntities.point3d import p3d_leanest

        pts = [p3d_leanest(1, 2, 0), p3d_leanest(3, 4, 1)]
    """

    __slots__ = ('_x', '_y', '_z')

    def __init__(self, x, y, z):
        """Initialise the leanest 3D point with ``x``, ``y``, ``z`` coordinates."""
        self._x, self._y, self._z = x, y, z

    def __repr__(self):
        """Return string representation of self."""
        return f'Lean 3D point at ({self._x}, {self._y}, {self._z}, {id(self)})'


class Point3d(UPXO_Point):
    """
    3D point entity for UPXO.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    z : float, optional
        z-coordinate (default 0.0).

    Notes
    -----
    Inherits from :class:`~upxo.geoEntities.bases.UPXO_Point`. Pydantic
    validation is intentionally avoided to minimise instantiation cost and
    memory overhead.

    Examples
    --------
    .. code-block:: python

        from upxo.geoEntities.point3d import Point3d as p3d

        A, B, C = p3d(10, 12, 0), p3d(10, 12, 0), p3d(11, 12, 0)
        print(A, B, C)

        print(A == B, A != B, A == C, A != C)

        A * 2
        A + [10, 20, 30]
    """

    ε = 1E-8
    __slots__ = UPXO_Point.__slots__ + ('z', )

    def __init__(self, x, y, z=0.0):
        """Initialise a 3D point with Cartesian coordinates ``x``, ``y``, ``z``."""
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Return a string representation of point3d instance."""
        return f"uxpo-p3d ({self.x},{self.y},{self.z})"

    def __eq__(self, plist):
        """
        Test equality between self and one or more points.

        Parameters
        ----------
        plist : point or list of points
            A single point object, list of point objects, or nested
            coordinate list.

        Returns
        -------
        list of bool
            Element-wise equality results.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d, p3d_leanest

            print(p3d(3, 4, 5) == p3d_leanest(3, 4, 5))
            print(p3d(3, 4, 5) == [p3d_leanest(1, 4, 5), p3d_leanest(3, 4, 5)])
            print(p3d(3, 4, 5) == p3d(3, 4, 5))
            print(p3d(3, 4, 5) == [p3d(1, 2, 5), p3d(3, 4, 5)])
            print(p3d(3, 4, 5) == (p3d(1, 4, 5), p3d(3, 4, 5)))
            print(p3d(3, 4, 5) == [[1, 2, 5], [3, 4, 5], [5, 6, 5]])
            print(p3d(3, 4, 5) == [[1, 3, 5], [2, 4, 6]])
            print(p3d(3, 4, 5) == [[3, 4, 5]])
            print(p3d(3, 4, 5) == [[3], [4], [5]])
        """
        if not plist:
            raise ValueError("plist is empty.")
        spec_found = False
        # ---------------------------------------------------------
        point_spec = find_spec_of_points(plist)
        # ---------------------------------------------------------
        if point_spec in dth.opt.name_point2d_specs:
            """We are dealing with a 2D point"""
            cmp, spec_found = [False], True
        elif point_spec in dth.opt.name_point3d_specs:
            """We are dealing with a 3D point"""
            if point_spec == 'Point3d':
                cmp, spec_found = [(self.x,
                                    self.y,
                                    self.z) == (plist.x,
                                                plist.y,
                                                plist.z)], True
            # ---------------------------------------------------------
            if point_spec == '[Point3d]':
                cmp, spec_found = [(self.x, self.y, self.z) == (p.x, p.y, p.z)
                                   for p in plist], True
            # ---------------------------------------------------------
            if point_spec == 'p3d_leanest':
                cmp, spec_found = [(self.x,
                                    self.y,
                                    self.z) == (plist._x,
                                                plist._y,
                                                plist._z)], True
            # ---------------------------------------------------------
            if point_spec == '[p3d_leanest]':
                cmp, spec_found = [(self.x, self.y, self.z) == (p._x,
                                                                p._y,
                                                                p._z)
                                   for p in plist], True
            # ---------------------------------------------------------
            if point_spec == 'type-[1,2,3]':
                cmp, spec_found = [(self.x,
                                    self.y,
                                    self.z) == tuple(plist)], True
            # ---------------------------------------------------------
            if point_spec == 'type-[[1,2,3]]':
                cmp, spec_found = [(self.x,
                                    self.y,
                                    self.z) == tuple(plist[0])], True
            # ---------------------------------------------------------
            if point_spec == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
                cmp = [self.x == p[0] and self.y == p[1] and self.z == p[2]
                       for p in plist]
                spec_found = True
            # ---------------------------------------------------------
            if point_spec == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
                cmp = [self.x == _x and self.y == _y and self.z == _z
                       for _x, _y, _z in zip(plist[0], plist[1], plist[2])]
                spec_found = True
        # ---------------------------------------------------------
        if spec_found:
            return cmp
        else:
            return None

    def __ne__(self, plist, use_tol=True):
        """Return ``True`` if self is not equal to any point in ``plist``."""
        return not self.__eq__(plist, use_tol=use_tol)

    def eq(self, plist, use_tol=False, tolerance=None):
        """
        Overloaded equality check with optional tolerance.

        Parameters
        ----------
        plist : point or list of points
            Points to compare against self.
        use_tol : bool, optional
            If True, apply tolerance (not yet implemented).
        tolerance : float, optional
            Tolerance threshold (not yet implemented).

        Returns
        -------
        list of bool
            Element-wise equality results.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d, p3d_leanest

            print(p3d(3, 4, 5).eq(p3d_leanest(3, 4, 5)))
            print(p3d(3, 4, 5).eq([p3d_leanest(1, 4, 5), p3d_leanest(3, 4, 5)]))
            print(p3d(3, 4, 5).eq(p3d(3, 4, 5)))
            print(p3d(3, 4, 5).eq([p3d(1, 2, 5), p3d(3, 4, 5)]))
            print(p3d(3, 4, 5).eq((p3d(1, 4, 5), p3d(3, 4, 5))))
            print(p3d(3, 4, 5).eq([[1, 2, 5], [3, 4, 5], [5, 6, 5]]))
            print(p3d(3, 4, 5).eq([[1, 3, 5], [2, 4, 6]]))
            print(p3d(3, 4, 5).eq([[3, 4, 5]]))
            print(p3d(3, 4, 5).eq([[3], [4], [5]]))
        """
        if not use_tol:
            return self.__eq__(plist)

    def eq_fast(self, plist, use_tol=False, point_spec=1):
        """
        Fast equality check with explicit point-type specification.

        Parameters
        ----------
        plist : point or array-like
            Points to compare.
        use_tol : bool, optional
            Tolerance flag (not yet implemented).
        point_spec : int, optional
            Integer code identifying the layout of ``plist``:

            1 – ``Point3d``;
            2 – ``[Point3d]``;
            3 – ``p3d_leanest``;
            4 – ``[p3d_leanest]``;
            5 – ``[x, y, z]``;
            6 – ``[[x, y, z]]``;
            7 – ``[[x1,x2,...], [y1,y2,...], [z1,z2,...]]`` (row layout);
            8 – ``[[x1,...], [y1,...], [z1,...]]`` (column layout).

        Returns
        -------
        list of bool or None
            Element-wise equality results, or ``None`` for unrecognised spec.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d, p3d_leanest

            print(p3d(3, 4, 5).eq_fast(p3d(3, 4, 5), point_spec=1))
            print(p3d(3, 4, 5).eq_fast([p3d(1, 2, 5), p3d(3, 4, 5)], point_spec=2))
            print(p3d(3, 4, 5).eq_fast((p3d(1, 4, 5), p3d(3, 4, 5)), point_spec=2))
            print(p3d(3, 4, 5).eq_fast(p3d_leanest(3, 4, 5), point_spec=3))
            print(p3d(3, 4, 5).eq_fast([p3d_leanest(1, 4, 5), p3d_leanest(3, 4, 5)], point_spec=4))
            print(p3d(3, 4, 5).eq_fast([1, 2, 5], point_spec=5))
            print(p3d(3, 4, 5).eq_fast([[1, 2, 5]], point_spec=6))
            print(p3d(3, 4, 5).eq_fast([[1, 2, 5], [3, 4, 5], [5, 6, 5]], point_spec=7))
            print(p3d(3, 4, 5).eq_fast([[3], [4], [5]], point_spec=8))
        """
        cmp = None
        if point_spec == 1:
            """1: Point3d"""
            cmp = [(self.x, self.y, self.z) == (plist.x, plist.y, plist.z)]
        if point_spec == 2:
            """2: [Point3d]"""
            cmp = [(self.x, self.y, self.z) == (p.x, p.y, p.z) for p in plist]
        if point_spec == 3:
            """3: p3d_leanest"""
            cmp = [(self.x, self.y, self.z) == (plist._x, plist._y, plist._z)]
        if point_spec == 4:
            """4: [p3d_leanest]"""
            cmp = [(self.x, self.y, self.z) == (p._x, p._y, p._z)
                   for p in plist]
        if point_spec == 5:
            """5: type-[1,2,3]"""
            cmp = [(self.x, self.y, self.z) == tuple(plist)]
        if point_spec == 6:
            """6: type-[[1,2,3]]"""
            cmp = [(self.x, self.y, self.z) == tuple(plist[0])]
        if point_spec == 7:
            """7: type-[[1,2,3],[4,5,6],[7,8,9]]"""
            cmp = [self.x == p[0] and self.y == p[1] and self.z == p[2]
                   for p in plist]
        if point_spec == 8:
            """8: type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]"""
            cmp = [self.x == _x and self.y == _y and self.z == _z
                   for _x, _y, _z in zip(plist[0], plist[1], plist[2])]
        return cmp

    def add(self, d, update=True, throw=False, mydecatlen2NUM='b'):
        """Add scalar or vector ``d`` to self coordinates. Not yet implemented."""
        raise NotImplementedError("add is not yet implemented.")

    def __mul__(self, f=1.0, update=True, throw=False):
        """Scale self coordinates by factor ``f``."""
        # Validate f
        # ----------------------------------
        # DEVELOPMENT STAGE - 1
        # TARGET: succeffull working when f is a single number
        if not isinstance(f, dth.dt.NUMBERS):
            raise TypeError('Invald factor')
        if update:
            self.x *= f
            self.y *= f
            self.z *= f
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            return Point3d(self.x*f, self.y*f, self.z*f)

    @classmethod
    def from_three_planes(cls, plane1, plane2, plane3):
        """Find the point of intersection of three planes.

        Parameters
        ----------
        plane1 : Plane
            First plane.
        plane2 : Plane
            Second plane.
        plane3 : Plane
            Third plane.

        Returns
        -------
        Point3d or None
            Intersection point, or ``None`` if no unique solution exists
            (parallel or coincident planes).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d
            from upxo.geoEntities.plane import Plane

            plane1 = Plane(point=(0, 0, 1), normal=(1, 1, 1))
            plane2 = Plane(point=(0, 0, 0), normal=(0, 1, 0))
            plane3 = Plane(point=(0, 0, 0), normal=(0, 0, 1))
            intersection_point = p3d.from_three_planes(plane1, plane2, plane3)
            print(intersection_point)
        """
        A = np.array([plane1.normal, plane2.normal, plane3.normal])
        D = np.array([[plane1.point @ plane1.normal],
                      [plane2.point @ plane2.normal],
                      [plane3.point @ plane3.normal]])
        try:
            X = np.linalg.solve(A, D).flatten()
            return cls(X[0], X[1], X[2])
            # return X.flatten()  # Return as a 1D NumPy array
        except np.linalg.LinAlgError:
            return None  # No solution

    @property
    def coords(self):
        """Return ``[x, y]`` as a numpy array (2D projection of this 3D point)."""
        return np.array([self.x, self.y])

    def squared_distance(self, plist=None, point_spec=-1):
        """
        Calculate squared distances between self and one or more points.

        Parameters
        ----------
        plist : point or array-like
            Single point, list of points, or coordinate array. The expected
            layout depends on ``point_spec``.
        point_spec : int, optional
            Integer code identifying the type of ``plist`` (default ``-1``
            for automatic detection):

            -1 – auto-detect via :func:`~upxo.geoEntities.featmake.make_p3d`;
             1 – ``Point3d``;
             2 – ``[Point3d]``;
             3 – ``p3d_leanest``;
             4 – ``[p3d_leanest]``;
             5 – ``(x, y, z)`` tuple;
             6 – ``[(x, y, z)]``;
             7 – ``[[x,y,z], ...]`` row-major;
             8 – ``[[x1,...], [y1,...], [z1,...]]`` column-major or ``(3, n)`` array.

        Returns
        -------
        float or numpy.ndarray
            Squared Euclidean distance(s) from self to each input point.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d, p3d_leanest
            import numpy as np

            p3d(0, 0, 0).squared_distance(p3d(1, 1, 0), point_spec=-1)
            p3d(0, 0, 0).squared_distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=-1)
            points = np.random.random((3, 100000))
            p3d(0, 0, 0).squared_distance(points)
            p3d(0, 0, 0).squared_distance(p3d(1, 1, 0), point_spec=1)
            p3d(0, 0, 0).squared_distance([p3d(1, 1, 0)], point_spec=2)
            p3d(0, 0, 0).squared_distance(p3d_leanest(1, 1, 0), point_spec=3)
            p3d(0, 0, 0).squared_distance([p3d_leanest(1, 1, 0)], point_spec=4)
            p3d(0, 0, 0).squared_distance((1, 1, 0), point_spec=5)
            p3d(0, 0, 0).squared_distance([(1, 1, 0)], point_spec=6)
            p3d(0, 0, 0).squared_distance([[1,2,3],[4,5,6],[7,8,9]], point_spec=7)
            p3d(0, 0, 0).squared_distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=8)
            p3d(0, 0, 0).squared_distance(np.random.random((3, 100000)), point_spec=8)
        """
        # Validations
        if plist is None:
            # Not very apt. Need to change with more developmet.
            raise ValueError('Invalid list of points')
        # ------------------------------------------
        if point_spec == -1:
            plist = make_p3d(plist, return_type='leanest')
            X, Y, Z = np.array([[p._x, p._y, p._z] for p in plist]).T
        # ------------------------------------------
        if point_spec == 1:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 1: Point3d.
            p3d(0,0,0).squared_distance(p3d(1,1,0), point_spec=1)
            """
            X, Y, Z = plist.x, plist.y, plist.z
        if point_spec == 2:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 2: [Point3d]
            p3d(0,0,0).squared_distance([p3d(1,1,0)], point_spec=2)
            """
            X, Y, Z = plist[0].x, plist[0].y, plist[0].z
        if point_spec == 3:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            from upxo.geoEntities.point3d import p3d_leanest
            # point_spec = 3: p3d_leanest
            p3d(0,0,0).squared_distance(p3d_leanest(1,1,0), point_spec=3)
            """
            X, Y, Z = plist._x, plist._y, plist._z
        if point_spec == 4:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            from upxo.geoEntities.point3d import p3d_leanest
            # point_spec = 4: [p3d_leanest]
            p3d(0,0,0).squared_distance([p3d_leanest(1,1,0)], point_spec=4)
            """
            X, Y, Z = plist[0]._x, plist[0]._y, plist[0]._z
        if point_spec == 5:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 5: type-[1,2,3]
            p3d(0,0,0).squared_distance((1,1,0), point_spec=5)
            """
            X, Y, Z = plist
        if point_spec == 6:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 6: type-[[1,2,3]]
            p3d(0, 0, 0).squared_distance([(1,1,0)], point_spec=6)
            """
            X, Y, Z = plist[0]
        if point_spec == 7:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 7: type-[[1,2,3],[4,5,6],[7,8,9]]
            p3d(0, 0, 0).squared_distance([[1,2,3],[4,5,6],[7,8,9]],
                                          point_spec=7)
            """
            X, Y, Z = np.array(plist).T
        if point_spec == 8:
            """
            from upxo.geoEntities.point3d import Point3d as p3d
            # point_spec = 8: type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]
            p3d(0, 0, 0).squared_distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]],
                                          point_spec=8)

            p3d(0, 0, 0).squared_distance(np.random.random((3, 100000)),
                                          point_spec=8)
            """
            X, Y, Z = np.array(plist)
        return (self.x-X)**2 + (self.y-Y)**2 + (self.z-Z)**2

    def distance(self, plist=None, point_spec=-1):
        """
        Calculate Euclidean distances between self and one or more points.

        Parameters
        ----------
        plist : point or array-like
            See :meth:`squared_distance` for accepted formats.
        point_spec : int, optional
            Point layout code passed through to :meth:`squared_distance`.

        Returns
        -------
        float or numpy.ndarray
            Euclidean distance(s) from self to each input point.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d, p3d_leanest
            import numpy as np

            p3d(0, 0, 0).distance(p3d(1, 1, 0), point_spec=-1)
            p3d(0, 0, 0).distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=-1)
            p3d(0, 0, 0).distance(p3d(1, 1, 0), point_spec=1)
            p3d(0, 0, 0).distance([p3d(1, 1, 0)], point_spec=2)
            p3d(0, 0, 0).distance(p3d_leanest(1, 1, 0), point_spec=3)
            p3d(0, 0, 0).distance([p3d_leanest(1, 1, 0)], point_spec=4)
            p3d(0, 0, 0).distance((1, 1, 0), point_spec=5)
            p3d(0, 0, 0).distance([(1, 1, 0)], point_spec=6)
            p3d(0, 0, 0).distance([[1,2,3],[4,5,6],[7,8,9]], point_spec=7)
            p3d(0, 0, 0).distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=8)
            p3d(0, 0, 0).distance(np.random.random((3, 100000)), point_spec=8)
        """
        return np.sqrt(self.squared_distance(plist=plist,
                                             point_spec=point_spec))

    def translate(self, *, vector=None, dist=None, update=False,
                  throw=True):
        """
        Translate self along a vector by a given distance.

        Parameters
        ----------
        vector : array-like
            Direction vector of translation.
        dist : float
            Distance to translate along ``vector``.
        update : bool, optional
            If True, modify self in place.
        throw : bool, optional
            If True, return the translated point.

        Returns
        -------
        Point3d or None
            Translated point if ``throw`` is True, else ``None``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d

            A = p3d(0, 0, 0)
            A.translate(vector=[-1, -1, -1], dist=3.4641016151377544,
                        update=True, throw=False)
            print(A)
        """
        distances = (np.array(vector) / np.linalg.norm(vector)) * dist
        if update:
            self.x += distances[0]
            self.y += distances[1]
            self.z += distances[2]
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            return Point3d(self.x+distances[0],
                           self.y+distances[1],
                           self.y+distances[2])

    def translate_to(self, *, point=None, update=False, throw=True):
        """Translate self to the location of ``point``."""
        if not point:
            raise ValueError('Must provide point object. Could also be coord.')
        xloc, yloc, zloc = Point3d.validate_single_point_input(point)
        if update:
            self.x, self.y, self.z = xloc, yloc, zloc
        if update and throw:
            # Retain this here, as the behaviour may change later on
            return deepcopy(self)
        if not update and throw:
            return Point3d(xloc, yloc, zloc)

    @staticmethod
    def val_points_and_get_coords(points):
        """Docstring."""
        if not points:
            raise ValueError('Points OR coords not provided.')
        if type(points) in dth.dt.ITERABLES:
            if len(set(type(point) for point in points)) != 1:
                raise ValueError('Points array contains multiple datatypes.')
        else:
            if find_spec_of_points(points) not in ('Point2d', 'p2d_leanest'):
                raise ValueError('Invalid datatype of the proivided single point.')
            points = [points]
        # --------------------------------------------------
        valid = False
        if find_spec_of_points(points[0]) in 'Point3d':
            x, y, valid = [p.x for p in points], [p.y for p in points], True
            z = [p.z for p in points]
        if find_spec_of_points(points[0]) == 'p3d_leanest':
            x, y, valid = [p._x for p in points], [p.y for p in points], True
            z = [p.z for p in points]
        if find_spec_of_points(points) == 'type-[1,2,3]':
            x, y, valid = [points[0]], [points[1]], True
            z = [p[2] for p in points]
        if find_spec_of_points(points) == 'type-[[1,2,3]]':
            x, y, valid = [points[0][0]], [points[0][1]], True
            z = [points[0][2] for p in points]
        if find_spec_of_points(points) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
            x, y, valid = [p[0] for p in points], [p[1] for p in points], True
            z = [p[2] for p in points]
        if find_spec_of_points(points) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
            x, y, valid = points[0], points[1], True
            z = points[2]
        if not valid:
            raise ValueError('Invalid points input.')
        return x, y

    def attach_feature(self, *, feature=None, feature_id=None):
        """Attach a feature object to ``self.f[feature_id]``."""
        if not feature:
            raise ValueError('feature cannot be empty.')
        if not feature_id:
            raise ValueError('feature_id cannot be empty.')
        fname = feature.__class__.__name__
        if not hasattr(self, 'f'):
            self.f = {}
        if fname not in self.f:
            self.f[fname] = {}
        if feature_id in self.f[fname].keys():
            raise KeyError('Cannot attach feature. feature_id: '
                           f'{feature_id} already in dict {self.f[fname]}.')
        self.f[fname][feature_id] = feature

    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0,
                                     on_boundary=True,
                                     threshold_perp_dist=0.0):
        """
        Things to do:
            1. validations
            2. consider plane in calculations. If plane is None, then all
                point locatyions within or withon r will be returned. If
                plane is specified differenyly, then return locations of
                those points which aactually satisfy both r and contained on
                or close to the plane. The closeneess sho9uld be determined by
                threshold_perp_dist, which is sthe threshold perpendicualr
                distance between a candidate point and yhe plane.
        """
        plist = np.array(plist)
        if type(r) not in dth.dt.NUMBERS:
            raise TypeError('Invalid r type.')
        sd = self.squared_distance(plist)
        if r <= self.ε:
            return np.argwhere(sd == sd.min()).squeeze()
        else:
            if on_boundary:
                return np.argwhere(sd <= r)
            else:
                return np.argwhere(sd < r)

    def find_neigh_point_by_count(self, *, plist=None, n=None, plane='xy'):
        """
        Find the ``n`` nearest points in ``plist`` by squared distance.

        Parameters
        ----------
        plist : array-like
            Pool of candidate points.
        n : int
            Number of nearest neighbours to return.
        plane : str, optional
            Reserved for plane-filtered searches. Currently unused.

        Returns
        -------
        tuple of numpy.ndarray
            Indices of the ``n`` nearest points in ``plist``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d

            p3d(0, 0, 0).find_neigh_point_by_count(
                plist=[[1, 2, 0], [10, 12, 0], [0, -5, 0], [0, 0, 0]], n=2)
        """
        # Validate plist
        # Validate n
        if not isinstance(n, int) or n == 0:
            raise TypeError('n must be an int type and non-zero.')
        if n > len(plist):
            raise ValueError('n is greater than len(plist).')
        sd = self.squared_distance(plist)
        return np.where(np.in1d(sd, np.sort(sd)[:n]))

    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        """Find neighbouring multi-point objects within radius ``r``. Not yet implemented."""
        # Use the ckdtree option.
        raise NotImplementedError("find_neigh_mulpoint_by_distance is not yet implemented.")

    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """Find neighbouring edges within radius ``r``. Not yet implemented."""
        raise NotImplementedError("find_neigh_edge_by_distance is not yet implemented.")

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """Find neighbouring multi-edges within radius ``r``. Not yet implemented."""
        raise NotImplementedError("find_neigh_muledge_by_distance is not yet implemented.")

    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """Find neighbouring crystals within radius ``r``. Not yet implemented."""
        raise NotImplementedError("find_neigh_xtal_by_distance is not yet implemented.")

    def set_gmsh_props(self, prop_dict):
        """Apply GMSH mesh properties from ``prop_dict`` to this point. Not yet implemented."""
        raise NotImplementedError("set_gmsh_props is not yet implemented.")

    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        """Generate an array of translated copies along ``vector``. Not yet implemented."""
        raise NotImplementedError("array_translation is not yet implemented.")

    def lies_on_which_line(self, *, llist=None, consider_ends=True):
        """Return which line(s) in ``llist`` this point lies on. Not yet implemented."""
        raise NotImplementedError("lies_on_which_line is not yet implemented.")

    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """Return which crystal(s) in ``xlist`` contain this point. Not yet implemented."""
        raise NotImplementedError("lies_in_which_xtal is not yet implemented.")

    def make_vtk_point(self, z=0):
        """
        Create a VTK point dataset from self.

        Parameters
        ----------
        z : float, optional
            Unused; retained for API compatibility.

        Returns
        -------
        dict
            Dictionary with keys ``'id'`` (point ID), ``'pd'``
            (``vtkPolyData``), and ``'help'`` (usage hint string).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.point3d import Point3d as p3d

            A = p3d(10, 12, 100)
            vtkobj = A.make_vtk_point()
            x, y, z = vtkobj['pd'].GetPoint(vtkobj['id'])
            print(x, y, z)
        """
        points = vtk.vtkPoints()
        point_id = points.InsertNextPoint(self.x,
                                          self.y,
                                          self.z)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        return {'id': point_id,
                'pd': poly_data,
                'help': "return['pd'].GetPoint(return['id'])"}

    def make_shape(self):
        """Create a geometric shape representation of this point. Not yet implemented."""
        raise NotImplementedError("make_shape is not yet implemented.")


'''def isinstance_many(tocheck, dtype):
    """
    Check if all elements of tocheck belongs to a valid dtype.

    Arguments
    ---------
    tocheck: An iterable of data.
    dtype: Valid datatype, in dth.dt.ITERABLES

    Return
    ------
    list of bools. True indicates element belonging to dtype

    Example
    -------
    from upxo.geoEntities.point3d import p2d_leanest, p3d_leanest
    a = [p2d_leanest(1, 2), p3d_leanest(1, 2, 1)]
    isinstance_many(a, p3d_leanest)

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    if type(tocheck) not in dth.dt.ITERABLES:
        tocheck = (tocheck, )
    return [isinstance(tc, dtype) for tc in tocheck]

def all_isinstance(dtype, *args):
    if len(args) > 0:
        print(args)
        return all(isinstance(arg, dtype) for arg in args)
'''
