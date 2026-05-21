"""
2D Point geometric entity module for UPXO (Universal PolyXtal Operations).

This module provides two-dimensional point representations with support for
lightweight (leanest) and feature-rich implementations. It includes geometric
operations (distance, translation, rotation), type conversions (UPXO, Shapely,
VTK, PyVista, GMSH), and point array generation methods.

Import
------
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point2d import Point2d
    from upxo.geoEntities.point2d import Point2d as p2d

Classes
-------
p2d_leanest
    Minimal 2D point container (private coordinates, no features).
Point2d
    Full-featured 2D point with attachable features and extensive
    geometric operations.
_coord_
    Internal 3D coordinate storage used for z-level tracking.

Dependencies
------------
Core (module-level):
    numpy (np), math, copy.deepcopy,
    upxo._sup.dataTypeHandlers,
    upxo.geoEntities.bases,
    upxo.geoEntities.featmake,
    upxo._sup.validation_values,
    upxo.geoEntities.point3d

Optional (loaded on demand):
    numpy.matlib — used in ``array_by_angles()``.
    shapely.geometry — used in ``make_shape()`` and ``rettype='shapely'`` clustering.
    vtk — used in ``make_vtk_point()``.
    pyvista — used in clustering with ``rettype='pyvista'``.
    gmsh — used in clustering with ``rettype='gmsh'``.

Notes
-----
``p2d_leanest`` is stable and should not be extended further; use
``Point2d`` for all new feature development.

Limitations
-----------
- Pydantic validation is intentionally excluded to keep instantiation fast.
- ABC inheritance was removed to eliminate per-object overhead.

See Also
--------
upxo.geoEntities.sline2d : 2D line segments.
upxo.geoEntities.point3d : 3D points.

Examples
--------
Leanest (minimal) point:

>>> from upxo.geoEntities.point2d import p2d_leanest
>>> p = p2d_leanest(1.0, 2.0)
>>> p._x, p._y
(1.0, 2.0)

Full-featured point with distance:

>>> from upxo.geoEntities.point2d import Point2d as p2d
>>> p1, p2 = p2d(0, 0), p2d(3, 4)
>>> p1.distance(p2)
5.0

Generate a point array around a centroid:

>>> center = p2d(5, 5)
>>> points = center.array_by_clustering(n=10, r=2.0)
>>> len(points)
10

Rotate a point 90 degrees about the origin:

>>> p = p2d(1, 0)
>>> rotated = p.rotate_about_point(p2d(0, 0), 90, degree=True)

Convert to a Shapely Point:

>>> p = p2d(1, 1)
>>> sp = p.make_shape()
"""

import math
import numpy as np
from copy import deepcopy
import upxo._sup.dataTypeHandlers as dth
from upxo._sup.dataTypeHandlers import opt as OPT, strip_str as SSTR
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
# from upxo._sup.validation_values import find_pnt_spec_type_2d
np.seterr(divide='ignore')
from upxo.geoEntities.featmake import make_p2d, make_p3d
from upxo._sup.validation_values import find_spec_of_points
from upxo._sup.validation_values import val_point_and_get_coord
from upxo._sup.validation_values import isinstance_many
import upxo.geoEntities.featmake as fmake
from upxo.geoEntities.point3d import Point3d


NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES

class _coord_():
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        """Store raw 3D coordinate values in ``x``, ``y``, ``z``."""
        self.x, self.y, self.z = x, y, z

class p2d_leanest():
    """
    Minimal 2D point container for lightweight, high-frequency operations.

    Stores coordinates as private attributes (``_x``, ``_y``) and provides
    compact distance and radius-membership checks with minimal overhead.
    Intended for inner-loop use where only core geometric predicates are
    required; use ``Point2d`` when richer features are needed.

    Import
    ------
        from upxo.geoEntities.point2d import p2d_leanest

    Parameters
    ----------
    x : float
        X-coordinate of the point.
    y : float
        Y-coordinate of the point.

    Attributes
    ----------
    _x : float
        X-coordinate (private).
    _y : float
        Y-coordinate (private).

    Limitations
    -----------
    - This class is frozen: do not add new methods or attributes.
    - No feature dictionary, plane tracking, or type-conversion methods.
    - No tolerance-aware equality; use ``Point2d`` for that.

    Examples
    --------
    >>> from upxo.geoEntities.point2d import p2d_leanest
    >>> a = [p2d_leanest(1, 2), p2d_leanest(3, 4)]
    >>> a[0]._x
    1
    """

    __slots__ = ('_x', '_y')

    def __init__(self, x, y):
        """
        Initialize lean 2D point.

        Parameters
        ----------
        x, y : float
            Coordinates of the point.
        """
        self._x, self._y = x, y

    def __repr__(self):
        """Return string representation of self."""
        return f'Lean 2D point at ({self._x}, {self._y})'

    def squared_distance_to_coord(self, x, y):
        """
        Compute squared Euclidean distance to a coordinate.

        Parameters
        ----------
        x, y : float
            Target coordinate.

        Returns
        -------
        float
            Squared distance `(self._x-x)^2 + (self._y-y)^2`.
        """
        return (self._x-x)**2 + (self._y-y)**2

    def is_coord_within_cor(self, x, y, cor=1E-8, on_cor_flag=True):
        """
        Check if a coordinate is inside or on a circle centred at self.

        Parameters
        ----------
        x, y : float
            Coordinate to test.
        cor : float, optional
            Circle radius.
        on_cor_flag : bool, optional
            If True, include circle boundary (`<= cor`), else strict inside
            (`< cor`).

        Returns
        -------
        bool
            True if coordinate satisfies circle-membership criterion.

        Example
        -------
        from upxo.geoEntities.point2d import p2d_leanest
        p2d_leanest(0, 0).is_coord_within_cor(0, 1, 1E-8)
        """
        if on_cor_flag:
            return math.sqrt((self._x-x)**2+(self._y-y)**2) <= cor
        else:
            return math.sqrt((self._x-x)**2+(self._y-y)**2) < cor

    def is_p2dl_within_cor(self, point, cor=1E-8, on_cor_flag=True):
        """
        Check if a lean-point is inside or on a circle centred at self.

        Parameters
        ----------
        point : p2d_leanest
            Lean point to test.
        cor : float, optional
            Circle radius.
        on_cor_flag : bool, optional
            If True, include circle boundary (`<= cor`), else strict inside
            (`< cor`).

        Returns
        -------
        bool
            True if point satisfies circle-membership criterion.

        Example
        -------
        from upxo.geoEntities.point2d import p2d_leanest
        p2d_leanest(0, 0).is_p2dl_within_cor(p2d_leanest(0, 0), 1E-8)
        """
        if on_cor_flag:
            return math.sqrt((self._x-point._x)**2+(self._y-point._y)**2) <= cor
        else:
            return math.sqrt((self._x-point._x)**2+(self._y-point._y)**2) < cor

    def is_p2d_within_cor(self, point, cor=1E-8, on_cor_flag=True):
        """
        Check if a Point2d is inside or on a circle centred at self.

        Parameters
        ----------
        point : Point2d
            UPXO Point2d object to test.
        cor : float, optional
            Circle radius.
        on_cor_flag : bool, optional
            If True, include circle boundary (`<= cor`), else strict inside
            (`< cor`).

        Returns
        -------
        bool
            True if point satisfies circle-membership criterion.

        Example
        -------
        from upxo.geoEntities.point2d import p2d_leanest, Point2d
        p2d_leanest(0, 0).is_p2d_within_cor(Point2d(0, 0), 1E-8)
        """
        if on_cor_flag:
            return math.sqrt((self._x-point.x)**2+(self._y-point.y)**2) <= cor
        else:
            return math.sqrt((self._x-point.x)**2+(self._y-point.y)**2) < cor

class Point2d():
    """
    Full-featured 2D point for UPXO geometry workflows.

    The primary 2D point representation in UPXO. Supports rich geometric
    operations (distance, rotation, reflection, translation), equality and
    comparison utilities with tolerance, feature-dictionary attachment, and
    interoperability with Shapely, VTK, PyVista, and GMSH.

    Import
    ------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.point2d import Point2d as p2d

    Parameters
    ----------
    x : float
        First coordinate of the point (interpreted according to ``plane``).
    y : float
        Second coordinate of the point (interpreted according to ``plane``).
    plane : str, optional
        Plane identifier — one of ``'xy'``, ``'yx'``, ``'yz'``, ``'zy'``,
        ``'xz'``, ``'zx'``. Determines how ``x`` and ``y`` map to 3D axes
        in mixed-plane workflows. Default is ``'xy'``.

    Raises
    ------
    ValueError
        Raised by ``__eq__`` / ``__ne__`` when ``plist`` is empty or has an
        unrecognised point specification.

    Limitations
    -----------
    - ABC inheritance was removed to eliminate per-object overhead; subclassing
      is possible but not recommended.
    - Pydantic validation is excluded intentionally to keep instantiation fast.

    Notes
    -----
    Refer to the Jupyter notebook demos in the repository for guided
    usage examples covering clustering, meshing, and crystal workflows.

    Examples
    --------
    >>> from upxo.geoEntities.point2d import Point2d
    >>> Point2d(10, 12)
    uxpo-p2d (10,12)

    Intersection of two 2D lines:

    >>> from upxo.geoEntities.point2d import Point2d
    >>> from upxo.geoEntities.sline2d import Sline2d as sl2d
    >>> la = sl2d.by_coord([0, 0], [1, 1])
    >>> lb = sl2d.by_coord([0.1, 0.1], [1.8, 1.8])
    >>> Point2d.from_intersection_two_lines(la, lb, tool='upxo')

    Point at a parametric position along a line:

    >>> from upxo.geoEntities.point2d import Point2d
    >>> from upxo.geoEntities.sline2d import Sline2d
    >>> Point2d.from_line_factor(Sline2d(-1, -1, 1, 1), 0.25)
    """

    ε = 1E-8
    # __slots__ = UPXO_Point.__slots__ + ('f', 'plane')
    __slots__ = ('x', 'y', 'f', 'plane')

    def __init__(self, x, y, plane='xy'):
        """
        Initialize a 2D UPXO point.

        Parameters
        ----------
        x, y : float
            Point coordinates.
        plane : str, optional
            Plane identifier (`xy`, `yx`, `yz`, `zy`, `xz`, `zx`) used for
            interpreting coordinate semantics in mixed-plane workflows.
        """
        # super().__init__(x, y)
        self.x = x
        self.y = y
        self.plane = plane

    def __repr__(self):
        """Instance object string representation."""
        return f"uxpo-p2d ({self.x},{self.y})"

    def __eq__(self, plist):
        """
        Element-wise equality between self and one or more point representations.

        Parameters
        ----------
        plist : Point2d, p2d_leanest, list, or tuple
            Single point object, a list/tuple of point objects, or a nested
            coordinate array. Accepted specification types are documented in
            ``eq_fast``.

        Returns
        -------
        list of bool
            One boolean per comparison pair.

        Raises
        ------
        ValueError
            If ``plist`` is empty or has an unrecognised point specification.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d, p2d_leanest
        >>> p2d(3, 4) == p2d_leanest(3, 4)
        [True]
        >>> p2d(3, 4) == [p2d_leanest(1, 4), p2d_leanest(3, 4)]
        [False, True]
        >>> p2d(3, 4) == [[1, 2], [3, 4], [5, 6]]
        [False, True, False]
        """
        if not plist:
            raise ValueError("plist is empty.")
        spec_found = False
        # ---------------------------------------------------------
        if find_spec_of_points(plist) == 'Point2d':
            cmp, spec_found = [(self.x, self.y) == (plist.x, plist.y)], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == '[Point2d]':
            cmp, spec_found = [(self.x, self.y) == (p.x, p.y)
                               for p in plist], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == 'p2d_leanest':
            cmp, spec_found = [(self.x, self.y) == (plist._x, plist._y)], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == '[p2d_leanest]':
            cmp, spec_found = [(self.x, self.y) == (p._x, p._y)
                               for p in plist], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == 'type-[1,2]':
            cmp, spec_found = [(self.x, self.y) == tuple(plist)], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == 'type-[[1,2]]':
            cmp, spec_found = [(self.x, self.y) == tuple(plist[0])], True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == 'type-[[1,2],[3,4],[5,6]]':
            cmp = [self.x == p[0] and self.y == p[1] for p in plist]
            spec_found = True
        # ---------------------------------------------------------
        elif find_spec_of_points(plist) == 'type-[[1,2,3,4],[5,6,7,8]]':
            cmp = [self.x == _x and self.y == _y for _x, _y in zip(plist[0],
                                                                   plist[1])]
            spec_found = True
        elif find_spec_of_points(plist) == 'type-shapely':
            pass
        elif find_spec_of_points(plist) == 'type-[shapely]':
            pass
        elif find_spec_of_points(plist) == 'type-gmsh':
            pass
        elif find_spec_of_points(plist) == 'type-[gmsh]':
            pass
        elif find_spec_of_points(plist) == 'type-vtk':
            pass
        elif find_spec_of_points(plist) == 'type-[vtk]':
            pass
        elif find_spec_of_points(plist) == 'type-pyvista':
            pass
        elif find_spec_of_points(plist) == 'type-[pyvista]':
            pass
        # ---------------------------------------------------------
        if spec_found:
            return cmp
        else:
            raise ValueError('Invalid points list provided.')


    def eq(self, plist, *, use_tol=False):
        """
        Named alias for ``__eq__``, with optional tolerance support.

        Parameters
        ----------
        plist : Point2d, p2d_leanest, list, or tuple
            Point(s) to compare against self. See ``__eq__`` for accepted
            specification types.
        use_tol : bool, optional
            If ``True``, apply tolerance-aware comparison. Default is
            ``False`` (exact comparison via ``__eq__``).

        Returns
        -------
        list of bool
            One boolean per comparison pair.

        Warnings
        --------
        Tolerance-aware comparison (``use_tol=True``) is not yet implemented
        and will silently return ``None``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d, p2d_leanest
        >>> p2d(3, 4).eq(p2d_leanest(3, 4))
        [True]
        >>> p2d(3, 4).eq([p2d(1, 2), p2d(3, 4)])
        [False, True]
        """
        if not use_tol:
            return self.__eq__(plist)
        else:
            # TODO
            pass

    def eq_fast(self, plist, use_tol=False, point_spec=1):
        """
        Fast equality check skipping type detection — caller must supply spec.

        Unlike ``__eq__``, no runtime type detection is performed; the caller
        declares the format of ``plist`` via ``point_spec``. This avoids the
        overhead of ``find_spec_of_points()`` in tight loops.

        Parameters
        ----------
        plist : various
            Point(s) to compare. Format must match ``point_spec``.
        use_tol : bool, optional
            Reserved for future tolerance-aware comparison. Default ``False``.
        point_spec : int, optional
            Integer ID declaring the type/format of ``plist``:

            =====  ==================================
            ID     Format
            =====  ==================================
            1      ``Point2d``
            2      ``[Point2d]`` or ``(Point2d, …)``
            3      ``p2d_leanest``
            4      ``[p2d_leanest]``
            5      ``[x, y]`` (flat, two numbers)
            6      ``[[x, y]]``
            7      ``[[x1,y1], [x2,y2], …]``
            8      ``[[x1,x2,…], [y1,y2,…]]``
            9–16   Shapely/GMSH/VTK/PyVista variants
            =====  ==================================

            Default is ``1``.

        Returns
        -------
        list of bool or None
            Comparison result(s), or ``None`` for unimplemented spec IDs.

        Warnings
        --------
        No input validation is performed. Passing a ``plist`` whose format
        does not match ``point_spec`` will raise an ``AttributeError`` or
        return a wrong answer silently.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d, p2d_leanest
        >>> p2d(3, 4).eq_fast(p2d(3, 4), point_spec=1)
        [True]
        >>> p2d(3, 4).eq_fast([p2d(1, 2), p2d(3, 4)], point_spec=2)
        [False, True]
        >>> p2d(3, 4).eq_fast([3, 4], point_spec=5)
        [True]
        """
        cmp = None
        if point_spec == 1:
            """1: Point2d"""
            cmp = [(self.x, self.y) == (plist.x, plist.y)]
        elif point_spec == 2:
            """2: [Point2d]"""
            cmp = [(self.x, self.y) == (p.x, p.y) for p in plist]
        elif point_spec == 3:
            """3: p2d_leanest"""
            cmp = [(self.x, self.y) == (plist._x, plist._y)]
        elif point_spec == 4:
            """4: [p2d_leanest]"""
            cmp = [(self.x, self.y) == (p._x, p._y)
                   for p in plist]
        elif point_spec == 5:
            """5: type-[1,2]"""
            cmp = [(self.x, self.y) == tuple(plist)]
        elif point_spec == 6:
            """6: type-[[1,2]]"""
            cmp = [(self.x, self.y) == tuple(plist[0])]
        elif point_spec == 7:
            """7: type-[[1,2],[3,4],[5,6]]"""
            cmp = [self.x == p[0] and self.y == p[1] for p in plist]
        elif point_spec == 8:
            """8: type-[[1,2,3,4],[5,6,7,8]]"""
            cmp = [self.x == _x and self.y == _y for _x, _y in zip(plist[0],
                                                                   plist[1])]
        elif point_spec == 9:
            '''9: shapely'''
            pass
        elif point_spec == 10:
            '''[shapely]'''
            pass
        elif point_spec == 11:
            '''gmsh'''
            pass
        elif point_spec == 12:
            '''[gmsh]'''
            pass
        elif point_spec == 13:
            '''vtk'''
            pass
        elif point_spec == 14:
            '''[vtk]'''
            pass
        elif point_spec == 15:
            '''pyvista'''
            pass
        elif point_spec == 16:
            '''[pyvista]'''
            pass

        return cmp


    def __ne__(self, plist):
        """
        Element-wise inequality between self and one or more point representations.

        Parameters
        ----------
        plist : Point2d, p2d_leanest, list, or tuple
            Point(s) to compare. See ``__eq__`` for accepted formats.

        Returns
        -------
        list of bool
            Logical negation of the result of ``__eq__``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d, p2d_leanest
        >>> Point2d(3, 4) != p2d_leanest(3, 4)
        [False]
        >>> Point2d(3, 4) != [Point2d(1, 2), Point2d(3, 4)]
        [True, False]
        """
        return [not eq for eq in self.__eq__(plist)]

    def ne(self, plist, *, use_tol=False):
        """
        Named alias for ``__ne__``, with optional tolerance support.

        Parameters
        ----------
        plist : Point2d, p2d_leanest, list, or tuple
            Point(s) to compare. See ``__eq__`` for accepted formats.
        use_tol : bool, optional
            If ``True``, apply tolerance-aware comparison. Default ``False``.

        Returns
        -------
        list of bool
            Inequality results.

        Warnings
        --------
        Tolerance-aware comparison (``use_tol=True``) is not yet implemented.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d, p2d_leanest
        >>> Point2d(3, 4).ne(p2d_leanest(3, 4))
        [False]
        """
        if not use_tol:
            return self.__ne__(plist)
        else:
            pass

    def add(self, d, update=True, throw=False, mydecatlen2NUM='b'):
        """
        Translate self by distance(s) ``d``, in-place and/or returning new point(s).

        Parameters
        ----------
        d : float, list of float, list of list, or list of point objects
            Displacement to apply. Accepted forms:

            - ``scalar`` — added to both ``x`` and ``y``.
            - ``[dx]`` — added to both ``x`` and ``y``.
            - ``[dx, dy]`` — interpretation depends on ``mydecatlen2NUM``.
            - ``[[dx, dy], …]`` — one new point per sub-list.
            - ``[[x1,x2,…], [y1,y2,…]]`` — column-wise displacement arrays.
            - ``[point_obj, …]`` — list of UPXO/Shapely/VTK/PyVista/GMSH
              2D or 3D point objects.

        update : bool, optional
            If ``True`` and ``d`` is a single scalar or ``(dx, dy)`` pair,
            update ``self.x`` and ``self.y`` in-place. Default ``True``.
        throw : bool, optional
            If ``True``, return a new ``Point2d`` (or ``deepcopy`` of self
            when ``update=True``). Default ``False``.
        mydecatlen2NUM : {'b', 'atxy', 'a', 'ta2sd'}, optional
            Disambiguation rule when ``d`` is a length-2 list of numbers:

            - ``'b'`` / ``'atxy'`` — treat as ``[dx, dy]`` (add to x and y
              separately). Default.
            - ``'a'`` / ``'ta2sd'`` — treat as two separate scalar
              displacements; returns two new points.

        Returns
        -------
        Point2d or list of Point2d or None
            When ``throw=True`` returns the translated point(s). When
            ``update=True, throw=False`` returns ``None`` (side-effect only).

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> A = p2d(10, 12)
        >>> A.add(5)
        >>> A
        uxpo-p2d (15,17)

        Translate by separate dx/dy:

        >>> A = p2d(10, 12)
        >>> A.add([3, 4], mydecatlen2NUM='b')
        >>> A
        uxpo-p2d (13,16)

        Return a new point without mutating self:

        >>> A = p2d(10, 12)
        >>> B = A.add(5, update=False, throw=True)
        >>> B
        uxpo-p2d (15,17)
        """
        NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES
        if type(d) in NUMBERS:
            '''
            Ex. @ d --> 10

            from upxo.geoEntities.point2d import Point2d as p2d
            d = 10
            # --------------------------------
            # Case A
            # ------
            A = p2d(10, 12)
            A.add(d)
            print(A)
            # --------------------------------
            # Case B
            # ------
            A = p2d(10, 12)
            A.add(d, update=True, throw=False)  # Same as case A
            print(A)
            # --------------------------------
            # Case C
            # ------
            A = p2d(10, 12)
            A.add(d, update=True, throw=True)
            print(A)
            # --------------------------------
            # Case D
            # ------
            A = p2d(10, 12)
            A.add(d, update=False, throw=True)
            print(A)
            # --------------------------------
            '''
            # add d to both x and y
            if update:
                self.x += d
                self.y += d
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d, self.y+d)
        # =======================================================
        if type(d) in ITERABLES:
            # add d contents seperatrely to x and y
            # ................
            # CASE - 1
            if len(d) == 1 and type(d[0]) in NUMBERS:
                '''
                Ex. @ d --> [10]
                from upxo.geoEntities.point2d import Point2d as p2d
                d = [10]
                # --------------------------------
                # Case A
                # ------
                A = p2d(10, 12)
                A.add(d)
                print(A)
                # --------------------------------
                # Case B
                # ------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False)  # Same as case A
                print(A)
                # --------------------------------
                # Case C
                # ------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True)
                print(A)
                # --------------------------------
                # Case D
                # ------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True)
                print(A)
                # --------------------------------
                '''
                if update:
                    self.x += d[0]
                    self.y += d[0]
                if update and throw:
                    return deepcopy(self)
                if not update and throw:
                    return Point2d(self.x+d[0], self.y+d[0])
            # ................
            # CASE - 2
            if len(d) == 1 and type(d[0]) in ITERABLES and len(d[0]) == 2:
                """
                from upxo.geoEntities.point2d import Point2d as p2d
                d = [[10, 12]]
                # --------------------------------
                # Case A1
                # -------
                A = p2d(10, 12)
                A.add(d)
                print(A)
                # --------------------------------
                # Case B1 # Same as Case - a
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case C1
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case D1
                # -------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case A2
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='a')
                # throw and update ignored.
                # results will be returned by default.
                print(A)  # Remains unaltere3d as update is ignored
                # --------------------------------
                """
                if mydecatlen2NUM in ('a', 'ta2sd'):
                    return [Point2d(self.x+_, self.y+_) for _ in d[0]]
                if mydecatlen2NUM in ('b', 'atxy'):
                    if update:
                        self.x += d[0][0]
                        self.y += d[0][1]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0][0], self.y+d[0][1])

            # ................
            # CASE - 3
            if len(d) == 2 and all(isinstance_many(d, NUMBERS)):
                """
                from upxo.geoEntities.point2d import Point2d as p2d
                d = [10, 12]
                # --------------------------------
                # Case A1
                # -------
                A = p2d(10, 12)
                A.add(d)
                print(A)
                # --------------------------------
                # Case B1 # Same as Case - a
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case C1
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case D1
                # -------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                print(A)
                # --------------------------------
                # Case A2
                # -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='a')
                # throw and update ignored.
                # results will be returned by default.
                print(A)  # Remains unaltere3d as update is ignored
                # --------------------------------
                """
                if mydecatlen2NUM in ('a', 'ta2sd'):
                    return [Point2d(self.x+_, self.y+_) for _ in d]
                if mydecatlen2NUM in ('b', 'atxy'):
                    if update:
                        self.x += d[0]
                        self.y += d[1]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0], self.y+d[1])
            # ................
            if all(_.__class__.__name__ == 'p2d_leanest' for _ in d):
                """
                from upxo.geoEntities.point2d import Point2d as p2d
                from upxo.geoEntities.point2d import p2d_leanest
                P = [p2d_leanest(-10, -12), p2d_leanest(-2, 2)]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.
                # --------------------------------
                # Case A
                # ------
                A = p2d(10, 12)
                A.add(P, update=True, throw=False)
                print(A)
                # --------------------------------
                """
                return [Point2d(self.x+_._x, self.y+_._y) for _ in d]
            # ................
            if all(_.__class__.__name__ == 'Point2d' for _ in d):
                """
                from upxo.geoEntities.point2d import Point2d as p2d
                P = [p2d(-10, -12), p2d(-2, 2)]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.
                # --------------------------------
                # Case A
                # ------
                A = p2d(10, 12)
                A.add(P, update=True, throw=False)
                print(A)
                # --------------------------------
                """
                return [Point2d(self.x+_.x, self.y+_.y) for _ in d]
            # ................
            if len(d) == 2 and all(isinstance_many(d, ITERABLES)):
                if len(d[0]) == 1 and len(d[1]) == 1:
                    """
                    from upxo.geoEntities.point2d import Point2d as p2d
                    d = [[10], [12]]

                    # Case A
                    # ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)
                    # --------------------------------
                    # Case A1 # Same as case A
                    # ------
                    A = p2d(10, 12)
                    A.add(d, update=True, throw=False, mydecatlen2NUM='b')
                    print(A)
                    # --------------------------------
                    # Case B
                    # ------
                    A = p2d(10, 12)
                    A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                    print(A)
                    # --------------------------------
                    # Case C
                    # ------
                    A = p2d(10, 12)
                    A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                    print(A)
                    # --------------------------------
                    # Case D  # NOTHING CHANGES!
                    # ------
                    A = p2d(10, 12)
                    A.add(d, update=False, throw=False, mydecatlen2NUM='b')
                    print(A)
                    # --------------------------------
                    """
                    if update:
                        self.x += d[0][0]
                        self.y += d[1][0]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0][0], self.y+d[1][0])
                elif len(d[0]) > 1 and (len(d[0]) == len(d[1])):
                    """
                    from upxo.geoEntities.point2d import Point2d as p2d
                    d = [[10, 11, 12, 13], [12, 13, 14, 15]]

                    EXAMPLE CASES
                    -------------
                    # Only case possible
                    # update and throw input arguments will be ignored.
                    # --------------------------------
                    # Case A
                    # ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)
                    # --------------------------------
                    """
                    return [Point2d(self.x+_x, self.y+_y)
                            for _x, _y in zip(d[0], d[1])]
            # ................
            # CASE - 4
            if len(d) > 2 and all(isinstance_many(d, NUMBERS)):
                """
                from upxo.geoEntities.point2d import Point2d as p2d
                d = [10, 11, 12, 13]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.
                # --------------------------------
                # Case A
                # ------
                A = p2d(10, 12)
                A.add(d)
                print(A)
                # --------------------------------
                """
                return [Point2d(self.x+_d, self.y+_d) for _d in d]
            # ................
            # CASE - 5
            if len(d) > 2 and all(isinstance_many(d, ITERABLES)):
                if all(len(_) == 2 for _ in d):
                    print('i am in')
                    """
                    from upxo.geoEntities.point2d import Point2d as p2d
                    d = [[2, 3], [4, 5], [5, 6], [0, 10]]

                    EXAMPLE CASES
                    -------------
                    # Only case possible
                    # update and throw input arguments will be ignored.
                    # --------------------------------
                    # Case A
                    # ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)
                    # --------------------------------
                    """
                    return make_p2d(d, return_type='Point2d')
                    # return [Point2d(self.x+_d[0], self.y+_d[1]) for _d in d]
                else:
                    '''Ex. @ d --> [[2, 3, 5], [4, 5], [5, 6], [0, 10]]'''
                    '''Ex. @ d --> [[2, 3, 6], [4], [5, 6], [0, 10]]'''
                    '''Ex. @ d --> [[2, 3, 6], [4, 5, 6], [0, 5, 10]]'''
                    raise ValueError('Invalid distances.')
        # =======================================================
        if d.__class__.__name__ == 'Point2d':
            """
            from upxo.geoEntities.point2d import Point2d as p2d
            P = p2d(-10, -12)
            # --------------------------------
            # Case A
            # ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=False)
            print(A)
            # --------------------------------
            # Case B
            # ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=True)
            print(A)
            # --------------------------------
            # Case C
            # ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=True)
            print(A)
            # --------------------------------
            # Case D
            # ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=False)
            print(A)
            # --------------------------------
            """
            if update:
                self.x += d.x
                self.y += d.y
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d.x, self.y+d.y)
        # =======================================================
        if d.__class__.__name__ == 'p2d_leanest':
            """
            from upxo.geoEntities.point2d import Point2d as p2d
            from upxo.geoEntities.point2d import p2d_leanest
            P = p2d_leanest(-10, -12)
            # --------------------------------
            # Case A
            # ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=False)
            print(A)
            # --------------------------------
            # Case B
            # ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=True)
            print(A)
            # --------------------------------
            # Case C
            # ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=True)
            print(A)
            # --------------------------------
            # Case D
            # ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=False)
            print(A)
            """
            if update:
                self.x += d._x
                self.y += d._y
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d._x, self.y+d._y)

    def __mul__(self, f=1.0):
        """
        Scale self point in-place by scalar or per-axis factor.

        Parameters
        ----------
        f : float or list of float
            Scalar: multiply both ``x`` and ``y`` by ``f``.
            Length-2 list ``[fx, fy]``: multiply ``x`` by ``fx`` and
            ``y`` by ``fy`` independently.

        Returns
        -------
        None
            Modifies self in-place.

        Raises
        ------
        TypeError
            If ``f`` is neither a number nor a length-2 iterable.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> a = Point2d(-10, -12)
        >>> a * 2
        >>> a
        uxpo-p2d (-20,-24)

        >>> a = Point2d(-10, -12)
        >>> a * [2, 4]
        >>> a
        uxpo-p2d (-20,-48)
        """
        if type(f) in NUMBERS:
            self.x *= f
            self.y *= f
        elif type(f) in ITERABLES and len(f) == 2:
            self.x *= f[0]
            self.y *= f[1]
        else:
            raise TypeError('Invald factor')

    def mul(self, f=1.0, update=True, throw=False):
        """
        Named alias for ``__mul__`` with optional mutation and return control.

        Parameters
        ----------
        f : float or list of float
            Scaling factor. Scalar applies to both axes; an iterable of
            scalars applies each element as a separate scalar (one result
            per element).
        update : bool, optional
            If ``True``, scale self in-place. Default ``True``.
        throw : bool, optional
            If ``True``, return the scaled point or a ``deepcopy`` of self.
            Default ``False``.

        Returns
        -------
        Point2d or list of Point2d or None
            Returns scaled point(s) when ``throw=True``, else ``None``.

        Raises
        ------
        TypeError
            If ``f`` is not a number or iterable.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> point = p2d(-10, -12)
        >>> point.mul(f=-2.5, update=False, throw=True)
        uxpo-p2d (25.0,30.0)
        """
        valid = False
        # --------------------------------
        if isinstance(f, NUMBERS):
            valid = True
            if update:
                self.x *= f
                self.y *= f
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x*f, self.y*f)
        # --------------------------------
        if isinstance(f, ITERABLES):
            valid = True
            # Validations here
            return [self.mul(f=_f_[0], update=False, throw=True) for _f_ in f]
        # --------------------------------
        if not valid:
            raise TypeError('Invalid factor.')

    @classmethod
    def from_intersection_two_lines(cls, la, lb, tool='upxo',
                                    return_type='upxo'):
        """
        Construct a point at the intersection of two 2D lines.

        The return value is a list so callers handle collinear cases (multiple
        points or infinitely many) uniformly.

        Parameters
        ----------
        la : Sline2d
            First 2D straight-line segment.
        lb : Sline2d
            Second 2D straight-line segment.
        tool : {'upxo', 'shapely'}, optional
            Backend to use for the intersection calculation. Default
            ``'upxo'``.
        return_type : str, optional
            Format of returned point(s). Default ``'upxo'``.

        Returns
        -------
        list of Point2d
            Intersection point(s). Empty list if lines do not intersect.

        Notes
        -----
        Collinear lines (same direction) return an empty list even when they
        overlap, as no unique intersection point exists.

        Examples
        --------
        >>> from upxo.geoEntities.sline2d import Sline2d
        >>> from upxo.geoEntities.point2d import Point2d
        >>> la = Sline2d.by_coord([0, 0], [1, 1])
        >>> lb = Sline2d.by_coord([0, 1], [1, 0])
        >>> Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        Collinear lines (no unique intersection):

        >>> la = Sline2d.by_coord([0, 0], [1, 1])
        >>> lb = Sline2d.by_coord([0.1, 0.1], [1.8, 1.8])
        >>> Point2d.from_intersection_two_lines(la, lb, tool='upxo')
        []
        """
        if tool == 'upxo':
            return fmake.intersect_slines2d(la, lb, Point2d,
                                            return_type=return_type)

        if tool == 'shapely':
            from shapely.geometry import LineString
            lineA = LineString([(la[0], la[1]), (la[2], la[3])])
            lineB = LineString([(lb[0], lb[1]), (lb[2], lb[3])])
            if lineA.intersects(lineB):
                return lineA.intersection(lineB)

    @classmethod
    def from_line_factor(cls, line, factor):
        """
        Construct a point at a parametric position along a line.

        Computes ``(x0 + factor*dx, y0 + factor*dy)`` where ``(x0, y0)``
        is the line start and ``(dx, dy)`` is its direction vector.

        Parameters
        ----------
        line : Sline2d
            UPXO 2D straight-line segment.
        factor : float
            Parametric factor. ``0`` → start point; ``1`` → end point;
            values outside ``[0, 1]`` extrapolate beyond the segment.

        Returns
        -------
        Point2d
            Point at position ``start + factor * direction``.

        Examples
        --------
        >>> from upxo.geoEntities.sline2d import Sline2d
        >>> from upxo.geoEntities.point2d import Point2d
        >>> Point2d.from_line_factor(Sline2d(-1, -1, 1, 1), 0.25)
        >>> Point2d.from_line_factor(Sline2d(-1, -1, 1, 1), -0.25)
        >>> Point2d.from_line_factor(Sline2d(-1, -1, 1, 1), 1.25)
        """
        # Validations
        return Point2d(line.x0 + factor*line.dx,
                       line.y0 + factor*line.dy)

    @classmethod
    def from_intersection_lines_regions(cls, lines, regions):
        """
        Instantiate point(s) from intersections between lines and regions.

        Parameters
        ----------
        lines : iterable
            Collection of line-like entities.
        regions : iterable
            Collection of region-like entities.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("from_intersection_lines_regions is not yet implemented.")

    @property
    def coords(self):
        """
        X and Y coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``[x, y]``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> A = Point2d(1.125, 0.456)
        >>> A.coords
        array([1.125, 0.456])
        """
        return np.array([self.x, self.y])

    @property
    def shapely(self):
        """
        Shapely Point representation of self.

        Returns
        -------
        shapely.geometry.Point
            Point at ``(self.x, self.y)``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> A = Point2d(1.125, 0.456)
        >>> A.shapely
        <POINT (1.125 0.456)>
        """
        from shapely.geometry import Point as ShPnt
        return ShPnt(self.x, self.y)

    def inside_line(self, line, consider_ends=True,
                    consider_tol=False, tol=0.0):
        """
        Test whether self lies on (within) a 2D line segment.

        Parameters
        ----------
        line : Sline2d
            UPXO 2D straight-line segment to test against.
        consider_ends : bool, optional
            If ``True``, points coinciding with the segment endpoints count
            as inside (``<=`` comparison). If ``False``, endpoints are
            excluded (``<`` comparison). Default ``True``.
        consider_tol : bool, optional
            Reserved for tolerance-aware testing. Not yet implemented.
        tol : float, optional
            Tolerance value for future use. Default ``0.0``.

        Returns
        -------
        bool
            ``True`` if self lies on the segment, ``False`` otherwise.

        Warnings
        --------
        Tolerance-aware testing (``consider_tol=True``) is not yet
        implemented and will have no effect.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.sline2d import Sline2d
        >>> line = Sline2d.by_coord([0, 0], [1, 1])
        >>> Point2d(0.5, 0.5).inside_line(line)
        True
        >>> Point2d(-0.5, -0.5).inside_line(line)
        False
        >>> Point2d(0, 0).inside_line(line, consider_ends=False)
        False
        """
        # Validations
        # TODO: Include tolerance consideratiopns
        pointA, pointB = [[line.x0, line.y0], [line.x1, line.y1]]
        length, inside_line = line.length, None
        if consider_ends:
            inside_line = all([d <= length
                               for d in self.distance([pointA, pointB])])
        else:
            inside_line = all([d < length
                               for d in self.distance([pointA, pointB])])
        return inside_line

    def squared_distance(self, plist=None):
        """
        Compute squared Euclidean distances from self to one or more points.

        Parameters
        ----------
        plist : point spec or list of point specs
            Accepted formats match those of ``__eq__``: single or list of
            ``Point2d``, ``p2d_leanest``, ``[x, y]``, ``[[x, y], …]``,
            or column arrays ``[[x1,…], [y1,…]]``.

        Returns
        -------
        numpy.ndarray
            Squared distances ``(self.x - X)² + (self.y - Y)²`` for each
            target point.

        Raises
        ------
        ValueError
            If ``plist`` is empty.

        Limitations
        -----------
        - Only 2D point targets are supported; 3D, Shapely, GMSH, and VTK
          targets are not yet implemented.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d, p2d_leanest
        >>> Point2d(0, 0).squared_distance(p2d_leanest(3, 4))
        array([25.])
        >>> Point2d(0, 0).squared_distance([Point2d(1, 2), Point2d(3, 4)])
        array([ 5., 25.])
        >>> Point2d(0, 0).squared_distance([[1, 2, -1, -3], [4, 5, 5, 6]])
        array([17., 29., 26., 45.])
        """
        if type(plist) not in ITERABLES:
            plist = [plist]

        if len(plist) == 0:
            raise ValueError('points list cannot be empty.')
        plist = make_p2d(plist, return_type='leanest')
        X, Y = np.array([[p._x, p._y] for p in plist]).T
        return (self.x-X)**2 + (self.y-Y)**2

    def distance(self, plist=None):
        """
        Compute Euclidean distances from self to one or more points.

        Parameters
        ----------
        plist : point spec or list of point specs
            Same accepted formats as ``squared_distance``.

        Returns
        -------
        numpy.ndarray
            Euclidean distances ``sqrt((self.x-X)² + (self.y-Y)²)``.

        Limitations
        -----------
        - Only 2D point targets are supported currently.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d, p2d_leanest
        >>> Point2d(0, 0).distance(p2d_leanest(3, 4))
        array([5.])
        >>> Point2d(0, 0).distance([Point2d(1, 2), Point2d(3, 4)])
        array([2.23606798, 5.        ])
        """
        return np.sqrt(self.squared_distance(plist))

    def translate(self, *, vector=None, dist=None, update=False,
                  throw=True, make3d=False, zloc=0):
        """
        Translate self along ``vector`` by scalar distance ``dist``.

        The displacement applied is ``(vector / |vector|) * dist``.

        Parameters
        ----------
        vector : list of float
            Direction vector; length 2 for 2D translation, length 3 when
            ``make3d=True`` and a Z-component is needed.
        dist : float
            Distance to translate along ``vector``.
        update : bool, optional
            If ``True``, update self's coordinates in-place. Default ``False``.
        throw : bool, optional
            If ``True``, return a point object. Default ``True``.
        make3d : bool, optional
            If ``True``, return a ``Point3d`` instead of ``Point2d``.
            Default ``False``.
        zloc : float, optional
            Z-coordinate for the returned ``Point3d`` when ``make3d=True``
            and ``vector`` has length 2. Default ``0``.

        Returns
        -------
        Point2d or Point3d or None
            Translated point when ``throw=True``; ``None`` otherwise.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> A = Point2d(0, 0)
        >>> A.translate(vector=[1, 0], dist=5, update=True, throw=True)
        uxpo-p2d (5.0,0.0)

        Translate and return a 3D point:

        >>> A = Point2d(0, 0)
        >>> A.translate(vector=[-1, -1], dist=5, throw=True,
        ...             make3d=True, zloc=10)
        """
        distances = (np.array(vector) / np.linalg.norm(vector)) * dist
        if update:
            if len(vector) != 2 and make3d == False:
                self.x += distances[0]
                self.y += distances[1]
            # self.add(distances, update=update, throw=throw)
        if update and throw:
            if not make3d:
                self.x += distances[0]
                self.y += distances[1]
                return deepcopy(self)
            else:
                if len(distances) == 2:
                    if type(zloc) in NUMBERS:
                        return Point3d(self.x+distances[0],
                                       self.y+distances[1], zloc)
                    else:
                        return Point3d(self.x+distances[0],
                                       self.y+distances[1], 0)
                elif len(distances) == 3:
                    return Point3d(self.x+distances[0],
                                   self.y+distances[1], distances[2])
                else:
                    raise ValueError('INvalid distance calculation.')
        if not update and throw:
            if not make3d:
                return Point2d(self.x+distances[0], self.y+distances[1])
            else:
                return Point3d(self.x+distances[0],
                               self.y+distances[1], distances[2])

    @staticmethod
    def val_point_and_get_coord(point):
        """Validate a single point specification and return its (x, y) coordinates."""
        _ = False
        if not point:
            raise ValueError('Point OR coord not provided.')
        if find_spec_of_points(point) == 'Point2d':
            target_loc_x, target_loc_y, _ = point.x, point.y, True
        if find_spec_of_points(point) == '[Point2d]':
            target_loc_x, target_loc_y, _ = point[0].x, point[0].y, True
        if find_spec_of_points(point) == 'p2d_leanest':
            target_loc_x, target_loc_y, _ = point._x, point._y, True
        if find_spec_of_points(point) == '[p2d_leanest]':
            target_loc_x, target_loc_y, _ = point[0]._x, point[0]._y, True
        if find_spec_of_points(point) == 'type-[1,2]':
            target_loc_x, target_loc_y, _ = point[0], point[1], True
        if find_spec_of_points(point) == 'type-[[1,2]]':
            target_loc_x, target_loc_y, _ = point[0][0], point[0][1], True
        if find_spec_of_points(point) == 'type-[[1,2,3,4],[5,6,7,8]]':
            if len(point[0]) * len(point[1]) == 1:
                target_loc_x, target_loc_y, _ = point[0][0], point[1][0], True
        if not _:
            raise ValueError('Invalid point input.')
        return target_loc_x, target_loc_y

    @staticmethod
    def val_points_and_get_coords(points):
        """Validate a collection of point specifications and return coordinate arrays (x, y)."""
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
        if find_spec_of_points(points[0]) in 'Point2d':
            x, y, valid = [p.x for p in points], [p.y for p in points], True
        if find_spec_of_points(points[0]) == 'p2d_leanest':
            x, y, valid = [p._x for p in points], [p._y for p in points], True
        if find_spec_of_points(points) == 'type-[1,2]':
            x, y, valid = [points[0]], [points[1]], True
        if find_spec_of_points(points) == 'type-[[1,2]]':
            x, y, valid = [points[0][0]], [points[0][1]], True
        if find_spec_of_points(points) == 'type-[[1,2],[3,4],[5,6]]':
            x, y, valid = [p[0] for p in points], [p[1] for p in points], True
        if find_spec_of_points(points) == 'type-[[1,2,3,4],[5,6,7,8]]':
            x, y, valid = points[0], points[1], True
        if not valid:
            raise ValueError('Invalid points input.')
        return x, y

    def translate_to(self, *, point=None, update=False, throw=True):
        """
        Move self to an absolute target position.

        Unlike ``translate``, which applies a displacement, this method sets
        self's coordinates to those of ``point`` directly.

        Parameters
        ----------
        point : Point2d, p2d_leanest, tuple, or list
            Target position. Any valid UPXO 2D point specification.
        update : bool, optional
            If ``True``, overwrite ``self.x`` and ``self.y``. Default ``False``.
        throw : bool, optional
            If ``True``, return the moved point. Default ``True``.

        Returns
        -------
        Point2d or Point3d or None
            Moved point when ``throw=True``, else ``None``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> A = p2d(0, 0)
        >>> A.translate_to(point=p2d(1, 1), update=True, throw=True)
        uxpo-p2d (1,1)

        >>> A = p2d(0, 0)
        >>> A.translate_to(point=(1, 1), update=True, throw=True)
        uxpo-p2d (1,1)
        """
        coord = val_point_and_get_coord(point,
                                        return_type='coord',
                                        safe_exit=True)
        # ---------------------------------------------------------
        if update and len(coord) == 2:
            self.x, self.y = coord
            if throw:
                return deepcopy(self)
        elif update and len(coord) == 3:
            # Retain this here, as the behaviour may change later on
            self.x, self.y = coord[:-1]
            if throw:
                return Point3d(*coord)
        if not update and throw:
            if len(coord) == 2:
                return Point2d(*coord)
            elif len(coord) == 2:
                return Point3d(*coord)

    def rotate_about_point(self, point=None, angle=0, *, degree=True,
                           update=False, throw=True, dec=8):
        """
        Rotate self about an arbitrary pivot point by a given angle.

        Parameters
        ----------
        point : Point2d, p2d_leanest, tuple, or list
            Pivot point to rotate about.
        angle : float
            Rotation angle. Interpreted as degrees when ``degree=True``,
            radians otherwise.
        degree : bool, optional
            If ``True``, treat ``angle`` as degrees. Default ``True``.
        update : bool, optional
            If ``True``, update self's coordinates in-place. Default ``False``.
        throw : bool, optional
            If ``True``, return the rotated point. Default ``True``.
        dec : int, optional
            Decimal places for rounding the output coordinates. Default ``8``.

        Returns
        -------
        Point2d or None
            Rotated point when ``throw=True``. A ``deepcopy`` of self is
            returned when ``update=True, throw=True``; a new ``Point2d``
            when ``update=False, throw=True``.

        Raises
        ------
        ValueError
            If ``point`` is empty or ``angle`` is not a number.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(1, 0).rotate_about_point(p2d(0, 0), 90,
        ...                              degree=True, update=True, throw=True)
        uxpo-p2d (0.0,1.0)
        >>> p2d(1, 0).rotate_about_point((0, 0), 45,
        ...                              degree=True, update=True, throw=True)
        """
        if not point:
            raise ValueError('Must provide point object. Could also be coord.')
        if type(angle) not in dth.dt.NUMBERS:
            raise ValueError('Invalid angle.')
        if degree:
            A = np.radians(angle)
        # ----------------------------------------------------
        # Coordinates of the point to rotate baout
        locx, locy = Point2d.val_point_and_get_coord(point)
        # ----------------------------------------------------
        st, ct, delx, dely = math.sin(A), math.cos(A), self.x-locx, self.y-locy
        xnew, ynew = delx*ct - dely*st + locx, delx*st + dely*ct + locy
        # ----------------------------------------------------
        if update:
            self.x, self.y = round(xnew, dec), round(ynew, dec)
        if update and throw:
            # Retain this here, as the behaviour may change later on
            return deepcopy(self)
        if not update and throw:
            return Point2d(round(xnew, dec), round(ynew, dec))

    def rotate_points(self, points=None, angles=0.0, *, degree=True, dec=8,
                      return_type='p2d'):
        """
        Rotate one or more target points about self as the pivot.

        Parameters
        ----------
        points : Point2d, p2d_leanest, tuple, list, or column-array tuple
            Target point(s) to rotate. Accepted forms include single point
            objects, lists of point objects, ``(x, y)`` tuples, and
            column-array tuples ``([x1, x2, …], [y1, y2, …])``.
        angles : float or array-like of float
            Rotation angle(s). Single scalar applied to all points; array
            must match the number of target points.
        degree : bool, optional
            If ``True``, treat ``angles`` as degrees. Default ``True``.
        dec : int, optional
            Decimal places for rounding output coordinates. Default ``8``.
        return_type : str, optional
            Reserved for future return-format control. Default ``'p2d'``.

        Returns
        -------
        numpy.ndarray, shape (N, 2)
            Rotated coordinates ``[[x1, y1], [x2, y2], …]``.

        Raises
        ------
        ValueError
            If ``points`` is empty or ``angles`` length mismatches points.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(0, 0).rotate_points(p2d(1, 0), 45, degree=True)
        >>> p2d(0, 0).rotate_points(([1, 2, 3], [0, 0, 0]), 45, degree=True)
        """
        _ITER, _NUMB = dth.dt.ITERABLES, dth.dt.NUMBERS
        # ----------------------------------------------------
        # Validations
        if not points:
            raise ValueError('Must provide point object. Could also be coord.')
        # Coordinates of the point to rotate baout
        locx, locy = Point2d.val_points_and_get_coords(points)
        locx, locy = np.array(locx), np.array(locy)
        # ----------------------------------------------------
        # Validations
        if type(angles) in _ITER:
            if not all(type(angle) in _NUMB for angle in angles):
                raise ValueError('Invalid angle.')
            if angles.size != len(locx):
                raise ValueError('Invalid angles input length.')
        if type(angles) in _NUMB:
            angles = np.array(angles)
        if angles.size == 1:
            import numpy.matlib
            angles = numpy.matlib.repmat(angles, 1, len(locx)).squeeze()
        if degree:
            angles = np.radians(angles)
        # ----------------------------------------------------
        st, ct = np.sin(angles), np.cos(angles)
        delx, dely = locx-self.x, locy-self.y
        xnew, ynew = delx*ct - dely*st + self.x, delx*st + dely*ct + self.y
        return np.array([xnew, ynew]).T

    def attach_feature_(self, *, feature=None, feature_id=None):
        """
        Attach a feature object to the point feature dictionary.

        Parameters
        ----------
        feature : object
            Feature instance to attach.
        feature_id : hashable
            Key used to store/retrieve the feature within feature class bucket.

        Raises
        ------
        ValueError
            If `feature` or `feature_id` is empty.
        KeyError
            If `feature_id` already exists for the same feature class.
        """
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

    def find_closest_points(self, plist=None, *, plane='xy', on_boundary=True):
        """
        Return indices of the point(s) in ``plist`` closest to self.

        Delegates to ``find_neigh_points_by_distance`` with ``r=0``.

        Parameters
        ----------
        plist : list of Point2d, p2d_leanest, or coordinate arrays
            Collection of candidate points.
        plane : str, optional
            Plane specifier. Default ``'xy'``.
        on_boundary : bool, optional
            Passed through to ``find_neigh_points_by_distance``.
            Default ``True``.

        Returns
        -------
        numpy.ndarray of int
            Indices of the closest point(s) in ``plist``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(0, 0).find_closest_points([p2d(0, 0), p2d(0, 1), p2d(0, 0)])
        array([0, 2])
        >>> p2d(0, 0).find_closest_points([[1, 2], [2, 3], [0, -5],
        ...                                [1, 2], [1, 2]])
        """
        return self.find_neigh_points_by_distance(plist=plist, plane='xy', r=0,
                                                  on_boundary=True)

    def find_neigh_points_by_distance(self, plist=None, plane='xy', r=0,
                                      on_boundary=True):
        """
        Return indices of points in ``plist`` within radius ``r`` of self.

        Parameters
        ----------
        plist : list or array-like
            Candidate points.
        plane : str, optional
            Plane specifier. Default ``'xy'``.
        r : float, optional
            Search radius. When ``r <= ε`` (≈ 0), returns the index/indices
            of the globally closest point(s). Default ``0``.
        on_boundary : bool, optional
            If ``True``, include points at exactly ``r``. Default ``True``.

        Returns
        -------
        numpy.ndarray of int
            Indices of neighbouring points.

        Raises
        ------
        TypeError
            If ``r`` is not a number.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(0, 0).find_neigh_points_by_distance([[1, 2], [10, 12], [0, 0]])
        array([2])
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

    def find_neigh_points_by_count(self, plist=None, n=None, plane='xy'):
        """
        Return indices of the ``n`` nearest points in ``plist`` to self.

        Parameters
        ----------
        plist : list or array-like
            Candidate points.
        n : int
            Number of nearest neighbours to return.
        plane : str, optional
            Plane specifier. Default ``'xy'``.

        Returns
        -------
        tuple of numpy.ndarray
            Indices of the ``n`` closest points (from ``numpy.where``).

        Raises
        ------
        TypeError
            If ``n`` is not a non-zero integer.
        ValueError
            If ``n`` exceeds the length of ``plist``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(0, 0).find_neigh_points_by_count(
        ...     [[1, 2], [10, 12], [0, -5], [0, 0]], n=2)
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
        """
        Find neighboring multi-points within a distance criterion.

        Parameters
        ----------
        mplist : iterable
            Collection of multi-point objects.
        plane : str, optional
            Plane specifier. Defaults to `xy`.
        r : float, optional
            Search radius.
        tolf : float, optional
            Tolerance factor for membership decision.

        Notes
        -----
        To be developed.
        """
        # Use the ckdtree option.
        raise NotImplementedError("find_neigh_mulpoint_by_distance is not yet implemented.")

    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find neighboring edges using distance-based search.

        Parameters
        ----------
        elist : iterable
            Collection of edge objects.
        plane : str, optional
            Plane specifier. Defaults to `xy`.
        refloc : str, optional
            Edge reference location used for distance calculation.
        r : float, optional
            Search radius.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("find_neigh_edge_by_distance is not yet implemented.")

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find neighboring multi-edges using distance-based search.

        Parameters
        ----------
        melist : iterable
            Collection of multi-edge objects.
        plane : str, optional
            Plane specifier. Defaults to `xy`.
        refloc : str, optional
            Edge reference location used for distance calculation.
        r : float, optional
            Search radius.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("find_neigh_muledge_by_distance is not yet implemented.")

    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find neighboring crystals using distance-based search.

        Parameters
        ----------
        xlist : iterable
            Collection of crystal objects.
        plane : str, optional
            Plane specifier. Defaults to `xy`.
        refloc : str, optional
            Crystal reference location used for distance calculation.
        r : float, optional
            Search radius.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("find_neigh_xtal_by_distance is not yet implemented.")

    def set_gmsh_props(self, prop_dict):
        """
        Set Gmsh-related properties for point export/workflows.

        Parameters
        ----------
        prop_dict : dict
            Dictionary of Gmsh properties.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("set_gmsh_props is not yet implemented.")

    def array_by_translation(self,
                             ncopies=10,
                             vector=[0, 0, 0],
                             spacing='constant'):
        """
        Create translated copies of the point.

        Parameters
        ----------
        ncopies : int, optional
            Number of copies.
        vector : iterable, optional
            Translation direction/magnitude specifier.
        spacing : str, optional
            Spacing mode.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("array_by_translation is not yet implemented.")

    def array_by_rotation(self,
                          ncopies=10,
                          vector=[0, 0, 0],
                          spacing='constant'):
        """
        Create rotated copies of the point.

        Parameters
        ----------
        ncopies : int, optional
            Number of copies.
        vector : iterable, optional
            Rotation axis/reference specifier.
        spacing : str, optional
            Spacing mode.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("array_by_rotation is not yet implemented.")

    def array_on_arc(self, ncopies=10, r=1, angles=[0.0, 360.0], degree=True):
        """
        Create point copies distributed on an arc.

        Parameters
        ----------
        ncopies : int, optional
            Number of copies.
        r : float, optional
            Arc radius.
        angles : iterable, optional
            Start and end angles.
        degree : bool, optional
            If True, interpret angles in degrees.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("array_on_arc is not yet implemented.")

    def array_by_clustering(self, n=10, r=1,
                            distribution='urand', dmin=None,
                            return_type='coord_list', zloc=0.0,
                            gmsh_model_name='Model-1'):
        """
        Generate a random cluster of ``n`` points within radius ``r`` of self.

        Self acts as the approximate centroid of the cluster. The larger ``n``
        is, the closer the actual centroid converges to self.

        Parameters
        ----------
        n : int, optional
            Number of points to generate. Default ``10``.
        r : float, optional
            Cluster radius. Default ``1``.
        distribution : {'urand'}, optional
            Spatial distribution type. Currently only uniform random
            (``'urand'``) is implemented. Default ``'urand'``.
        dmin : float or None, optional
            Minimum inter-point distance constraint. Not yet implemented.
        return_type : str, optional
            Format of the returned points. Accepted values:

            ======================  ============================================
            ``'coord_list'``        ``[[x0,…], [y0,…]]`` (default)
            ``'coords_2d'``         ``[[x0,y0], …]``
            ``'upxo_2d'``           List of ``Point2d``
            ``'upxo_2d_leanest'``   List of ``p2d_leanest``
            ``'coord_list_3d'``     ``[[x0,…], [y0,…], [z0,…]]``
            ``'coords_3d'``         ``[[x0,y0,z0], …]``
            ``'shapely'``           List of ``shapely.geometry.Point``
            ``'gmsh'``              List of GMSH point tags
            ``'pyvista'``           PyVista PolyData point cloud
            ``'mulpoint2d'``        UPXO mulpoint2d object
            ======================  ============================================

        zloc : float, optional
            Z-coordinate for 3D or GMSH output. Default ``0.0``.
        gmsh_model_name : str, optional
            GMSH model name created if GMSH is not yet initialised.
            Default ``'Model-1'``.

        Returns
        -------
        varies
            Format depends on ``return_type``; see parameter description.

        Limitations
        -----------
        - Only ``distribution='urand'`` is implemented.
        - ``dmin`` constraint is not yet enforced.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> p2d(0, 0).array_by_clustering(n=10, r=1)
        >>> p2d(0, 0).array_by_clustering(n=10, r=1, return_type='coords_2d')
        >>> p2d(0, 0).array_by_clustering(n=10, r=1, return_type='upxo_2d')
        >>> p2d(0, 0).array_by_clustering(n=10, r=1, return_type='shapely')
        """
        if distribution == 'urand':
            ang = np.random.uniform(0, 2*np.pi, n)
            rad = np.sqrt(np.random.uniform(0, r, n)) * r
        # --------------------------------------
        xy = np.array([self.x+rad*np.cos(ang), self.y+rad*np.sin(ang)])
        # --------------------------------------
        if return_type in ('coord_list_2d', 'coord_list'):
            return xy
        elif return_type in ('coords_2d', 'coords2d', 'coord'):
            return xy.T
        elif return_type in ('upxo_2d', 'upxo2d', 'upxo'):
            return make_p2d(xy, return_type='p2d')
        elif return_type in ('upxo_2d_leanest', 'upxo2dleanest',
                             'upxoleanest', 'lean'):
            return make_p2d(xy, return_type='p2d_leanest')
        elif return_type in ('coord_list_3d'):
            return np.vstack((xy, np.zeros(n)))
        elif return_type in ('coords_3d', 'coords3d'):
            return np.vstack((xy, np.zeros(n))).T
        elif return_type in ('upxo_3d', 'upxo3d'):
            pass
        elif return_type in ('shapely'):
            '''Returns a list of shjapely point obejct.'''
            from shapely.geometry import Point as ShPnt
            return [ShPnt(x, y) for x, y in xy.T]
        elif return_type in ('gmsh'):
            '''Returnsa list of gmsh piont tags'''
            import gmsh
            if not gmsh.isInitialized():
                gmsh.initialize()
            if not gmsh.model.getCurrent():
                gmsh.model.add(gmsh_model_name)
            point_tags = [gmsh.model.geo.addPoint(x, y, zloc)
                          for x, y in xy.T]
            return point_tags
        elif return_type in ('pyvista'):
            import pyvista
            points = np.vstack((xy, np.zeros(n))).T
            point_cloud = pyvista.PolyData(points)
            point_cloud.verts = np.vstack([[1, i]
                                           for i in range(n)]).flatten()
            return point_cloud
        elif return_type in OPT.name_mulpoint2d:  # mulpoint2d
            from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
            return mp2d.from_xy(xy)

    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        """
        Return indices of edges on which the point lies.

        Parameters
        ----------
        elist : iterable
            Collection of edge objects.
        consider_ends : bool, optional
            If True, consider edge endpoints as valid hits.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("lies_on_which_edge is not yet implemented.")

    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """
        Return indices of crystals containing this point.

        Parameters
        ----------
        xlist : iterable
            Collection of crystal objects.
        cosider_boundary : bool, optional
            If True, include boundary checks.
        consider_boundary_ends : bool, optional
            If True, include boundary-endpoint checks.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("lies_in_which_xtal is not yet implemented.")

    def set_z(self, z=0):
        """
        Attach a Z-coordinate to this 2D point via the feature dictionary.

        Stores a ``_coord_`` feature at key ``-1`` under ``self.f``.

        Parameters
        ----------
        z : float, optional
            Z-coordinate value to attach. Default ``0``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> A = p2d(10, 12)
        >>> A.set_z(z=100)
        >>> A.f['_coord_'][-1].z
        100
        """
        from upxo.geoEntities.point2d import _coord_
        self.attach_feature(feature=_coord_(self.x, self.y, z),
                            feature_id=-1)

    def make_vtk_point(self, z=0):
        """
        Wrap this point as a VTK PolyData point.

        Parameters
        ----------
        z : float, optional
            Z-coordinate for the VTK point. If ``self.f`` does not already
            contain a ``_coord_`` entry, ``set_z(z)`` is called first.
            Default ``0``.

        Returns
        -------
        dict
            ``{'id': int, 'pd': vtkPolyData, 'help': str}`` where ``id``
            is the inserted-point index and ``pd`` holds the vtkPolyData.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d as p2d
        >>> vtkobj = p2d(10, 12).make_vtk_point(z=100)
        >>> x, y, z = vtkobj['pd'].GetPoint(vtkobj['id'])
        """
        if not hasattr(self, 'f'):
            self.set_z(z=z)
        import vtk
        points = vtk.vtkPoints()
        point_id = points.InsertNextPoint(self.x,
                                          self.y,
                                          self.f['_coord_'][-1].z)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        return {'id': point_id,
                'pd': poly_data,
                'help': "return['pd'].GetPoint(return['id'])"}

    def make_shape(self):
        """
        Create geometric shape representation of point.

        Notes
        -----
        To be developed.
        """
        raise NotImplementedError("make_shape is not yet implemented.")


def all_isinstance(dtype, *args):
    """
    Check whether all provided arguments are instances of a given type.

    Parameters
    ----------
    dtype : type
        Target Python/UPXO type to validate against.
    *args : various
        Objects to test.

    Returns
    -------
    bool or None
        ``True`` if every argument is an instance of ``dtype``;
        ``None`` if no arguments are provided.

    Examples
    --------
    >>> from upxo.geoEntities.point2d import all_isinstance, p2d_leanest
    >>> all_isinstance(p2d_leanest, p2d_leanest(0, 0), p2d_leanest(1, 1))
    True
    """
    if len(args) > 0:
        print(args)
        return all(isinstance(arg, dtype) for arg in args)
