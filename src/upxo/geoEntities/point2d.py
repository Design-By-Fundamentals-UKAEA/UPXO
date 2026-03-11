"""
2D Point geometric entity module for UPXO (Universal PolyXtal Operations).

This module provides two-dimensional point representations with support for
lightweight (leanest) and feature-rich implementations. It includes geometric
operations (distance, translation, rotation), type conversions (UPXO, Shapely,
VTK, PyVista, GMSH), and point array generation methods.

Core Classes
------------
- ``p2d_leanest`` : Minimal 2D point container (private coordinates, no features).
- ``Point2d`` : Full-featured 2D point with attachable features and extensive
  geometric operations.
- ``_coord_`` : Internal 3D coordinate storage (used for z-level tracking).

Key Features
------------
- Distance calculations (Euclidean, squared).
- Geometric transformations (translation, rotation, reflection).
- Point array generation (clustering, angles, grid patterns).
- Type conversions to/from multiple formats (lists, arrays, Shapely, VTK).
- Feature attachment and intersection detection.
- Neighbor search (by distance, by count).

Imports (Core/Runtime)
---------------------
**Core imports** (module-level):
  - numpy (np): Array/numerical operations.
  - math: Trigonometry, distance calculations.
  - copy.deepcopy: Object cloning.

**Lazy imports** (loaded on demand):
  - numpy.matlib: For ``repmat()`` in ``array_by_angles()``.
  - shapely.geometry: For Shapely point/polygon conversions (``make_shape()``,
    ``array_by_clustering()`` with 'shapely' return type).
  - vtk: For VTK point object creation (``make_vtk_point()``).
  - pyvista: For PyVista point cloud generation (``array_by_clustering()``
    with 'pyvista' return type).
  - gmsh: For GMSH point tagging (``array_by_clustering()`` with 'gmsh'
    return type).

**UPXO imports** (module-level):
  - upxo._sup.dataTypeHandlers: Type checking, constants.
  - upxo.geoEntities.bases: UPXO_Point, UPXO_Edge base classes.
  - upxo.geoEntities.featmake: Point/edge construction utilities.
  - upxo._sup.validation_values: Point validation/specification.
  - upxo.geoEntities.point3d: 3D point for conversion operations.

Metadata
--------
* Module: upxo.geoEntities.point2d
* Author: Dr. Sunil Anandatheertha
* Email: vaasu.anandatheertha@ukaea.uk
* Status: Active (p2d_leanest: stable, Point2d: full-featured)
* Last updated: 2026-03-11
* Version: 1.1

A Few Usage Examples
--------------------
**Leanest (minimal) point:**
    >>> from upxo.geoEntities.point2d import p2d_leanest
    >>> p = p2d_leanest(1.0, 2.0)
    >>> p._x, p._y
    (1.0, 2.0)

**Full-featured point with distance:**
    >>> from upxo.geoEntities.point2d import Point2d as p2d
    >>> p1, p2 = p2d(0, 0), p2d(3, 4)
    >>> p1.distance(p2)
    5.0

**Generate point array around centroid:**
    >>> center = p2d(5, 5)
    >>> points = center.array_by_clustering(n=10, r=2.0)
    >>> len(points)
    10

**Rotate point around origin:**
    >>> p = p2d(1, 0)
    >>> rotated = p.rotate_about_point(p2d(0, 0), 90, degree=True)

**Convert to Shapely:**
    >>> p = p2d(1, 1)
    >>> sp = p.make_shape()  # Returns Shapely Point

See Also
--------
- ``upxo.geoEntities.sline2d`` : 2D line segments.
- ``upxo.geoEntities.point3d`` : 3D points.
- ``upxo.geoEntities.polygon2d`` : 2D polygon/crystal entities.

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
        self.x, self.y, self.z = x, y, z

class p2d_leanest():
    """
    Leanest redefinition of 2D point class.

    This class is a minimal coordinate container intended for lightweight,
    high-frequency point operations where only core geometric checks are
    needed. It stores coordinates as private attributes (`_x`, `_y`) and
    provides compact distance/radius checks with minimal overhead.

    Attributes
    ----------
    _x, _y : float
        Point coordinates in 2D space.

    Metadata
    --------
    * Class: p2d_leanest
    * Module: upxo.geoEntities.point2d
    * Author: Dr. Sunil Anandatheertha
    * Email: vaasu.anandatheertha@ukaea.uk
    * Status: Active
    * Last updated: 2026-03-11

    Import
    ------
    from upxo.geoEntities.point2d import p2d_leanest

    Author
    ------
    Dr. Sunil Anandatheertha

    @dev
    ----
    Restrict any further development

    Examples
    --------
    from upxo.geoEntities.point2d import p2d_leanest
    a = [p2d_leanest(1, 2), p2d_leanest(1, 2)]

    # Extension: Check if all of the above list belong to the same type
    all_isinstance(p2d_leanest, a)
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
    UPXO Point2d object, new version.

    This is the primary 2D point representation in UPXO. It supports rich
    geometric operations, distance and comparison utilities, feature handling,
    and interoperability with other UPXO geometry entities.

    Parameters
    ----------
    pln: Denotes plane which contains the self point.
    x: 1st coordinate of the point.
    y: 2nd coordinate of the point.
    f: Feature dictionary containsing features attached to the point.
    plane: Indicates whether the points is in xy/yx/yz/zy/xz/zx plane.
        If zx for example, then x and y of Point2d should be / will be
        interpreted as z and x.

    Metadata
    --------
    * Class: Point2d
    * Module: upxo.geoEntities.point2d
    * Author: Dr. Sunil Anandatheertha
    * Email: vaasu.anandatheertha@ukaea.uk
    * Status: Active development
    * Last updated: 2026-03-11

    Import
    ------
    from upxo.geoEntities.point2d import Point2d

    Recommended alias import:
    from upxo.geoEntities.point2d import Point2d as p2d

    Notes to users
    --------------
    @user: Please refer to examples and Jupyter notebook demos before use.

    Notes to developers and maintainers
    -----------------------------------
    @dev: Inherits from ABC: Point.: NOT ANYORE. Puts overhead cost in object
        instantiation. Removed.
    @dev: Lets not use pydantic in the interest of maintaining speed of
        instantiation and reducing memory overhead.

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d
    Point2d(10, 12)

    from upxo.geoEntities.point2d import Point2d
    from upxo.geoEntities.sline2d import Sline2d
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0.1, 0.1], [1.8, 1.8])
    p2d.from_intersection_two_lines(la, lb, tool='upxo')

    from upxo.geoEntities.point2d import Point2d
    from upxo.geoEntities.sline2d import Sline2d
    Point2d.from_line_factor(Sline2d(-1,-1, 1,1), 0.25)

    Refer to method docs for more examples.
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
        Boolean inequality.

        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.point2d import p2d_leanest

        print(p2d(3, 4) == p2d_leanest(3, 4))
        print(p2d(3, 4) == [p2d_leanest(1, 4), p2d_leanest(3, 4)])
        print(p2d(3, 4) == p2d(3, 4))
        print(p2d(3, 4) == [p2d(1, 2), p2d(3, 4)])
        print(p2d(3, 4) == (p2d(1, 4), p2d(3, 4)))

        print(p2d(3, 4) == [[1, 2], [3, 4], [5, 6]])
        print(p2d(3, 4) == [[1, 3, 5], [2, 4, 6]])
        print(p2d(3, 4) == [[3, 4]])
        print(p2d(3, 4) == [[3], [4]])
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
        Overloaded __eq__.

        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.point2d import p2d_leanest

        print(p2d(3, 4).eq(p2d_leanest(3, 4)))
        print(p2d(3, 4).eq([p2d_leanest(1, 4), p2d_leanest(3, 4)]))
        print(p2d(3, 4).eq(p2d(3, 4)))
        print(p2d(3, 4).eq([p2d(1, 2), p2d(3, 4)]))
        print(p2d(3, 4).eq((p2d(1, 4), p2d(3, 4))))
        print(p2d(3, 4).eq([[1, 2], [3, 4], [5, 6]]))
        print(p2d(3, 4).eq([[1, 3, 5], [2, 4, 6]]))
        print(p2d(3, 4).eq([[3, 4]]))
        print(p2d(3, 4).eq([[3], [4]]))
        """
        if not use_tol:
            return self.__eq__(plist)
        else:
            # TODO
            pass

    def eq_fast(self, plist, use_tol=False, point_spec=1):
        """
        Fast equality check of point with list of point specifications.

        Exaplanations
        -------------
        Overloaded eq. Note: User is responsible for valid type. No validations
        will be carried out.

        Parameters
        ----------
        plist: List of point specificatrions.
        point_spec: Thge specification ID of the point, from the following:
            * ID = 1: Point2d
            * ID = 2: [Point2d]
            * ID = 3: p2d_leanest
            * ID = 4: [p2d_leanest]
            * ID = 5: type-[1,2]
            * ID = 6: type-[[1,2]]
            * ID = 7: type-[[1,2],[3,4],[5,6]]
            * ID = 8: type-[[1,2,3,4],[5,6,7,8]]
            * ID = 9: shapely
            * ID = 10: [shapely]
            * ID = 11: gmsh
            * ID = 12: [gmsh]
            * ID = 13: vtk
            * ID = 14: [vtk]
            * ID = 15: pyvista
            * ID = 16: [pyvista]

        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.point2d import p2d_leanest

        print(p2d(3,4).eq_fast(p2d(3,4), point_spec=1))
        print(p2d(3,4).eq_fast([p2d(1,2), p2d(3,4)], point_spec=2))
        print(p2d(3,4).eq_fast((p2d(1,4), p2d(3,4)), point_spec=2))
        print(p2d(3,4).eq_fast(p2d_leanest(3,4), point_spec=3))
        print(p2d(3,4).eq_fast([p2d_leanest(1,4), p2d_leanest(3, 4)], point_spec=4))
        print(p2d(3,4).eq_fast([1,2], point_spec=5))
        print(p2d(3,4).eq_fast([[1,2]], point_spec=6))
        print(p2d(3,4).eq_fast([[1,2],[3,4],[5,6]], point_spec=7))
        print(p2d(3,4).eq_fast([[1,3,5],[2,4,6]]))  # INvalid comparison
        print(p2d(3,4).eq_fast([[3],[4],[5]], point_spec=8))
        print(p2d(3,4).eq_fast([[3,3,4],[4,4,4],[5,5,5]], point_spec=8))
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
        Boolean inequality.

        from upxo.geoEntities.point2d import Point2d, p2d_leanest

        print(Point2d(3, 4) != p2d_leanest(3, 4))
        print(Point2d(3, 4) != [p2d_leanest(1, 4), p2d_leanest(3, 4)])
        print(Point2d(3, 4) != Point2d(3, 4))
        print(Point2d(3, 4) != [Point2d(1, 2), Point2d(3, 4)])
        print(Point2d(3, 4) != (Point2d(1, 4), Point2d(3, 4)))
        print(Point2d(3, 4) != [[1, 2], [3, 4], [5, 6]])
        print(Point2d(3, 4) != [[1, 3, 5], [2, 4, 6]])
        print(Point2d(3, 4) != [[3, 4]])
        print(Point2d(3, 4) != [[3], [4]])
        """
        return [not eq for eq in self.__eq__(plist)]

    def ne(self, plist, *, use_tol=False):
        """
        Overloaded __ne__.

        from upxo.geoEntities.point2d import Point2d, p2d_leanest

        print(Point2d(3, 4).ne(p2d_leanest(3, 4)))
        print(Point2d(3, 4).ne([p2d_leanest(1, 4), p2d_leanest(3, 4)]))
        print(Point2d(3, 4).ne(Point2d(3, 4)))
        print(Point2d(3, 4).ne([Point2d(1, 2), Point2d(3, 4)]))
        print(Point2d(3, 4).ne((Point2d(1, 4), Point2d(3, 4))))
        print(Point2d(3, 4).ne([[1, 2], [3, 4], [5, 6]]))
        print(Point2d(3, 4).ne([[1, 3, 5], [2, 4, 6]]))
        print(Point2d(3, 4).ne([[3, 4]]))
        print(Point2d(3, 4).ne([[3], [4]]))
        """
        if not use_tol:
            return self.__ne__(plist)
        else:
            pass

    def add(self, d, update=True, throw=False, mydecatlen2NUM='b'):
        """
        Add distances to point coord & update self or return new point objects.

        All descriptions in parameters below, naturally extend to 3D.

        Parameters
        ----------
        d: list of distances. Depending on distances, functionaliy changes as
        below.
            * [1, 2, 3, 4]: Each entry is added to both x and y. 4 new point
            objects gets created.
            * [[1, 2], [3, 4]]: [1, 2] denote first set of x and y distances.
            They get added with self.x and self.y to make a new point. Similar
            operation extewnds to [3, 4]. Two new points are created.
            * [[1, 2, 3, 4], [5, 6, 7, 8]]: These are X and Y arrays. Each x
            and y in X and Y, gets added with self.x and self.y to make n
            points, where n = len(distances[0]).
            * [po1, po2, po3]: List of point objects. Point objects could be
            2D or 3D. UPXO, GMSH, VTK, PyVista, Shapely types are allowed.

        update: If True and if distances is either K or Iterable(P, Q), where,
            K, P and Q are dth.dt.NUMBERS, self will be updated as self.x+K and
            self.y+K or self.x+P and self.y+P.

        throw: If True and if additional conditions provided in update are
            atisfied, then the deepcopy of the point will be returned. If,
            however, update is False, a new point with coordiates self.x+K and
            self.y+K or self.x+P and self.y*Q, shall be created and returned.

        mydecatlen2NUM: My Decision At len(d)=2 when all in d are NUMBERS.
        Options include the following:
            * 'a' OR 'ta2sd': Treat as two seperate distances. In this case,
            d[0] and d[1] will be added seperately and two pint objects shall
            be made.
            * 'b' OR 'atxy': Add d[0] to x and d[1] to y. Self point may
            update and/or new point may be returned.
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
        Multiply point by factor. Use it to scale/translate self point.

        Parameters
        ----------
        f: factor: int/float/Iterable

        Return
        ------
        None

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        a = Point2d(-10, -12)
        a*2
        print(a)

        from upxo.geoEntities.point2d import Point2d
        a = Point2d(-10, -12)
        a*[2, 4]
        print(a)
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
        Overloaded __mul__ with update and throw options.

        Parameters
        ----------
        f: factor: int/float/Iterable
        update: If True, self point gets updated to new coordinates.
        throw: If True, new point will be created and returned.

        Return
        ------
        If throw is True, point object(s) will be returned, else inactive.

        Import
        ------
        from upxo.geoEntities.point2d import Point2d as p2d
        point = p2d(-10, -12)
        point.mul(f=-2.5, update=False, throw=True)
        point.mul(f=[-2.5], update=False, throw=True)
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
        Create point from intersection of two lines. Overloaded return.

        The return is overloaded and is a list containing the intersection
        point(s). In the strictest sense, this is not a classmethod by design.

        Parameters
        ----------
        lineA: 1st line.
        lineB: 2nd line.
        tool: Tool to use. UPXO preferred.

        Return
        ------
        List of points of intersection. If no points of intersection are
        found, then empty list is created.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        from upxo.geoEntities.point2d import Point2d

        # Example-1: Collinear lines Case 1
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([0.1, 0.1], [1.8, 1.8])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-2: Collinear lines Case 2
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([0.1, 0.1], [0.8, 0.8])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-3: Collinear lines Case 3
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([-0.1, -0.1], [1.8, 1.8])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-4: Collinear lines Case 4
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([1.8, 1.8], [0.1, 0.1])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-5: non-collinear lines Case 5
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([0, 1], [1, 0])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-6: Collinear lines Case 6
        # ----------------------------------
        la = Sline2d.by_coord([0.1, 0.1], [0.8, 0.8])
        lb = Sline2d.by_coord([0, 0], [1, 1])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-7: Non-Collinear lines Case 7a
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([0, 0], [1, 0])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-8: Non-Collinear lines Case 7b
        # ----------------------------------
        la = Sline2d.by_coord([0, 0], [1, 1])
        lb = Sline2d.by_coord([-0,-0], [1, 0])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-9: Collinear lines Case 8
        # ----------------------------------
        la = Sline2d.by_coord([0,0], [1,1])
        lb = Sline2d.by_coord([0,0], [1,1])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-10: Collinear lines Case 9
        # ----------------------------------
        la = Sline2d.by_coord([0,0], [1,1])
        lb = Sline2d.by_coord([0,0], [-1,-1])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')

        # Example-11: Collinear lines Case 10
        # ----------------------------------
        la = Sline2d.by_coord([0,0], [1,0])
        lb = Sline2d.by_coord([0,1], [1,1])
        Point2d.from_intersection_two_lines(la, lb, tool='upxo')
        la.plot(sl2d=lb)
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
        Instantiate Point2d using line and factor the point divides the line.

        Parameters
        ----------
        line: UPXO 2D straight line, Sline2d.
        factor: NUmerical value.

        Return
        ------
        Point2d object.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        Point2d.from_line_factor(Sline2d(-1,-1, 1,1), +0.25)
        Point2d.from_line_factor(Sline2d(-1,-1, 1,1), -0.25)
        Point2d.from_line_factor(Sline2d(-1,-1, 1,1), +1.25)
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
        pass

    @property
    def coords(self):
        """
        Returns x and y coordinates of the point.

        Parameters
        ----------
        None

        Return
        ------
        numpy array of coordinates

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        A = Point2d(1.125, 0.456)
        A.coords
        """
        return np.array([self.x, self.y])

    @property
    def shapely(self):
        """
        Returns a shapely point object.

        Pareameters
        -----------
        None

        Return
        ------
        Shapely point representation of self.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        A = Point2d(1.125, 0.456)
        A.shapely
        A.x, A.y
        """
        from shapely.geometry import Point as ShPnt
        return ShPnt(self.x, self.y)

    def inside_line(self, line, consider_ends=True,
                    consider_tol=False, tol=0.0):
        """
        Return True if a point lies inside the line else return False.

        Parameters
        ----------
        line: Single UPXO Sline2d object.
        consider_ends: If True, point being on the end points of the line
            will also be considered to be inside the line, and True will be
            returned. If False, even if the point is one of the end points of
            the line, False will be returned.

        Return
        ------
        inside_line: True if point inside line, False if not.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d
        Point2d(-0.5,-0.5).inside_line(Sline2d.by_coord([0, 0], [1, 1]))
        Point2d(0.5,0.5).inside_line(Sline2d.by_coord([0, 0], [1, 1]))
        Point2d(0,0).inside_line(Sline2d.by_coord([0, 0], [1, 1]), consider_ends=False)
        Point2d(0,0).inside_line(Sline2d.by_coord([0, 0], [1, 1]), consider_ends=True)
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
        Calculate the squared distances between self point and plist.

        Parameters
        ----------
        plist: List of valid point specifications.

        Return
        ------
        numpy array of computed distances.

        Code development phases
        -----------------------
        Phase 1: basic working code. DONE
        Phase 2: include option to point_type. This will speed up process. This
            would mean to bypass/remove the make_2d

        Feature inclusion phases
        ------------------------
        Phase 1: 2D case: basic. DONE
        Phase 2: 3D case: # TODO. Include squared_distance claculation
            with 3D points as well!!
        Phase 3: shapely: # TODO. INclude squared_distance calcualtion with
            shapely point objects as well.
        Phase 4: gmsh: # TODO. Include squared_distance cauclation with gmsh
            point objects as well.
        Phase 5: VTK: # TODO. Include squared_distance cauclation with VTK
            point objects as well.
        Phase 6: include distance computation to 3D points.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d, p2d_leanest

        Point2d(0, 0).squared_distance(p2d_leanest(3, 4))
        Point2d(0, 0).squared_distance([p2d_leanest(1, 4), p2d_leanest(3, 4)])
        Point2d(0, 0).squared_distance(Point2d(3, 4))
        Point2d(0, 0).squared_distance([Point2d(1, 2), Point2d(3, 4)])
        Point2d(0, 0).squared_distance((Point2d(1, 4), Point2d(3, 4)))

        Point2d(0, 0).squared_distance( [1, 2] )
        Point2d(0, 0).squared_distance( [[1, 2]] )
        Point2d(0, 0).squared_distance( [[1, 2], [10, 12]] )
        Point2d(0, 0).squared_distance( [[1, 2], [10, 12], [0, -5]] )
        Point2d(0, 0).squared_distance( [[1, 2, -1, -3], [4, 5, 5, 6]] )
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
        Calculate the distances between self point and plist.

        Parameters
        ----------
        plist: List of valid point specifications.

        Return
        ------
        numpy array of computed distances.

        Code development phases
        -----------------------
        Phase 1: basic working code. DONE
        Phase 2: include option to point_type. This will speed up process. This
            would mean to bypass/remove the make_2d

        Feature inclusion phases
        ------------------------
        Phase 1: 2D case: basic. DONE
        Phase 2: 3D case: # TODO. Include squared_distance claculation
            with 3D points as well!!
        Phase 3: shapely: # TODO. INclude squared_distance calcualtion with
            shapely point objects as well.
        Phase 4: gmsh: # TODO. Include squared_distance cauclation with gmsh
            point objects as well.
        Phase 5: VTK: # TODO. Include squared_distance cauclation with VTK
            point objects as well.
        Phase 6: include distance computation to 3D points.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d, p2d_leanest

        Point2d(0, 0).distance(p2d_leanest(3, 4))
        Point2d(0, 0).distance([p2d_leanest(1, 4), p2d_leanest(3, 4)])
        Point2d(0, 0).distance(Point2d(3, 4))
        Point2d(0, 0).distance([Point2d(1, 2), Point2d(3, 4)])
        Point2d(0, 0).distance((Point2d(1, 4), Point2d(3, 4)))

        Point2d(0, 0).distance( [1, 2] )
        Point2d(0, 0).distance( [[1, 2]] )
        Point2d(0, 0).distance( [[1, 2], [10, 12]] )
        Point2d(0, 0).distance( [[1, 2], [10, 12], [0, -5]] )
        Point2d(0, 0).distance( [[1, 2, -1, -3], [4, 5, 5, 6]] )
        """
        return np.sqrt(self.squared_distance(plist))

    def translate(self, *, vector=None, dist=None, update=False,
                  throw=True, make3d=False, zloc=0):
        """
        Translate the self along the vector by dist.

        Explanations
        ------------
        Please look at all examples to understand the behaviours.

        Parameters
        ----------
        vector: translation vector specification.
        dist: distance.
        update: If True, self gets updated.
        throw: If True, either self or new point will be returned.
        make3d: If True, a 3D point will be created and thrown.

        Return
        ------
        Either return nothing or return Point2d or Point3d depending on
        throw.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        A = Point2d(0, 0)
        A.translate(vector=[1, 1], dist=5, update=True, throw=True)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[1, 0], dist=5, update=True, throw=True)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1], dist=5, update=True, throw=True)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1], dist=5, update=True, throw=True,
                    make3d=True, zloc=10)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1], dist=5, update=True, throw=True,
                    make3d=False, zloc=10)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1], dist=5, update=False, throw=True,
                    make3d=False, zloc=10)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1, 3], dist=5, update=True, throw=True,
                    make3d=True, zloc=0)
        print(A)

        A = Point2d(0, 0)
        A.translate(vector=[-1, -1, 3], dist=5, update=True, throw=True,
                    make3d=True, zloc=20)
        print(A)
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
        """Docstring."""
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
        Translate the self to another point.

        Parameters
        ----------
        point:
        update:
        throw:

        Return
        ------
        Either update self or make new point object deending on throw.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.point2d import p2d_leanest as p2dl

        A = p2d(0,0)
        A.translate_to(point=p2d(1,1), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=(p2d(1,1),), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=p2dl(1,1), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=(p2dl(1,1),), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=(1,1), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=((1,1),), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=((1,1,2),), update=True, throw=True)

        A = p2d(0,0)
        A.translate_to(point=((1,1,2),), update=False, throw=False)

        A = p2d(0,0)
        A.translate_to(point=((1,1,2),), update=True, throw=False)

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
        Rotate self point about another point.

        Parameters
        ----------
        point
        angle
        degree
        update
        throw
        dec

        4.30 to 6
        12 to 2
        4 to time of sunset

        Return
        ------
        deepcopy of self or Point2d depending on update and throw
        specifications. No return action if throw is False.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d as p2d
        dut = {'degree': True, 'update': True, 'throw': True}

        p2d(1, 0).rotate_about_point(p2d(0, 0), -45, **dut)
        p2d(1, 0).rotate_about_point(p2d(0, 0), +45, **dut)
        p2d(0, 0).rotate_about_point(p2d(-1, 0), 45, **dut)
        p2d(0, 0).rotate_about_point(p2d(-1, 0), -45, **dut)
        p2d(0, 0).rotate_about_point(p2d(-1, 0), 45, **dut)
        p2d(0.70710678,0.70710678).rotate_about_point(p2d(0, 0), 45, **dut)
        p2d(0.70710678,0.70710678).rotate_about_point(p2d(0, 0), -45, **dut)
        p2d(-0.70710678,0.70710678).rotate_about_point(p2d(0, 0), -45, **dut)

        p2d(1, 0).rotate_about_point((0, 0), 45, **dut)
        p2d(1, 0).rotate_about_point(([0], [0]), 45, **dut)
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
        Rotate self point about another point.

        Parameters
        ----------
        point
        angle
        degree
        update
        throw
        dec

        Return
        ------

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d as p2d
        dd = {'degree': True, 'dec': 8}

        p2d(0, 0).rotate_points(p2d(1, 0), 45, **dd)
        p2d(1, 0).rotate_points(p2d(0, 0), +45, **dd)
        p2d(0, 0).rotate_points(p2d(-1, 0), 45, **dd)
        p2d(0, 0).rotate_points(p2d(-1, 0), -45, **dd)
        p2d(0, 0).rotate_points(p2d(-1, 0), 45, **dd)
        p2d(0.70710678, 0.70710678).rotate_points(p2d(0, 0), 45, **dd)
        p2d(0.70710678, 0.70710678).rotate_points(p2d(0, 0), -45, **dd)
        p2d(-0.70710678, 0.70710678).rotate_points(p2d(0, 0), -45, **dd)

        p2d(0, 0).rotate_points((1, 0), 45, **dd)
        p2d(0, 0).rotate_points(([1], [0]), 45, **dd)
        p2d(0, 0).rotate_points(([1, 0], [2, 0], [3, 0]), 45, **dd)
        p2d(0, 0).rotate_points(([1, 0], [2, 0], [3, 0]), -45, **dd)
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
        Find index/indices of closest point(s) from input point collection.

        Parameters
        ----------
        plist : iterable
            Input points/coordinates.
        plane : str, optional
            Plane specifier. Defaults to `xy`.
        on_boundary : bool, optional
            If True, include points exactly on search radius when delegated
            neighbor query is radius-based.

        Returns
        -------
        numpy.ndarray
            Index/indices of closest points.

        Examples
        --------

        Example-1
        from upxo.geoEntities.point2d import Point2d as p2d

        p2d(0, 0).find_closest_point( [p2d(0, 0), p2d(0, 1), p2d(0, 0)])
        Expected >> array([0, 2], dtype=int64)

        Example-2
        p2d(0, 0).find_closest_point( [[1, 2], [2, 3], [10, 12], [0, -5],
                                       [1, 2], [1, 2]] )
        Expected >> array([0, 4, 5], dtype=int64)
        """
        return self.find_neigh_points_by_distance(plist=plist, plane='xy', r=0,
                                                  on_boundary=True)

    def find_neigh_points_by_distance(self, plist=None, plane='xy', r=0,
                                      on_boundary=True):
        """
        from upxo.geoEntities.point2d import Point2d as p2d
        p2d(0, 0).find_neigh_points_by_distance( [[1, 2], [10, 12], [0, -5]] )
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
        from upxo.geoEntities.point2d import Point2d as p2d
        p2d(0, 0).find_neigh_points_by_count( [[1, 2], [10, 12], [0, -5], [0, 0]], 2)
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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def array_by_clustering(self, n=10, r=1,
                            distribution='urand', dmin=None,
                            return_type='coord_list', zloc=0.0,
                            gmsh_model_name='Model-1'):
        """
        Make an array of points with self as centroid (approx).

        Parameters
        ----------
        n: number of points to create.
        r: radius
        distribution: Specifies the distribution type. Options include:
            urand - Uniform random
            nrand - random uniform
            qrand - quadratic random
            crand - cubic random
        return_type: Type of the return object. Options include:
            coord_list_2d - [[x0, ..., xn], [y0, ..., yn]]
            coords_2d - [[x0, y0], ..., [xn, yn]]
            upxo_2d - upxo 2D point object
            coord_list_3d - [[x0, ..., xn], [y0, ..., yn], [z0, ..., zn]]
            coords_3d - [[x0, y0, z0], ..., [xn, yn, zn]]
            upxo_3d - upxo 3D point object
            gmsh - GMSH poit object
            pyvista - Py-Vista object
        zloc: Value of z-coordinate if return type is 3D.

        Return
        ------
        LIst of point coordinates.

        Explanations
        ------------
        When random, greater the value of n, nearer would be the self to
        the centroid.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d as p2d
        p2d(0,0).array_by_clustering(n=10, r=1)
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='coords_2d')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='upxo_2d')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='upxo_2d_leanest')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='shapely')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='gmsh')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='pyvista')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='coords_3d')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='coord_list_3d')
        p2d(0,0).array_by_clustering(n=10, r=1, return_type='mulpoint2d')
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
        pass

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
        pass

    def set_z(self, z=0):
        """
        from upxo.geoEntities.point2d import Point2d as p2d
        A, z = p2d(10, 12), 100
        A.set_z(z=100)
        A.f['_coord_'][-1].z
        """
        from upxo.geoEntities.point2d import _coord_
        self.attach_feature(feature=_coord_(self.x, self.y, z),
                            feature_id=-1)

    def make_vtk_point(self, z=0):
        """
        from upxo.geoEntities.point2d import Point2d as p2d
        A, z = p2d(10, 12), 100
        vtkobj = A.make_vtk_point(z=100)

        # Accessing data in the vtk_point
        x, y, z = vtkobj['pd'].GetPoint(vtkobj['id'])
        print(x, y, z)
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
        pass


def all_isinstance(dtype, *args):
    """
    Check whether all provided arguments are instances of a given type.

    Parameters
    ----------
    dtype : type
        Target Python/UPXO type to validate against.
    *args : tuple
        Variable number of objects to test.

    Returns
    -------
    bool or None
        Returns `True` only if every argument is an instance of `dtype`.
        Returns `None` when no arguments are provided.

    Example
    -------
    from upxo.geoEntities.point2d import all_isinstance, p2d_leanest
    all_isinstance(p2d_leanest, p2d_leanest(0, 0), p2d_leanest(1, 1))
    """
    if len(args) > 0:
        print(args)
        return all(isinstance(arg, dtype) for arg in args)
