"""
Core module of UKAEA Poly-XTAL Operations.

Authors
-------
Dr. Sunil Anandatheertha

@Developer notes
----------------
We could benefit a lot from the below links.
https://docs.sympy.org/latest/modules/geometry/points.html
https://www.geeksforgeeks.org/python-sympy-segment-perpendicular_bisector-method/?ref=next_article
https://www.geeksforgeeks.org/python-sympy-line-is_parallel-method/
https://www.geeksforgeeks.org/python-sympy-line-smallest_angle_between-method/
https://www.geeksforgeeks.org/python-sympy-line-parallel_line-method/
https://www.geeksforgeeks.org/python-sympy-line-are_concurrent-method/
https://www.geeksforgeeks.org/python-sympy-ellipse-equation-method/
https://www.geeksforgeeks.org/python-sympy-ellipse-method/
https://www.geeksforgeeks.org/python-sympy-plane-equation-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-polygon-cut_section-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-is_coplanar-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-perpendicular_plane-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-projection-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-line-intersection-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-curve-translate-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-triangle-is_right-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-triangle-is_isosceles-method/?ref=ml_lbp
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
        self.x, self.y, self.z = x, y, z


class p3d_leanest():
    """
    Leanest redefinition of 3d point class. Intended for private use only.

    Author
    ------
    Dr. Sunil Anandatheertha

    @dev
    ----
    Restrict any further development

    Examples
    --------
    from upxo.geoEntities.point3d import p3d_leanest
    a = [p3d_leanest(1, 2, 0), p3d_leanest(1, 2, 0)]
    """

    __slots__ = ('_x', '_y', '_z')

    def __init__(self, x, y, z):
        self._x, self._y, self._z = x, y, z

    def __repr__(self):
        """Return string representation of self."""
        return f'Lean 3D point at ({self._x}, {self._y}, {self._z}, {id(self)})'


class Point3d(UPXO_Point):
    """
    UPXO Point2d object, new version.

    DEVELOPMENTAL PHASES AND PROGRESS
    ---------------------------------
    __eq__: DONE
    __ne__: DONE


    Parameters
    ----------
    pln: Denotes plane which contains the self point.
    i: 1st coordinate of the point.
    j: 2nd coordinate of the point.
    f: Feature dictionary containsing features attached to the point.

    Explanations
    ------------
    If pln is 'ij' or 'ji': x, y = x_, y_: True representation
    If pln is 'jk' or 'kj': x, y = y_, z_: False representation
    If pln is 'ki' or 'ik': x, y = x_, z_: False representation
    Where, x_, y_ and z_ are actual coordinate axes.

    Notes to users
    --------------
    @user: Please refer to examples and Jupyter notebook demos before use.

    Notes to developers and maintainers
    -----------------------------------
    @dev: Inherits from ABC: Point.
    @dev: Lets not use pydantic in the interest of maintaining speed of
        instantiation and reducing memory overhead.

    Import statement
    ----------------
    from upxo.geoEntities.point3d import Point2d as p2d
    Example 1: Creation
    -------------------
    A, B, C = p3d(10, 12), p3d(10, 12), p3d(11, 12)
    print(A, B, C)

    Example 2: equality check
    -------------------------
    print(A == B, A != B, A == C, A != C)

    Example 3: Addition and subtraction
    -----------------------------------
    A + 10
    print(A)
    A - 10
    print(A)
    A + [10, 20, 30]
    """

    ε = 1E-8
    __slots__ = UPXO_Point.__slots__ + ('z', )

    def __init__(self, x, y, z=0.0):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Return a string representation of point3d instance."""
        return f"uxpo-p3d ({self.x},{self.y},{self.z})"

    def __eq__(self, plist):
        """
        Boolean inequality.


        from upxo.geoEntities.point3d import p3d_leanest

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
        return not self.__eq__(plist, use_tol=use_tol)

    def eq(self, plist, use_tol=False, tolerance=None):
        """
        Overloaded __eq__.

        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.point3d import p3d_leanest

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
        Overloaded eq.

        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.point3d import p3d_leanest

        print(p3d(3,4,5).eq_fast(p3d(3,4,5), point_spec=1))
        print(p3d(3,4,5).eq_fast([p3d(1,2,5), p3d(3,4,5)], point_spec=2))
        print(p3d(3,4,5).eq_fast((p3d(1,4,5), p3d(3,4,5)), point_spec=2))
        print(p3d(3,4,5).eq_fast(p3d_leanest(3,4,5), point_spec=3))
        print(p3d(3,4,5).eq_fast([p3d_leanest(1,4,5), p3d_leanest(3, 4, 5)], point_spec=4))
        print(p3d(3,4,5).eq_fast([1,2,5], point_spec=5))
        print(p3d(3,4,5).eq_fast([[1,2,5]], point_spec=6))
        print(p3d(3,4,5).eq_fast([[1,2,5],[3,4,5],[5,6,5]], point_spec=7))
        print(p3d(3,4,5).eq_fast([[1,3,5],[2,4,6]]))  # INvalid comparison
        print(p3d(3,4,5).eq_fast([[3],[4],[5]], point_spec=8))
        print(p3d(3,4,5).eq_fast([[3,3,4],[4,4,4],[5,5,5]], point_spec=8))
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
        pass

    def __mul__(self, f=1.0, update=True, throw=False):
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
        """Finds the point of intersection of three planes, if it exists.

        Args:
            plane1, plane2, plane3: Plane objects.

        Returns:
            A NumPy array representing the intersection point, or None if no solution exists.

        Example
        -------
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
        return np.array([self.x, self.y])

    def squared_distance(self, plist=None, point_spec = -1):
        """
        Calculate the squared distances between self point and plist, having
        a list of 3D points.

        Parameters
        ----------
        plist: List of point objects
        point_spec: Integer representaing the type of point specification

        DEVELOPMENT PHASES AND PROGRESS
        -------------------------------
        PHASE 1: 2D case: DONE
        PHASE 2: 3D case: DONE
        PHASE 3: Improve validations

        Examples
        --------
        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.point3d import p3d_leanest

        p3d(0,0,0).squared_distance(p3d(1,1,0), point_spec=-1)
        p3d(0,0,0).squared_distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]],
                                    point_spec=-1)
        points = np.random.random((3, 100000))
        p3d(0,0,0).squared_distance(points)

        p3d(0,0,0).squared_distance(p3d(1,1,0), point_spec=1)
        p3d(0,0,0).squared_distance([p3d(1,1,0)], point_spec=2)
        p3d(0,0,0).squared_distance(p3d_leanest(1,1,0), point_spec=3)
        p3d(0,0,0).squared_distance([p3d_leanest(1,1,0)], point_spec=4)
        p3d(0,0,0).squared_distance((1,1,0), point_spec=5)
        p3d(0,0,0).squared_distance([(1,1,0)], point_spec=6)
        p3d(0,0,0).squared_distance([[1,2,3],[4,5,6],[7,8,9]], point_spec=7)
        p3d(0,0,0).squared_distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]],
                                    point_spec=8)
        p3d(0,0,0).squared_distance(np.random.random((3, 100000)),
                                    point_spec=8)
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
        Calculate the distances between self point and plist.

        DEVELOPMENT PHASES AND PROGRESS
        -------------------------------
        PHASE 1: 2D case: DONE
        PHASE 2: 3D case: DONE

        DEVELOPMENT PHASES AND PROGRESS
        -------------------------------
        PHASE 1: 2D case: DONE
        PHASE 2: 3D case: DONE
        PHASE 3: Improve validations

        Examples
        --------
        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.point3d import p3d_leanest

        p3d(0,0,0).distance(p3d(1,1,0), point_spec=-1)
        p3d(0,0,0).distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=-1)
        points = np.random.random((3, 100000))
        p3d(0,0,0).distance(points)
        p3d(0,0,0).distance(p3d(1,1,0), point_spec=1)
        p3d(0,0,0).distance([p3d(1,1,0)], point_spec=2)
        p3d(0,0,0).distance(p3d_leanest(1,1,0), point_spec=3)
        p3d(0,0,0).distance([p3d_leanest(1,1,0)], point_spec=4)
        p3d(0,0,0).distance((1,1,0), point_spec=5)
        p3d(0,0,0).distance([(1,1,0)], point_spec=6)
        p3d(0,0,0).distance([[1,2,3],[4,5,6],[7,8,9]], point_spec=7)
        p3d(0,0,0).distance([[1,2,3,4],[1,2,3,4],[1,2,3,4]], point_spec=8)
        p3d(0,0,0).distance(np.random.random((3, 100000)), point_spec=8)
        """
        return np.sqrt(self.squared_distance(plist=plist,
                                             point_spec=point_spec))

    def translate(self, *, vector=None, dist=None, update=False,
                  throw=True):
        """
        Translate the self along the vector by dist.

        Development phases
        ------------------
        PHASE 1:
        PHASE 2: Validation for dist
        PHASE 3: Validation for vector

        Examples
        --------
        from upxo.geoEntities.point3d import Point3d as p3d
        A = p3d(0, 0, 0)

        Example-1
        ---------
        A.translate(vector=[-1,-1,-1], dist=3.4641016151377544,
                    update=True,
                    throw=False)
        A
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

    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        """
        from upxo.geoEntities.point3d import Point3d as p3d
        p2d(0,0,0).find_neigh_point_by_count( [[1, 2], [10, 12], [0, -5], [0, 0]], 2)
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
        # Use the ckdtree option.
        pass

    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        pass

    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    def set_gmsh_props(self, prop_dict):
        pass

    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        pass

    def lies_on_which_line(self, *, llist=None, consider_ends=True):
        pass

    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        pass

    def make_vtk_point(self, z=0):
        """
        from upxo.geoEntities.point3d import Point2d as p2d
        A, z = p2d(10, 12), 100
        vtkobj = A.make_vtk_point(z=100)

        # Accessing data in the vtk_point
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
        pass


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
