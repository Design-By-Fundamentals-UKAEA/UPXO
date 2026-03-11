"""
2D straight line.

Applications
------------
* Non-conformal geometry to conformal geometry conversion
* Hierarchical grain structure feature generation
* General geometry use

Classes
-------
* Sline2d_leanest
* Sline2d

Definitions
-----------
None

Coordinate system
-----------------
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

"""

import math
import numpy as np
import numpy.matlib
from copy import deepcopy
from scipy.spatial import cKDTree
import vtk
from shapely.geometry import Point as ShPnt, Polygon as ShPol
from functools import wraps
import matplotlib.pyplot as plt
import upxo._sup.dataTypeHandlers as dth
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
from upxo.geoEntities.point2d import Point2d
np.seterr(divide='ignore')
from upxo._sup.validation_values import isinstance_many
from upxo.geoEntities.mulpoint2d import MPoint2d
from upxo._sup.validation_values import val_points_and_get_coords


NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES


class Sline2d_leanest():
    """
    Lean 2D straight line class.

    This is the minimal, lightweight representation of a 2D straight line
    in UPXO. It stores only endpoint coordinates `(x0, y0)` and `(x1, y1)`
    and provides essential geometric utilities such as:
    * line length
    * gradient
    * endpoint/index access
    * point containment checks relative to the finite segment

    Compared to :class:`Sline2d`, this class is intentionally compact and is
    suitable for high-volume geometry operations where only core line behavior
    is required.

    Attributes
    ----------
    x0, y0 : float
        Coordinates of start point.
    x1, y1 : float
        Coordinates of end point.

    Import
    ------
    from upxo.geoEntities.sline2d import Sline2d_leanest as sl2dl

    Examples
    --------
    sl2dl(-2, 3, 4, 5)
    sl2dl()

    # Example-2
    for coord in e:
        print(coord)

    # Example-3
    print(e[1])

    Author: Dr. Sunil Anandatheertha
    """

    __slots__ = ('x0', 'y0', 'x1', 'y1')

    def __init__(self, x0=0, y0=0, x1=1, y1=0):
        """
        Initialize a lean 2D line from two endpoint coordinates.

        Parameters
        ----------
        x0, y0 : float, optional
            Coordinates of the start point. Defaults to `(0, 0)`.
        x1, y1 : float, optional
            Coordinates of the end point. Defaults to `(1, 0)`.
        """
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def __repr__(self):
        """Repr function."""
        return f'UPXO-sl2d-lean ({round(self.x0, 6)},{round(self.y0, 6)})-({round(self.x1, 6)},{round(self.y1, 6)})'

    def __iter__(self):
        """
        Make self iterable over endpoint coordinate pairs.

        Returns
        -------
        generator
            Generator yielding two tuples:
            1. `(x0, y0)`
            2. `(x1, y1)`
        """
        return (i for i in ((self.x0, self.y0), (self.x1, self.y1)))

    def __getitem__(self, index):
        """
        Make self indexable by endpoint position.

        Parameters
        ----------
        index : int
            Endpoint index.
            * `0` -> start point `(x0, y0)`
            * `1` -> end point `(x1, y1)`

        Returns
        -------
        tuple
            Endpoint coordinates corresponding to index.

        Raises
        ------
        IndexError
            If index is outside valid bounds.
        """
        return ((self.x0, self.y0), (self.x1, self.y1))[index]

    @classmethod
    def by_coord(cls, start, end):
        """
        Create Sline2d by specifying end coordinates.

        Parameters
        ----------
        start: Starting point coordinate [x0, y0]
        end: Ending point coordinate [x1, y1]

        Returns
        -------
        Sline2d_leanest
            Lean line connecting `start` and `end`.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d_leanest as sl2dlean
        A = sl2dlean.by_coord([-1, 2], [3, 4])
        """
        return cls(start[0], start[1], end[0], end[1])

    @property
    def length(self):
        """
        Return length of self.

        Returns
        -------
        float
            Euclidean distance between `(x0, y0)` and `(x1, y1)`.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d_leanest as sl2dl
        e = sl2dl(-2, 3, 4, 5)
        e.length
        """
        return math.sqrt((self.x0-self.x1)**2 + (self.y0-self.y1)**2)

    @property
    def gradient(self):
        """
        Return the gradient of the self line.

        Parameters
        ----------
        None

        Returns
        -------
        Gradient of the line.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d_leanest as sl2dl

        Example-1
        ---------
        e = sl2dl(-2, 3, 4, 5)
        e.gradient

        Example-2
        ---------
        e = sl2dl(0, 1, 0, 2)
        e.gradient
        """
        if self.x0 == self.x1:
            return math.inf
        else:
            return (self.y1-self.y0)/(self.x1-self.x0)

    def contains_point(self, obj=None):
        """
        Assess relative positioning of a point with respect to self edge.
        Output helps determine whether the point:
            1. is fully contained inside the self edge
            2. is coincident with one of the points of the self edge
            3. is located on the extended part of the self edge
            4. none of the above. Relative position unknown.

        Parameters
        ----------
        obj : coord, UPXO point2d object
            Represents a point in space. The default is None.

        Notes
        -----
        Point containment is exact (no tolerance). The method evaluates:
        * perpendicular distance of point from the line,
        * distances from point to line endpoints,
        and then classifies the position.

        Returns
        -------
        intersection : [bool, bool, bool]
            Provides the relative position of point with respect to self edge.

            1. Contains the point. It coincides with one of the edge points.
               The truth values in 'intersection' are [True, False, True]
            2. Contains the point. Point is fully inside the edge.
               The truth values in 'intersection' are [True, False, False]
            3. Point is on the extended edge.
               The truth values in 'intersection' are [False, True, False]
            4. Relative position of point unknown.
               The truth values in 'intersection' are [False, False, False]

        EXAMPLES
        --------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d_leanest as sl2dlean
        e = sl2dlean.by_coord([-1, 0], [1, 0])

        e.contains_point([-0.5, 0])
        e.contains_point([0, 0])
        e.contains_point([-1, 0])
        e.contains_point([1, 0])
        e.contains_point([-1.1, 0])
        e.contains_point([-1.1, 1])

        e.contains_point(Point2d(-0.5, 0))
        e.contains_point(Point2d(0, 0))
        e.contains_point(Point2d(-1, 0))
        e.contains_point(Point2d(1, 0))
        e.contains_point(Point2d(-1.1, 0))
        e.contains_point(Point2d(-1.1, 1))

        e = Sline2d(pnta=Point2d(1, 0), pntb=Point2d(1, 0))
        e.contains_point([0.8, 0.2])
        e.contains_point(Point2d(0.8, 0.2))
        """
        SQRT = math.sqrt
        if dth.IS_CPAIR(obj):
            # Entered obj is a coordinate pair
            if self.y0 == self.y1:
                # When edge has zero slope
                pdist = abs(obj[1]-self.y0)
            elif self.x0 == self.x1:
                # When edge has infinite slope
                pdist = abs(obj[0]-self.x0)
            else:
                m = self.slope
                # Calculate the y-intercept of the line
                yintercept = self.y0 - m * self.x0
                # Calculate the perpendicular distance from the point to the line
                pdist = abs(m*obj[0] - obj[1] + yintercept) / SQRT(m**2 + 1)
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.x0-obj[0])**2 +
                                           (self.y0-obj[1])**2),
                                      SQRT((self.x1-obj[0])**2 +
                                           (self.y1-obj[1])**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]
        elif isinstance(obj, Point2d):
            if self.y0 == self.y1:
                # When edge has zero slope
                pdist = abs(obj.y-self.y0)
            elif self.x0 == self.x1:
                # When edge has infinite slope
                pdist = abs(obj.x-self.x0)
            else:
                m = self.slope
                # Calculate the y-intercept of the line
                yintercept = self.y0 - m * self.x0
                # Calculate the perpendicular distance from the point to the line
                pdist = abs(m*obj.x - obj.y + yintercept) / math.sqrt(m**2 + 1)
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.x0-obj.x)**2 +
                                           (self.y0-obj.y)**2),
                                      SQRT((self.x1-obj.x)**2 +
                                           (self.y1-obj.y)**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]
        return intersection


class Sline2d():
    """
    Sline2d: 2D Straight line object.

    Rich 2D line entity used across UPXO geometry, meshing and analysis
    workflows. In addition to endpoint coordinates, this class maintains UPXO
    point objects (`pnta`, `pntb`) and provides construction helpers,
    geometric properties, containment checks, splitting, translation and
    neighborhood utilities.

    Use this class when mutable point objects, feature extraction or advanced
    geometric operations are needed. For lightweight coordinate-only behavior,
    use :class:`Sline2d_leanest`.

    Points are:
        1. start (i.e. i): x0 & y0
        2. end (i.e. j): x1 & y1

    Following are the creation methods:
        * Default creation is by specifying __init__(x0, y0, x1, y1).
        * by_coord(start, end).
        * by_point_slope(point, slope).
        * by_slope_intercept(slope, intercept).
        * by_parametric(point1, point2, N).
        * by_coeff_const(a, b, c).
        * by_vector(point, xyproj).
        * by_loc_len_ang(ref='i', loc=[0, 0, 0], length=1, ang=0, degree=True).
        * by_perp_bisector(line, point).
        * by_transform(refedge=None, shiftxy=[0, 1], rot=+45, degree=True,
                       rot_pnt_f=0.5).
        * by_dist_bw_points(refpoint=None, points=None, f).

    Following are the property attributes:
        * length: Length of the line
        * gradient: Gradient of the line
        * mid: midpoint of the line
        * ang: angle in radians
        * angd: angle in degrees
        * vert: True if line is vertical
        * horz: True if horizontal
        * lean: Return lean representation of self.

    Import
    ------
    from upxo.geoEntities.sline2d import Sline2d
    """

    ε = 1E-8
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'f', 'pnta', 'pntb')

    def __init__(self, x0=0, y0=0, x1=1, y1=0, pnta=None, pntb=None):
        """
        Initialize a 2D line from coordinates or UPXO points.

        Parameters
        ----------
        x0, y0 : float, optional
            Start point coordinates when `pnta` and `pntb` are not provided.
        x1, y1 : float, optional
            End point coordinates when `pnta` and `pntb` are not provided.
        pnta, pntb : Point2d, optional
            Start and end UPXO point objects. If both are provided, they take
            precedence over coordinate inputs.
        """
        if pnta is None and pntb is None:
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
            self.pnta, self.pntb = Point2d(x0, y0), Point2d(x1, y1)
        if pnta is not None and pntb is not None:
            self.pnta, self.pntb = pnta, pntb
            self.x0, self.y0 = pnta.x, pnta.y
            self.x1, self.y1 = pntb.x, pntb.y

    def __repr__(self):
        """Repr function."""
        return f'UPXO-sl2d ({round(self.x0, 6)},{round(self.y0, 6)})-({round(self.x1, 6)},{round(self.y1, 6)}). {id(self)}'

    def __iter__(self):
        """Make self an iterable over its two points."""
        return (i for i in ((self.x0, self.y0), (self.x1, self.y1)))

    def __getitem__(self, index):
        """Make self indexable. 0: 1st point, 1: 2nd point, other: Error."""
        return ((self.x0, self.y0), (self.x1, self.y1))[index]

    def __eq__(self, lines):
        """Check for == @length."""
        # Validate lines
        return [self.length == e.length for e in lines]

    def __ne__(self, lines):
        """Check for != @length."""
        # No need of validations here.
        return self == lines

    def __lt__(self, lines):
        """
        Check for < @length.

        True for a line in lines if self line length is < line length.
        """
        # No need of validations here.
        length = self.length
        return [length < line for line in lines]

    def __le__(self, lines):
        """
        Check for <= @length.

        True for a line in lines if self line length is <= line length.
        """
        # No need of validations here.
        length = self.length
        return [length <= line for line in lines]

    def __gt__(self, lines):
        """
        Check for > @length.

        True for a line in lines if self line length is > line length.
        """
        # No need of validations here.
        length = self.length
        return [length > line for line in lines]

    def __ge__(self, lines):
        """
        Check for >= @length.

        True for a line in lines if self line length is >= line length.
        """
        # No need of validations here.
        length = self.length
        return [length >= line for line in lines]

    @classmethod
    def by_coord(cls, start, end):
        """
        Create Sline2d by specifying end coordinates.

        Parameters
        ----------
        start: Starting point coordinate [x0, y0]
        end: Ending point coordinate [x1, y1]

        Returns
        -------
        Sline2d
            Line connecting `start` and `end`.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        A = sl2d.by_coord([-1, 2], [3, 4])
        """
        return cls(start[0], start[1], end[0], end[1])

    @classmethod
    def by_p2d(cls, start, end):
        """
        Create Sline2d by specifying end UPXO points.

        Parameters
        ----------
        start: Starting point
        end: Ending point

        Returns
        -------
        Sline2d
            Line created from two UPXO point objects.

        Example
        -------
        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_p2d(p2d(-1, 2), p2d(3, 4))
        """
        return cls(pnta=start, pntb=end)

    @classmethod
    def by_MCL(cls, gradient, intercept, length):
        """
        Instantiate the Sline2d using slope, intercept and length.

        Parameters
        ----------
        gradient: Slope of the 2D line
        intercept: Y-intercept of the straight line
        length: Length of the straight line

        Return
        ------
        Instance of Sline2d

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d
        Sline2d.by_MCL(1.0, -1, 1)
        """
        if np.isinf(gradient):
            x0, y0 = 0, intercept
            x1, y1 = 0, y0 + length
        else:
            delta_x = np.sqrt(length**2 / (1 + gradient**2))
            delta_y = gradient * delta_x
            x0, y0 = 0, intercept
            x1, y1 = x0 + delta_x, y0 + delta_y
        return cls(x0, y0, x1, y1)

    @classmethod
    def by_MCLC(cls, gradient, intercept, length, centre):
        """
        Instantiate Sline2d using m, c and L, centred at centre-(cx, cy).

        Explanations
        ------------
        MCLC: Gradient, Y-intercept, Length, Centre

        Parameters
        ----------
        gradient: Slope of the 2D line
        intercept: Y-intercept of the straight line
        length: Length of the straight line
        centre: Proposed x- and y-location of line midpoint: (cx, cy)

        Return
        ------
        Instance of Sline2d

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d
        Sline2d.by_MCLC(1.0, 0, 2.0, (-10.0, -10.0))
        Sline2d.by_MCLC(1.0, -1.0, 2.0, (-10.0, -5.0))
        """
        if gradient == float('inf'):
            x0 = centre[0]
            y0 = centre[1] - length / 2
            x1 = centre[0]
            y1 = centre[1] + length / 2
        else:
            delta_x = length / (2 * math.sqrt(1 + gradient**2))
            delta_y = gradient * delta_x

            x0 = centre[0] - delta_x
            y0 = centre[1] - delta_y
            x1 = centre[0] + delta_x
            y1 = centre[1] + delta_y
        return cls(x0, y0, x1, y1)

    @classmethod
    def by_parametric(cls, point1, point2, N):
        """
        A line can be represented using parametric equations.
        line = [[x1 + t * (x2 - x1), y1 + t * (y2 - y1)] for t in range(n)]

        Notes
        -----
        To be developed.
        """
        pass

    @classmethod
    def by_general_form(cls, A, B, C):
        """
        Instantiate a line represented in the general form (Ax+By+C=0).

        line = [A, B, C]

        Parameters
        ----------
        A, B, C : float
            Coefficients of the line equation `Ax + By + C = 0`.

        Returns
        -------
        Sline2d
            Constructed line consistent with the supplied general-form
            coefficients.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        Sline2d.by_general_form(1, 1, 0.2)
        """
        if B != 0:
            x0, y0, x1, y1 = 0, -C/B, -C/A, 0
        elif A != 0:
            x0, y0, x1, y1 = -C/A, 0, 0, -C/B
        else:
            raise ValueError("Invalid A and B: both cannot be zero.")
        return cls(x0, y0, x1, y1)

    @classmethod
    def by_point_dxdy(cls, start_point, dxdy):
        """
        A line can be represented using a point on the line and a dir. vector.

        line = [[x, y], [dx, dy]]

        Parameters
        ----------
        start_point : iterable
            Starting coordinate pair `[x, y]`.
        dxdy : iterable
            Direction increments `[dx, dy]`.

        Returns
        -------
        Sline2d
            Line from `start_point` to `start_point + dxdy`.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        Sline2d.by_point_dxdy([1, 1], [2, 3])
        """
        return cls(start_point[0], start_point[1],
                   start_point[0]+dxdy[0], start_point[1]+dxdy[1])

    @classmethod
    def by_LFGL(cls, location=[0, 0], factor=0.0, gradient=0, length=1,
                _skip_val_=False):
        """
        Create Sline2d by specifying location, factor, gradient and length.

        Import
        ------
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        Examples
        --------
        sl2d.by_LFGL(location=[0, 0], factor=0.0, gradient=0, length=1)
        sl2d.by_LFGL(location=[0, 0], factor=0.0, gradient=1, length=1)
        sl2d.by_LFGL(location=[0, 0], factor=0.0, gradient=-1, length=1)
        sl2d.by_LFGL(location=[0, 0], factor=1.0, gradient=0, length=1)

        Issue-1
        -------
        Line creation issue when gradient is infinity.
        sl2d.by_LFGL(location=[0, 0], factor=0.0, gradient=math.inf, length=1)

        Author: Dr. Sunil Anandatheertha

        Returns
        -------
        Sline2d
            Line generated by location-factor-gradient-length specification.
        """
        if not _skip_val_:
            # Validations
            if not type(factor) in dth.dt.NUMBERS or factor < 0.0 or factor > 1.0:
                raise ValueError('Invalid factor specification.')
        # ---------------------------------
        unit_dir = (1, gradient) / np.sqrt(1 + gradient**2)
        x0, y0 = location - length*factor*unit_dir
        x1, y1 = location + length*(1-factor)*unit_dir
        return cls(x0, y0, x1, y1)

    @classmethod
    def by_LFAL(cls, location=[0, 0], factor=0.0, angle=0, length=1,
                           degree=True):
        """
        Create Sline2d by specifying location, factor, angle and length.

        Parameters
        ----------
        ref: Specifies which point on the line is used to specify the edge.

        loc: Specifies the ref point.

        length: Length of the line to be made.

        angle: Angle(s) of inclination of the line. If 2D, single value. If 3D,
        this specifies a list of three angles. First angle

        degree: ang considered in degree if degree is True, in radians if
        otherwise.

        Returns
        -------
        Sline2d
            Line generated by location-factor-angle-length specification.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        _cdef_ = sl2d.by_LFAL  # Class definition i.e. class method

        loc = [0, 0]
        _cdef_(location=loc, factor=0.0, angle=0, length=1, degree=True)
        _cdef_(location=loc, factor=1.0, angle=0, length=1, degree=True)
        _cdef_(location=loc, factor=0.5, angle=0, length=1, degree=True)

        _cdef_(location=loc, factor=0.0, angle=90, length=1, degree=True)
        _cdef_(location=loc, factor=0.0, angle=-90, length=1, degree=True)
        _cdef_(location=loc, factor=1.0, angle=-90, length=1, degree=True)

        loc = [10, 10]
        _cdef_(location=loc, factor=0.0, angle=45, length=1, degree=True)
        _cdef_(location=loc, factor=0.0, angle=-45, length=1, degree=True)
        _cdef_(location=loc, factor=1.0, angle=45, length=1, degree=True)
        _cdef_(location=loc, factor=1.0, angle=-45, length=1, degree=True)

        _cdef_(location=loc, factor=0.5, angle=45, length=1, degree=True)
        _cdef_(location=loc, factor=0.5, angle=-45, length=1, degree=True)
        _cdef_(location=loc, factor=0.2, angle=45, length=1, degree=True)
        _cdef_(location=loc, factor=0.8, angle=-45, length=1, degree=True)

        Author: Dr. Sunil Anandatheertha
        """
        # Validations
        if not type(factor) in dth.dt.NUMBERS or factor < 0.0 or factor > 1.0:
            raise ValueError('Invalid factor specification.')
        # ---------------------------------
        loc_x, loc_y = location
        # ---------------------------------
        if degree:
            angle = np.radians(angle)
        # ---------------------------------
        dx, dy = length*np.array([np.cos(angle), np.sin(angle)])
        x0, y0 = loc_x-dx*factor, loc_y-dy*factor
        x1, y1 = loc_x+dx*(1-factor), loc_y+dy*(1-factor)
        return cls(x0, y0, x1, y1)

    @classmethod
    def by_perp_bisector(cls, line, point):
        """
        Calculate and make the perpendicular bisector Sline2d b/w line and a
        point.

        Parameters
        ----------
        e: Edge specification. Preferred: UPXO edge2d_leanest
        p: Point specification. Preferred: UPXO point2d_leanest

        Examples
        --------
        from upxo.geoEntities.point2d import Edge2d as e2d
        from upxo.geoEntities.point2d import edge2d_leanest
        from upxo.geoEntities.point2d import p2d_leanest

        e = edge2d_leanest(-2, 3, 4, 5)
        e[1]
        p = p2d_leanest(1, 2)

        from sympy import Point, Segment
        s = Segment(Point(e[0], e[1]), Point(e[2], e[3]))

        Notes
        -----
        To be developed.
        """
        pass

    @classmethod
    def by_transform(cls, refedge=None, shiftxy=[0, 1],
                     rot=+45, degree=True, rot_pnt_f=0.5):
        """
        Calculate and make the new Sline2d by transforming refedge.

        Parameters
        ----------
        refedge: Reference sline which is to be transformed. Preferably,
            provide UPXO edge2d object.
        shiftxy: Translation along x and y axes.
        rot: Rotation angle about a point. This point location is determined
            using input rot_pnt_f. Positive value indicates CCW from x+ axis.
            Negative value indicated CW from x+ axis. Domain is [0, 180] and
            [-(180-eps), 0-eps], eps is very small number in python (value?).
        degree: If True, provided rot value will be considered in degrees,
            else, in radians.
        rot_pnt_f: Valid domain [0.0, 1.0]. This is a factor of length. For
            example, if rot_pnt_f = 0.25, then the point about which the edge
            shall be rotated by rot angle will be located at 25% length away
            from start (i.e, (x0, y0)) point of the refedge.

        Notes
        -----
        To be developed.
        """
        pass

    @property
    def mid_coord(self):
        """
        Return midpoint coordinates of the line.

        Returns
        -------
        list
            Midpoint as `[xmid, ymid]`.
        """
        return [(self.x0+self.x1)/2, (self.y0+self.y1)/2]

    @property
    def mid_point(self):
        """
        Return midpoint as a UPXO point object.

        Returns
        -------
        Point2d
            Midpoint represented as `Point2d`.
        """
        return Point2d(*self.mid_coord)

    @property
    def gradient(self):
        """
        Return the slope of the line.

        Returns
        -------
        float
            Gradient of the line. Returns `math.inf` for vertical lines.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_coord([-1, 2], [3, 4]).gradient
        sl2d.by_coord([-1, -1], [1, 1]).gradient
        sl2d.by_coord([0, 0], [0, 1]).gradient
        sl2d.by_coord([0, 0], [1, 0]).gradient
        """
        if self.x0 == self.x1:
            return math.inf
        else:
            return (self.y1-self.y0)/(self.x1-self.x0)

    @property
    def dxdy(self):
        """
        Return the length increments along x and y.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        a = sl2d(0,0, 1,1)
        a.dxdy
        """
        return self.dx, self.dy

    @property
    def dx(self):
        """
        Return the length increment along x.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.dx
        """
        return self.x1-self.x0

    @property
    def dy(self):
        """
        Return the length increments along y.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.dy
        """
        return self.y1-self.y0

    @property
    def yint(self):
        """
        Return the y-intercept of the line.

        Returns
        -------
        float
            Y-intercept if finite; `math.inf` for vertical lines.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_coord([-1, 2], [3, 4]).yint
        sl2d.by_coord([-1, -1], [1, 1]).yint
        sl2d.by_coord([0, 0], [0, 1]).yint
        sl2d.by_coord([0, 0], [1, 0]).yint
        sl2d.by_coord([0, -1], [1, -1]).yint
        """
        if self.x0 == self.x1:
            return math.inf
        else:
            return self.y0 - self.gradient * self.x0

    @property
    def ang(self):
        """
        Return the ccw + angle in radians.

        Returns
        -------
        float
            Angle from positive x-axis to line direction in radians.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_coord([-1, 2], [3, 4]).ang
        sl2d.by_coord([-1, -1], [1, 1]).ang
        sl2d.by_coord([0, 0], [0, 1]).ang
        sl2d.by_coord([0, 0], [1, 0]).ang
        sl2d.by_coord([0, -1], [1, -1]).ang
        """
        return math.atan2(self.y1-self.y0, self.x1-self.x0)

    @property
    def angd(self):
        """
        Return the ccw + angle in degrees.

        Returns
        -------
        float
            Angle from positive x-axis to line direction in degrees.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_coord([-1, 2], [3, 4]).angd
        sl2d.by_coord([-1, -1], [1, 1]).angd
        sl2d.by_coord([0, 0], [0, 1]).angd
        sl2d.by_coord([0, 0], [1, 0]).angd
        sl2d.by_coord([0, -1], [1, -1]).angd
        """
        return math.degrees(self.ang)

    @property
    def length(self):
        """
        Calculate and return self length.

        Returns
        -------
        float
            Euclidean distance between line endpoints.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d.by_coord([-1, 2], [3, 4]).length
        sl2d.by_coord([-1, -1], [1, 1]).length
        sl2d.by_coord([0, 0], [0, 1]).length
        sl2d.by_coord([0, 0], [1, 0]).length
        sl2d.by_coord([0, -1], [1, -1]).length
        """
        return math.sqrt((self.x0-self.x1)**2 + (self.y0-self.y1)**2)

    @property
    def vert(self):
        """
        Return True if the line is vertical.

        Returns
        -------
        bool
            `True` when `|x0 - x1| <= Sline2d.ε`.
        """
        return abs((self.x0 - self.x1)) <= Sline2d.ε

    @property
    def horz(self):
        """
        Return True if the line is horizontal.

        Returns
        -------
        bool
            `True` when `|y0 - y1| <= Sline2d.ε`.
        """
        return abs((self.y0 - self.y1)) <= Sline2d.ε

    @property
    def lean(self):
        """
        Return lean representation of self.

        Returns
        -------
        Sline2d_leanest
            Coordinate-only lean representation.
        """
        return Sline2d_leanest(self.x0, self.y0, self.x1, self.y1)

    @property
    def points(self):
        """
        Return point – A, mid-point and point – B

        Returns
        -------
        list
            `[pnta, mid_point, pntb]` as UPXO point objects.
        """
        return [self.pnta, self.mid_point, self.pntb]

    @property
    def coords(self):
        """
        Return coordinate array.

        Returns
        -------
        list
            Coordinates `[x0, y0, x1, y1]`.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        a = sl2d(0,0, 1,1)
        a.coords
        """
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def coord_list(self):
        """
        Return coordinate array.

        Returns
        -------
        list
            Endpoint coordinates `[[x0, y0], [x1, y1]]`.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        a = sl2d(0,0, 1,1)
        a.coord_list
        """
        return [[self.x0, self.y0], [self.x1, self.y1]]

    @property
    def coord_i(self):
        """Return coordinates of starting point as `[x0, y0]`."""
        return [self.x0, self.y0]

    @property
    def coord_j(self):
        """Return coordinates of ending point as `[x1, y1]`."""
        return [self.x1, self.y1]

    @property
    def general_form(self):
        """
        Return coefficients of the general form of the self.

        Returns
        -------
        list
            Coefficients `[A, B, C]` for equation `Ax + By + C = 0`.

        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d(0,0,0,1).general_form
        sl2d(1,0,0,1).general_form
        sl2d(-1.06,+10.6854,0.156,-1.685463).general_form
        """
        if self.vert:
            A, B, C = 1, 0, -self.x0
        else:
            gradient = (self.y1 - self.y0) / (self.x1 - self.x0)
            A, B, C = -gradient, 1, gradient*self.x0 - self.y0
        return [A, B, C]

    def flip(self):
        """ Flip the line coordinates and points. MIDs of point objects
        do not change, but only their coordinate values change.

        Returns
        -------
        None

        from upxo.geoEntities.sline2d import Sline2d as sl2d
        a = sl2d(0,0,0,1)
        a.flip()
        a
        a.coord_list
        """
        startx, starty = deepcopy(self.x0), deepcopy(self.y0)
        endx, endy = deepcopy(self.x1), deepcopy(self.y1)
        self.x0, self.y0, self.x1, self.y1 = endx, endy, startx, starty
        self.reset_points_to_coords()

    def reset_coords_to_points(self):
        """When the points coordinates have been changed, either by
        changing the coordinates of the individual points or by changing the
        points itself, then use this to update the coordinates.

        Returns
        -------
        None
        """
        self.x0, self.y0 = self.pnta.x, self.pnta.y
        self.x1, self.y1 = self.pntb.x, self.pntb.y

    def reset_points_to_coords(self):
        """When the coordinates of the line end points have been updated, use
        this to update the point objects of the line.

        Returns
        -------
        None
        """
        self.pnta.x, self.pnta.y = self.x0, self.y0
        self.pntb.x, self.pntb.y = self.x1, self.y1

    def is_point_endpoint(self, point):
        """
        Return True if point is one of the end points on the line.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        sl2d(0,0,1,1).is_point_endpoint((0,0))
        sl2d(0,0,1,1).is_point_endpoint([1, 3])
        """
        is_endpoint = False
        if np.any(np.all(np.array(self.coord_list) == point, axis=1)):
            is_endpoint = True
        return is_endpoint

    def invert(self):
        """
        Invert start and end points.

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        a = sl2d(0,0,1,1)
        a.invert()
        a.coord_list
        """
        ends = self.coord_list
        self.x0, self.y0 = ends[1]
        self.x1, self.y1 = ends[0]
        self.reset_points_to_coords()

    def move_i(self, point):
        """
        Move start point of the line to a new coordinate.

        Parameters
        ----------
        point : iterable
            New start-point coordinate as `(x, y)`.
        """
        self.x0, self.y0 = point
        self.pnta.translate_to(point=p2d(self.x0, self.y0),
                               update=True, throw=False)

    def move_j(self, point):
        """
        Move end point of the line to a new coordinate.

        Parameters
        ----------
        point : iterable
            New end-point coordinate as `(x, y)`.
        """
        self.x1, self.y1 = point
        self.pntb.translate_to(point=p2d(self.x1, self.y1),
                               update=True, throw=False)

    def move_to_location(self, coord=None, ref='mid', saa=True, throw=False):
        """
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        line = sl2d(0, 0, 1, 1)
        line.move_to_location(coord=[0,0],ref='mid',saa=True,throw=False)
        line

        line = sl2d(0, 0, 1, 1)
        line.move_to_location(coord=[0,0],ref='mid',saa=False,throw=True)
        line
        """
        if not saa and not throw:
            return
        # ---------------------------------------
        px, py = coord
        # ---------------------------------------
        if ref == 'mid':
            midx, midy = self.mid_coord
            dx, dy = px - midx, py - midy
        elif ref == 'i':
            ix, iy = self.coord_i
            dx, dy = px - ix, py - iy
        elif ref == 'j':
            jx, jy = self.coord_j
            dx, dy = px - jx, py - jy
        # ---------------------------------------
        if saa:
            self.x0, self.x1 = self.x0+dx, self.x1+dx
            self.y0, self.y1 = self.y0+dy, self.y1+dy
            if throw:
                return self
        if not saa and throw:
            return Sline2d(self.x0+dx, self.x1+dx, self.y0+dy, self.y1+dy)

    def break_up(self, n):
        """
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-3,-5,2,2)
        n = 5
        line.break_up(n)
        """
        n, ang, dx, dy, length = n+1, self.ang, self.dx, self.dy, self.length
        x0, y0 = self.coord_i
        # Make points on a unit horizontal radius vector
        r = np.linspace(0, 1, n)
        # Incline these vectors to the same inclination as the self.
        # Scale the line.
        # Then, apply the shift.
        xy = np.array([x0+length*math.cos(ang)*r,
                       y0+length*math.sin(ang)*r])
        return [Sline2d(xy[0][i], xy[1][i], xy[0][i+1], xy[1][i+1])
                for i in range(n-1)]

    def fully_contains_point(self, p2d=None, method='through'):
        """
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d.by_coord([-1, 0], [1, 0])
        points = [Point2d(0, 0), Point2d(-1, 0), Point2d(1, 0), Point2d(0.2, 1),
                  Point2d(0.2, 0), Point2d(0, 1), Point2d(-1, 1), Point2d(1, 1)]
        for point in points:
            print('=======================')
            print('LINE: ', line, 'POINT: ', point)
            print(line.perpendicular_distance(point))
            print(line.fully_contains_point(point, method='through'))
        """
        # Validations
        # -------------------------------
        if method == 'simple':
            if self.perpendicular_distance(p2d) == 0:
                conditiona = self.pnta.eq_fast(p2d, use_tol=False, point_spec=1)[0]
                conditionb = self.pntb.eq_fast(p2d, use_tol=False, point_spec=1)[0]
                if not(conditiona or conditionb):
                    return True
                else:
                    return False
            else:
                return False
        # ------------------------------
        if method == 'through':
            flag = self.contains_point(p2d)
            if flag[0] and not flag[1] and flag[2]:
                '''
                Contains the point. It coincides with one of the edge points.
                '''
                return False
            elif flag[0] and not flag[1] and not flag[2]:
                '''
                Contains the point. Point is fully inside the edge.
                '''
                return True
            elif not flag[0] and flag[1] and not flag[2]:
                '''
                Point is on the extended edge.
                '''
                return False
            elif not flag[0] and not flag[1] and not flag[2]:
                '''
                Relative position of point unknown.
                '''
                return False


    def contains_point(self, obj=None, return_bools=True):
        """
        Assess relative positioning of a point with respect to self edge.
        Output helps determine whether the point:
            1. is fully contained inside the self edge
            2. is coincident with one of the points of the self edge
            3. is located on the extended part of the self edge
            4. none of the above. Relative position unknown.

        Parameters
        ----------
        obj : coord, UPXO point2d object
            Represents a point in space. The default is None.
        return_bools : bool, optional
            If True, return classification as a boolean triplet.
            If False, return classification as integer case id in [1, 4].

        Notes
        -----
        This method performs exact geometric checks (no tolerance parameter).
        It classifies the point with respect to the finite segment and its
        infinite extension.

        Returns
        -------
        intersection : [bool, bool, bool]
            Provides the relative position of point with respect to self edge.

            1. Contains the point. It coincides with one of the edge points.
               The truth values in 'intersection' are [True, False, True].
               This is case number 1. A return_bools=False would return
               1 if contains check operation results in [True, False, True].
            2. Contains the point. Point is fully inside the edge.
               The truth values in 'intersection' are [True, False, False].
               This is case number 2. A return_bools=False would return
               2 if contains check operation results in [True, False, False].
            3. Point is on the extended edge.
               The truth values in 'intersection' are [False, True, False].
               This is case number 3. A return_bools=False would return
               3 if contains check operation results in [False, True, False].
            4. Relative position of point unknown.
               The truth values in 'intersection' are [False, False, False].
               This is case number 4. A return_bools=False would return
               4 if contains check operation results in [False, False, False].

        EXAMPLES
        --------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d
        pnta, pntb = Point2d(-1, 0), Point2d(1, 0)
        e = Sline2d.by_p2d(pnta, pntb)
        # CHECK for mids: id(pnta), id(e.pnta), id(pntb), id(e.pntb)

        e.contains_point([-0.5, 0])
        e.contains_point([-0.5, 0], return_bools=True)
        e.contains_point([-0.5, 0], return_bools=False)
        e.contains_point([0, 0])
        e.contains_point([-1, 0])
        e.contains_point([1, 0])
        e.contains_point([-1.1, 0])
        e.contains_point([-1.1, 1])

        e.contains_point(Point2d(-0.5, 0))
        e.contains_point(Point2d(0, 0))
        e.contains_point(Point2d(-1, 0))
        e.contains_point(Point2d(1, 0))
        e.contains_point(Point2d(-1.1, 0))
        e.contains_point(Point2d(-1.1, 1))

        e = Sline2d(pnta=Point2d(1, 0), pntb=Point2d(1, 0))
        e.contains_point([0.8, 0.2], return_bools=False)
        e.contains_point(Point2d(0.8, 0.2))
        e.contains_point(Point2d(0.8, 0.2), return_bools=False)
        """
        SQRT = math.sqrt
        if dth.IS_CPAIR(obj):
            # Entered obj is a coordinate pair
            if self.pnta.y == self.pntb.y:
                # When edge has zero slope
                pdist = abs(obj[1]-self.pnta.y)
            elif self.pnta.x == self.pntb.x:
                # When edge has infinite slope
                pdist = abs(obj[0]-self.pnta.x)
            else:
                pdist = self.perpendicular_distance(Point2d(obj[0], obj[1]))
                '''
                m = self.gradient
                # Calculate the y-intercept of the line
                yintercept = self.pnta.y - m * self.pnta.x
                # Calculate the perpendicular distance from the point to the line
                pdist = abs(m*obj[0] - obj[1] + yintercept) / SQRT(m**2 + 1)
                '''
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.pnta.x-obj[0])**2 +
                                           (self.pnta.y-obj[1])**2),
                                      SQRT((self.pntb.x-obj[0])**2 +
                                           (self.pntb.y-obj[1])**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]
        elif isinstance(obj, Point2d) or obj.__class__.__name__ == 'Point2d':
            if self.pnta.y == self.pntb.y:
                # When edge has zero slope
                pdist = abs(obj.y-self.pnta.y)
            elif self.pnta.x == self.pntb.x:
                # When edge has infinite slope
                pdist = abs(obj.x-self.pnta.x)
            else:
                pdist = self.perpendicular_distance(Point2d(obj.x, obj.y))
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.pnta.x-obj.x)**2 +
                                           (self.pnta.y-obj.y)**2),
                                      SQRT((self.pntb.x-obj.x)**2 +
                                           (self.pntb.y-obj.y)**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]

        if return_bools:
            intersection = intersection
        else:
            choices = np.array([[True, False, True],
                                [True, False, False],
                                [False, True, False],
                                [False, False, False]])
            intersection = np.array(intersection)
            ind = np.where((choices == intersection).all(axis=1))[0][0]
            intersection = ind + 1
        return intersection

    def perpendicular_distance(self, point):
        """
        Compute perpendicular distance from a point to the infinite line.

        Parameters
        ----------
        point : Point2d
            Point for distance evaluation.

        Returns
        -------
        float
            Perpendicular distance from `point` to the line through self.
        """
        m = self.gradient
        # Calculate the y-intercept of the line
        yintercept = self.pnta.y - m * self.pnta.x
        # Calculate the perpendicular distance from the point to the line
        pdist = abs(m*point.x - point.y + yintercept) / math.sqrt(m**2 + 1)
        return pdist

    def contains_sl2d(self, obj=None, otype='sl2d'):
        """
        Checks whether an edge is contained in self edge

        Parameters
        ----------
        obj : Multiple types
            Point object or coordinate pair OR line object or pair of
            coordinate pairs. Accepts following types:
                * Coordinate pair * Pair of coordinate pair
                * UPXO point2d * UPXO edge object
                * Shapely point * Shapely line object
                * VTK point * VTK line object
                * GMSH point * GMSH line object
        otype : str
            Specify type of the object

        Notes
        -----
        Valid `otype` values used in this method are:
        * `clist` for coordinate pair-of-pairs
        * `up2d` for list/tuple of two UPXO points
        * `sl2d` for `Sline2d` instance

        Returns
        -------
        tuple( list(bool, bool), list(bool, bool) )
            Two truth value pairs. Description:
                1st value: [bool1, bool2]
                2nd value: [bool3, bool4]
                All values indicate location of points on the user input edge
                bool1:
                    True if pnta inside self edge
                    True if pnta coincides with any of two self edge points
                    False if pnta lies outside self edge
                bool2:
                    True only if pnta of input edge lies on extended self edge

            Third tuple item is a single bool indicating whether both end
            points of input line are contained in self.


        PRE-REQUISITE DATA
        ------------------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d
        pnta, pntb = Point2d(0, 0), Point2d(1, 0)
        e = Sline2d.by_p2d(pnta, pntb)

        EXAMPLE-1
        ---------
        obj = [[0.2, 0], [0.8, 0]]
        k = e.contains_sl2d(obj=obj, otype='clist')
        print(k)
        > k[0] = [True, False]
        > k[1] = [True, False]
        > k[2] = True

        EXAMPLE-2
        ---------
        obj = [[-0.1, 0], [1.0, 0]]
        k = e.contains_sl2d(obj=obj, otype='clist')
        print(k)
        > k[0] = [False, True]
        > k[1] = [True, False]
        > k[2] = False

        EXAMPLE-3
        ---------
        obj = [[-0.1, 1], [1.0, 0]]
        k = e.contains_sl2d(obj=obj, otype='clist')
        print(k)
        > k[0] = [False, False]
        > k[1] = [True, False]
        > k[2] = False

        EXAMPLE-4
        ---------
        obj = [[0, 0], [1, 0]]
        k = e.contains_sl2d(obj=obj, otype='clist')
        print(k)
        > k[0] = [True, False]
        > k[1] = [True, False]
        > k[2] = True

        EXAMPLE-5
        ---------
        obj = [[0, 0], [0, 0]]
        k = e.contains_sl2d(obj=obj, otype='clist')
        print(k)
        > k[0] = [True, False]
        > k[1] = [True, False]
        > k[2] = True

        EXAMPLE-6
        ---------
        obj = [point2d(0.2, 0), point2d(0.8, 0)]
        k = e.contains_sl2d(obj=obj, otype='up2d')
        print(k)
        > k[0] = [True, False]
        > k[1] = [True, False]

        EXAMPLE-7
        ---------
        obj = [point2d(-0.1, 0), point2d(1.0, 0)]
        k = e.contains_sl2d(obj=obj, otype='up2d')
        print(k)
        > k[0] = [False, True]
        > k[1] = [True, False]

        EXAMPLE-8
        ---------
        obj = Sline2d(pnta=Point2d(0.1, 0), pntb=Point2d(0.5, 0))
        print(obj.pnta, obj.pntb)
        k = e.contains_sl2d(obj=obj, otype='sl2d')
        print(k)
        > k[0] = [False, True]
        > k[1] = [True, False]

        EXAMPLE-9
        ---------
        obj = Sline2d(pnta=Point2d(0.1, 0), pntb=Point2d(1.5, 0))
        print(obj.pnta, obj.pntb)
        k = e.contains_sl2d(obj=obj, otype='sl2d')
        print(k)
        > k[0] = [False, True]
        > k[1] = [True, False]
        """
        if obj:
            if otype == 'clist':
                pnta = Point2d(obj[0][0], obj[0][1])
                pntb = Point2d(obj[1][0], obj[1][1])
            if otype == 'up2d':
                pnta, pntb = obj[0], obj[1]
            if otype == 'sl2d':
                pnta, pntb = obj.pnta, obj.pntb
            # Evaluate contains_point
            _pnta_ = self.contains_point(obj=pnta)
            _pntb_ = self.contains_point(obj=pntb)
            if _pnta_[0] and _pntb_[0]:
                _edge_ = True
            else:
                _edge_ = False
        else:
            _pnta_, _pntb_, _edge_ = None, None, None
            print('Please enter valid inputs')
        return (_pnta_, _pntb_, _edge_)


    def distribute_points(self,
                          n=5,
                          spacing='constant',
                          factor=0,
                          sub_spacing='constant',
                          subfactors=[0, 0],
                          trim_ij=False,
                          symexp=None,
                          _coord_rounding_=(False, 8),
                          _plot_=False
                          ):
        """
        Distribute points over a straight line.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(0, 0, 0, 1)

        line.distribute_points(n=9, spacing='constant', factor=0.5,
                               trim_ij=True, _plot_=True)
        line.distribute_points(n=9, spacing='constant', factor=0,
                               trim_ij=True, _plot_=True)
        # MORE EXAMPLES NEEDED
        line.distribute_points(n=[10, 10],
                               spacing='constant', factor=0.5,
                               sub_spacing=['cubic','cubic'],
                               subfactors=[0, 1],
                               trim_ij=True,
                               _coord_rounding_=(True, 8),
                               _plot_=True)
        """
        valid_spacing = ('constant', 'linear', 'quadratic', 'cubic',
                         'symbolic')
        valid_subfactors = (0.0, 1.0)
        NUM, ITER = dth.dt.NUMBERS, dth.dt.ITERABLES
        # ----------------------------------------------------------------
        # Validatios
        if type(n) not in (NUM + ITER):
            raise ValueError('Invalid n specified.')
        # .........
        if spacing not in valid_spacing:
            raise ValueError('Invalid spacing specified.')
        # .........
        if factor < 0.0 or factor > 1.0:
            raise ValueError('Invalid factor specified.')
        # .........
        if type(subfactors) not in (NUM + ITER):
            raise ValueError('Invalid subfactors specified. 1.')
        if type(subfactors) in ITER:
            for sf in subfactors:
                if sf not in valid_subfactors:
                    raise ValueError('Invalid subfactors specified. 2.')
        if type(subfactors) in NUM:
            subfactors = [subfactors, subfactors]
            for sf in subfactors:
                if sf < 0.0 or sf > 1.0:
                    raise ValueError('Invalid subfactors specified. 3.')
        # ----------------------------------------------------------------
        x0, y0 = self.coord_i
        ang, length = self.ang, self.length
        # .........
        def apply_spacing(r, spacing, fac):
            # Valid only for factors = 0.0, 1.0
            spacing_actions = {
                'constant': lambda r: r,
                'linear': lambda r: r,
                'quadratic': lambda r: r**2 if fac == 0 else np.flip(1-r**2),
                'cubic': lambda r: r**3 if fac == 0 else np.flip(1-r**3)
            }
            # Default to no-op
            return spacing_actions.get(spacing, lambda r: r)(r)
        # ----------------------------------------------------------------
        if factor == 0.0 or factor == 1.0:
            if type(n) in NUM:
                n = n + 2
            elif type(n) in ITER:
                n = n[0] + 2
            # Make points on a unit horizontal radius vector
            r = np.linspace(0, 1, n)
            r = apply_spacing(r, spacing, factor)
        # ----------------------------------------------------------------
        if factor > 0.0 and factor < 1.0:
            # factor = 0.5
            _, lines, __ = Sline2d(0, 0, 1, 0).divide_at_ratios(factor)
            # .........
            if type(n) in NUM:
                N = [int(n/2), int(n/2)]
            elif type(n) in ITER and len(n) >= 2:
                N = [abs(n[0]), abs(n[1])]
            # .........
            if type(sub_spacing) in ITER and len(sub_spacing) >= 2:
                sub_spacing0, sub_spacing1 = sub_spacing
            else:
                sub_spacing0, sub_spacing1 = [sub_spacing, sub_spacing]
            # .........
            points0 = lines[0].distribute_points(N[0], spacing=sub_spacing0,
                                                 factor=subfactors[0],
                                                 trim_ij=False)
            # .........
            points1 = lines[1].distribute_points(N[1], spacing=sub_spacing1,
                                                 factor=subfactors[1],
                                                 trim_ij=False)
            r = np.vstack((points0, points1[1:, :]))[:, 0]
        # ----------------------------------------------------------------
        # Incline these vectors to the same inclination as the self.
        # Scale the line. Then, apply the shift.
        xy = np.array([x0+length*math.cos(ang)*r, y0+length*math.sin(ang)*r])
        # ----------------------------------------------------------------
        if trim_ij:
            xy = xy[:, 1:-1].T
        else:
            xy = xy.T
        # ----------------------------------------------------------------
        if _coord_rounding_[0]:
            # toldp: Tol on decimal places in rounding
            toldp, XY = _coord_rounding_[1], []
            for _xy_ in xy:
                XY.append([round(_xy_[0], toldp),
                           round(_xy_[1], toldp)])
            xy = np.array(XY)
        # ----------------------------------------------------------------
        if _plot_:
            plt.figure(dpi=60, figsize=(5, 5))
            plt.plot(xy[:, 0], xy[:, 1], 'kx', ms=8)
        return xy

    def divide_at_ratios(self, ratios):
        """
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        LINE = sl2d(-3,-5,  2,2)
        ratios = [0.2, 0.3, 0.4]
        points, lines, mullines = LINE.divide_at_ratios(ratios)
        mullines.lines
        """
        # ---------------------------
        # PREVENT CIRCULAR IMPORT
        from upxo.geoEntities.mulsline2d import MSline2d
        # ---------------------------
        if type(ratios) not in dth.dt.ITERABLES:
            ratios = [ratios]
        # VALIDATIONS
        # ---------------------------
        ratios = list(np.sort(ratios))
        if all([sf >= 0.0 and sf <= 1.0 for sf in ratios]):
            if ratios[0] != 0.0:
                ratios = [0.0] + ratios
            if ratios[-1] != 1.0:
                ratios = ratios + [1.0]
        else:
            raise ValueError('One or more ratios is not in [0.0, 1.0].')
        # ---------------------------
        points = [Point2d.from_line_factor(self, f) for f in ratios]
        POINTS = [[points[i], points[i+1]] for i in range(len(points)-1)]
        LINES = [Sline2d.by_p2d(point[0], point[1]) for point in POINTS]
        MULLINES = MSline2d.from_lines(LINES, close=False)
        return points, LINES, MULLINES

    def move(self, dx, dy):
        """
        Translate line endpoints by uniform x and y increments.

        Parameters
        ----------
        dx : float
            Translation along x-axis.
        dy : float
            Translation along y-axis.
        """
        self.x0 += dx
        self.x1 += dx
        self.y0 += dy
        self.y1 += dy

    def is_normal(self, lines, _tol_decplace_=8):
        """
        Return Truth value list for normality check between self and lines.

        Parameters
        ----------
        lines: Iterable of UPXO lines.
        _tol_decplace_: Number of rounding decimal places for gradient product
            check. Defaults to 8.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        LINE = sl2d(-1.25, 1.068, 6.163, -8.012)
        nv1 = LINE.normal_vector(ratio=0.0, return_type='sl2d')
        nv2 = LINE.normal_vector(ratio=0.5, return_type='sl2d')
        nv3 = LINE.normal_vector(ratio=1.0, return_type='sl2d')
        LINE.is_normal((nv1, nv2, nv2))
        """
        if type(lines) not in dth.dt.ITERABLES:
            lines = [lines]
        # -------------------------------
        _tol_decplace_, normcheck = _tol_decplace_, []
        for line in lines:
            if round(self.gradient*line.gradient, _tol_decplace_) == -1:
                normcheck.append(True)
            else:
                normcheck.append(False)
        return normcheck

    def normal_vector(self, ratio=0.0, return_type='sl2d'):
        """
        Find normal vector centred at starting point.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        LINE = sl2d(-1.25, 1.068, 6.163, -8.012)
        nv1 = LINE.normal_vector(ratio=0.0, return_type='sl2d')
        nv2 = LINE.normal_vector(ratio=0.5, return_type='sl2d')
        nv3 = LINE.normal_vector(ratio=1.0, return_type='sl2d')
        LINE.is_normal((nv1, nv2))

        normal = LINE.normal_vector(ratio=0.5, return_type='sl2d')
        # CHECK:
        line.is_normal(normal)
        line.distance_to_lines(normal, refi='i', refj='all')
        """
        locx, locy = Point2d.from_line_factor(self, ratio).coords
        normal = Sline2d(self.x0, self.y0, self.x0-self.dy, self.y0+self.dx)
        normal.move(locx-normal.mid_coord[0], locy-normal.mid_coord[1])
        '''
        @CHECK:
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        LINE = sl2d(-1.25, 1.068, 6.163, -8.012)
        ratio = 0.2
        locx, locy = Point2d.from_line_factor(LINE, ratio).coords
        normal = Sline2d(LINE.x0, LINE.y0, LINE.x0-LINE.dy, LINE.y0+LINE.dx)
        normal.move(locx-normal.mid[0], locy-normal.mid[1])
        print(LINE.intersection_lines([normal])[0]) # A
        print([locx, locy]) # B
        # If A and B are same, we are correct.
        '''
        # --------------------------------------
        if return_type == 'coords':
            return normal.coords
        elif return_type == 'sl2d':
            return normal

    def generate_factors_0_and_1(self, dx1, dx2, dmean, k=0.0, th_res=1e-3, max_iter=50):
        """
        Generate internal split factors between two boundary offsets in [0, 1].

        Parameters
        ----------
        dx1, dx2 : float
            Left and right boundary offsets from 0 and 1 respectively.
        dmean : float
            Target mean spacing for generated factors.
        k : float, optional
            Relative random perturbation factor in [0, 0.25].
        th_res : float, optional
            Convergence tolerance on final residual.
        max_iter : int, optional
            Maximum attempts for stochastic convergence.

        Returns
        -------
        numpy.ndarray
            Monotonic factors beginning at `dx1` and ending near `1-dx2`.
        """
        if not (0 < dx1 < 1 and 0 < dx2 < 1 and dx1 + dx2 < 1):
            raise ValueError("dx1 and dx2 must be in (0, 1), and dx1 + dx2 < 1.")
        if not (0 <= k <= 0.25):
            raise ValueError("k must be in [0, 0.25].")
        if dmean <= 0 or th_res <= 0:
            raise ValueError("dmean and th_res must be > 0.")

        total_length = 1 - dx1 - dx2

        for _ in range(max_iter):
            n_points = max(2, int(round(total_length / dmean)))

            # Create random spacings with perturbation
            base_spacing = dmean
            perturbations = np.random.uniform(1 - k, 1 + k, size=n_points)
            raw_spacings = base_spacing * perturbations

            # Normalize to match total length
            scale_factor = total_length / np.sum(raw_spacings)
            spacings = raw_spacings * scale_factor

            # Generate points
            points = np.cumsum(spacings) + dx1

            if points[-1] > 1 - dx2:
                continue  # overshot; retry

            residual = abs((1 - dx2) - points[-1])
            if residual <= th_res:
                points = np.append(dx1, points)
                return points

        raise RuntimeError(f"Unable to converge within {max_iter} iterations. Final residual too high.")

    def stretch(self, point, sf):
        """
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(0, 0, 1, 0)
        line.stretch([0, 0], 1)
        line
        line.stretch([0, 0], 0.5)
        line
        line.stretch([0.5, 0], 0.5)
        line
        line.stretch([0, 0], 1)
        line
        """
        v1 = (self.coords[0]-point[0], self.coords[1]-point[1])
        v2 = (self.coords[2]-point[0], self.coords[3]-point[1])

        v1_scaled = (v1[0]*sf, v1[1]*sf)
        v2_scaled = (v2[0]*sf, v2[1]*sf)

        x0 = point[0] + v1_scaled[0]
        y0 = point[1] + v1_scaled[1]
        x1 = point[0] + v2_scaled[0]
        y1 = point[1] + v2_scaled[1]

        self.move_i((x0, y0))
        self.move_j((x1, y1))

    def distribute_normal_vectors(self,
                                  method='by_spacing',
                                  spacing_opt={'n': [10, 10],
                                               'spacing': 'constant',
                                               'factor': 0.5,
                                               'sub_spacing': ['cubic',
                                                               'cubic'],
                                               'subfactors': [0, 1],
                                               'trim_ij': True,
                                               '_coord_rounding_': (True, 8),
                                               '_plot_': True},
                                  points=None,
                                  ratios=None,
                                  perform_checks=False,
                                  _check_eps_=1E-8):
        """
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        Example-1. Using default values
        -------------------------------
        line = Sline2d(*np.random.randint(-10, 20, 3))
        normals = line.distribute_normal_vectors(5)
        line.plot(sl2d=normals)

        Examples-2, 3, 4
        ----------------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(*np.random.randint(-10, 20, 3))
        Example-2
        .........
        spacing_opt={'n': [10, 10], 'spacing': 'constant',
                     'factor': 0.25, 'sub_spacing': ['constant', 'constant'],
                     'subfactors': [0, 1], 'trim_ij': True,
                     '_coord_rounding_': (True, 8), '_plot_': True}
        normals = line.distribute_normal_vectors(method='by_spacing',
                                                 spacing_opt=spacing_opt)
        line.plot(sl2d=normals)
        Example-3
        .........
        spacing_opt={'n': [10, 10], 'spacing': 'constant',
                     'factor': 0.25, 'sub_spacing': ['constant', 'quadratic'],
                     'subfactors': [0, 1], 'trim_ij': True,
                     '_coord_rounding_': (True, 8), '_plot_': True}
        normals = line.distribute_normal_vectors(method='by_spacing',
                                                 spacing_opt=spacing_opt)
        line.plot(sl2d=normals)
        Example-4
        .........
        spacing_opt={'n': [10, 50], 'spacing': 'constant',
                     'factor': 0.25, 'sub_spacing': ['constant', 'cubic'],
                     'subfactors': [0, 1], 'trim_ij': True,
                     '_coord_rounding_': (True, 8), '_plot_': True}
        normals = line.distribute_normal_vectors(method='by_spacing',
                                                 spacing_opt=spacing_opt)
        line.plot(sl2d=normals)

        Example-5
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(*np.random.randint(-10, 20, 3))
        points=line.distribute_points(n=[10, 10],
                                      spacing='constant', factor=0.5,
                                      sub_spacing=['cubic','cubic'],
                                      subfactors=[0, 1],
                                      trim_ij=True,
                                      _coord_rounding_=(True, 8),
                                      _plot_=False)
        normals = line.distribute_normal_vectors(method='by_points', points=points)
        line.plot(sl2d=normals)

        Example-5
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(*np.random.randint(-10, 20, 3))
        line = Sline2d(0, 0, 1, 0)
        points, lines, mullines = line.divide_at_ratios([0.1, 0.2, 0.3, 0.4,
                                                         0.5, 0.6, 0.7, 0.8,
                                                         0.9])
        normals = line.distribute_normal_vectors(method='by_points', points=points)
        line.plot(sl2d=normals)

        Example-6
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(*np.random.randint(-10, 20, 3))
        f = line.generate_factors_0_and_1(dx1=0.15, dx2=0.15, dmean=0.15, k=0.2, th_res=0.02)
        points, lines, mullines = line.divide_at_ratios(f)
        normals = line.distribute_normal_vectors(method='by_points', points=points)
        for normal in normals:
            normal.stretch(normal.mid_coord, 10)
        line.plot(sl2d=normals)
        """
        if method not in ('by_spacing', 'by_points', 'by_ratios'):
            return ValueError('Invalid method specified.')
        # .........
        NORMAL = self.normal_vector(return_type='sl2d')
        if perform_checks:
            d = self.distance_to_lines(NORMAL,
                                       refi='i',
                                       refj='ij')[0].squeeze()
            if abs(d[0] - d[1]) <= _check_eps_ and self.is_normal(NORMAL):
                pass
            else:
                raise ValueError('Calculation unsuccessful. Check _check_eps_ value.')
        # ---------------------------------------------
        if method == 'by_spacing':
            # Define a bunch of default values
            '''@developer, maintainer: For internal control only.'''
            _DEF_n_ = 3
            _DEF_spac_ = 'constant'
            _DEF_fac_ = 0.5
            _DEF_subspac_ = None
            _DEF_subfac_ = None
            _DEF_trim_ij_ = False
            _DEF_coround_ = (True, 8)
            # Distribute points on the line.
            coords = self.distribute_points(n=spacing_opt.get('n', _DEF_n_),
                                            spacing=spacing_opt.get('spacing', _DEF_spac_),
                                            factor=spacing_opt.get('factor', _DEF_fac_),
                                            sub_spacing=spacing_opt.get('sub_spacing', _DEF_subspac_),
                                            subfactors=spacing_opt.get('subfactors', _DEF_subfac_),
                                            trim_ij=spacing_opt.get('trim_ij', _DEF_trim_ij_),
                                            _coord_rounding_=spacing_opt.get('_coord_rounding_', _DEF_coround_),
                                            _plot_=False)
        # ---------------------------------------------
        if method == 'by_points':
            coords = val_points_and_get_coords(points)
        # ---------------------------------------------
        if method == 'by_ratios':
            points, _, _ = self.divide_at_ratios(ratios)
            coords = val_points_and_get_coords(points[1:-1])
        # ---------------------------------------------
        normals = [deepcopy(NORMAL) for coord in coords]
        for normal, coord in zip(normals, coords):
            normal.move(coord[0]-self.x0, coord[1]-self.y0)
        return normals

    def plot(self, p2d=None, sl2d=None):
        """
        Quick-plot self line with optional additional points and lines.

        Parameters
        ----------
        p2d : Point2d or iterable, optional
            Point(s) to overlay.
        sl2d : Sline2d or iterable, optional
            Line(s) to overlay.
        """
        plt.plot([self.x0, self.x1], [self.y0, self.y1],
                 '-ko', markersize=12)
        if sl2d is not None:
            if type(sl2d) not in dth.dt.ITERABLES:
                sl2d = [sl2d]
            for line in sl2d:
                plt.plot([line.x0, line.x1], [line.y0, line.y1],
                         '--x', markersize=10)
        if p2d is not None:
            if type(p2d) not in dth.dt.ITERABLES:
                p2d = [p2d]
            for point in p2d:
                plt.plot(point.x, point.y, 'b+', markersize=12)

        plt.gca().set_aspect('equal')

    def distance_to_points(self, points=None, *, ref='all'):
        """
        Calculate the Baudhāyana distance between self and list of points.

        Parameters
        ----------
        points: List of points
        ref: Refers to point(s) location on the sline. Options include:
            * 'all': uses location i, j and the mid on the line.
            * 'i': starting point, (x0, y0)
            * 'j': starting point, (x1, y1)
            * 'mid': middle point.

        Returns
        -------
        list or numpy.ndarray
            Distances from the selected self reference location(s) to input
            points. For `ref='all'`, returns three distance arrays/lists for
            start, midpoint and end respectively.

        Example-1
        ---------
        from upxo.geoEntities.point2d import Point2d as p2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0)
        points = [p2d(xy[0], xy[1]) for xy in np.random.random((10,2))]
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

    def distance_to_lines(self, lines=None, method='ref',
                          refi='mid', refj='mid'):
        """
        Calculate the Baudhāyana distance between self and list of edges.

        Parameters
        ----------
        lines: List of lines
        method: Indicate the method of calculation. Can take following values:
            * 'ref': Use the reference location specifiers to calculate
            distance. That is, use refi and refj.
            * 'min': Minimum Baudhāyana distance
            * 'max': MAximum Baudhāyana distance
            * 'mean': Mean Baudhāyana distance
        refi: reference location for self i.e. i
        refj: referecne location for the other edge i.e. j

        Notes
        -----
        Current implementation computes pairwise Euclidean distances between
        selected reference points only. The `method` argument is currently not
        branched into `min`/`max`/`mean` logic inside this function.

        Return
        ------
        List of ndarray distances to all edges.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        rline = sl2d(0, 0, 1, 0)
        lines = [sl2d(0, i, 1, i) for i in range(1, 10)]
        rline.distance_to_lines(lines, refi='mid', refj='all')

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        if type(lines) not in dth.dt.ITERABLES:
            lines = [lines]
        # ---------------------------------
        if refi == 'all':
            refpnts = np.array([self.coord_i, self.mid_coord, self.coord_j])
        elif refi == 'i':
            refpnts = np.array([self.coord_i])
        elif refi == 'mid':
            refpnts = np.array([self.mid_coord])
        elif refi == 'j':
            refpnts = np.array([self.coord_j])
        elif refi == 'ij':
            refpnts = np.array([self.coord_i, self.coord_j])
        elif refi == 'imid':
            refpnts = np.array([self.coord_i, self.mid_coord])
        elif refi == 'midj':
            refpnts = np.array([self.mid_coord, self.coord_j])
        # ---------------------------------
        if refj == 'all':
            LinePoints = [np.array([line.coord_i, line.mid_coord, line.coord_j])
                          for line in lines]
        elif refj == 'i':
            LinePoints = [np.array([line.coord_i]) for line in lines]
        elif refj == 'mid':
            LinePoints = [np.array([line.mid_coord]) for line in lines]
        elif refj == 'j':
            LinePoints = [np.array([line.coord_j]) for line in lines]
        elif refj == 'ij':
            LinePoints = [np.array([line.coord_i, line.coord_j]) for line in lines]
        elif refj == 'imid':
            LinePoints = [np.array([line.coord_i, line.mid_coord]) for line in lines]
        elif refj == 'midj':
            LinePoints = [np.array([line.mid_coord, line.coord_j]) for line in lines]
        # ---------------------------------
        distances = []
        for linepoints in LinePoints:
            distances.append(np.linalg.norm(refpnts[:,None] - linepoints, axis=2))
        # ---------------------------------
        return distances

    def translate_by(self, *, vector=None, refloc = 'i', dist=None,
                     update=False, throw=True):
        """
        Translate the Edge by a Euclidean distance.

        Translate the Edge along the vector by a given distance. If
        update is True, then coords of the self will be updated.

        If throw is True and update is False, a new edge of the new
        coordinates shall be returned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Development phases
        ------------------
        Phase - 1: Without any validations

        Parameters
        ----------
        vector: Direction of translation. Two specifications allowed are:
             Specification 1: [vector start point coords,
                               vector end point coords]
             Specification 2: 'x+', 'z-'

        dist: Euclidean distance. If None and not a Number, then the
            translation distance will be the length of the vector. If a
            number, then dist will be the translation distance, in which case
            the vector will only be used to know the translation direction.

        update: Update the current point if True, do not update if False.

        throw: Return a edge if True, else return nothing if False.

        Return
        ------
        UPXO edge object: Conditional, depending on input throw (refer to
                                                                 description).

        Author
        ------
        Dr. Sunil Anandatheertha

        Examples
        --------

        Notes
        -----
        To be developed.
        """
        pass

    def intersection_lines(self, lines, dim=2):
        """
        Development milestones
        ----------------------
        * All in lines are UPXO type 2D line objects.
        * Consider 2D coordinate listings
        * Consider UPXO type 3D line objects.
        * Consider 3D coordinate listings

        Parameters
        ----------
        lines: Data representing 2D and/or 3D lines.
        dim: dimensionality. Default to 2. Options are:
            2: 2D.
            3: 3D.
            4: 2D/3D - indicates mixed collection in lines user input.

        Example
        -------
        import random
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d.by_coord([-1, -1], [3, 3]),
                 sl2d.by_coord([-1, 1], [1, -1]),
                 sl2d.by_coord([0, 0], [1, 0]),
                 sl2d.by_coord([0, 0], [0, 1])
                 ]
        sl2d.by_coord([0, 0], [0, 1]).intersection_lines(lines)
        sl2d.by_coord([1, 0], [1, 1]).intersection_lines(lines)
        """
        # Validations
        # ------------------------------------
        # When all are UPXO type 2D lines
        gradients = [l.gradient for l in lines]
        m = self.gradient
        c = self.yint  # b1
        yintercepts = [l.yint for l in lines]  # b2
        x = [(yint-c)/(m-grad) for yint, grad in zip(yintercepts, gradients)]
        y = [m*_x + c for _x in x]
        return [[_x, _y] for _x, _y in zip(x, y)]

    def rectangle(self, width, vis=False):
        """
        Return rectangle form of line.

        Convert self line into rectangle of length equal to line.length and
        width equal to the user specified width. Rectangle will completely
        bound the line. Ednd points of line will be at midpoints of
        corresponding opposite lines of the rectangle.

        Parameters
        ----------
        width: width of the rectangle.
        vis: bool, optional
            If True, plot a quick visualization of line and rectangle.

        Return
        ------
        coords: list of [p1, p2, p3, p4]. CCW from lower left point, p1.
        rectangle: Shapely polygon object

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        # =======================================
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        # ---------------------------
        line = sl2d(0, 0, 1, 0)
        _, r = line.rectangle(1, vis=True)
        # ---------------------------
        line = sl2d(1, 0, 0, 0)
        _, r = line.rectangle(1, vis=True)
        # =======================================
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        # ---------------------------
        line = sl2d(0, 0, 1, 1)
        _, r = line.rectangle(1, vis=True)
        # ---------------------------
        line = sl2d(1, 1, 0, 0)
        _, r = line.rectangle(1, vis=True)
        # ---------------------------
        line = sl2d(-1, -1, 1, 1)
        _, r = line.rectangle(1, vis=True)
        # ---------------------------
        line = sl2d(1, -1, -1, 1)
        _, r = line.rectangle(1, vis=True)
        """
        # Validate user input
        # ------------------------------------------
        def splot(r):
            x, y = r.boundary.xy
            # ..........
            # plt.plot(x[0], y[0], 'ro', markersize=6)
            # plt.plot(x[1], y[1], 'ro', markersize=8)
            # plt.plot(x[2], y[2], 'ro', markersize=10)
            # plt.plot(x[3], y[3], 'ro', markersize=12)
            # ..........
            # plt.plot(self.x0, self.y0, 'gs', markersize=12)
            # plt.plot(self.x1, self.y1, 'gs', markersize=16)
            # ..........
            plt.plot([self.x0, self.x1], [self.y0, self.y1], '-gs', markersize=8)
            # ..........
            plt.plot([x[0], x[1]], [y[0], y[1]], '--ko', markersize=6)
            plt.plot([x[1], x[2]], [y[1], y[2]], '--ko', markersize=6)
            plt.plot([x[2], x[3]], [y[2], y[3]], '--ko', markersize=6)
            plt.plot([x[3], x[0]], [y[3], y[0]], '--ko', markersize=6)

        # ------------------------------------------
        hw, ang = width/2, self.ang
        # Inlusivity drive: jya and upajya are for sin and cos in samskrita
        hw_jya, hw_upajya = hw*math.sin(ang), hw*math.cos(ang)
        # ----------------------------------------------------
        if self.vert:
            coords = [[self.x0-hw, self.y0], [self.x1+hw, self.y0],
                      [self.x1+hw, self.y1], [self.x0-hw, self.y1]]
        elif self.horz:
            coords = [[self.x0, self.y0-hw], [self.x1, self.y1-hw],
                      [self.x1, self.y1+hw], [self.x0, self.y0+hw]]
        else:
            coords = [[self.x0-hw_upajya, self.y0+hw_jya],
                      [self.x0+hw_upajya, self.y0-hw_jya],
                      [self.x1+hw_upajya, self.y1-hw_jya],
                      [self.x1-hw_upajya, self.y1+hw_jya]]
        rectangle = ShPol(coords)
        if vis:
            splot(rectangle)
        return coords, rectangle

    def identify_points_in_rectangle(self, points, width=None,
                                     boundary_points=True, vis=False):
        """
        Identify points which lie inside rectangle of the line.

        For a given set of points list, this function returns the
        indices of those points which are inside (and, on) the boundary of
        rectangle made from the self line (using width, of course).

        Parameters
        ----------
        points: coordinates as [[x-coords],[y-coords]]
        width: rectangle width
        boundary_points: If True, points on the boundary of the rectangle will
            be considered, else not.
        vis: If True, a quick visualization will be provided. Green is for
            self line, where smaller is starting point i and larger is ending
            point j. Marker sizes indicating corners of rectangle follow
            shapely polygon coordinate order. Black dots are points. Blue
            crosses are points which satisfy the geometric condition.
            NOTE: The above order may differ from UPXO line.rectangle.

        Return
        ------
        inside: Numpy arrau of bools. A True indicates position in input
            points which satisfy the geometric criteria.

        Notes
        -----
        The rectangle is created from `self.rectangle(width)`. If
        `boundary_points=True`, points on polygon boundaries are included.

        Examples
        --------
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0)
        _x, _y = np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)
        x, y = np.meshgrid(_x, _y)
        inside = line.identify_points_in_rectangle([x.ravel(), y.ravel()],
                                                   width=1,
                                                   boundary_points=True,
                                                   vis=True)
        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, 0, 1, 0)
        _x, _y = np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)
        x, y = np.meshgrid(_x, _y)
        points = [x.ravel(), y.ravel()]
        inside = line.identify_points_in_rectangle(points, 1, True, True)

        Example-3
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, 1, 1, 0)
        _x, _y = np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)
        x, y = np.meshgrid(_x, _y)
        inside = line.identify_points_in_rectangle(points, 1, True, True)

        Example-4
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 0, 1)
        _x, _y = np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)
        x, y = np.meshgrid(_x, _y)
        inside = line.identify_points_in_rectangle(points, 1, True, True)

        Example-5
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(1, -1, -1, 1)
        _x, _y = np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1)
        x, y = np.meshgrid(_x, _y)
        inside = line.identify_points_in_rectangle(points, 1, True, True)

        Example-6
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(1, -1, -1, 1)
        _x, _y = np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1)
        x, y = np.meshgrid(_x, _y)
        inside = line.identify_points_in_rectangle(points, 1, True, True)
        """
        # ------------------------------------------
        # Validate: points, width=None, boundary_points=True, vis=False
        # ------------------------------------------

        def splot(r, x, y, inside):
            rx, ry = r.boundary.xy
            # ..........
            plt.plot([rx[0], rx[1]], [ry[0], ry[1]], '--ko', markersize=6)
            plt.plot([rx[1], rx[2]], [ry[1], ry[2]], '--ko', markersize=6)
            plt.plot([rx[2], rx[3]], [ry[2], ry[3]], '--ko', markersize=6)
            plt.plot([rx[3], rx[0]], [ry[3], ry[0]], '--ko', markersize=6)
            # ..........
            plt.plot([self.x0, self.x1], [self.y0, self.y1], '-gs', markersize=8)
            # ..........
            plt.plot(x, y, 'k.')
            plt.plot(x[inside], y[inside], 'bx')
        # ------------------------------------------
        _, r = self.rectangle(width)
        x, y = points
        shPoints = [ShPnt(_x, _y) for _x, _y in zip(x, y)]
        inside = list(map(lambda shPoints: r.contains(shPoints), shPoints))
        if boundary_points:
            on_boundary = list(map(lambda shPoints: r.touches(shPoints),
                                   shPoints))
            inside = [_in or _onb for _in, _onb in zip(inside, on_boundary)]
        if vis:
            splot(r, x, y, inside)
        return inside

    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        """
        Translate self to the specified location.

        New location is specified by point object. POint object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be returned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        ref: Location on the edge which translates to new point. Values of ref
        could be:
            * 'i': Starting point of the edge
            * 'j': Ending point of the edge
            * 'mid': Middle point of the edge
            * [x, y, (z)]: Coordinate value

        point: New position. UPXO / direct point specification.

        update: Update the current point if True, do not update if False.

        throw: Return a point if True, else return nothing if False.

        Return
        ------
        UPXO point object: Conditional, depending on input throw (refer to
                                                                  description).

        Notes
        -----
        To be developed.
        """
        pass

    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        """
        Rotate point about the specified axis by the specified angle.

        New location is specified by point object. Point object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be returned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        axis: Axis of rotation. Two specifications allowed are:
             Specification 1: [axis start point coords,
                               axis end point coords]
             Specification 2: 'x+', 'z-'

        angle: Counter-Clockwise positive, angle of rotation

        degree: angle considered in degrees if True, radians if False.

        update: Update the current point if True, do not update if False.

        throw: Return a point if True, else return nothing if False.

        Return
        ------
        UPXO point object: Conditional, depending on input throw (refer to
                                                                  description).

        Notes
        -----
        To be developed.
        """
        pass

    def attach_mp(self, *, mp=None, name=None):
        """Attach a UPXO multi-point object and a name."""
        self.mp[name] = mp

    def attach_xtal(self, *, xtals=None):
        """
        Attach a list of UPXO xtal objects.

        Notes
        -----
        To be developed.
        """
        pass

    def perp_distance(self, plist, ptype='coord_list'):
        """
        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        from upxo.geoEntities.point2d import Point2d as p2d
        line = sl2d(0, 0, 1, 1)
        plist = (0.1, 0.1)
        line.perp_distance(plist)

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0.0)
        plist = np.random.random((4, 2)).T
        line.perp_distance(plist)

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0.0)
        plist = [p2d(1, 1), p2d(1, 2), p2d(1, 3), p2d(1, 4), p2d(1, 5)]
        line.perp_distance(plist, ptype = '[point2d]')
        """
        if ptype == 'coord_list':
            x, y = plist
        elif ptype in ('[point2d]', 'point2d', 'p2d'):
            if type(plist) not in dth.dt.ITERABLES:
                plist = [plist]
            x, y = np.array([[p.x, p.y] for p in plist]).T
        pd = abs((self.x1-self.x0)*(y-self.y0)-(x-self.x0)*(self.y1-self.y0))
        pd = pd/self.length
        return pd

    def find_neigh_point_by_perp_distance(self, points=None, r=0.25,
                                          use_bounding_rec=False, epsfactor=1E2,
                                          vis=False):
        """
        Find the neighbouring point(s) in a list of points by perp distance.

        Point subselection depends on whether a point is within OR on the
        cut-off perpendicular distance, r.

        Parameters
        ----------
        points: Elements of points must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        r: Cut-off perpendicular diatance.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        use_bounding_rec : bool, optional
            If True, perform candidate filtering using rectangle containment
            around the line with width `r` (plus optional epsilon padding).
        epsfactor : float, optional
            Multiplier for `Sline2d.ε` when expanding rectangle width in
            bounding-rectangle mode.
        vis : bool, optional
            If True, plot the selected neighboring points.

        Return
        ------
        list
            Indices in `points` satisfying selection criteria.
            Empty list if no points are inside cut-off.

        Example set-1
        -------------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0)
        x, y = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
        points = (x.ravel(), y.ravel())
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=False, epsfactor=1E6, vis=True)

        Example set-2
        -------------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, -1, 0)
        x, y = np.meshgrid(np.arange(-0.2, 1, 0.1), np.arange(-0.2, 1, 0.1))
        points = (x.ravel(), y.ravel())
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=0, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=1E7, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=False, epsfactor=0, vis=True)

        Example set-3
        -------------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0.5)
        x, y = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
        points = (x.ravel(), y.ravel())
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=0, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=1E7, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=False, epsfactor=0, vis=True)

        Example set-4
        -------------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-0.5, -1, 1, 0.5)
        x, y = np.meshgrid(np.arange(-0.2, 1, 0.1), np.arange(-0.2, 1, 0.1))
        points = (x.ravel(), y.ravel())
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=0, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=True, epsfactor=1E7, vis=True)
        line.find_neigh_point_by_perp_distance(points, 0.2, use_bounding_rec=False, epsfactor=0, vis=True)
        """
        def splot(line, x, y, xnearest, ynearest):
            plt.plot(line.x0, line.y0, 'gs', markersize=12)
            plt.plot(line.x1, line.y1, 'gs', markersize=16)
            # ..........
            plt.plot(x, y, 'k.')
            plt.plot(xnearest, ynearest, 'bx')
        # -------------------------------------------
        # Validate user inputs
        # -------------------------------------------
        nearest = self.perp_distance(points) <= r
        # -------------------------------------------
        if use_bounding_rec:
            width = r + (epsfactor > 0)*epsfactor*Sline2d.ε
            nearest = self.identify_points_in_rectangle(points,
                                                        width=width,
                                                        boundary_points=True,
                                                        vis=False)
        # -------------------------------------------
        if vis:
            x, y = points
            splot(self, x, y, x[nearest], y[nearest])
        return list(np.argwhere(nearest).T.squeeze())

    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        """
        Find n nearest neighbouring points in a specified list of points.

        Parameters
        ----------
        plist:  Elements of plist must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        n: Number of nearest neighbours to return. If not entered, a single
        point shall be returned.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        Return
        ------
        Indices in plist.

        Notes
        -----
        To be developed.
        """
        pass

    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        """
        Find the nearest UPXO multi-point in specified list of UPXO mulpoints.

        If tolerance factor, tolf is provided to be -1, then, even if a single
        point in a mp falls in  or on r, the index of mp in mplist will be
        added to the list to be returned. If 0 < tolf <= 1, then even if tolf
        factor of total number of points in a mp falls inside r, then the index
        of this mp in mplist will be added to the list to be returned.

        Parameters
        ----------
        mplist: Elements of plist must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.

        Notes
        -----
        To be developed.
        """
        pass

    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO edge in specified list of UPXO edges.

        Parameters
        ----------
        elist: Elements of elist must contain edges in either of the following
            two formats:
                1. 2D/3D UPXO edge objects.
                2. Iterable object with elements starting point: [x0, y0, z0]
                   and ending point: [x1, y1, z1]

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the following options:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No alternate.
            * 'all'. Both start and end points of the edge.  No alternate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.

        Notes
        -----
        To be developed.
        """
        pass

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
                1. 2D/3D UPXO medge objects.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the following options:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No alternate.
            * 'all'. Both start and end points of the edge.  No alternate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.

        Notes
        -----
        To be developed.
        """
        pass

    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO xtal in specified list of UPXO xtals.

        Parameters
        ----------
        xlist: Elements of xlist must contain xtals in either of the
        following three formats:
                1. 2D/3D UPXO xtal objects.
                2. Shapely polygon object.
                3. GMSH closed region.
                4. VTK polyhedra object.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the following options:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No alternate.
            * 'all'. Both start and end points of the edge.  No alternate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.

        Notes
        -----
        To be developed.
        """
        pass

    def find_colinear_lines(self, lines, line_repr='upxo'):
        """
        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, -1, 0)
        lines = [sl2d(0, 0, -1, 0), sl2d(0, 0, 1, 2),
                 sl2d(-1, 0, -2, 0), sl2d(4, 0, 3, 0),
                 sl2d(0, 0, 1, 0), sl2d(0, 1, 1, 1)]
        line.find_colinear_lines(lines, line_repr='upxo')

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(2, 7, 7, 3)
        lines = [[[0, 9], [0, 1]],
                 [[1, 8], [6, 2]],
                 [[3, 8], [8, 4]],
                 [[3, 6], [8, 4]],
                 ]
        line.find_colinear_lines(lines, line_repr='coord_list')

        Example-3
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, -1, 0)
        lines = [[[0, 0], [-1, 0]],
                 [[0, 0], [1, 2]],
                 [[-1, 0], [-2, 0]],
                 [[4, 0], [3, 0]],
                 [[0, 0], [1, 0]],
                 [[0, 1], [1, 1]],
                 [[0, 0], [-1, 0]]]
        line.find_colinear_lines(lines, line_repr='coord_list')
        """
        parallel = self.find_parallel_lines(lines, line_repr=line_repr)
        if parallel:
            if line_repr == 'upxo':
                midpoints = np.array([list(lines[prl].mid_coord)
                                      for prl in parallel]).T
            elif line_repr == 'coord_list':
                lines = np.array(lines)
                x1, y1 = lines[:, 0].T
                x2, y2 = lines[:, 1].T
                midpoints = np.array([[(xi+xj)/2, (yi+yj)/2] for xi, yi, xj, yj in zip(x1, y1, x2, y2)])
                midpoints = midpoints[parallel].T
            collinear_locs = self.perp_distance(midpoints) == 0
            if any(collinear_locs):
                return [parallel[i] for i in np.where(collinear_locs)[0]]
            else:
                return None
        else:
            return None

    def find_parallel_lines(self, lines, line_repr='upxo'):
        """
        Find line amongst lines parallel to self.

        Parameters
        ----------
        lines: list of data representing lines
        line_repr: specifies the type of line representation. Valid
            options include (see examples below for specific information.):
                * 'upxo'
                * 'coord_list'

        Return
        ------
        Location indices of those lines in lines parallel to self.

        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, -1, 0)
        lines = [sl2d(0, 0, -1, 0), sl2d(0, 0, 1, 2)]
        line.find_parallel_lines(lines, line_repr='upxo')

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(2, 7, 7, 3)
        lines = [[[0, 9], [0, 1]],
                 [[1, 8], [6, 2]],
                 [[2, 7], [7, 3]],
                 [[3, 6], [8, 4]]]
        line.find_parallel_lines(lines, line_repr='coord_list')
        """
        # Validate user inputs
        # -----------------------------------
        if line_repr == 'upxo':
            return [i for i, l in enumerate(lines) if self.gradient == l.gradient]
        if line_repr == 'coord_list':
            '''
            Expected: [[X1, Y1], [X2, Y2]]
            For example, lines = [[[0, 9], [0, 1]],
                                  [[1, 8], [6, 2]],
                                  [[2, 7], [7, 3]],
                                  [[3, 6], [8, 4]]]
            '''
            lines = np.array(lines)
            x1, y1 = lines[:, 0].T
            x2, y2 = lines[:, 1].T
            gradients = (y2-y1)/(x2-x1)
            locations = np.where(gradients == self.gradient)
            if locations[0].size:
                return list(locations[0])
            else:
                return None

    def make_points(self, n, spacing='linear', threshold_factor=1.0, start='i',
                    store_as_feature=False, feature_replace=False, vis=False):
        """
        Make n points on the line.

        Parameters
        ----------
        n: Number of points to make
        spacing: mathematical spacing to apply
        threshold_factor
        start: spacifies the location about which point creation starts.
        store_as_feature: If True, points will be stored in self.f dictionary.
            Defaults to False.
        feature_replace: If True, any existing feature points will be erased.
            DEfaults to False.
        vis: If True, result shall be visualized. Defaults to False.

        Returns
        -------
        points:
        increments:

        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, -1, 1, 1)
        line.make_points(10, spacing='linear', threshold_factor=1.0, vis=True)

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, -1, 1, 1)
        line.make_points(5, spacing='quadratic', threshold_factor=1.0, vis=True)

        Example-3
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, -1, 1, 1)
        line.make_points(5, spacing='linear', threshold_factor=0.5, vis=True)

        Example-4
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(-1, -1, 1, 1)
        line.make_points(5, spacing='linear', threshold_factor=0.5, start='j', vis=True)

        Example-5
        ---------
        # TODO: Relevant codes for store_as_feature and feature_replace use
        inputs is yet to be written.
        """
        # Validate user inputs
        # ------------------------------------
        def plot(points):
            plt.figure()
            plt.plot(self.x0, self.y0, 'ks', markersize=6)
            plt.plot(self.x1, self.y1, 'ks', markersize=8)
            x, y = np.array([[p.x, p.y] for p in points]).T
            plt.plot(x, y, 'bo', markersize=8, mfc='none')
        # ------------------------------------
        pi, _, pj = self.points
        incr = threshold_factor/(n+1)
        increments = np.arange(incr, threshold_factor-incr, incr)
        # ------------------------------------
        if spacing == 'linear':
            pass
        if spacing == 'quadratic':
            increments = increments**2
        # ------------------------------------
        increments *= self.length
        # ------------------------------------
        points = []
        for incr in increments:
            if start == 'i':
                points.append(pi.translate(vector=[pj.x, pj.y], dist=incr,
                                           update=False, throw=True))
            elif start == 'j':
                points.append(pj.translate(vector=[pi.x, pi.y], dist=incr,
                                           update=False, throw=True))
        # ------------------------------------
        if vis:
            plot(points)
        return points, increments

    def set_gmsh_props(self, prop_dict):
        """
        Set dictionary of gmsh properties.

        Notes
        -----
        To be developed.
        """
        pass

    def make_shapely(self):
        """
        Return shapely point object. Only valid for 2D.

        Notes
        -----
        To be developed.
        """
        pass

    def make_vtk(self):
        """
        Make VTK line object.

        Notes
        -----
        To be developed.
        """
        pass

    def generate_points(self, dxmean, pert_factor=0.0):
        """
        dxmean: Mean spacing
        pert_factor

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(0, 0, 1, 1)
        line.generate_points(np.sqrt(2)/10, pert_factor=1)
        """
        x0, y0, x1, y1 = self.coords

        if dxmean <= 0:
            raise ValueError("mean spacing 'dxmean' must be positive.")
        dx, dy = x1-x0, y1-y0
        n_points = max(1, int(np.floor(np.sqrt(dx**2+dy**2)/dxmean)))
        t_values = np.linspace(0, 1, n_points+1)
        pert = (np.random.random((n_points))-0.5)*dxmean*pert_factor

        points = [(x0+t*dx*(1+p), y0+t*dy*(1+p))
                  for t, p in zip(t_values, pert)][1:]
        return points

    def translate_along_normals(self, method='by_distances',
                                nlines=1, pert=0.0, spacing=1.0, dmax=None):
        """
        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)
        # ----------------------------------------------------------
        lines = line.translate_along_normals(method='by_count', nlines=[3, 2])
        line.plot(sl2d=lines)
        # ----------------------------------------------------------
        lines = line.translate_along_normals(method='by_distances',
                                             nlines=[[1.0, 2.0], [1.5, 3.0]],
                                             pert=0.1)
        line.plot(sl2d=lines)
        # ----------------------------------------------------------
        spacing={'dist': 'normal', 'mean': 0.25, 'std_factor': 0.025}
        lines = line.translate_along_normals(method='by_count', nlines=[5, 5],
                                             spacing=spacing)
        line.plot(sl2d=lines)
        # ----------------------------------------------------------
        spacing = {'dist': 'normal', 'mean': 0.25, 'std_factor': 0.1015}
        lines = line.translate_along_normals(method='by_count', nlines=[20,20],
                                      spacing=spacing, dmax=[12.5, 12.5])
        line.plot(sl2d=lines)
        # ----------------------------------------------------------
        """
        # Validations
        if method not in ['by_distances', 'by_count']:
            raise ValueError("Invalid method. Choose 'by_distances' or 'by_count'.")

        if method == 'by_distances':
            if not (isinstance(nlines, list) and len(nlines) == 2 and all(isinstance(lst, list) for lst in nlines)):
                raise ValueError("For 'by_distances', nlines must be a list of two lists.")
            if not (0.0 <= pert <= 0.25):
                raise ValueError("Perturbation must be between 0 and 0.25.")

        elif method == 'by_count':
            if not (isinstance(nlines, list) and len(nlines) == 2 and all(isinstance(v, int) and v >= 0 for v in nlines)):
                raise ValueError("For 'by_count', nlines must be a list of two non-negative integers.")
        # ---------------------------------------------------------------------
        dx, dy = self.dx, self.dy
        norm = np.sqrt(dx**2+dy**2)
        if norm == 0:
            raise ValueError("Zero-length line.")
        dx /= norm
        dy /= norm
        # ---------------------------------------------------------------------
        lines = []

        if method == 'by_distances':
            for i, dist_list in enumerate(nlines):
                sign = 1 if i == 0 else -1
                for dist in dist_list:
                    if pert > 0:
                        perturb = dist*np.random.uniform(-pert, pert)
                    else:
                        perturb = 0
                    dd = sign*(dist+perturb)
                    nx, ny = dy*dd, -dx*dd
                    lines.append(Sline2d(self.x0+nx, self.y0+ny,
                                         self.x1+nx, self.y1+ny))

        if method == 'by_count':
            def get_spacing(k):
                if isinstance(spacing, dict) and spacing.get("dist") == "normal":
                    mu = spacing.get("mean", 1.0)
                    std_factor = spacing.get("std_factor", 0.0)
                    sigma = abs(std_factor)*mu
                    return max(0.0, np.random.normal(mu, sigma))*k
                elif isinstance(spacing, (int, float)):
                    return spacing * k
                else:
                    raise ValueError("Invalid spacing. Must be float or dict with 'dist':'normal'.")

            for i, count in enumerate(nlines):
                sign = 1 if i == 0 else -1
                max_dist = None
                if isinstance(dmax, list) and len(dmax) == 2:
                    max_dist = dmax[i] - spacing.get("mean", 1.0) if isinstance(spacing, dict) else dmax[i] - spacing
                    max_dist = max(0.0, max_dist)
                cum_dist, k = 0.0, 1
                while k <= count:
                    sk = get_spacing(k)
                    if max_dist is not None and cum_dist+sk > max_dist:
                        break
                    cum_dist += sk
                    dd = sign*cum_dist
                    nx, ny = dy*dd, -dx*dd
                    lines.append(Sline2d(self.x0+nx, self.y0+ny,
                                         self.x1+nx, self.y1+ny))
                    k += 1
        return lines

    def translate_along_normals_2(self, method='by_distances', d=1, perturbation=0.0):
        """
        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)

        # Create 3 copies in one direction and 2 in the other, spaced unit distance apart
        translated_lines = line.translate_along_normals_2(
            method='by_count',
            d=[3, 2]
        )
        line.plot(sl2d=translated_lines)

        translated_lines = line.translate_along_normals_2(
    method='by_distances',
    d=[[1.0, 2.0], [1.5, 3.0]],
    perturbation=0.1
)
        line.plot(sl2d=translated_lines)
        """
        import numpy as np
        from copy import deepcopy

        if method not in ['by_distances', 'by_count']:
            raise ValueError("Invalid method. Choose from 'by_distances' or 'by_count'.")

        if method == 'by_distances':
            if not (isinstance(d, list) and len(d) == 2 and all(isinstance(sublist, list) for sublist in d)):
                raise ValueError("For 'by_distances', d must be a list of two lists.")

            if not (0.0 <= perturbation <= 0.25):
                raise ValueError("Perturbation must be between 0 and 0.25.")

        elif method == 'by_count':
            if not (isinstance(d, list) and len(d) == 2 and all(isinstance(val, int) and val >= 0 for val in d)):
                raise ValueError("For 'by_count', d must be a list of two non-negative integers.")

        # Unit normal
        dx, dy = self.dx, self.dy
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm == 0:
            raise ValueError("Zero-length line.")
        dx /= norm
        dy /= norm

        translated_lines = []

        if method == 'by_distances':
            for i, dist_list in enumerate(d):
                sign = 1 if i == 0 else -1
                for dist in dist_list:
                    perturb = dist * np.random.uniform(-perturbation, perturbation) if perturbation > 0 else 0
                    dd = sign * (dist + perturb)
                    nx = dy * dd
                    ny = -dx * dd
                    x0_new, y0_new = self.x0 + nx, self.y0 + ny
                    x1_new, y1_new = self.x1 + nx, self.y1 + ny
                    translated_lines.append(Sline2d(x0_new, y0_new, x1_new, y1_new))

        elif method == 'by_count':
            spacing = 1.0  # unit distance
            for i, count in enumerate(d):
                sign = 1 if i == 0 else -1
                for k in range(1, count + 1):
                    dd = sign * spacing * k
                    nx = dy * dd
                    ny = -dx * dd
                    x0_new, y0_new = self.x0 + nx, self.y0 + ny
                    x1_new, y1_new = self.x1 + nx, self.y1 + ny
                    translated_lines.append(Sline2d(x0_new, y0_new, x1_new, y1_new))

        return translated_lines


    def translate_along_normals_1(self, d=1):
        """
        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)
        line.translate_along_normals_1(d=[1, 20])

        line = Sline2d(-0.5, 0, 0.5, 0)
        # line.plot()
        line.plot(sl2d=line.translate_along_normals_1(d=[1, 3]))
        """
        # Validations
        if type(d) in ITERABLES:
            if not d:
                raise ValueError('Invalid distance specification.')
            if len(d) == 1:
                if type(d[0]) in NUMBERS:
                    d = [d[0], d[0]]
                else:
                    raise ValueError('Invalid distance specification.')
        else:
            if type(d) in NUMBERS:
                d = [d, d]
            else:
                 raise ValueError('Invalid distance specification.')

        # -------------------------------------------
        dx, dy = self.dx, self.dy
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx /= norm
            dy /= norm
        normal1_x = dy * d[0]
        normal1_y = -dx * d[0]
        normal2_x = -dy * d[1]
        normal2_y = dx * d[1]

        translated_lines = []
        for normal_x, normal_y in [(normal1_x, normal1_y), (normal2_x, normal2_y)]:
            x_new0 = self.x0 + normal_x
            y_new0 = self.y0 + normal_y
            x_new1 = self.x1 + normal_x
            y_new1 = self.y1 + normal_y
            translated_lines.append(Sline2d(x_new0, y_new0, x_new1, y_new1))

        return translated_lines

    def array_translation(self,
                          ncopies=10,
                          vector=[0, 1],
                          spacing='constant',
                          trim_self=True
                          ):
        """
        Make an array of points by repeat-translating self.

        Parameters
        ----------
        ncopies:
        vector:
        spacing:

        Return
        ------
        list of point objects

        Example-1
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)
        lines = line.array_translation(ncopies=2,
                                       vector=[0, 1],
                                       spacing='constant')
        line.plot(sl2d=lines)

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)
        nv = line.normal_vector(ratio=0.0, return_type='sl2d')
        lines = line.array_translation(ncopies=1,
                                       vector=[nv.x0, nv.y0],
                                       spacing='constant', trim_self=True)
        line.plot(sl2d=lines)

        Example-3
        ---------
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-0.5, -1, 1, 0.5)
        nv = line.normal_vector(ratio=0.0, return_type='sl2d')
        lines_A = line.array_translation(ncopies=1,
                                       vector=[nv.x0, nv.y0],
                                       spacing='constant', trim_self=True)
        lines_B = line.array_translation(ncopies=1,
                                         vector=[-nv.x0, -nv.y0],
                                         spacing='constant', trim_self=True)
        lines = lines_A + lines_B
        line.plot(sl2d=lines)
        """
        if not ncopies or not isinstance(ncopies, int) or ncopies < 0:
            raise ValueError('ncopies must be int > 0.')
        # More validations
        # ------------------------------------
        center_x, center_y = self.mid_coord
        dx, dy = vector
        # ------------------------------------
        if dx == 0:
            DX = [0 for _ in range(ncopies)]
        else:
            DX = np.arange(0, center_x+ncopies*dx, dx)
        # ------------------------------------
        if dy == 0:
            DY = [0 for _ in range(ncopies)]
        else:
            DY = np.arange(0, center_y+ncopies*dy, dy)
        # ------------------------------------
        lines = [Sline2d(self.x0+dx, self.y0+dy, self.x1+dx, self.y1+dy)
                 for dx, dy in zip(DX, DY)]
        # ------------------------------------
        if trim_self:
            lines = lines[1:]
        return lines

    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        """
        Get indices from a list of edges, which contain the self point.

        The point could lie on the end points of an edge or in-between the
        two end points of an edge.

        Parameters
        ----------
        elist: list of edge objects

        consider_ends: If True, index of the edge containing the selfpoint
        coordinates at one of its end points will also be returned. If False,
        the index will be included only if the point is not on the end
        points but completely inside the edge's end points.

        Return
        ------
        List of indices of points which satisfy the condition.

        Notes
        -----
        To be developed.
        """
        pass

    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """
        Get indices from a list of xtals, which contain the self point.

        The point could lie inside the xtal, on the boundaries of the xtal
        or on one of the end points of the many edges of the xtal.

        Parameter
        ---------
        xlist: list of edge objects

        consider_boundary: If True, search will be carried out to see if the
        self point lies on one of the boundary edges of the xtal. How this
        search behaves is decided by consider_boundary_ends.

        consider_boundary_ends: If True, index of the xtal containing the
        selfpoint coordinates at one of the end points of its the xtal's
        many boundary edges will be returned. If False, the index will be
        included only if the point is not on the end points but completely
        inside the edge's end points, provided that the cosider_boundary is
        True.

        Return
        ------
        List of indices of points which satisfy the condition.

        Notes
        -----
        To be developed.
        """
        pass

    def split(self, method='byfactor', f=0.5, divider=None,
              saa=False, throw=True, update='pntb',
              perform_containment_check=True):
        """
        Split the self.line at location(s) specified.

        Prameters
        ---------
        method: str
            Split specification mode. Supported values in current
            implementation are `factor`, `p2d` and `coord`.
        f: specifies the locations. Can be a numeric value or an iterable of
            numerical values. All values must be in the domain (0, 1).
        divider:
            Divider location used by `p2d` and `coord` modes.
        saa: bool
            If True, modifies current object and creates complementary segment.
        throw: bool
            If True, returns `(self, new_line)` after split.
        update: str
            Which endpoint of self is retained/updated (`pnta` or `pntb`).
        perform_containment_check: bool
            If True, validates divider is fully on and within self.

        Returns
        -------
        tuple
            `(self, new_line)` when `throw=True` and split succeeds.

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d as sl2d

        line = sl2d(0,0, 1,0)
        line.split(method='factor', f=0.75, saa=True, throw=True, update='pnta')

        line = sl2d(0,0, 1,0)
        line.split(method='factor', f=0.75, saa=True, throw=True, update='pntb')

        line = sl2d(0,0, 1,0)
        line.split(method='p2d', divider=Point2d(0.05, 0), saa=True, throw=True, update='pntb')

        line = sl2d(0,0, 1,0)
        line.split(method='p2d', divider=Point2d(0.05, 0), saa=True, throw=True, update='pnta')

        line = sl2d(0,0, 1,0)
        line.split(method='p2d', divider=Point2d(0.00, 0), saa=True, throw=True, update='pntb')

        line = sl2d(0,0, 1,0)
        line.split(method='coord', divider=(0.05, 0), saa=True, throw=True, update='pnta')

        line = sl2d(0,0, 1,0)
        line.split(method='coord', divider=(0.0, 0), saa=True, throw=True, update='pnta')
        """
        if method=='factor':
            if type(f) not in dth.dt.NUMBERS:
                raise TypeError('Invalid type spec for factor f.')
            if f > 0.0 and f < 1.0:
                point = (self.x0+f*self.dx, self.y0+f*self.dy)
                if saa:
                    if update == 'pnta':
                        startpoint = deepcopy(self.coord_list[0])
                        self.x0, self.y0 = point
                        self.pnta.x, self.pnta.y = point
                        new_line = Sline2d.by_coord(startpoint, point)
                    elif update == 'pntb':
                        endpoint = deepcopy(self.coord_list[1])
                        self.x1, self.y1 = point
                        self.pntb.x, self.pntb.y = point
                        new_line = Sline2d.by_coord(point, endpoint)
                if throw:
                    return self, new_line
            else:
                print('Point not fully inside line.')
        # -----------------------------------------------------
        if method=='p2d':
            # Validations
            # -----------------------------
            check_pass = True
            # Calculate the relative position and assess whether to proceed.
            if perform_containment_check:
                check_pass = self.fully_contains_point(divider)
            # -----------------------------
            if check_pass:
                dist_to_pnta = self.pnta.distance(plist=[divider])[0]
                factor = dist_to_pnta / self.length
                if factor > 0.0 and factor < 1.0:
                    if saa:
                        if update == 'pnta':
                            startpoint = deepcopy(self.coord_list[0])
                            self.x0, self.y0 = divider.x, divider.y
                            self.pnta = divider
                            new_line = Sline2d.by_p2d(Point2d(startpoint[0],
                                                              startpoint[1]
                                                              ),
                                                      divider)
                        elif update == 'pntb':
                            endpoint = deepcopy(self.coord_list[1])
                            self.x1, self.y1 = divider.x, divider.y
                            self.pntb = divider
                            new_line = Sline2d.by_p2d(divider,
                                                      Point2d(endpoint[0],
                                                              endpoint[1]
                                                              ))
                    if throw:
                        return self, new_line
                else:
                    print('Point not fully inside line.')
            else:
                raise ValueError('divider point not fully on and inside line.')
        # -----------------------------------------------------
        if method=='coord':
            # Validations
            # -----------------------------
            return self.split(method='p2d',
                              divider=Point2d(divider[0], divider[1],
                                              plane='xy'),
                              saa=saa, throw=throw, update=update)
