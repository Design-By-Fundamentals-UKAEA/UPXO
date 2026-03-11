"""
3D straight line.

Applications
------------
* Non-conformal geometry to conformal geometry conversion
* Heirarchical grain structure feature generation
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
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
np.seterr(divide='ignore')
from upxo._sup.validation_values import isinstance_many
from upxo.geoEntities.point3d import Point3d

class Sline3d_leanest():
    """
    Lean 2D straight line class.

    Import
    ------
    from upxo.geoEntities.sline3d import Sline3d_leanest as sl3dl

    Examples
    --------
    sl3dl(-2,3,4, 5,1,2)

    # Example-2
    for coord in e:
        print(coord)

    # Example-3
    print(e[1])
    """

    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1')

    def __init__(self, x0=0, y0=0, z0=0, x1=1, y1=0, z1=0):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1

    def __repr__(self):
        """Repr function."""
        return f'UPXO-sl3d-lean ({self.x0},{self.y0},{self.z0})-({self.x1},{self.y1},{self.z1})'

    def __iter__(self):
        """Make self an iterable over its two points."""
        return (i for i in ((self.x0, self.y0, self.z0),
                            (self.x1, self.y1, self.z1)))

    def __getitem__(self, index):
        """Make self indexable. 0: 1st point, 1: 2nd point, other: Error."""
        return ((self.x0, self.y0, self.z0),
                (self.x1, self.y1, self.z1))[index]

    @property
    def length(self):
        """Calculate and return self length."""
        return math.sqrt((self.x1-self.x0)**2+(self.y1-self.y0)**2+(self.z1-self.z0)**2)


class Sline3d():
    """
    UPXO code class.

    Examples
    --------
    from upxo.geoEntities.sline3d import Sline3d as sl3d
    sl3d(0,0,0, 1,1,1)
    """
    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'f', 'pnta', 'pntb')

    def __init__(self, x0=0, y0=0, z0=0, x1=1, y1=0, z1=0, pnta=None, pntb=None):
        if pnta is None and pntb is None:
            self.x0, self.y0, self.z0 = x0, y0, z0
            self.x1, self.y1, self.z1 = x1, y1, z1
            self.pnta, self.pntb = Point3d(x0, y0, z0), Point3d(x1, y1, z1)
        if pnta is not None and pntb is not None:
            self.pnta, self.pntb = pnta, pntb
            self.x0, self.y0, self.z0 = pnta.x, pnta.y, pnta.z
            self.x1, self.y1, self.z1 = pntb.x, pntb.y, pnta.z

    def __repr__(self):
        return f'UPXO-sl3d ({self.x0},{self.y0},{self.z0})-({self.x1},{self.y1},{self.z1}). {id(self)}'

    def __iter__(self):
        """Make self an iterable over its two points."""
        return (i for i in ((self.x0, self.y0, self.z0),
                            (self.x1, self.y1, self.z1)))

    def __getitem__(self, index):
        """Make self indexable. 0: 1st point, 1: 2nd point, other: Error."""
        return ((self.x0, self.y0, self.z0),
                (self.x1, self.y1, self.z1))[index]

    def __eq__(self, lines):
        """
        Check if the two edges are coincident.

        Examples
        --------
        """
        # Validate elist
        length = self.length
        return [length == e.length for e in lines]

    def __ne__(self, lines):
        """Check if the two edges are not coincident."""
        # No need of validations here.
        return [not eeq for eeq in self == lines]

    def __lt__(self, lines):
        """Check if the two edges are not coincident."""
        # No need of validations here.
        length = self.length
        return [length < l for l in lines]

    @classmethod
    def by_coord(cls, start, end):
        """
        Create Sline2d by specifying end coordinates.

        Parameters
        ----------
        start: Starting point coordinate [x0, y0]
        end: Ending point coordinate [x1, y1]

        Example
        -------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        A = sl2d.by_coord([-1, 2], [3, 4])
        """
        return cls(start[0], start[1], start[2], end[0], end[1], end[2])

    @classmethod
    def by_p3d(cls, start, end):
        """
        Create Sline3d by specifying end UPXO points.

        Parameters
        ----------
        start: Starting point
        end: Ending point

        Example
        -------
        from upxo.geoEntities.point3d import Point3d
        from upxo.geoEntities.sline3d import Sline3d
        Sline3d.by_p3d(Point3d(-1, 2, 3), Point3d(3, 4, 1))
        """
        return cls(pnta=start, pntb=end)

    @classmethod
    def by_vector(cls, point, xyproj):
        """
        A line can be represented using end point on the line and a dir. vector.
        line = [[x, y], [dx, dy]]
        """
        pass

    @property
    def mid(self):
        """
        Return the mid point.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.mid
        """
        return (self.xmid, self.ymid, self.zmid)

    @property
    def xmid(self):
        """
        Return the mid point along x.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.xmid
        """
        return (self.x0 + self.x1)/2

    @property
    def ymid(self):
        """
        Return the mid point along y.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.ymid
        """
        return (self.y0 + self.y1)/2

    @property
    def zmid(self):
        """
        Return the mid point along z.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.zmid
        """
        return (self.z0 + self.z1)/2

    @property
    def gradient(self):
        """
        Return the gradient of the line.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.gradient
        """
        grad = [math.inf, math.inf]
        dx, dy, dz = self.dxdydz
        if self.z0 != self.z1:
            grad[0] = (dx)/(dz)
        if self.z0 != self.z1:
            grad[1] = (dy)/(dz)
        return grad

    @property
    def delxyz(self):
        """
        Return the length increments along x, y and z.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.dxdydz
        """
        return self.dx, self.dy, self.dz

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
    def dz(self):
        """
        Return the length increments along z.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.dz
        """
        return self.z1-self.z0

    @property
    def ang(self):
        """
        Return the ccw + angle in radians.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.ang
        """
        dx, dy, dz = self.dxdydz
        angle_x = math.atan2(dy, dz)
        angle_y = math.atan2(dx, dz)
        angle_z = math.atan2(dy, dx)
        return [angle_x, angle_y, angle_z]

    @property
    def angd(self):
        """
        Return the ccw + angle in radians.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.angd
        """
        return [math.degrees(ang) for ang in self.ang]

    @property
    def length(self):
        """
        Calculate and return self length.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.length
        """
        return math.sqrt((self.x1-self.x0)**2+(self.y1-self.y0)**2+(self.z1-self.z0)**2)

    @property
    def dc(self):
        """
        Calculate and return direction cosines.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.dc
        """
        dx, dy, dz = self.dxdydz
        length = self.length
        return dx/length, dy/length, dz/length

    @property
    def coords(self):
        """
        Return coordinate array.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.coords
        """
        return [self.x0, self.y0, self.z0, self.x1, self.y1, self.z1]

    @property
    def coord_list(self):
        """
        Return coordinate array.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        a = sl3d(0,0,0, 1,1,1)
        a.coord_list
        """
        return [[self.x0, self.y0, self.z0], [self.x1, self.y1, self.z1]]

    @property
    def coord_i(self):
        return [self.x0, self.y0, self.z0]

    @property
    def coord_j(self):
        return [self.x1, self.y1, self.z1]

    @property
    def points(self):
        from upxo.geoEntities.point3d import Point3d
        mp = self.mid
        return [Point3d(self.x0, self.y0, self.z0),
                Point3d(mp[0], mp[1], mp[2]),
                Point3d(self.x1, self.y1, self.z1)]

    def is_point_endpoint(self, point):
        """
        Return True if point is one of the end points on the line.

        Example
        -------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        sl3d(0,0,0, 1,1,1).is_point_endpoint((0,0,0))
        sl3d(0,0,0, 1,1,1).is_point_endpoint([1, 2, 3])
        """
        is_endpoint = False
        if np.any(np.all(np.array(self.coord_list) == point, axis=1)):
            is_endpoint = True
        return is_endpoint

    def invert(self):
        ends = self.coord_list
        self.x0, self.y0, self.z0 = ends[1]
        self.x1, self.y1, self.z1 = ends[0]

    def move_i(self, point):
        self.x0, self.y0, self.z0 = point

    def move_j(self, point):
        self.x1, self.y1, self.z1 = point

    def distance_to_points(self, points=None, *, ref='all'):
        """
        Calculate the Baudhāyana distance between self and list of points.

        Parameters
        ----------
        points: LIst of points
        ref: Refers to point(s) location on the sline. Options include:
            * 'all': uses location i, j and the mid on the line.
            * 'i': starting point, (x0, y0)
            * 'j': starting point, (x1, y1)
            * 'mid': middle point.

        Example-1
        ---------
        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d

        line = sl3d(0,0,0, 1,1,1)
        points = [p3d(xy[0], xy[1], xy[2]) for xy in np.random.random((10,3))]

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
        """
        Example-1
        ---------
        from upxo.geoEntities.point3d import Point3d as p3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d

        line = sl3d(0,0,0, 1,0,0)
        plist = (1, 1, 1)
        line.perp_distance(plist)

        Example-2
        ---------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        line = sl3d(0,0,0, 1,0,0)

        plist = np.random.random((10000, 3))

        line.perp_distance_vectorized(plist, ptype='coord_list')

        Example-2
        ---------
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        line = sl2d(0, 0, 1, 0.0)
        plist = [p2d(1, 1), p2d(1, 2), p2d(1, 3), p2d(1, 4), p2d(1, 5)]
        line.perp_distance(plist, ptype = '[point2d]')
        """
        if ptype == 'coord_list':
            x, y, z = np.array(point)
        elif ptype in ('[point3d]', 'point3d', 'p3d'):
            if type(point) not in dth.dt.ITERABLES:
                plist = [point]
            x, y, z = np.array([[p.x, p.y, p.z] for p in plist]).T
        # ------------------------------------
        point = np.array([x, y, z])
        line_start = np.array(self.coord_list[0])
        line_end = np.array(self.coord_list[1])
        # ------------------------------------
        # Vector along the line
        line_vec = line_end - line_start
        # Vector from line start to the point
        point_vec = point - line_start
        # Project point_vec onto line_vec
        projection = np.dot(point_vec,
                            line_vec) / np.dot(line_vec, line_vec) * line_vec
        # Compute the perpendicular vector from the point to the line
        perpendicular_vec = point_vec - projection
        # Compute the distance (magnitude) of the perpendicular vector
        distance = np.linalg.norm(perpendicular_vec)
        return distance

    def perp_distance_vectorized(self, points, ptype='coord_list'):
        """
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        line = sl3d(0,0,0, 0.5,0.5,0.5)

        x = np.linspace(0, 1, 4)
        points = np.meshgrid(x, x, x)
        points = np.stack(points, axis=-1).reshape(-1, 3)

        line.perp_distance_vectorized(points, ptype='coord_list')
        """
        if ptype == 'coord_list':
            points = np.array(points)
        elif ptype in ('[point3d]', 'point3d', 'p3d'):
            points = np.array([[p.x, p.y, p.z] for p in points])

        # Extract line start and end coordinates
        line_start = np.array(self.coord_list[0])
        line_end = np.array(self.coord_list[1])

        # Vector along the line
        line_vec = line_end - line_start

        # Vector from line start to all points
        point_vec = points - line_start

        # Project point_vec onto line_vec
        line_vec_norm_sq = np.dot(line_vec, line_vec)
        projection = (np.dot(point_vec, line_vec)[:, None] / line_vec_norm_sq) * line_vec

        # Compute the perpendicular vectors
        perpendicular_vec = point_vec - projection

        # Compute the distances (magnitudes of the perpendicular vectors)
        distances = np.linalg.norm(perpendicular_vec, axis=1)

        return distances

    def extend(self, dincr, direction='both', saa=True, throw=False):
        # -------------------------------------
        P1 = np.array([self.x0, self.y0, self.z0])
        P2 = np.array([self.x1, self.y1, self.z1])
        # -------------------------------------
        line_vec = P2 - P1
        unit_vec = line_vec / np.linalg.norm(line_vec)
        # -------------------------------------
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
        # -------------------------------------
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
        """
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        LINE = sl3d(0,0,0, 0.5,0.5,0.5)

        x = np.linspace(0, 5, 4)
        points = np.meshgrid(x, x, x)
        points = np.stack(points, axis=-1).reshape(-1, 3)

        LINE.extend_until_exhaustion(points,
                                     voxel_size=x[1]-x[0],
                                     dincr_factor=0.1,
                                     direction='end')
        LINE
        """
        threshold_distance = 1.7320508075688772 * voxel_size

        # Initial setup
        P1 = np.array([LINE.x0, LINE.y0, LINE.z0])
        P2 = np.array([LINE.x1, LINE.y1, LINE.z1])
        points = np.array(points)

        # Calculate initial number of points within the threshold
        pdist = LINE.perp_distance_vectorized(points, ptype='coord_list')
        initial_count = np.sum(pdist <= threshold_distance)
        prev_count = initial_count

        max_iterations = max_iterations
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # Extend the line using the existing extend method
            P1_extended, P2_extended = LINE.extend(dincr_factor*LINE.length,
                                                   direction=direction,
                                                   saa=False, throw=True)
            _line_ = sl3d(P1_extended[0], P1_extended[1], P1_extended[2],
                          P2_extended[0], P2_extended[1], P2_extended[2])

            # Update the line endpoints for distance calculation
            self.x0, self.y0, self.z0 = P1_extended
            self.x1, self.y1, self.z1 = P2_extended

            # Calculate the new number of points within the threshold
            pdist = _line_.perp_distance_vectorized(points, ptype='coord_list')
            new_count = np.sum(pdist <= threshold_distance)

            # Stop extending if the count no longer increases
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
                        print('The start and end point oindices are:')
                        return actual_start_point, actual_end_point
                elif direction == 'start':
                    pass
                elif direction == 'end':
                    far_pnt = np.argmax(_line_.distance_to_points(points,
                                                                ref='end'))
                    actual_end_point = points[far_pnt]
                    print(f'No. of iterations: {iteration}')
                    if saa_final:
                        # self.x0, self.y0, self.z0 = actual_start_point
                        self.x1, self.y1, self.z1 = actual_end_point
                    if throw_final:
                        print('The start and end point oindices are:')
                        start_point = np.array([self.x1, self.y1, self.z1])
                        return start_point, actual_end_point
                break

            # Update endpoints and previous count
            P1, P2 = P1_extended, P2_extended
            prev_count = new_count

    def split(self, f=0.5, saa=False, throw=True, retain=0):
        """
        Split the self.line at location(s) specified.

        Prameters
        ---------
        f: specifies the locat5ions. Can be a numeric value or an iterable of
            numerical values. All values must be in the domain (0, 1).

        Examples
        --------
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        line = sl3d(0,0,0, 1,0,0)
        line.split(f=0.75, saa=False, throw=True)
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
            # TODO.
            pass
