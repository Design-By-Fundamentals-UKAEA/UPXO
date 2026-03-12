"""
2D multi-point collection.

This module provides a container class for managing ordered collections of
2D points in UPXO. It supports construction from coordinate arrays, individual
Point2d objects, rectangular grids, and intersection results, and exposes
numerical, geometric, and spatial-query operations over the full point set.

Imports
-------
from upxo.geoEntities.mulpoint2d import MPoint2d

Recommended alias imports:
--------------------------
from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d

Metadata
--------
* Module: upxo.geoEntities.mulpoint2d
* Package: upxo
* License: GPL-3.0-only
* Author: Dr. Sunil Anandatheertha
* Email: vaasu.anandatheertha@ukaea.uk
* Status: Active development
* Last updated: 2026-03-12

Applications
------------
* Storing and manipulating polycrystal grain-boundary vertex sets
* Batch spatial queries (distances, centroid proximity, containment)
* Rectangular-grid point generation for structured-domain discretization
* Clustering-based synthetic point-cloud generation
* Intersection-point collection from multi-line geometry operations

Classes
-------
* MPoint2d - ordered 2D multi-point collection backed by a (N x 2) NumPy array

Definitions
-----------
coords : np.ndarray, shape (N, 2)
    Array of (x, y) coordinate pairs for all points in the collection.
points : list of Point2d
    Corresponding list of UPXO Point2d objects.
n : int
    Number of points in the collection (read-only property).
centroid : np.ndarray, shape (2,)
    Arithmetic mean of all (x, y) coordinates.
"""
import math
import numpy as np
import numpy.matlib
from copy import deepcopy
# from icecream import ic
import itertools
from scipy.spatial import cKDTree
import vtk
from shapely.geometry import Point as ShPnt, Polygon as ShPol
from shapely.geometry import LineString
from functools import wraps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import upxo._sup.dataTypeHandlers as dth
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
# from upxo._sup.validation_values import find_pnt_spec_type_2d
np.seterr(divide='ignore')
from upxo.geoEntities.featmake import make_p2d, make_p3d
from upxo._sup.validation_values import find_spec_of_points
from upxo._sup.validation_values import isinstance_many
import upxo.geoEntities.featmake as fmake
from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.point3d import Point3d
from upxo._sup.validation_values import val_point_and_get_coord, val_points_and_get_coords

class MPoint2d():
    """
    UPXO core class. Collection of 2D points. Offers wide spectrucm of operations.

    Stores points both as a NumPy array (``coords``) for vectorised numerical
    operations and as a list of UPXO ``Point2d`` objects (``points``) for
    object-level access. Supports construction from raw coordinates, existing
    ``Point2d`` instances, rectangular grids, clustering distributions, and
    multi-line intersection results.

    Parameters
    ----------
    coords : array-like of shape (N, 2), optional
        Sequence of (x, y) coordinate pairs.  If provided, ``points`` must be
        None; individual ``Point2d`` objects are created automatically.
    points : list of Point2d, optional
        Pre-constructed UPXO ``Point2d`` objects.  If provided, ``coords``
        must be None; the coordinate array is derived from the points.

    Attributes
    ----------
    coords : np.ndarray, shape (N, 2)
        Flat array of (x, y) coordinate pairs.
    points : list of Point2d
        Corresponding UPXO ``Point2d`` object list.
    EPS : float
        Small tolerance constant (1e-8) used in geometric comparisons.

    Properties
    ----------
    n : int
        Number of points in the collection.
    centroid : np.ndarray, shape (2,)
        Arithmetic mean of all (x, y) coordinates.
    x : np.ndarray
        Array of x-coordinates.
    y : np.ndarray
        Array of y-coordinates.

    Class Methods (constructors)
    ----------------------------
    from_coords(point_coords)
        Construct from an (Nx2) coordinate array or list.
    from_xy(xy)
        Construct from a (2xN) array of x and y coordinate rows.
    from_upxo_points2d(points)
        Construct from a list of ``Point2d`` objects.
    from_rect_grid(xstart, xinc, xend, ystart, yinc, yend)
        Construct from a regular rectangular point grid.
    from_clustering_around_centroid(centroid, n, r, distribution, dmin)
        Generate a clustered random point cloud around a centroid.
    from_intersection_linesA_linesB(La, Lb, ...)
        Collect all pairwise intersection points of two line sets.

    Standard data format
    --------------------
    coords: np.array([[0, 0], [1, 1], [2, 3], [4, 5]])

    Import
    ------
    from upxo.geoEntities.mulpoint2d import MPoint2d
    from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
    """
    EPS = 1E-8
    __slots__ = ('coords', 'points')

    def __init__(self, coords=None, points=None):
        if coords is not None and points is None:
            self.coords = np.array(coords)
            self.points = [Point2d(xy[0], xy[1]) for xy in coords]
        if coords is None and points is not None:
            self.points = points
            self.coords = np.array([[p.x, p.y] for p in points])

    def __repr__(self):
        return f'UPXO-mp2d. n={self.n}.'

    def __iter__(self):
        """
        Return an iterable of point coordsinates in self.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        for coord in mulpoint2d:
            print(coord)
        """
        return iter(self.coords)

    def __getitem__(self, i):
        """
        Make self indexable. i: index location.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d[9]
        mulpoint2d[10]
        """
        if i >= self.n:
            raise ValueError('Index exceeds maximum number of coordinates.')
        return self.coords[i]

    def add(self, toadd=None, operation='add'):
        """
        Add toadd to self.coords.

        Example-1
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d.coords
        mulpoint2d.add(toadd=10, operation='add')
        mulpoint2d.coords

        Example-2
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d.coords
        mulpoint2d.add(toadd=[-10, 20], operation='add')
        mulpoint2d.coords

        Example-3
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d.coords
        mulpoint2d.add(toadd=np.random.random((mulpoint2d.n, 2)), operation='add')
        mulpoint2d.coords

        Example-4
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d.coords
        mulpoint2d.add(toadd=np.random.random((mulpoint2d.n, 2)).T, operation='add')
        mulpoint2d.coords

        Example-5
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((10,2)))
        mulpoint2d.coords
        mulpoint2d.add(toadd=np.random.random((10,2)), operation='append')
        mulpoint2d.coords
        """
        if toadd is None:
            return
        else:
            if operation == 'add':
                if type(toadd) in dth.dt.NUMBERS:
                    self.coords += toadd
                if type(toadd) in dth.dt.ITERABLES:
                    if find_spec_of_points(toadd) == 'type-[1,2]':
                        '''
                        toadd = [0, 0]
                        find_spec_of_points(toadd)
                        '''
                        self.coords += np.array(toadd)
                    if find_spec_of_points(toadd) == 'type-[[1,2]]':
                        '''
                        toadd = [[0, 0]]
                        find_spec_of_points(toadd)
                        '''
                        self.coords += np.array(toadd[0])
                    if find_spec_of_points(toadd) == 'type-[[1,2],[4,5],[7,8]]':
                        '''
                        toadd = [[1,2],[4,5],[7,8]]
                        find_spec_of_points(toadd)
                        '''
                        if len(toadd) == self.n:
                            self.coords += np.array(toadd)
                        else:
                            raise ValueError('Invalid length of toadd.')
                    if find_spec_of_points(toadd) == 'type-[[1,2,3,4],[5,6,7,8]]':
                        '''
                        toadd = [[1,2,3,4],[5,6,7,8]]
                        find_spec_of_points(toadd)
                        '''
                        if len(toadd[0]) == self.n:
                            self.coords += np.array(toadd).T
                        else:
                            raise ValueError('Invalid length of toadd.')
            elif operation == 'append':
                if type(toadd) in dth.dt.ITERABLES:
                    if find_spec_of_points(toadd) == 'type-[1,2,3]':
                        '''
                        toadd = [0, 0, 0]
                        find_spec_of_points(toadd)
                        '''
                        self.coords = np.array(list(self.coords) + list(toadd))
                    if find_spec_of_points(toadd) == 'type-[[1,2,3]]':
                        '''
                        toadd = [[0, 0, 0]]
                        find_spec_of_points(toadd)
                        '''
                        self.coords = np.array(list(self.coords) + list(toadd[0]))
                    if find_spec_of_points(toadd) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
                        '''
                        toadd = [[1,2,3],[4,5,6],[7,8,9],[7,8,9]]
                        find_spec_of_points(toadd)
                        '''
                        toadd = [list(ta) for ta in toadd]
                        coords = [list(coord) for coord in self.coords]
                        self.coords = np.array(coords+toadd)
                    if find_spec_of_points(toadd) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
                        '''
                        toadd = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
                        find_spec_of_points(toadd)
                        '''
                        toadd = [list(ta) for ta in np.array(toadd).T]
                        coords = [list(coord) for coord in self.coords]
                        self.coords += np.array(toadd).T

    @classmethod
    def from_coords(cls, point_coords):
        """
        Instantiate mulpoint2d using list of point coordinates.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        point_coords = np.array([[0, 0], [1, 1], [2, 3], [4, 5]])
        MULPOINT2D = mp2d.from_coords(point_coords)
        MULPOINT2D.coords
        MULPOINT2D.points
        """
        # Validations
        return cls(coords=np.array(point_coords))

    @classmethod
    def from_xy(cls, xy):
        """
        Instantiate mulpoint2d using array of x, y and z coordinate lists.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        xy = np.array([[0, 0], [1, 1], [2, 3], [4, 5]]).T
        MULPOINT2D = mp2d.from_xy(xy)
        MULPOINT2D.coords
        """
        # Validations
        return cls(coords = xy.T)

    @classmethod
    def from_upxo_points2d(cls, points, zloc=0.0):
        return cls(coords=None, points=points)

    @classmethod
    def from_mulpoint2d(cls, mp2d, zloc=0.0):
        pass

    @classmethod
    def from_rect_grid(cls, xstart, xinc, xend, ystart, yinc, yend):
        """
        Example-1
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        mp = MPoint2d.from_rect_grid(0, 1, 5, 0, 1, 3)
        """
        # Validations
        # ------------------------------
        x, y = np.meshgrid(np.arange(xstart, xend, xinc),
                           np.arange(ystart, yend, yinc))
        coords = np.c_[x.ravel(), y.ravel()]
        return cls(coords=coords)

    @classmethod
    def from_clustering_around_centroid(cls, centroid, n=10, r=1,
                                        distribution='urand', dmin=None):
        """
        Examnple-1
        ----------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        MP2D = MPoint2d.from_clustering_around_centroid((-5, -10),
                                                        n=1000, r=1,
                                                        distribution='urand',
                                                        dmin=0.1)
        plt.scatter(MP2D.coords[:,0],MP2D.coords[:,1])

        Examnple-2
        ----------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        cenx, ceny = [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]
        for cx, cy in zip(cenx, ceny):
            MP2D = MPoint2d.from_clustering_around_centroid((cx, cy),
                                                            n=250, r=0.25,
                                                            distribution='urand',
                                                            dmin=0.1)
            plt.scatter(MP2D.coords[:,0],MP2D.coords[:,1])

        Examnple-3
        ----------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        from upxo.geoEntities.sline2d import Sline2d
        line = Sline2d(-1, 0, 0, 1)
        refpoints = line.distribute_points(n=[5, 3],
                                           spacing='constant', factor=0.6,
                                           sub_spacing=['cubic','cubic'],
                                           subfactors=[0, 1],
                                           trim_ij=True,
                                           _coord_rounding_=(True, 8),
                                           _plot_=False)
        for refpnt in refpoints:
            MP2D = MPoint2d.from_clustering_around_centroid((refpnt[0],
                                                             refpnt[1]),
                                                            n=12, r=0.1,
                                                            distribution='urand',
                                                            dmin=0.1)
            plt.plot(MP2D.coords[:,0],MP2D.coords[:,1], '.', ms=5)

        Example-4
        ---------
        in this example, you will create point distributions at a certain
        distance, normal to a line. We will uise both normals. A practical
        application of this is inserting precipitates near to a grain
        boundary edge.

        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.sline2d import Sline2d
        from upxo.geoEntities.mulpoint2d import MPoint2d
        #  . . . . . . . . . . . . . . . . . . . . . .
        # Create the line of interest
        line = Sline2d(-1, 0, 0, 1)
        #  . . . . . . . . . . . . . . . . . . . . . .
        # Generate parallel lines along normal at distances of 0.25 and 0.4
        PAR_lines = line.translate_along_normals(d=[0.25, 0.4])
        #  . . . . . . . . . . . . . . . . . . . . . .
        # Distribute points, which would then function as centroids of
        # point clusters, to be introduced later on. We will do this on botrh
        # lines.
        refpoints_A = PAR_lines[0].distribute_points(n=[5, 3],
                                           spacing='constant', factor=0.6,
                                           sub_spacing=['cubic','cubic'],
                                           subfactors=[0, 1],
                                           trim_ij=True,
                                           _coord_rounding_=(True, 8),
                                           _plot_=False)
        refpoints_B = PAR_lines[1].distribute_points(n=[8, 4],
                                           spacing='constant', factor=0.6,
                                           sub_spacing=['cubic','quadratic'],
                                           subfactors=[0, 1],
                                           trim_ij=True,
                                           _coord_rounding_=(True, 8),
                                           _plot_=False)
        #  . . . . . . . . . . . . . . . . . . . . . .
        # Lets visualize thewse centroidal points in relation to the
        # original line.
        plt.plot([line.x0, line.x1], [line.y0, line.y1], '-ko', ms=12, lw=3)
        for refpnt_A in refpoints_A:
            plt.plot(refpnt_A[0], refpnt_A[1], '.', ms=5)
        for refpnt_B in refpoints_B:
            plt.plot(refpnt_B[0], refpnt_B[1], '.', ms=5)
        #  . . . . . . . . . . . . . . . . . . . . . .
        # We will now generate points around each centroid. We will use
        # multipoint method, where we first create a multipoint with points
        # being polar translations of the centroid itself. This is recommended
        # as this avoids the increase density of pointsa near the centroid.
        MP2D_A, MP2D_B = [], []
        for rf_A in refpoints_A:
            MP2D_A.append(MPoint2d.from_clustering_around_centroid((rf_A[0], rf_A[1]),
                                                                    n=12,
                                                                    r=0.2,
                                                                    distribution='urand',
                                                                    dmin=0.1))

        for rf_B in refpoints_B:
            MP2D_B.append(MPoint2d.from_clustering_around_centroid((rf_B[0], rf_B[1]),
                                                                    n=25,
                                                                    r=0.075,
                                                                    distribution='urand',
                                                                    dmin=0.1))
        #  . . . . . . . . . . . . . . . . . . . . . .
        # Lets now visualize all the points in relation to the or9iginal line.
        plt.plot([line.x0, line.x1], [line.y0, line.y1], '-ko', ms=12, lw=3)
        for mp2d_a in MP2D_A:
            plt.plot(mp2d_a.x, mp2d_a.y, '.', ms=5)
        for mp2d_b in MP2D_B:
            plt.plot(mp2d_b.x, mp2d_b.y, '.', ms=5)
        """
        # Validations
        centroid = val_point_and_get_coord(centroid, return_type='upxo',
                                           safe_exit=False)
        # --------------------------------------
        return cls(centroid.array_by_clustering(n=n, r=r,
                                                return_type='coords_2d'))

    @classmethod
    def from_intersection_linesA_linesB(cls, La, Lb,
                                        return_ordered_points=True,
                                        plot=False):
        """
        Create mulpoint from intersection of many lines.

        Checks lines for intersections and creates UPXO point object at
        every intersection.NOTE: This method can potentially cause a
        bottleneck in case of large lines array size.

        Parameters
        ----------
        la: list of upxo line objects
        lb: list of upxo line objects
        return_ordered_points: if True, ordered points will also be return3ed
        plot: if True, lines and intersection points will be visualized.

        Return
        ------
        cls: MPoint2d of the list of intersection points.
        points: ordered list of points. Number of rows is len(lb) and number
            of columns is len(la)

        Examples
        --------
        from upxo.geoEntities.sline2d import Sline2d
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.mulpoint2d import MPoint2d

        na, nb = 5, 5
        R = np.random.rand
        La = [Sline2d.by_coord([-10-R(),-10+R()], [10+R(),10-R()])
              for _ in range(na)]
        Lb = [Sline2d.by_coord([-10+R(),-10-R()], [10-R(),10+R()])
              for _ in range(nb)]
        MPoint2d.from_intersection_linesA_linesB(La, Lb,
                                                 return_ordered_points=True,
                                                 plot=True)
        MP2D = MPoint2d.from_intersection_linesA_linesB(La, Lb,
                                                        return_ordered_points=False,
                                                        plot=False)
        MP2D.points, MP2D.coords
        """
        # Validations
        '''Validation: La and Lb must be non-empty Iterables.'''
        # ----------------------------------------------
        na, nb = len(La), len(Lb)
        points = [[None for _ in range(na)] for __ in range(nb)]
        for lai, la in enumerate(La):
            for lbi, lb in enumerate(Lb):
                points[lbi][lai] = Point2d.from_intersection_two_lines(la, lb,
                                                                       tool='upxo',
                                                                       return_type='upxo')
        # ----------------------------------------------
        points_all = []
        for lai in range(na):
            for lbi in range(nb):
                for pnts in points[lbi][lai]:
                    points_all.append(pnts)
        # ----------------------------------------------
        if plot:
            La[0].plot(p2d=points_all, sl2d=La[1:]+Lb)
        # return points, points_all
        if return_ordered_points:
            return cls([[pnt.x, pnt.y] for pnt in points_all]), points
        else:
            return cls([[pnt.x, pnt.y] for pnt in points_all])

    @property
    def n(self):
        return len(self.coords)

    @property
    def centroid(self):
        return np.mean(self.coords, axis=0)

    @property
    def get_points(self):
        return [Point2d(x, y) for x, y in self.coords]

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    def __contains__(self, point, validate=False):
        """
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        xy = np.array([[0, 0], [1, 1], [2, 3], [4, 5]]).T
        MULPOINT2D = mp2d.from_xy(xy)
        [0, 0] in MULPOINT2D
        """
        return any(self.squared_distances_to_point(point,
                                                   validate=validate) <= self.EPS)

    def squared_distances_to_point(self, point, validate=True):
        """
        When validate is True, then point can be any of the permitted forms.
        When validate is False, then point must be in UPXO point format.
        """
        if validate:
            point = val_point_and_get_coord(point, return_type='coord',
                                            safe_exit=False)
            return (self.x-point[0])**2 + (self.y-point[1])**2
        else:
            return (self.x-point.x)**2 + (self.y-point.y)**2

    def distances_to_point(self, point):
        return np.sqrt(self.squared_distances_to_point(point))

    def squared_distance_to_centroid(self, points,
                                     validate_points=True,
                                     points_type='numpy'):
        """
        Calculates squared distances between self.centroid and other 2D points.

        Parameters
        ----------
        points: list of points

        validate_points: If True, validation will be used. When confident that
            points are provided as a numpy array of coordinate pairs, it is
            advised to keep this False. When unknown, keep it True. True will
            may increase computation time depending on the number of points.

        points_type: If validate_points is False, then points_type must be
            'numpy'. You could also use 'coord' but, this would include an
            additional overhead of conversion from coord to numpy array. This
            is provided to ensure safe claculation.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        from upxo.geoEntities.point2d import Point2d
        MULPOINT2D = MPoint2d.from_coords(np.random.random((10, 2)))
        POINTS = make_p2d(2+np.random.random((10, 2)), return_type='p2d')
        MULPOINT2D.squared_distance_to_centroid(POINTS, validate_points=True)
        POINTS = 2+np.random.random((10, 2))
        MULPOINT2D.squared_distance_to_centroid(POINTS, validate_points=False,
                                                points_type='numpy')
        """
        cen = self.centroid
        if validate_points:
            pnts = val_points_and_get_coords(points,
                                             return_type='numpy',
                                             safe_exit=False)
        else:
            if points_type in ('upxo', 'shapely'):
                pnts = val_points_and_get_coords(points,
                                                 return_type='numpy',
                                                 safe_exit=False)
            elif points_type in ('coord', 'coord_pair'):
                pnts = val_points_and_get_coords(np.array(points),
                                                 return_type='numpy',
                                                 safe_exit=False)
            elif points_type in ('np', 'numpy'):
                pnts = points
        return (pnts[:, 0]-cen[0])**2 + (pnts[:, 1]-cen[1])**2

    def distance_to_centroid(self, points, validate_points=True,
                             points_type='numpy'):
        """
        Calculates squared distances between self.centroid and other 2D points.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d
        from upxo.geoEntities.point2d import Point2d
        MULPOINT2D = MPoint2d.from_coords(np.random.random((10, 2)))
        POINTS = make_p2d(2+np.random.random((10, 2)), return_type='p2d')
        MULPOINT2D.squared_distance_to_centroid(POINTS, validate_points=True)
        POINTS = 2+np.random.random((10, 2))
        MULPOINT2D.distance_to_centroid(POINTS, validate_points=False,
                                        points_type='numpy')
        """
        return np.sqrt(self.squared_distance_to_centroid(points,
                                                         validate_points=validate_points,
                                                         points_type=points_type))

    def linreg(self):
        pass

    def relax(self):
        pass

    def convex_hull(self):
        pass

    def find_boundary(self, boundary_type='chull'):
        """
        Use convex hull and find boundaries
        """
        pass

    def bbox(self):
        """
        Return bounding box of the mulpoint.

        Parameters
        ----------
        None

        Return
        ------
        bbox: dict of keys 'bl', 'br', 'tr' and 'tl', standing for bottom left,
            bottom right, top right and top left.

        Example-1
        ---------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        MP2D = mp2d.from_coords(np.random.random((10,2)))
        MP2D.bbox()
        """
        x, y = self.x, self.y
        bl, br = [x.min(), y.min()], [x.max(), y.min()]
        tr, tl = [x.max(), y.max()], [x.min(), y.max()]
        return {'bl': bl, 'br': br, 'tr': tr, 'tl': tl}

    def maketree(self, treeType='ckdtree'):
        """
        Use tree structure to deal with a very large system of points.

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((25, 3)))
        mulpoint2d.coords
        from scipy.spatial import cKDTree as ckdt
        a = ckdt(mulpoint2d.coords, copy_data=False, balanced_tree=True)
        a.data
        """
        if treeType in ('ckdtree', 'kdtree'):
            # Scipy ckdtree
            from scipy.spatial import cKDTree as ckdt
            # Make the tree data-structure
            return ckdt(self.coords, copy_data=False, balanced_tree=True)

    def plot(self, points=None, primary_ms=None, secondary_ms=None):
        """
        Scatter plot points and choose to overlay over specifried points.

        Parameters
        ----------
        points: List of secondary points
        primary_ms: marker size to use for primary list of points
        secondary_ms: marker size to use for secondary list of points

        Example
        -------
        from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
        mulpoint2d = mp2d.from_coords(np.random.random((25, 3)))
        MULPOINT2D = mp2d.from_mulpoint2d(mulpoint2d=mulpoint2d,
                                          dxy=[0.0, 0.0],
                                          translate_ref=mulpoint2d.centroid,
                                          rot=[10, 0.0, 0.0],
                                          rot_ref=mulpoint2d.centroid,
                                          degree=True)
        mulpoint2d.plot(MULPOINT2D.coords, primary_ms=50)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # -----------------------------
        # PRIMARY POINT SET
        if primary_ms is None:
            primary_ms = 100
        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                   c='b', marker='o', alpha=0.2, s=primary_ms)
        # -----------------------------
        if points is not None:
            # SECONDARY POINT SET
            if secondary_ms is None:
                secondary_ms = 50
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c='r', marker='x', s=50)
        # -----------------------------
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot')
        plt.show()
