import math
import numpy as np
import numpy.matlib
from copy import deepcopy
# from icecream import ic
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
from upxo.geoEntities.point3d import Point3d
from upxo._sup.validation_values import val_point_and_get_coord, val_points_and_get_coords
from scipy.spatial.distance import pdist

class MPoint3d():
    """
    Standard data formats
    ---------------------
    coords: np.array([[0, 0, 0],
                      [1, 1, 1],
                      [2, 3, 3],
                      [4, 5, 6]])
    """
    __slots__ = ('coords', 'tree', 'pdist')

    def __init__(self, coords=None):
        self.coords = coords
        self.pdist = pdist

    def __repr__(self):
        return f'UPXO-mp3d. n={self.n}.'

    def __iter__(self):
        """
        Return an iterable of point coordsinates in self.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        for coord in mulpoint3d:
            print(coord)
        """
        return iter(self.coords)

    def __getitem__(self, i):
        """
        Make self indexable. i: index location.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d[9]
        mulpoint3d[10]
        """
        if i >= self.n:
            raise ValueError('Index exceeds maximum number of coordinates.')
        return self.coords[i]

    def add(self, toadd=None, operation='add'):
        """
        Add toadd to self.coords.

        Example-1
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d.coords
        mulpoint3d.add(toadd=10, operation='add')
        mulpoint3d.coords

        Example-2
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d.coords
        mulpoint3d.add(toadd=[-10, 20, 0], operation='add')
        mulpoint3d.coords

        Example-3
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d.coords
        mulpoint3d.add(toadd=np.random.random((mulpoint3d.n, 3)), operation='add')
        mulpoint3d.coords

        Example-4
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d.coords
        mulpoint3d.add(toadd=np.random.random((mulpoint3d.n, 3)).T, operation='add')
        mulpoint3d.coords

        Example-5
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((10,3)))
        mulpoint3d.coords
        mulpoint3d.add(toadd=np.random.random((10,3)), operation='append')
        mulpoint3d.coords
        """
        if toadd is None:
            return
        else:
            if operation == 'add':
                if type(toadd) in dth.dt.NUMBERS:
                    self.coords += toadd
                if type(toadd) in dth.dt.ITERABLES:
                    if find_spec_of_points(toadd) == 'type-[1,2,3]':
                        '''
                        toadd = [0, 0, 0]
                        find_spec_of_points(toadd)
                        '''
                        self.coords += np.array(toadd)
                    if find_spec_of_points(toadd) == 'type-[[1,2,3]]':
                        '''
                        toadd = [[0, 0, 0]]
                        find_spec_of_points(toadd)
                        '''
                        self.coords += np.array(toadd[0])
                    if find_spec_of_points(toadd) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
                        '''
                        toadd = [[1,2,3],[4,5,6],[7,8,9],[7,8,9]]
                        find_spec_of_points(toadd)
                        '''
                        if len(toadd) == self.n:
                            self.coords += np.array(toadd)
                        else:
                            raise ValueError('Invalid length of toadd.')
                    if find_spec_of_points(toadd) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
                        '''
                        toadd = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
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
        Instantiate mulpoint3d using list of point coordinates.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]])
        MULPOINT3D = mp3d.from_coords(point_coords)
        MULPOINT3D.coords
        """
        # Validations
        return cls(coords=np.array(point_coords))

    @classmethod
    def from_x_y_z(cls, x, y, z):
        """
        Instantiate mulpoint3d using lists of x, y and z coordinate values.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        x, y, z = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]]).T
        MULPOINT3D = mp3d.from_x_y_z(x, y, z)
        MULPOINT3D.coords
        """
        # Validations
        return cls(coords = np.array([x, y, z]).T)

    @classmethod
    def from_xyz(cls, xyz):
        """
        Instantiate mulpoint3d using array of x, y and z coordinate lists.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]]).T
        MULPOINT3D = mp3d.from_xyz(xyz)
        MULPOINT3D.coords
        """
        # Validations
        return cls(coords = xyz.T)

    @classmethod
    def from_mulpoint2d(cls, mp2d, zloc=0.0):
        pass

    @classmethod
    def from_mulpoint3d(cls,
                        mulpoint3d=None,
                        dxyz=[0.0, 0.0, 0.0],
                        translate_ref=[0.0, 0.0, 0.0],
                        rot=[0.0, 0.0, 0.0],
                        rot_ref=[0.0, 0.0, 0.0],
                        degree=True
                        ):
        """
        Instantiate mulpoint3d by operating on another mulpoint3d.

        Note
        ----
        Use is detailed. Please refer to examples to know behaviour.

        Parametrs
        ---------
        mulpoint3d: UPXO multi-point 3D object
        dxyz: Translations to apply along x, y and z axes.
        translate_ref: Reference point for translation operation.
        rot: Rotations to apply about x, y and z axes (CCW +ve abovt +ve axes).
        rot_ref: Reference point for rotation operation.
        degree: If True, rot will be considered in degrees, else in radians.

        Example-1
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[0.0, 0.0, 0.0],
                                          rot_ref=[0.0, 0.0, 0.0],
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-2
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[45, 0.0, 0.0],
                                          rot_ref=[0.0, 0.0, 0.0],
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-3
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=[0.0, 0.0, 0.0],
                                          rot=[45, 0.0, 0.0],
                                          rot_ref=[2.0, 0.0, 0.0],
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-4
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[45, 0.0, 0.0],
                                          rot_ref=[2.0, 0.0, 0.0],
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-5
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[45, 0.0, 0.0],
                                          rot_ref=mulpoint3d.centroid,
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-6
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[1.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[45, 0.0, 0.0],
                                          rot_ref=mulpoint3d.centroid,
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)

        Example-7
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        point_coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]])
        mulpoint3d = mp3d.from_coords(point_coords)
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[1.0, 1.0, -0.5],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[0, 0.0, 0.0],
                                          rot_ref=mulpoint3d.centroid,
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords)
        """
        # Validations
        # ------------------------------------
        if degree:
            rot = np.radians(rot)
        # ------------------------------------
        # Apply the rotation operation.
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rot[0]), -np.sin(rot[0])],
                       [0, np.sin(rot[0]), np.cos(rot[0])]])
        Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])],
                       [0, 1, 0],
                       [-np.sin(rot[1]), 0, np.cos(rot[1])]])
        Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0],
                       [np.sin(rot[2]), np.cos(rot[2]), 0],
                       [0, 0, 1]])
        # Make the totalk rotation matrix
        R = np.dot(Rz, np.dot(Ry, Rx))
        # Translate points to the origin
        translated_points = mulpoint3d.coords - rot_ref
        # Apply rotation
        rotated_points = np.dot(translated_points, R.T)
        # Translate points back to original position
        rotated_points += rot_ref
        # ------------------------------------
        # Offset to new required position by translation aloing x, y and z
        coords = rotated_points - (mulpoint3d.centroid - translate_ref) + dxyz
        # ------------------------------------
        return cls(coords=coords)

    @classmethod
    def from_mulsline3d(cls, msline3d):
        pass

    @classmethod
    def from_xyz_grid(cls,
                      xspec=[0, 1, 0.25],
                      yspec=[0, 1, 0.25],
                      zspec=[0, 1, 0.25],
                      dxyz=[0.0, 0.0, 0.0],
                      translate_ref=[0.0, 0.0, 0.0],
                      rot=[0.0, 0.0, 0.0],
                      rot_ref=[0.0, 0.0, 0.0],
                      degree=True
                      ):
        """
        Instantiate mulpoint3d for a regular x, y, z grid.

        Parameters
        ----------
        xspec: [xstart, xend, xincrement]
        zspec: [ystart, yend, yincrement]
        xspec: [zstart, zend, zincrement
        dxyz: Translations to apply along x, y and z axes.
        translate_ref: Reference point for translation operation.
        rot: Rotations to apply about x, y and z axes (CCW +ve abovt +ve axes).
        rot_ref: Reference point for rotation operation.
        degree: If True, rot will be considered in degrees, else in radians.

        Example-1
        ---------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        xspec, yspec, zspec = [0, 1, 0.1], [0, 1, 0.1], [0, 1, 0.1]
        dxyz, translate_ref = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        mulpoint3d = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                                        dxyz=dxyz, translate_ref=translate_ref,
                                        rot=[0.0, 0.0, 0.0],
                                        rot_ref=[0.0, 0.0, 0.0],
                                        degree=True)
        MULPOINT3D = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                                        dxyz=dxyz, translate_ref=translate_ref,
                                        rot=[5.0, 5.0, 5.0],
                                        rot_ref=[0.0, 0.0, 0.0],
                                        degree=True)
        MULPOINT3D.plot(mulpoint3d.coords, primary_ms=50, secondary_ms=5)
        """
        # Validations
        # --------------------------
        X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1]+xspec[2], xspec[2]),
                              np.arange(yspec[0], yspec[1]+yspec[2], yspec[2]),
                              np.arange(zspec[0], zspec[1]+zspec[2], zspec[2]))
        coords = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        # --------------------------
        mulpoint3d = MPoint3d.from_coords(coords)
        # --------------------------
        if isinstance(translate_ref, str):
            if translate_ref == 'centroid':
                translate_ref = mulpoint3d.centroid
            else:
                raise ValueError('Invalid translate_ref specification.')
        elif type(translate_ref) in dth.dt.ITERABLES:
            # Do nothing, as user input is just the coordinate values.
            pass
        else:
            raise ValueError('Invalid translate_ref specification.')
        # --------------------------
        return MPoint3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                        dxyz=dxyz,
                                        translate_ref=translate_ref,
                                        rot=rot,
                                        rot_ref=rot_ref,
                                        degree=True)

    @property
    def n(self):
        return len(self.coords)

    @property
    def centroid(self):
        return np.mean(self.coords, axis=0)

    @property
    def points(self):
        return [Point3d(x, y, z) for x, y, z in zip(self.x, self.y, self.z)]

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]

    @property
    def ckd_tree(self):
        return self.maketree(treeType='ckdtree')

    def squared_distances_to_point(self, point):
        point = val_point_and_get_coord(point, return_type='coord',
                                        safe_exit=False)
        return (self.x-point[0])**2 + (self.y-point[1])**2 + (self.z-point[2])**2

    def distances_to_point(self, point):
        return np.sqrt(self.squared_distances_to_point(point))

    def squared_distance_to_centroid(self, points,
                                     validate_points=True,
                                     points_type='numpy'):
        """
        Calculates squared distances between self.centroid and other 3D points.

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
        from upxo.geoEntities.mulpoint3d import MPoint3d
        from upxo.geoEntities.point3d import Point3d
        MULPOINT3D = MPoint3d.from_coords(np.random.random((10, 3)))
        POINTS = make_p3d(2+np.random.random((10, 3)), return_type='p3d')
        MULPOINT3D.squared_distance_to_centroid(POINTS, validate_points=True)

        POINTS = 2+np.random.random((10, 3))
        MULPOINT3D.squared_distance_to_centroid(POINTS, validate_points=False,
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
        return (pnts[:, 0]-cen[0])**2 + (pnts[:, 1]-cen[1])**2 + (pnts[:, 2]-cen[2])**2

    def distance_to_centroid(self, points, validate_points=True,
                             points_type='numpy'):
        """
        Calculates squared distances between self.centroid and other 3D points.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d
        from upxo.geoEntities.point3d import Point3d
        MULPOINT3D = MPoint3d.from_coords(np.random.random((10, 3)))
        POINTS = make_p3d(2+np.random.random((10, 3)), return_type='p3d')
        MULPOINT3D.squared_distance_to_centroid(POINTS, validate_points=True)
        POINTS = 2+np.random.random((10, 3))
        MULPOINT3D.distance_to_centroid(POINTS, validate_points=False,
                                        points_type='numpy')
        """
        return np.sqrt(self.squared_distance_to_centroid(points,
                                                         validate_points=validate_points,
                                                         points_type=points_type))

    def convex_hull(self):
        pass

    def maketree(self, treeType='ckdtree', saa=False,
                 throw=False, balance=True):
        """
        Use tree structure to deal with a very large system of points.

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((25, 3)))
        mulpoint3d.coords
        from scipy.spatial import cKDTree as ckdt
        a = ckdt(mulpoint3d.coords, copy_data=False, balanced_tree=True)
        a.data
        """
        if treeType not in ('ckdtree', 'kdtree'):
            return None

        # Scipy ckdtree
        from scipy.spatial import cKDTree as ckdt
        # Make the tree data-structure
        tree = ckdt(self.coords, copy_data=False, balanced_tree=balance)
        if saa:
            self.tree = tree
        if throw:
            return tree

    def get_self_distance_max(self):
        '''
        Return the maximum distance possible in self.coords
        '''
        return self.pdist(self.coords).max()

    def get_self_distance_min(self):
        '''
        Return the minimum distance possible in self.coords
        '''
        return self.pdist(self.coords).min()

    def find_first_order_neigh_CUBIC(self, coord, vox_size,
                                     return_indices=True,
                                     return_coords=True,
                                     return_input_coord=False,
                                     k=1.000001):
        '''
        let variable COORDS is a list of voxel coordinates in the form of a
        numopy array [x1, y1, z1], so on. Variable coord is a member of COORDS.
        This definition finds the first order nearest neighbours. I define the
        first order nearest neighbours as all the possible coords in COORDS
        which can be placed in a 3x3 voxel arrangement centred at coord. The
        name suffix CUBIC is to indicate this definition is designed to
        get the neighbours in a CUBIC type lattice.

        Note 1:
            CUBIC lattice: 3D version of 2D square lattice.

        Development
        -----------
        A voxel [x,y,z] is a first-order neighbor of coord [cx,cy,cz] if:
            |x−cx| <= A,  |y−cy| <= B,  |z−cz| <= C.
            Where, A, B and C are voxel sizes alonmg x, y, and z axes
            respectively. In this definition, A = B = C = vox_size
        This means the neighbor lies within a 3x3x3 grid centered at coord.

        vox_size = 2
        X, Y, Z = np.meshgrid(np.arange(0, 10, vox_size),
                              np.arange(0, 10, vox_size),
                              np.arange(0, 10, vox_size))
        COORDS = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        coord = np.array([4, 4, 4])

        diffs = np.abs(COORDS - coord)
        coords = COORDS[np.argwhere(np.prod(diffs <= vox_size, axis=1)).T]

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        vs = 0.1  # Voxel size
        xspec, yspec, zspec = [0, 1, vs], [0, 1, vs], [0, 1, vs]
        X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1], xspec[2]),
                              np.arange(yspec[0], yspec[1], yspec[2]),
                              np.arange(zspec[0], zspec[1], zspec[2]))
        mp = mp3d.from_coords(np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T)
        mp.find_first_order_neigh_CUBIC((0.5, 0.5, 0.5), vs)
        '''
        coord = np.array(coord)
        diffs = np.abs(self.coords - coord)
        coords_indices = np.argwhere(np.prod(diffs <= vox_size*k, axis=1)).T
        coords = self.coords[coords_indices]

        if return_indices and not return_coords:
            if not return_input_coord:
                return coords_indices
            else:
                return coords_indices, coord

        if not return_indices:
            if not return_input_coord:
                return coords
            else:
                return coords, coord

        if return_indices and return_coords:
            if not return_input_coord:
                return coords_indices, coords, coord
            else:
                return coords_indices, coords

    def check_if_point_can_host_a_single_surface_CUBIC(self, coord, vs):
        """
        Check if coord in self.coords can have a single surface through it.

        Given that there will be 27 voxels in the 3x3x3 arramgement, where
        some neighbours are in "ON" state and while rest are in "OFF" state,
        the conditiion for a single syurface to pass through all the ON state
        points and the central point is that the number of ON state members
        should be a maximum of 5. This is purely empirical value, derived by
        some common sense and some spatial thinking.

        The name suffix CUBIC is to indicate this definition is designed to
        work reliably on CUBIC type lattice only.

        coord: coordinates of the concerned point
        vs: Voxel size

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        vs = 0.1  # Voxel size
        xspec, yspec, zspec = [0, 1, vs], [0, 1, vs], [0, 1, vs]
        X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1], xspec[2]),
                              np.arange(yspec[0], yspec[1], yspec[2]),
                              np.arange(zspec[0], zspec[1], zspec[2]))
        mp = mp3d.from_coords(np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T)
        coords = mp.find_first_order_neigh_CUBIC((0.5, 0.5, 0.5),
                                                 vs,
                                                 return_indices=False,
                                                 return_coords=True,
                                                 return_input_coord=False)[0]
        coord = np.array([0.5, 0.5, 0.5])

        coord_loc = np.argwhere(np.all(coords == coord, axis=1)).squeeze()
        rand_4_locs = np.sort(np.random.choice(range(coords.shape[0]), 4, replace=False))
        points_5_locs = np.unique(np.hstack((coord_loc, rand_4_locs)))

        coords_ON_state = coords[points_5_locs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c='c', marker='o', alpha=0.1,
                   s=200, edgecolors='black')
        ax.scatter(coords_ON_state[:, 0], coords_ON_state[:, 1], coords_ON_state[:, 2],
                   c='b', marker='o', alpha=0.8,
                   s=50, edgecolors='black')
        """
        # Validations
        # -----------------------------------
        coord = np.array(coord)
        '''Find the coords in self.coords which are first order neigh.'''
        coords = self.find_first_order_neigh_CUBIC(coord, vs,
                                                   return_indices=False,
                                                   return_coords=True,
                                                   return_input_coord=False,
                                                   k=1.000001)
        '''Check if coord is in coords. If not, return None'''
        coord_in_coords = np.argwhere(np.all(coords[0] == coord, axis=1)).squeeze()
        if coord_in_coords.size == 0:
            print('coord is not in self.coords !!')
            return None
        '''Extract the coords other than coord'''  # Note: a1
        coords_ = self.coords[~np.all(self.coords == coord, axis=1)]
        '''Assess capability to form single non-intersecting surface.'''
        npnt = coords_.shape[0]
        if npnt in (2, 3, 4):
            '''
            Here, 2 is the lower limit, because, along with these 2, coord
            can form a plane and hence can have a normal defined at it.
            Here, upper limit of 5 actually becomes 4, as the user input
            coord has been removed from computation (see note a1).
            '''
            return True # A surface can be formed
        else:
            # A surface cannot be formed
            return False

    def get_local_tn(self, coord, k=5):
        '''
        For the input coordinate, find the local tangent plane and normal
        vector. For this, the input coord must be a member of self.coords.
        '''
        # 1. Find the minimum possible distance
        d0 = self.get_self_distance_min()

    def find_intersection_voxels_with_line(self, sl3d, cod):
        '''
        Find all the voxels which intersect with the given upxo line 3d
        within a distance specified by cod (cut off distance).
        '''
        pass

    def find_intersection_voxels_with_plane(self, plane, cod):
        '''
        Find all the voxels which intersect with the given upxo plane 3d
        within a distance specified by cod (cut off distance).
        '''
        pass

    def plot(self,
             points=None,
             primary_ms=None, primary_alpha=0.2,
             secondary_ms=None, secondary_alpha=None,
             xbound=None, ybound=None, zbound=None):
        """
        Scatter plot points and choose to overlay over specifried points.

        Parameters
        ----------
        points: List of secondary points
        primary_ms: marker size to use for primary list of points
        secondary_ms: marker size to use for secondary list of points

        Example
        -------
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        mulpoint3d = mp3d.from_coords(np.random.random((25, 3)))
        MULPOINT3D = mp3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                          dxyz=[0.0, 0.0, 0.0],
                                          translate_ref=mulpoint3d.centroid,
                                          rot=[10, 0.0, 0.0],
                                          rot_ref=mulpoint3d.centroid,
                                          degree=True)
        mulpoint3d.plot(MULPOINT3D.coords, primary_ms=50)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # -----------------------------
        # PRIMARY POINT SET
        if primary_ms is None:
            primary_ms = 100
        # -----------------------------
        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                   c='b', marker='o', alpha=primary_alpha, s=primary_ms,
                   edgecolors='black')
        # -----------------------------
        if points is not None:
            # SECONDARY POINT SET
            if secondary_ms is None:
                secondary_ms = 50
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c='r', marker='s', s=50, edgecolors='black',
                       alpha=secondary_alpha)
        # -----------------------------
        if (zbound is not None) and (ybound is not None) and (zbound is not None):
            # Need stronger validations for if conditional !!
            vertices = np.array([[xbound[0], ybound[0], zbound[0]],  # 0
                                 [xbound[1], ybound[0], zbound[0]],  # 1
                                 [xbound[1], ybound[1], zbound[0]],  # 2
                                 [xbound[0], ybound[1], zbound[0]],  # 3
                                 [xbound[0], ybound[0], zbound[1]],  # 4
                                 [xbound[1], ybound[0], zbound[1]],  # 5
                                 [xbound[1], ybound[1], zbound[1]],  # 6
                                 [xbound[0], ybound[1], zbound[1]]])  # 7
            # Define the edges of the cuboid
            edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            for edge in edges:
                ax.plot(*zip(*vertices[edge]), color='k', linewidth=2.5)
        # -----------------------------
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
