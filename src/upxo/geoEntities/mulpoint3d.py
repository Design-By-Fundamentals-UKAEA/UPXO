"""
Multi-point 3D geometric entity module for UPXO.

Provides ``MPoint3d``, a collection class for N 3-D points stored as a
single ``(N, 3)`` NumPy array.  Supports construction from coordinate arrays,
separated x/y/z lists, regular grids, and other UPXO point collections;
rigid-body operations (translation, rotation); spatial queries (kd-tree,
nearest neighbours, distance computations); and surface-topology checks for
voxel-based meshes.

Classes
-------
MPoint3d
    Collection of 3-D points backed by an ``(N, 3)`` NumPy array.

Usage
-----
::

    from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d

@author: Dr. Sunil Anandatheertha
"""
import math
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree
import vtk
from shapely.geometry import Point as ShPnt, Polygon as ShPol
from shapely.geometry import LineString
from functools import wraps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import upxo._sup.dataTypeHandlers as dth
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
np.seterr(divide='ignore')
from upxo.geoEntities.featmake import make_p2d, make_p3d
from upxo._sup.validation_values import find_spec_of_points
from upxo._sup.validation_values import isinstance_many
import upxo.geoEntities.featmake as fmake
from upxo.geoEntities.point3d import Point3d
from upxo._sup.validation_values import val_point_and_get_coord, val_points_and_get_coords
from scipy.spatial.distance import pdist


class MPoint3d():
    """Collection of N 3-D points stored as a single ``(N, 3)`` NumPy array.

    Provides construction class-methods, rigid-body operations (translation,
    rotation), spatial-query helpers (kd-tree, neighbour search, distance
    calculations), and surface-topology checks for voxel-based meshes.

    Attributes
    ----------
    coords : numpy.ndarray, shape (N, 3)
        Row-major array of 3-D coordinates: each row is ``[x, y, z]``.
    tree : scipy.spatial.cKDTree or None
        Spatial index, populated on demand by :meth:`maketree`.
    pdist : callable
        Reference to ``scipy.spatial.distance.pdist`` for pairwise distances.

    Standard coordinate format
    --------------------------
    ::

        coords = np.array([[0, 0, 0],
                           [1, 1, 1],
                           [2, 3, 3],
                           [4, 5, 6]])
    """
    __slots__ = ('coords', 'tree', 'pdist')

    def __init__(self, coords=None):
        """Initialise from an ``(N, 3)`` numpy array of 3D coordinates."""
        self.coords = coords
        self.pdist = pdist

    def __repr__(self):
        """Return ``UPXO-mp3d. n=<N>.`` summary string."""
        return f'UPXO-mp3d. n={self.n}.'

    def __iter__(self):
        """Iterate over the point coordinates in ``self.coords``.

        Yields
        ------
        numpy.ndarray, shape (3,)
            One ``[x, y, z]`` coordinate row per iteration.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            for coord in mulpoint3d:
                print(coord)
        """
        return iter(self.coords)

    def __getitem__(self, i):
        """Return the point at index ``i`` as a ``(3,)`` coordinate array.

        Parameters
        ----------
        i : int
            Zero-based index into ``self.coords``.  Must be less than ``self.n``.

        Returns
        -------
        numpy.ndarray, shape (3,)
            The ``[x, y, z]`` coordinate at position ``i``.

        Raises
        ------
        ValueError
            If ``i >= self.n``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            print(mulpoint3d[9])   # last element
        """
        if i >= self.n:
            raise ValueError('Index exceeds maximum number of coordinates.')
        return self.coords[i]

    def add(self, toadd=None, operation='add'):
        """Add to or append coordinates in ``self.coords``.

        Parameters
        ----------
        toadd : scalar, list, or numpy.ndarray, optional
            Value(s) to add or append.  Accepted shapes / types:

            * scalar number — broadcast-added to every coordinate.
            * ``[x, y, z]`` — added to every row as a 3-element offset.
            * ``[[x, y, z]]`` — same as above, single-row list.
            * ``(N, 3)`` array — element-wise add; must match ``self.n``.
            * ``(3, N)`` array (transposed) — transposed before adding.

            When ``operation='append'``, the same shapes are supported but
            the rows are appended instead of added.
        operation : {'add', 'append'}, optional
            ``'add'`` modifies coordinates in place; ``'append'`` grows
            ``self.coords`` by the supplied rows.  Default is ``'add'``.

        Returns
        -------
        None
            Modifies ``self.coords`` in place.

        Examples
        --------
        **Example 1** — scalar broadcast addition:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            mulpoint3d.add(toadd=10, operation='add')

        **Example 2** — 3-element offset vector:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            mulpoint3d.add(toadd=[-10, 20, 0], operation='add')

        **Example 3** — element-wise ``(N, 3)`` array addition:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            mulpoint3d.add(toadd=np.random.random((mulpoint3d.n, 3)), operation='add')

        **Example 4** — transposed ``(3, N)`` array addition:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            mulpoint3d.add(toadd=np.random.random((mulpoint3d.n, 3)).T, operation='add')

        **Example 5** — append rows:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((10, 3)))
            mulpoint3d.add(toadd=np.random.random((10, 3)), operation='append')
        """
        if toadd is None:
            return
        else:
            if operation == 'add':
                if type(toadd) in dth.dt.NUMBERS:
                    self.coords += toadd
                if type(toadd) in dth.dt.ITERABLES:
                    if find_spec_of_points(toadd) == 'type-[1,2,3]':
                        self.coords += np.array(toadd)
                    if find_spec_of_points(toadd) == 'type-[[1,2,3]]':
                        self.coords += np.array(toadd[0])
                    if find_spec_of_points(toadd) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
                        if len(toadd) == self.n:
                            self.coords += np.array(toadd)
                        else:
                            raise ValueError('Invalid length of toadd.')
                    if find_spec_of_points(toadd) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
                        if len(toadd[0]) == self.n:
                            self.coords += np.array(toadd).T
                        else:
                            raise ValueError('Invalid length of toadd.')
            elif operation == 'append':
                if type(toadd) in dth.dt.ITERABLES:
                    if find_spec_of_points(toadd) == 'type-[1,2,3]':
                        self.coords = np.array(list(self.coords) + list(toadd))
                    if find_spec_of_points(toadd) == 'type-[[1,2,3]]':
                        self.coords = np.array(list(self.coords) + list(toadd[0]))
                    if find_spec_of_points(toadd) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
                        toadd = [list(ta) for ta in toadd]
                        coords = [list(coord) for coord in self.coords]
                        self.coords = np.array(coords+toadd)
                    if find_spec_of_points(toadd) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
                        toadd = [list(ta) for ta in np.array(toadd).T]
                        coords = [list(coord) for coord in self.coords]
                        self.coords += np.array(toadd).T

    @classmethod
    def from_coords(cls, point_coords):
        """Instantiate from an ``(N, 3)`` array or list of coordinate triples.

        Parameters
        ----------
        point_coords : array-like, shape (N, 3)
            Each row is a 3-D coordinate ``[x, y, z]``.

        Returns
        -------
        MPoint3d
            New instance with ``coords`` set from ``point_coords``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            point_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]])
            MULPOINT3D = mp3d.from_coords(point_coords)
            print(MULPOINT3D.coords)
        """
        return cls(coords=np.array(point_coords))

    @classmethod
    def from_x_y_z(cls, x, y, z):
        """Instantiate from separate x, y, and z coordinate arrays.

        Parameters
        ----------
        x, y, z : array-like, shape (N,)
            Coordinate components of the N points.

        Returns
        -------
        MPoint3d
            New instance with ``coords`` of shape (N, 3).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            x, y, z = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]]).T
            MULPOINT3D = mp3d.from_x_y_z(x, y, z)
            print(MULPOINT3D.coords)
        """
        return cls(coords=np.array([x, y, z]).T)

    @classmethod
    def from_xyz(cls, xyz):
        """Instantiate from a ``(3, N)`` coordinate matrix.

        Parameters
        ----------
        xyz : numpy.ndarray, shape (3, N)
            Row 0 is x-coords, row 1 is y-coords, row 2 is z-coords.

        Returns
        -------
        MPoint3d
            New instance with ``coords`` of shape (N, 3) (transposed from ``xyz``).

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 3], [4, 5, 6]]).T
            MULPOINT3D = mp3d.from_xyz(xyz)
            print(MULPOINT3D.coords)
        """
        return cls(coords=xyz.T)

    @classmethod
    def from_mulpoint2d(cls, mp2d, zloc=0.0):
        """Construct from a ``MPoint2d`` by appending a constant z-value. Not yet implemented."""
        raise NotImplementedError("from_mulpoint2d is not yet implemented.")

    @classmethod
    def from_mulpoint3d(cls,
                        mulpoint3d=None,
                        dxyz=[0.0, 0.0, 0.0],
                        translate_ref=[0.0, 0.0, 0.0],
                        rot=[0.0, 0.0, 0.0],
                        rot_ref=[0.0, 0.0, 0.0],
                        degree=True
                        ):
        """Instantiate by applying rotation and translation to an existing ``MPoint3d``.

        Parameters
        ----------
        mulpoint3d : MPoint3d
            Source point collection to transform.
        dxyz : list of float, optional
            Translation offsets ``[dx, dy, dz]`` applied after rotation.
            Default is ``[0.0, 0.0, 0.0]``.
        translate_ref : list of float, optional
            Reference point for the translation step; the cloud is shifted so
            that ``translate_ref`` maps to the origin before rotation.
            Default is ``[0.0, 0.0, 0.0]``.
        rot : list of float, optional
            Rotation angles ``[rx, ry, rz]`` about the x, y, and z axes (CCW
            positive about positive axes).  Default is ``[0.0, 0.0, 0.0]``.
        rot_ref : list of float, optional
            Centre of rotation in 3-D space.  Default is ``[0.0, 0.0, 0.0]``.
        degree : bool, optional
            If ``True``, ``rot`` values are interpreted as degrees; if
            ``False``, as radians.  Default is ``True``.

        Returns
        -------
        MPoint3d
            New instance with transformed coordinates.

        Notes
        -----
        Rotation is applied as successive Rx → Ry → Rz matrix multiplication
        about ``rot_ref``.  Translation is applied last by centering on
        ``translate_ref`` and adding ``dxyz``.  Refer to the examples for a
        concrete demonstration of each degree of freedom.

        Examples
        --------
        **Example 1** — no rotation, no translation (identity):

        .. code-block:: python

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

        **Example 2** — 45° rotation about x-axis, centred at origin:

        .. code-block:: python

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

        **Example 3** — 45° rotation about x-axis, non-origin rotation centre:

        .. code-block:: python

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

        **Example 4** — rotation about x with centroid as both translate_ref and rot_ref:

        .. code-block:: python

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

        **Example 5** — rotation with rot_ref at centroid:

        .. code-block:: python

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

        **Example 6** — combined rotation and x-translation:

        .. code-block:: python

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

        **Example 7** — 3-axis translation, no rotation:

        .. code-block:: python

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
        if degree:
            rot = np.radians(rot)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rot[0]), -np.sin(rot[0])],
                       [0, np.sin(rot[0]), np.cos(rot[0])]])
        Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])],
                       [0, 1, 0],
                       [-np.sin(rot[1]), 0, np.cos(rot[1])]])
        Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0],
                       [np.sin(rot[2]), np.cos(rot[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        translated_points = mulpoint3d.coords - rot_ref
        rotated_points = np.dot(translated_points, R.T)
        rotated_points += rot_ref
        coords = rotated_points - (mulpoint3d.centroid - translate_ref) + dxyz
        return cls(coords=coords)

    @classmethod
    def from_mulsline3d(cls, msline3d):
        """Construct from a ``MSline3d`` endpoint collection. Not yet implemented."""
        raise NotImplementedError("from_mulsline3d is not yet implemented.")

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
        """Instantiate from a regular 3-D Cartesian grid with optional rigid-body transform.

        Builds a meshgrid from the three axis specifications, flattens it to an
        ``(N, 3)`` array, then delegates to :meth:`from_mulpoint3d` to apply
        the requested rotation and translation.

        Parameters
        ----------
        xspec : list of float, optional
            ``[xstart, xend, xincrement]`` for the x-axis grid.
            Default is ``[0, 1, 0.25]``.
        yspec : list of float, optional
            ``[ystart, yend, yincrement]`` for the y-axis grid.
            Default is ``[0, 1, 0.25]``.
        zspec : list of float, optional
            ``[zstart, zend, zincrement]`` for the z-axis grid.
            Default is ``[0, 1, 0.25]``.
        dxyz : list of float, optional
            Translation offsets ``[dx, dy, dz]``.  Default is
            ``[0.0, 0.0, 0.0]``.
        translate_ref : list of float or str, optional
            Reference point for translation.  Pass ``'centroid'`` to use the
            grid centroid, or a ``[x, y, z]`` coordinate list.
            Default is ``[0.0, 0.0, 0.0]``.
        rot : list of float, optional
            Rotation angles ``[rx, ry, rz]`` about x, y, z axes (CCW positive).
            Default is ``[0.0, 0.0, 0.0]``.
        rot_ref : list of float, optional
            Centre of rotation.  Default is ``[0.0, 0.0, 0.0]``.
        degree : bool, optional
            If ``True``, ``rot`` is in degrees; if ``False``, in radians.
            Default is ``True``.

        Returns
        -------
        MPoint3d
            New instance containing the grid points after the rigid-body
            transform.

        Examples
        --------
        **Example 1** — two grids, one base and one rotated by (5°, 5°, 5°):

        .. code-block:: python

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
        X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1]+xspec[2], xspec[2]),
                              np.arange(yspec[0], yspec[1]+yspec[2], yspec[2]),
                              np.arange(zspec[0], zspec[1]+zspec[2], zspec[2]))
        coords = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        mulpoint3d = MPoint3d.from_coords(coords)
        if isinstance(translate_ref, str):
            if translate_ref == 'centroid':
                translate_ref = mulpoint3d.centroid
            else:
                raise ValueError('Invalid translate_ref specification.')
        elif type(translate_ref) in dth.dt.ITERABLES:
            pass
        else:
            raise ValueError('Invalid translate_ref specification.')
        return MPoint3d.from_mulpoint3d(mulpoint3d=mulpoint3d,
                                        dxyz=dxyz,
                                        translate_ref=translate_ref,
                                        rot=rot,
                                        rot_ref=rot_ref,
                                        degree=True)

    @property
    def n(self):
        """Number of points in the collection."""
        return len(self.coords)

    @property
    def centroid(self):
        """Mean 3D coordinate of all points as a ``(3,)`` array."""
        return np.mean(self.coords, axis=0)

    @property
    def points(self):
        """Return a list of ``Point3d`` objects built from ``self.coords``."""
        return [Point3d(x, y, z) for x, y, z in zip(self.x, self.y, self.z)]

    @property
    def x(self):
        """x-coordinates of all points as a 1-D array."""
        return self.coords[:, 0]

    @property
    def y(self):
        """y-coordinates of all points as a 1-D array."""
        return self.coords[:, 1]

    @property
    def z(self):
        """z-coordinates of all points as a 1-D array."""
        return self.coords[:, 2]

    @property
    def ckd_tree(self):
        """Build and return a ``cKDTree`` for fast nearest-neighbour queries."""
        return self.maketree(treeType='ckdtree')

    def squared_distances_to_point(self, point):
        """Return squared Euclidean distances from all points to ``point``.

        Parameters
        ----------
        point : Point3d or array-like
            Target point.  Validated via ``val_point_and_get_coord``.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Squared distance from each point in ``self.coords`` to ``point``.
        """
        point = val_point_and_get_coord(point, return_type='coord',
                                        safe_exit=False)
        return (self.x-point[0])**2 + (self.y-point[1])**2 + (self.z-point[2])**2

    def distances_to_point(self, point):
        """Return Euclidean distances from all points to ``point``.

        Parameters
        ----------
        point : Point3d or array-like
            Target point.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Euclidean distance from each point in ``self.coords`` to ``point``.
        """
        return np.sqrt(self.squared_distances_to_point(point))

    def squared_distance_to_centroid(self, points,
                                     validate_points=True,
                                     points_type='numpy'):
        """Compute squared distances from ``self.centroid`` to a set of 3-D points.

        Parameters
        ----------
        points : list of Point3d or numpy.ndarray, shape (M, 3)
            Target points to measure from ``self.centroid``.
        validate_points : bool, optional
            When ``True`` the input is validated and converted automatically.
            When confident that ``points`` is an ``(M, 3)`` NumPy array, set
            to ``False`` to skip validation overhead.  Default is ``True``.
        points_type : {'numpy', 'upxo', 'shapely', 'coord', 'coord_pair'}, optional
            Type hint used only when ``validate_points=False``.  Use
            ``'numpy'`` for plain NumPy arrays.  Default is ``'numpy'``.

        Returns
        -------
        numpy.ndarray, shape (M,)
            Squared Euclidean distances from each target point to
            ``self.centroid``.

        Examples
        --------
        **Example 1** — validated UPXO point objects:

        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d
            MULPOINT3D = MPoint3d.from_coords(np.random.random((10, 3)))
            POINTS = make_p3d(2 + np.random.random((10, 3)), return_type='p3d')
            MULPOINT3D.squared_distance_to_centroid(POINTS, validate_points=True)

        **Example 2** — raw NumPy array, validation skipped:

        .. code-block:: python

            POINTS = 2 + np.random.random((10, 3))
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
        """Compute Euclidean distances from ``self.centroid`` to a set of 3-D points.

        Parameters
        ----------
        points : list of Point3d or numpy.ndarray, shape (M, 3)
            Target points.
        validate_points : bool, optional
            See :meth:`squared_distance_to_centroid`.  Default is ``True``.
        points_type : str, optional
            See :meth:`squared_distance_to_centroid`.  Default is ``'numpy'``.

        Returns
        -------
        numpy.ndarray, shape (M,)
            Euclidean distances from each target point to ``self.centroid``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d
            MULPOINT3D = MPoint3d.from_coords(np.random.random((10, 3)))
            POINTS = 2 + np.random.random((10, 3))
            MULPOINT3D.distance_to_centroid(POINTS, validate_points=False,
                                            points_type='numpy')
        """
        return np.sqrt(self.squared_distance_to_centroid(points,
                                                         validate_points=validate_points,
                                                         points_type=points_type))

    def convex_hull(self):
        """Compute the convex hull of the 3D point set. Not yet implemented."""
        raise NotImplementedError("convex_hull is not yet implemented.")

    def maketree(self, treeType='ckdtree', saa=False,
                 throw=False, balance=True):
        """Build a spatial index tree over ``self.coords``.

        Parameters
        ----------
        treeType : {'ckdtree', 'kdtree'}, optional
            Type of spatial index.  Currently only ``'ckdtree'`` is
            implemented.  Default is ``'ckdtree'``.
        saa : bool, optional
            If ``True``, store the built tree on ``self.tree``.
            Default is ``False``.
        throw : bool, optional
            If ``True``, return the tree object.  Default is ``False``.
        balance : bool, optional
            Passed as ``balanced_tree`` to ``cKDTree``.  Default is ``True``.

        Returns
        -------
        scipy.spatial.cKDTree or None
            The built tree when ``throw=True``; otherwise ``None``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            mulpoint3d = mp3d.from_coords(np.random.random((25, 3)))
            tree = mulpoint3d.maketree(treeType='ckdtree', throw=True)
            print(tree.data.shape)
        """
        if treeType not in ('ckdtree', 'kdtree'):
            return None
        from scipy.spatial import cKDTree as ckdt
        tree = ckdt(self.coords, copy_data=False, balanced_tree=balance)
        if saa:
            self.tree = tree
        if throw:
            return tree

    def get_self_distance_max(self):
        """Return the maximum pairwise distance among all points in ``self.coords``."""
        return self.pdist(self.coords).max()

    def get_self_distance_min(self):
        """Return the minimum pairwise distance among all points in ``self.coords``."""
        return self.pdist(self.coords).min()

    def find_first_order_neigh_CUBIC(self, coord, vox_size,
                                     return_indices=True,
                                     return_coords=True,
                                     return_input_coord=False,
                                     k=1.000001):
        """Find first-order (face+edge+vertex) neighbours of a voxel on a cubic lattice.

        A point ``p`` in ``self.coords`` is a first-order neighbour of
        ``coord`` if ``|p[d] - coord[d]| <= vox_size`` for all three
        dimensions d (i.e. it fits within a 3×3×3 cubic stencil centred at
        ``coord``).  The tolerance multiplier ``k`` avoids floating-point
        boundary misclassification.

        Parameters
        ----------
        coord : array-like, shape (3,)
            Centre voxel coordinate.  Must be a member of ``self.coords``.
        vox_size : float
            Voxel edge length; defines the stencil half-width.
        return_indices : bool, optional
            Include neighbour indices into ``self.coords`` in the output.
            Default is ``True``.
        return_coords : bool, optional
            Include neighbour coordinate arrays in the output.
            Default is ``True``.
        return_input_coord : bool, optional
            Append ``coord`` to the return tuple.  Default is ``False``.
        k : float, optional
            Tolerance multiplier applied to ``vox_size`` to avoid floating-point
            boundary misses.  Default is ``1.000001``.

        Returns
        -------
        tuple
            Contents depend on the flag combination:

            * ``(return_indices=True, return_coords=False)`` →
              ``(indices,)`` or ``(indices, coord)``
            * ``(return_indices=False, return_coords=True)`` →
              ``(coords,)`` or ``(coords, coord)``
            * ``(return_indices=True, return_coords=True)`` →
              ``(indices, coords, coord)`` or ``(indices, coords)``

        Notes
        -----
        Designed for cubic lattices only.  A voxel ``[x, y, z]`` is a
        first-order neighbour of ``[cx, cy, cz]`` when
        ``|x-cx| <= A``, ``|y-cy| <= B``, ``|z-cz| <= C``
        where A = B = C = ``vox_size``.  This includes up to 26 neighbours
        in a full 3×3×3 grid.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            vs = 0.1
            xspec, yspec, zspec = [0, 1, vs], [0, 1, vs], [0, 1, vs]
            X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1], xspec[2]),
                                  np.arange(yspec[0], yspec[1], yspec[2]),
                                  np.arange(zspec[0], zspec[1], zspec[2]))
            mp = mp3d.from_coords(np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T)
            mp.find_first_order_neigh_CUBIC((0.5, 0.5, 0.5), vs)
        """
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
        """Check whether a voxel can have a single non-self-intersecting surface through it.

        Given that the 3×3×3 neighbourhood contains 27 voxels in various
        ON/OFF states, a single surface can pass through the centre voxel and
        all ON-state neighbours only when the number of ON-state neighbours is
        at most 4 (empirical threshold; equivalent to at most 5 points
        including the centre).

        The ``CUBIC`` suffix indicates this method is designed for cubic
        lattices only.

        Parameters
        ----------
        coord : array-like, shape (3,)
            Coordinate of the voxel to assess.  Must be a member of
            ``self.coords``.
        vs : float
            Voxel size used to define the 3×3×3 stencil.

        Returns
        -------
        bool or None
            ``True`` if 2–4 same-state neighbours exist (a surface can be
            formed), ``False`` if outside that range, ``None`` if ``coord``
            is not found in ``self.coords``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            vs = 0.1
            xspec, yspec, zspec = [0, 1, vs], [0, 1, vs], [0, 1, vs]
            X, Y, Z = np.meshgrid(np.arange(xspec[0], xspec[1], xspec[2]),
                                  np.arange(yspec[0], yspec[1], yspec[2]),
                                  np.arange(zspec[0], zspec[1], zspec[2]))
            mp = mp3d.from_coords(np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T)
            coords = mp.find_first_order_neigh_CUBIC((0.5, 0.5, 0.5), vs,
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
                       c='c', marker='o', alpha=0.1, s=200, edgecolors='black')
            ax.scatter(coords_ON_state[:, 0], coords_ON_state[:, 1], coords_ON_state[:, 2],
                       c='b', marker='o', alpha=0.8, s=50, edgecolors='black')
            result = mp.check_if_point_can_host_a_single_surface_CUBIC(coord, vs)
            print("Can host single surface:", result)
        """
        coord = np.array(coord)
        coords = self.find_first_order_neigh_CUBIC(coord, vs,
                                                   return_indices=False,
                                                   return_coords=True,
                                                   return_input_coord=False,
                                                   k=1.000001)
        coord_in_coords = np.argwhere(np.all(coords[0] == coord, axis=1)).squeeze()
        if coord_in_coords.size == 0:
            print('coord is not in self.coords !!')
            return None
        coords_ = self.coords[~np.all(self.coords == coord, axis=1)]
        npnt = coords_.shape[0]
        if npnt in (2, 3, 4):
            return True
        else:
            return False

    def get_local_tn(self, coord, k=5):
        """Find the local tangent plane and normal vector at a coordinate.

        Parameters
        ----------
        coord : array-like, shape (3,)
            Query point.  Must be a member of ``self.coords``.
        k : int, optional
            Number of nearest neighbours used to fit the tangent plane.
            Default is 5.

        Returns
        -------
        None
            Not yet implemented.
        """
        d0 = self.get_self_distance_min()

    def find_intersection_voxels_with_line(self, sl3d, cod):
        """Find all voxels in ``self.coords`` that intersect a 3-D line within a cut-off distance.

        Parameters
        ----------
        sl3d : Sline3d
            UPXO 3-D straight-line object to intersect against.
        cod : float
            Cut-off distance; only voxels within this distance of ``sl3d``
            are considered to intersect.

        Returns
        -------
        None
            Not yet implemented.
        """
