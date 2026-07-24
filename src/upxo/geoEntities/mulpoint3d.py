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
    metadata : dict
        User/provenance metadata carried with this point cloud.

    Standard coordinate format
    --------------------------
    ::

        coords = np.array([[0, 0, 0],
                           [1, 1, 1],
                           [2, 3, 3],
                           [4, 5, 6]])
    """
    __slots__ = ('coords', 'tree', 'pdist', 'metadata')

    def __init__(self, coords=None, metadata=None):
        """Initialise from an ``(N, 3)`` numpy array of 3D coordinates."""
        self.coords = self._coerce_coords(coords)
        self.tree = None
        self.pdist = pdist
        self.metadata = {} if metadata is None else dict(metadata)

    @staticmethod
    def _coerce_coords(coords):
        """Return ``coords`` as a numeric contiguous ``(N, 3)`` array."""
        if coords is None:
            return np.empty((0, 3), dtype=float)

        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            if coords.size == 0:
                return np.empty((0, 3), dtype=float)
            if coords.size != 3:
                raise ValueError('coords must have shape (N, 3).')
            coords = coords.reshape(1, 3)
        elif coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('coords must have shape (N, 3).')

        return np.ascontiguousarray(coords)

    @staticmethod
    def _coerce_bounds(bounds):
        """Return axis-aligned RVE bounds as a numeric ``(3, 2)`` array."""
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError('bounds must have shape (3, 2).')
        if not np.all(np.isfinite(bounds)):
            raise ValueError('bounds must contain only finite values.')
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError('Each bounds row must satisfy min < max.')
        return np.ascontiguousarray(bounds)

    @staticmethod
    def _coerce_boundary(boundary):
        """Return canonical boundary label and per-axis periodic flags."""
        boundary = str(boundary).strip().lower()
        if boundary in ('aperiodic', 'nonperiodic', 'non-periodic', 'open'):
            return 'aperiodic', (False, False, False)
        if boundary == 'periodic':
            return 'periodic', (True, True, True)
        raise ValueError("boundary must be either 'aperiodic' or 'periodic'.")

    @staticmethod
    def _effective_bounds(bounds, face_clearance):
        """Return the bounds available after applying face clearance."""
        face_clearance = float(face_clearance)
        if not np.isfinite(face_clearance):
            raise ValueError('face_clearance must be finite.')
        if face_clearance < 0:
            raise ValueError('face_clearance must be non-negative.')

        lengths = bounds[:, 1] - bounds[:, 0]
        if np.any(2.0*face_clearance >= lengths):
            raise ValueError(
                'face_clearance must be smaller than half of every RVE length.'
            )

        effective = bounds.copy()
        effective[:, 0] += face_clearance
        effective[:, 1] -= face_clearance
        return effective

    @staticmethod
    def _seed_metadata(generator_type, boundary, periodic, bounds,
                       effective_bounds, face_clearance, extra=None):
        """Return standard Voronoi-seed provenance metadata."""
        seed_metadata = {
            'seed_role': 'voronoi_generator',
            'generator_type': generator_type,
            'boundary': boundary,
            'periodic': periodic,
            'bounds': bounds.tolist(),
            'effective_bounds': effective_bounds.tolist(),
            'face_clearance': float(face_clearance),
        }
        if extra is not None:
            seed_metadata.update(dict(extra))
        return seed_metadata

    @staticmethod
    def _points_in_bounds(coords, bounds):
        """Return mask for coordinates inside closed axis-aligned bounds."""
        coords = np.asarray(coords, dtype=float)
        return np.all((coords >= bounds[:, 0]) & (coords <= bounds[:, 1]),
                      axis=1)

    @staticmethod
    def _wrap_coords_to_bounds(coords, bounds):
        """Wrap coordinates into an axis-aligned periodic box."""
        lengths = bounds[:, 1] - bounds[:, 0]
        return ((coords - bounds[:, 0]) % lengths) + bounds[:, 0]

    @staticmethod
    def _periodic_delta(coords, refs, bounds):
        """Return minimum-image deltas from ``coords`` to ``refs``."""
        delta = coords[:, None, :] - refs[None, :, :]
        lengths = bounds[:, 1] - bounds[:, 0]
        return delta - lengths*np.round(delta/lengths)

    @classmethod
    def _nearest_seed_indices(cls, points, seeds, bounds=None,
                              periodic=False, batch_size=20000):
        """Return nearest seed index for every point."""
        points = np.asarray(points, dtype=float)
        seeds = np.asarray(seeds, dtype=float)
        nearest = np.empty(points.shape[0], dtype=int)
        for start in range(0, points.shape[0], batch_size):
            stop = min(start + batch_size, points.shape[0])
            batch = points[start:stop]
            if periodic:
                delta = cls._periodic_delta(batch, seeds, bounds)
            else:
                delta = batch[:, None, :] - seeds[None, :, :]
            nearest[start:stop] = np.argmin(np.einsum('ijk,ijk->ij',
                                                       delta, delta),
                                            axis=1)
        return nearest

    @staticmethod
    def _apply_jitter(coords, jitter, rng, bounds, periodic):
        """Apply bounded random jitter to generated lattice coordinates."""
        jitter = float(jitter)
        if jitter < 0:
            raise ValueError('jitter must be non-negative.')
        if jitter == 0 or coords.size == 0:
            return coords
        coords = coords + rng.uniform(-jitter, jitter, size=coords.shape)
        if periodic:
            return MPoint3d._wrap_coords_to_bounds(coords, bounds)
        mask = MPoint3d._points_in_bounds(coords, bounds)
        return coords[mask]

    @staticmethod
    def _select_points(coords, n, rng):
        """Select exactly ``n`` points without replacement when requested."""
        if n is None:
            return coords
        n = int(n)
        if n < 1:
            raise ValueError('n must be a positive integer.')
        if coords.shape[0] < n:
            raise ValueError(
                f'Only {coords.shape[0]} points generated; requested {n}. '
                'Use smaller spacing, larger bounds, or lower face_clearance.'
            )
        if coords.shape[0] == n:
            return coords
        selection = rng.choice(coords.shape[0], size=n, replace=False)
        return coords[np.sort(selection)]

    @staticmethod
    def _estimate_lattice_spacing(bounds, n, lattice, ca_ratio):
        """Estimate lattice spacing needed to generate roughly ``n`` points."""
        n = int(n)
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        lattice = str(lattice).lower()
        if lattice == 'bcc':
            return (2.0*volume/n)**(1.0/3.0)
        if lattice == 'fcc':
            return (4.0*volume/n)**(1.0/3.0)
        if lattice == 'hcp':
            cell_factor = np.sqrt(3.0)*ca_ratio
            return (2.0*volume/(n*cell_factor))**(1.0/3.0)
        raise ValueError("lattice must be one of 'fcc', 'bcc', or 'hcp'.")

    @staticmethod
    def _generate_lattice_points(lattice, bounds, spacing, ca_ratio):
        """Generate FCC, BCC, or HCP lattice points inside ``bounds``."""
        lattice = str(lattice).strip().lower()
        spacing = float(spacing)
        if spacing <= 0 or not np.isfinite(spacing):
            raise ValueError('spacing must be a positive finite value.')

        lengths = bounds[:, 1] - bounds[:, 0]
        if lattice in ('fcc', 'bcc'):
            if lattice == 'fcc':
                basis = np.array([[0.0, 0.0, 0.0],
                                  [0.0, 0.5, 0.5],
                                  [0.5, 0.0, 0.5],
                                  [0.5, 0.5, 0.0]])
            else:
                basis = np.array([[0.0, 0.0, 0.0],
                                  [0.5, 0.5, 0.5]])
            shape = np.ceil(lengths/spacing).astype(int) + 1
            i, j, k = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]),
                                  indexing='ij')
            cells = np.column_stack((i.ravel(), j.ravel(), k.ravel()))
            coords = (bounds[:, 0] + cells[:, None, :]*spacing
                      + basis[None, :, :]*spacing)
            coords = coords.reshape(-1, 3)
        elif lattice == 'hcp':
            a = spacing
            c = ca_ratio*a
            a1 = np.array([a, 0.0, 0.0])
            a2 = np.array([0.5*a, 0.5*np.sqrt(3.0)*a, 0.0])
            a3 = np.array([0.0, 0.0, c])
            basis = np.array([[0.0, 0.0, 0.0],
                              [2.0/3.0, 1.0/3.0, 0.5]])
            n1 = int(np.ceil(lengths[0]/a)) + 3
            n2 = int(np.ceil(2.0*lengths[1]/(np.sqrt(3.0)*a))) + 3
            n3 = int(np.ceil(lengths[2]/c)) + 2
            coords = []
            start = bounds[:, 0] - np.array([a, np.sqrt(3.0)*a, c])
            for i in range(n1):
                for j in range(n2):
                    for k in range(n3):
                        origin = start + i*a1 + j*a2 + k*a3
                        for b in basis:
                            coords.append(origin + b[0]*a1 + b[1]*a2
                                          + b[2]*a3)
            coords = np.asarray(coords, dtype=float)
        else:
            raise ValueError("lattice must be one of 'fcc', 'bcc', or 'hcp'.")

        return coords[MPoint3d._points_in_bounds(coords, bounds)]

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

        if operation not in ('add', 'append'):
            raise ValueError("operation must be either 'add' or 'append'.")

        if operation == 'add':
            if type(toadd) in dth.dt.NUMBERS:
                self.coords += toadd
                self.tree = None
                return

            toadd = np.asarray(toadd, dtype=float)
            if toadd.ndim == 1 and toadd.size == 3:
                self.coords += toadd
            elif toadd.ndim == 2 and toadd.shape == self.coords.shape:
                self.coords += toadd
            elif toadd.ndim == 2 and toadd.T.shape == self.coords.shape:
                self.coords += toadd.T
            else:
                raise ValueError('Invalid shape of toadd for add operation.')
            self.tree = None

        elif operation == 'append':
            toadd = self._coerce_coords(toadd)
            self.coords = np.vstack((self.coords, toadd))
            self.tree = None

    @classmethod
    def from_coords(cls, point_coords, metadata=None):
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
        return cls(coords=np.array(point_coords), metadata=metadata)

    @classmethod
    def from_random_uniform(cls, bounds=((0, 1), (0, 1), (0, 1)), n=100,
                            seed=None, boundary='aperiodic',
                            face_clearance=0.0, metadata=None):
        """Generate uniformly distributed random 3D Voronoi seed points.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            Axis-aligned RVE bounds as ``((xmin, xmax), (ymin, ymax),
            (zmin, zmax))``. Default is the unit cube.
        n : int, optional
            Number of seed points to generate. Default is 100.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        boundary : {'aperiodic', 'periodic'}, optional
            Intended downstream Voronoi boundary condition. The point
            coordinates are generated the same way, but the metadata carries
            periodic flags for tessellation code. Default is 'aperiodic'.
        face_clearance : float, optional
            Minimum Euclidean distance from every seed point to any RVE face.
            For an axis-aligned box this is implemented by sampling within the
            inner box obtained by offsetting all faces inward by this value.
            Default is 0.0.
        metadata : dict or None, optional
            Extra metadata to merge into the generated seed metadata.

        Returns
        -------
        MPoint3d
            Seed point cloud with generation metadata attached.
        """
        n = int(n)
        if n < 1:
            raise ValueError('n must be a positive integer.')

        bounds = cls._coerce_bounds(bounds)
        boundary, periodic = cls._coerce_boundary(boundary)
        effective_bounds = cls._effective_bounds(bounds, face_clearance)

        rng = np.random.default_rng(seed)
        lows = effective_bounds[:, 0]
        highs = effective_bounds[:, 1]
        coords = rng.uniform(lows, highs, size=(n, 3))

        seed_metadata = cls._seed_metadata(
            'random', boundary, periodic, bounds, effective_bounds,
            face_clearance, extra={
            'n_requested': n,
            'n_generated': n,
            'random_seed': seed,
            })
        if metadata is not None:
            seed_metadata.update(dict(metadata))
        return cls(coords=coords, metadata=seed_metadata)

    @classmethod
    def from_custom_seeds(cls, coords, bounds=None, boundary='aperiodic',
                          face_clearance=0.0, enforce_bounds=True,
                          metadata=None):
        """Create Voronoi seed points from user-supplied coordinates.

        Parameters
        ----------
        coords : array-like, shape (N, 3)
            User-provided seed coordinates.
        bounds : array-like, shape (3, 2), optional
            RVE bounds as ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.
            If provided, the coordinates may be checked against the effective
            bounds after applying ``face_clearance``.
        boundary : {'aperiodic', 'periodic'}, optional
            Boundary-condition label to store in seed metadata.
        face_clearance : float, optional
            Minimum distance expected between seeds and RVE faces. This is
            used for metadata and, when ``bounds`` and ``enforce_bounds`` are
            provided, for bounds validation.
        enforce_bounds : bool, optional
            If True, require all coordinates to lie within the effective
            bounds.
        metadata : dict, optional
            Additional metadata merged into the generated seed metadata.

        Returns
        -------
        MPoint3d
            Point cloud containing the custom seed coordinates and provenance
            metadata.

        Raises
        ------
        ValueError
            If ``face_clearance`` is non-zero without bounds, or if
            ``enforce_bounds`` is True and coordinates fall outside the
            effective bounds.
        """
        coords = cls._coerce_coords(coords)
        boundary, periodic = cls._coerce_boundary(boundary)

        if bounds is None:
            if face_clearance != 0.0:
                raise ValueError('bounds are required when face_clearance '
                                 'is non-zero.')
            seed_metadata = {
                'seed_role': 'voronoi_generator',
                'generator_type': 'custom',
                'boundary': boundary,
                'periodic': periodic,
                'bounds': None,
                'effective_bounds': None,
                'face_clearance': float(face_clearance),
                'n_generated': coords.shape[0],
            }
        else:
            bounds = cls._coerce_bounds(bounds)
            effective_bounds = cls._effective_bounds(bounds, face_clearance)
            if enforce_bounds and not np.all(cls._points_in_bounds(
                    coords, effective_bounds)):
                raise ValueError('custom seed coordinates must lie inside '
                                 'the effective bounds.')
            seed_metadata = cls._seed_metadata(
                'custom', boundary, periodic, bounds, effective_bounds,
                face_clearance, extra={'n_generated': coords.shape[0]})

        if metadata is not None:
            seed_metadata.update(dict(metadata))
        return cls(coords=coords, metadata=seed_metadata)

    @classmethod
    def from_lattice(cls, lattice, bounds=((0, 1), (0, 1), (0, 1)), n=None,
                     spacing=None, seed=None, boundary='aperiodic',
                     face_clearance=0.0, jitter=0.0,
                     ca_ratio=np.sqrt(8.0/3.0), metadata=None):
        """Generate FCC, BCC, or HCP Voronoi seed lattice points.

        Parameters
        ----------
        lattice : {'fcc', 'bcc', 'hcp'}
            Lattice family used to place candidate seed points.
        bounds : array-like, shape (3, 2), optional
            RVE bounds as ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.
        n : int, optional
            Number of seed points to return. If provided with ``spacing=None``,
            spacing is estimated and candidates are down-selected to exactly
            ``n`` points.
        spacing : float, optional
            Lattice spacing. If omitted, ``n`` must be supplied.
        seed : int or None, optional
            Random seed used for down-selection and jitter.
        boundary : {'aperiodic', 'periodic'}, optional
            Boundary-condition label stored in metadata.
        face_clearance : float, optional
            Inward offset from every RVE face used to define the effective
            lattice-generation bounds.
        jitter : float, optional
            Uniform random perturbation magnitude applied independently to
            generated lattice coordinates.
        ca_ratio : float, optional
            HCP ``c/a`` ratio. Ignored for FCC and BCC.
        metadata : dict, optional
            Additional metadata merged into the generated seed metadata.

        Returns
        -------
        MPoint3d
            Lattice seed point cloud.

        Raises
        ------
        ValueError
            If neither ``spacing`` nor ``n`` is supplied, if spacing is
            invalid, or if too few lattice candidates can be generated.
        """
        lattice = str(lattice).strip().lower()
        bounds = cls._coerce_bounds(bounds)
        boundary, periodic = cls._coerce_boundary(boundary)
        effective_bounds = cls._effective_bounds(bounds, face_clearance)
        rng = np.random.default_rng(seed)
        spacing_requested = spacing
        spacing_was_estimated = spacing is None

        if spacing is None:
            if n is None:
                raise ValueError('Either spacing or n must be supplied.')
            spacing = cls._estimate_lattice_spacing(effective_bounds, n,
                                                    lattice, ca_ratio)
        spacing = float(spacing)

        candidate_coords = None
        spacing_used = spacing
        for attempt in range(16):
            candidate_coords = cls._generate_lattice_points(
                lattice, effective_bounds, spacing_used, ca_ratio)
            candidate_coords = cls._apply_jitter(candidate_coords, jitter, rng,
                                                 effective_bounds, periodic)
            if n is None or candidate_coords.shape[0] >= int(n):
                break
            if not spacing_was_estimated:
                break
            spacing_used *= 0.90
        else:
            raise ValueError('Could not generate enough lattice points. '
                             'Try smaller spacing or lower face_clearance.')

        coords = cls._select_points(candidate_coords, n, rng)
        seed_metadata = cls._seed_metadata(
            lattice, boundary, periodic, bounds, effective_bounds,
            face_clearance, extra={
                'lattice': lattice,
                'spacing_requested': spacing_requested,
                'spacing_used': spacing_used,
                'jitter': float(jitter),
                'ca_ratio': float(ca_ratio),
                'n_requested': None if n is None else int(n),
                'n_candidates': int(candidate_coords.shape[0]),
                'n_generated': int(coords.shape[0]),
                'random_seed': seed,
            })
        if metadata is not None:
            seed_metadata.update(dict(metadata))
        return cls(coords=coords, metadata=seed_metadata)

    @classmethod
    def from_fcc_lattice(cls, **kwargs):
        """Generate FCC lattice Voronoi seed points.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`from_lattice`.

        Returns
        -------
        MPoint3d
            FCC lattice seed point cloud.
        """
        return cls.from_lattice('fcc', **kwargs)

    @classmethod
    def from_bcc_lattice(cls, **kwargs):
        """Generate BCC lattice Voronoi seed points.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`from_lattice`.

        Returns
        -------
        MPoint3d
            BCC lattice seed point cloud.
        """
        return cls.from_lattice('bcc', **kwargs)

    @classmethod
    def from_hcp_lattice(cls, **kwargs):
        """Generate HCP lattice Voronoi seed points.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`from_lattice`.

        Returns
        -------
        MPoint3d
            HCP lattice seed point cloud.
        """
        return cls.from_lattice('hcp', **kwargs)

    @classmethod
    def from_hard_core_random(cls, bounds=((0, 1), (0, 1), (0, 1)), n=100,
                              min_distance=0.05, seed=None,
                              boundary='aperiodic', face_clearance=0.0,
                              max_attempts=100000, metadata=None):
        """Generate random seeds with a minimum seed-to-seed distance.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds as ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.
        n : int, optional
            Number of accepted seed points to generate.
        min_distance : float, optional
            Minimum permitted Euclidean distance between any two seeds.
        seed : int or None, optional
            Random seed for reproducible rejection sampling.
        boundary : {'aperiodic', 'periodic'}, optional
            Boundary-condition label stored in metadata. Fully periodic
            clouds use minimum-image distance checks.
        face_clearance : float, optional
            Inward offset from every RVE face used to define the effective
            sampling bounds.
        max_attempts : int, optional
            Maximum number of candidate draws before failing.
        metadata : dict, optional
            Additional metadata merged into the generated seed metadata.

        Returns
        -------
        MPoint3d
            Hard-core random seed point cloud.

        Raises
        ------
        ValueError
            If inputs are invalid or if ``n`` seeds cannot be placed within
            ``max_attempts``.
        """
        n = int(n)
        if n < 1:
            raise ValueError('n must be a positive integer.')
        min_distance = float(min_distance)
        if min_distance < 0 or not np.isfinite(min_distance):
            raise ValueError('min_distance must be a non-negative finite '
                             'value.')

        bounds = cls._coerce_bounds(bounds)
        boundary, periodic = cls._coerce_boundary(boundary)
        effective_bounds = cls._effective_bounds(bounds, face_clearance)
        rng = np.random.default_rng(seed)
        max_attempts = int(max_attempts)
        if max_attempts < n:
            raise ValueError('max_attempts must be at least n.')

        lows = effective_bounds[:, 0]
        highs = effective_bounds[:, 1]
        accepted = []
        attempts = 0
        min_distance_sq = min_distance*min_distance
        while len(accepted) < n and attempts < max_attempts:
            attempts += 1
            candidate = rng.uniform(lows, highs, size=3)
            if not accepted:
                accepted.append(candidate)
                continue
            refs = np.asarray(accepted, dtype=float)
            if periodic == (True, True, True):
                delta = cls._periodic_delta(candidate.reshape(1, 3), refs,
                                            effective_bounds)[0]
            else:
                delta = refs - candidate
            if np.all(np.einsum('ij,ij->i', delta, delta) >=
                      min_distance_sq):
                accepted.append(candidate)

        if len(accepted) < n:
            raise ValueError(
                f'Could only place {len(accepted)} hard-core points after '
                f'{attempts} attempts. Reduce n, min_distance, or '
                'face_clearance, or increase max_attempts.'
            )

        coords = np.asarray(accepted, dtype=float)
        seed_metadata = cls._seed_metadata(
            'hard_core', boundary, periodic, bounds, effective_bounds,
            face_clearance, extra={
                'n_requested': n,
                'n_generated': int(coords.shape[0]),
                'min_distance': min_distance,
                'attempts': attempts,
                'max_attempts': max_attempts,
                'random_seed': seed,
            })
        if metadata is not None:
            seed_metadata.update(dict(metadata))
        return cls(coords=coords, metadata=seed_metadata)

    @classmethod
    def from_cvt(cls, bounds=((0, 1), (0, 1), (0, 1)), n=100, seed=None,
                 boundary='aperiodic', face_clearance=0.0, iterations=20,
                 samples_per_seed=40, batch_size=20000, metadata=None):
        """Generate approximate centroidal Voronoi tessellation seed points.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds as ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.
        n : int, optional
            Number of seed points to generate.
        seed : int or None, optional
            Random seed for reproducible initialization and sampling.
        boundary : {'aperiodic', 'periodic'}, optional
            Boundary-condition label stored in metadata. Fully periodic clouds
            use minimum-image assignments during Lloyd updates.
        face_clearance : float, optional
            Inward offset from every RVE face used to define the effective
            sampling bounds.
        iterations : int, optional
            Number of Monte-Carlo Lloyd iterations.
        samples_per_seed : int, optional
            Number of random sample points per seed and iteration.
        batch_size : int, optional
            Batch size used for nearest-seed assignment.
        metadata : dict, optional
            Additional metadata merged into the generated seed metadata.

        Returns
        -------
        MPoint3d
            Approximate CVT seed point cloud.

        Raises
        ------
        ValueError
            If count, iteration, sampling, bounds, or batch parameters are
            invalid.
        """
        n = int(n)
        iterations = int(iterations)
        samples_per_seed = int(samples_per_seed)
        if n < 1:
            raise ValueError('n must be a positive integer.')
        if iterations < 0:
            raise ValueError('iterations must be non-negative.')
        if samples_per_seed < 1:
            raise ValueError('samples_per_seed must be positive.')
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')

        bounds = cls._coerce_bounds(bounds)
        boundary, periodic = cls._coerce_boundary(boundary)
        effective_bounds = cls._effective_bounds(bounds, face_clearance)
        rng = np.random.default_rng(seed)
        lows = effective_bounds[:, 0]
        highs = effective_bounds[:, 1]
        seeds = rng.uniform(lows, highs, size=(n, 3))
        nsamples = n*samples_per_seed

        for _ in range(iterations):
            samples = rng.uniform(lows, highs, size=(nsamples, 3))
            nearest = cls._nearest_seed_indices(
                samples, seeds, bounds=effective_bounds,
                periodic=periodic == (True, True, True),
                batch_size=batch_size)
            next_seeds = seeds.copy()
            for idx in range(n):
                owned = samples[nearest == idx]
                if owned.size == 0:
                    next_seeds[idx] = rng.uniform(lows, highs, size=3)
                elif periodic == (True, True, True):
                    delta = cls._periodic_delta(owned, seeds[idx:idx+1],
                                                effective_bounds)[:, 0, :]
                    next_seeds[idx] = seeds[idx] + delta.mean(axis=0)
                else:
                    next_seeds[idx] = owned.mean(axis=0)
            if periodic == (True, True, True):
                next_seeds = cls._wrap_coords_to_bounds(next_seeds,
                                                        effective_bounds)
            seeds = next_seeds

        seed_metadata = cls._seed_metadata(
            'cvt', boundary, periodic, bounds, effective_bounds,
            face_clearance, extra={
                'n_requested': n,
                'n_generated': int(seeds.shape[0]),
                'iterations': iterations,
                'samples_per_seed': samples_per_seed,
                'random_seed': seed,
                'method': 'monte_carlo_lloyd',
            })
        if metadata is not None:
            seed_metadata.update(dict(metadata))
        return cls(coords=seeds, metadata=seed_metadata)

    def _resolve_seed_bounds(self, bounds=None):
        """Resolve RVE bounds from user input or seed metadata."""
        if bounds is None:
            bounds = self.metadata.get('bounds')
        if bounds is None:
            if self.n == 0:
                raise ValueError('bounds are required for an empty seed cloud.')
            mins = self.coords.min(axis=0)
            maxs = self.coords.max(axis=0)
            bounds = np.column_stack((mins, maxs))
        return self._coerce_bounds(bounds)

    def _resolve_periodic(self, periodic=None):
        """Resolve periodic flags from user input or seed metadata."""
        if periodic is None:
            periodic = self.metadata.get('periodic', (False, False, False))
        if isinstance(periodic, bool):
            periodic = (periodic, periodic, periodic)
        if len(periodic) != 3:
            raise ValueError('periodic must be a bool or length-3 iterable.')
        return tuple(bool(flag) for flag in periodic)

    def nearest_neighbour_distances(self, bounds=None, periodic=None):
        """Return nearest-neighbour distance for every seed point.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds used when periodic distances are requested. If omitted,
            bounds are resolved from ``self.metadata['bounds']`` or from the
            coordinate extents.
        periodic : bool or iterable of bool, optional
            Periodic boundary flags. If omitted, metadata is used.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Nearest-neighbour distance for each seed point.

        Notes
        -----
        Periodic distances use the minimum-image convention inside ``bounds``.
        A single-point cloud returns ``nan`` for that point.
        """
        if self.n == 0:
            return np.empty(0, dtype=float)
        if self.n == 1:
            return np.full(1, np.nan, dtype=float)

        periodic = self._resolve_periodic(periodic)
        if periodic == (True, True, True):
            bounds = self._resolve_seed_bounds(bounds)
            delta = self._periodic_delta(self.coords, self.coords, bounds)
            dist_sq = np.einsum('ijk,ijk->ij', delta, delta)
            np.fill_diagonal(dist_sq, np.inf)
            return np.sqrt(np.min(dist_sq, axis=1))

        distances, _ = self.ckd_tree.query(self.coords, k=2)
        return distances[:, 1]

    def distances_to_rve_faces(self, bounds=None):
        """Return distances to ``xmin, xmax, ymin, ymax, zmin, zmax`` faces.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds. If omitted, bounds are resolved from metadata or from
            coordinate extents.

        Returns
        -------
        numpy.ndarray, shape (N, 6)
            Distance from each seed to the six RVE faces, ordered as
            ``xmin, xmax, ymin, ymax, zmin, zmax``.
        """
        bounds = self._resolve_seed_bounds(bounds)
        return np.column_stack((
            self.coords[:, 0] - bounds[0, 0],
            bounds[0, 1] - self.coords[:, 0],
            self.coords[:, 1] - bounds[1, 0],
            bounds[1, 1] - self.coords[:, 1],
            self.coords[:, 2] - bounds[2, 0],
            bounds[2, 1] - self.coords[:, 2],
        ))

    def minimum_face_distances(self, bounds=None):
        """Return minimum distance from each seed to any RVE face.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Minimum face distance for each seed.
        """
        if self.n == 0:
            return np.empty(0, dtype=float)
        return np.min(self.distances_to_rve_faces(bounds=bounds), axis=1)

    def distances_to_rve_corners(self, bounds=None):
        """Return distances from every seed to the 8 RVE corners.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.

        Returns
        -------
        numpy.ndarray, shape (N, 8)
            Distances from each seed to all RVE corners.
        """
        bounds = self._resolve_seed_bounds(bounds)
        corners = np.array([[x, y, z]
                            for x in bounds[0]
                            for y in bounds[1]
                            for z in bounds[2]], dtype=float)
        delta = self.coords[:, None, :] - corners[None, :, :]
        return np.linalg.norm(delta, axis=2)

    def minimum_corner_distances(self, bounds=None):
        """Return minimum distance from each seed to any RVE corner.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Minimum corner distance for each seed.
        """
        if self.n == 0:
            return np.empty(0, dtype=float)
        return np.min(self.distances_to_rve_corners(bounds=bounds), axis=1)

    def distances_to_rve_edges(self, bounds=None):
        """Return distances from every seed to the 12 finite RVE edges.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.

        Returns
        -------
        numpy.ndarray, shape (N, 12)
            Distances from each seed to the finite RVE edges.
        """
        bounds = self._resolve_seed_bounds(bounds)
        x0, x1 = bounds[0]
        y0, y1 = bounds[1]
        z0, z1 = bounds[2]
        edges = []
        for y in (y0, y1):
            for z in (z0, z1):
                edges.append(([x0, y, z], [x1, y, z]))
        for x in (x0, x1):
            for z in (z0, z1):
                edges.append(([x, y0, z], [x, y1, z]))
        for x in (x0, x1):
            for y in (y0, y1):
                edges.append(([x, y, z0], [x, y, z1]))

        distances = []
        for start, end in edges:
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            direction = end - start
            length_sq = np.dot(direction, direction)
            t = np.dot(self.coords - start, direction)/length_sq
            closest = start + np.clip(t, 0.0, 1.0)[:, None]*direction
            distances.append(np.linalg.norm(self.coords - closest, axis=1))
        return np.column_stack(distances)

    def minimum_edge_distances(self, bounds=None):
        """Return minimum distance from each seed to any RVE edge.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Minimum edge distance for each seed.
        """
        if self.n == 0:
            return np.empty(0, dtype=float)
        return np.min(self.distances_to_rve_edges(bounds=bounds), axis=1)

    def boundary_zone_counts(self, bounds=None, threshold=None):
        """Count seeds close to no faces, one face, two faces, or three faces.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds.
        threshold : float, optional
            Distance threshold used to classify a seed as near a face. If
            omitted, ``self.metadata['face_clearance']`` is used when present.

        Returns
        -------
        dict
            Counts for ``interior_zone``, ``face_zone``, ``edge_zone``, and
            ``corner_zone`` together with the applied threshold.

        Notes
        -----
        This directly exposes the different practical effects of a face
        clearance near RVE faces, edges, and corners.
        """
        if threshold is None:
            threshold = self.metadata.get('face_clearance', 0.0)
        threshold = float(threshold)
        if threshold < 0.0:
            raise ValueError('threshold must be non-negative.')
        near_faces = self.distances_to_rve_faces(bounds=bounds) <= threshold
        n_near_faces = near_faces.sum(axis=1)
        return {
            'interior_zone': int(np.sum(n_near_faces == 0)),
            'face_zone': int(np.sum(n_near_faces == 1)),
            'edge_zone': int(np.sum(n_near_faces == 2)),
            'corner_zone': int(np.sum(n_near_faces >= 3)),
            'threshold': threshold,
        }

    def seed_quality_summary(self, bounds=None, periodic=None,
                             min_distance=None, face_clearance=None):
        """Return compact seed-cloud QA statistics as a dictionary.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds used for boundary-distance diagnostics.
        periodic : bool or iterable of bool, optional
            Periodic boundary flags used for nearest-neighbour distances.
        min_distance : float, optional
            Minimum seed-to-seed distance to check.
        face_clearance : float, optional
            Minimum distance from seeds to RVE faces to check. If omitted,
            metadata is used when available.

        Returns
        -------
        dict
            Summary containing nearest-neighbour statistics, face/edge/corner
            distances, zone counts, and violation counts.
        """
        bounds = self._resolve_seed_bounds(bounds)
        periodic = self._resolve_periodic(periodic)
        if face_clearance is None:
            face_clearance = self.metadata.get('face_clearance', 0.0)
        face_clearance = float(face_clearance)

        nn = self.nearest_neighbour_distances(bounds=bounds,
                                              periodic=periodic)
        min_face = self.minimum_face_distances(bounds=bounds)
        min_edge = self.minimum_edge_distances(bounds=bounds)
        min_corner = self.minimum_corner_distances(bounds=bounds)
        finite_nn = nn[np.isfinite(nn)]
        summary = {
            'n': int(self.n),
            'generator_type': self.metadata.get('generator_type'),
            'boundary': self.metadata.get('boundary'),
            'periodic': periodic,
            'bounds': bounds.tolist(),
            'face_clearance': face_clearance,
            'nn_min': None if finite_nn.size == 0 else float(finite_nn.min()),
            'nn_mean': None if finite_nn.size == 0 else float(finite_nn.mean()),
            'nn_max': None if finite_nn.size == 0 else float(finite_nn.max()),
            'face_distance_min': None if min_face.size == 0 else float(min_face.min()),
            'face_distance_mean': None if min_face.size == 0 else float(min_face.mean()),
            'edge_distance_min': None if min_edge.size == 0 else float(min_edge.min()),
            'corner_distance_min': None if min_corner.size == 0 else float(min_corner.min()),
            'boundary_zone_counts': self.boundary_zone_counts(
                bounds=bounds, threshold=face_clearance),
        }
        if min_distance is not None:
            min_distance = float(min_distance)
            summary['min_distance'] = min_distance
            summary['min_distance_violations'] = int(np.sum(finite_nn <
                                                            min_distance))
        summary['face_clearance_violations'] = int(np.sum(min_face <
                                                          face_clearance))
        return summary

    def validate_seed_cloud(self, bounds=None, periodic=None,
                            min_distance=None, face_clearance=None,
                            throw=False):
        """Validate seed cloud spacing and face clearance constraints.

        Parameters
        ----------
        bounds : array-like, shape (3, 2), optional
            RVE bounds used for boundary-distance diagnostics.
        periodic : bool or iterable of bool, optional
            Periodic boundary flags used for nearest-neighbour distances.
        min_distance : float, optional
            Minimum seed-to-seed distance to enforce.
        face_clearance : float, optional
            Minimum distance from seeds to RVE faces to enforce.
        throw : bool, optional
            If True, raise an exception when validation fails.

        Returns
        -------
        dict
            Seed-quality summary augmented with ``valid`` and ``errors`` keys.

        Raises
        ------
        ValueError
            If ``throw`` is True and one or more validation checks fail.
        """
        summary = self.seed_quality_summary(bounds=bounds, periodic=periodic,
                                            min_distance=min_distance,
                                            face_clearance=face_clearance)
        errors = []
        if summary['face_clearance_violations'] > 0:
            errors.append('face_clearance')
        if summary.get('min_distance_violations', 0) > 0:
            errors.append('min_distance')
        summary['valid'] = len(errors) == 0
        summary['errors'] = errors
        if throw and errors:
            raise ValueError(f'Seed cloud failed validation: {errors}')
        return summary

    @classmethod
    def from_x_y_z(cls, x, y, z, metadata=None):
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
        return cls(coords=np.array([x, y, z]).T, metadata=metadata)

    @classmethod
    def from_xyz(cls, xyz, metadata=None):
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
        return cls(coords=xyz.T, metadata=metadata)

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
        return cls(coords=coords,
                   metadata=deepcopy(getattr(mulpoint3d, 'metadata', {})))

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
                      degree=True,
                      metadata=None
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
        mulpoint3d = MPoint3d.from_coords(coords, metadata=metadata)
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
                                        degree=degree)

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
        if self.tree is None:
            self.tree = self.maketree(treeType='ckdtree', throw=True)
        return self.tree

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
        if self.n == 0:
            raise ValueError('Cannot build a tree for an empty MPoint3d.')
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
