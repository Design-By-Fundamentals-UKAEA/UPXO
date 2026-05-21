"""
3D surface geometric entity for UPXO.

Import
------
    from upxo.geoEntities.surface import Surface

Classes
-------
Surface : 3D surface representation with meshing and smoothing operations.

Limitations
-----------
- All methods are stubs awaiting implementation.
"""


class Surface():
    """
    Represents a 3D surface in UPXO.

    Attributes
    ----------
    x : array-like
        X-coordinates of surface points.
    y : array-like
        Y-coordinates of surface points.
    z : array-like
        Z-coordinates of surface points.

    Limitations
    -----------
    - All methods are stubs awaiting implementation.
    """

    __slots__ = ('x', 'y', 'z')

    def __init__(self):
        """Initialise an empty surface."""
        raise NotImplementedError("__init__ is not yet implemented.")

    def __repr__(self):
        """Return ``UPXO surface.``"""
        return "UPXO surface."

    @classmethod
    def from_points(self):
        """Construct a surface from a collection of 3D points."""
        raise NotImplementedError("from_points is not yet implemented.")

    @classmethod
    def from_vertices(self):
        """Construct a surface from a vertex array."""
        raise NotImplementedError("from_vertices is not yet implemented.")

    def compute_normals(self):
        """Compute surface normals at each vertex."""
        raise NotImplementedError("compute_normals is not yet implemented.")

    def shortest_path(self, point):
        """
        Find the shortest geodesic path from a reference point on the surface.

        Parameters
        ----------
        point : array-like
            Query point coordinates.
        """
        raise NotImplementedError("shortest_path is not yet implemented.")

    def triangulate(self, point):
        """
        Triangulate the surface from the given reference point.

        Parameters
        ----------
        point : array-like
            Reference point for triangulation.
        """
        raise NotImplementedError("triangulate is not yet implemented.")

    def distribute_points(self, n, min_distance=-1):
        """
        Distribute ``n`` points on the surface with optional minimum separation.

        Parameters
        ----------
        n : int
            Number of points to distribute.
        min_distance : float, optional
            Minimum distance between points. Default is -1 (no constraint).
        """
        raise NotImplementedError("distribute_points is not yet implemented.")

    def pyvista_mesh(self):
        """Return a PyVista mesh representation of this surface."""
        raise NotImplementedError("pyvista_mesh is not yet implemented.")

    def smooth_laplace(self, niterations):
        """
        Apply Laplacian smoothing to the surface.

        Parameters
        ----------
        niterations : int
            Number of smoothing iterations.
        """
        raise NotImplementedError("smooth_laplace is not yet implemented.")

    def smooth_taubin(self, niterations):
        """
        Apply Taubin smoothing to the surface.

        Parameters
        ----------
        niterations : int
            Number of smoothing iterations.
        """
        raise NotImplementedError("smooth_taubin is not yet implemented.")
