"""
Abstract base classes for UPXO geometric entities.

Usage
-----
    from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge

Classes
-------
UPXO_Point : Abstract base class for 2D and 3D point objects.
UPXO_Edge  : Abstract base class for 2D and 3D edge objects.

Notes
-----
Concrete implementations (e.g., ``point2d``, ``edge2d``) must override all
abstract methods.  The base classes define the common interface contract only
and carry no executable logic.
"""
from abc import ABC, abstractmethod


class UPXO_Point(ABC):
    """
    Abstract base class for UPXO point entities.

    Defines the minimum interface that all concrete 2D and 3D point
    implementations must satisfy.

    Attributes
    ----------
    x : float
    y : float
    pln : str
        Plane identifier (e.g. ``'ij'``).
    f : object
        Reserved for future use.

    Notes
    -----
    Concrete subclasses must provide all geometric operations declared by this
    abstract interface.
    """

    __slots__ = ('x', 'y', 'pln', 'f')

    @abstractmethod
    def __init__(self, x=.0, y=.0, pln='ij'):
        """
        Initialise the point at coordinates ``(x, y)``.

        Parameters
        ----------
        x : float, optional
            X-coordinate of the point.
        y : float, optional
            Y-coordinate of the point.
        pln : str, optional
            Plane identifier. Default is ``'ij'``.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a developer-readable string representation.

        Returns
        -------
        str
            Representation supplied by the concrete point class.
        """
        pass

    @abstractmethod
    def __eq__(self, plist, *, use_tol=True):
        """
        Check whether this point is coincident with candidate points.

        Parameters
        ----------
        plist : object or list
            Point or point collection to compare against.
        use_tol : bool, optional
            Whether to use coordinate tolerance in the comparison.

        Returns
        -------
        bool or list of bool
            Coincidence result supplied by the concrete point class.
        """
        pass

    @abstractmethod
    def __ne__(self, plist, *, use_tol=True):
        """
        Check whether this point is not coincident with candidate points.

        Parameters
        ----------
        plist : object or list
            Point or point collection to compare against.
        use_tol : bool, optional
            Whether to use coordinate tolerance in the comparison.

        Returns
        -------
        bool or list of bool
            Non-coincidence result supplied by the concrete point class.
        """
        pass

    @abstractmethod
    def add(self, distances, update=True, throw=False,
            mydecatlen2NUM='taxx'):
        """
        Translate this point by the given distances.

        Parameters
        ----------
        distances : object
            Translation distance specification accepted by the concrete class.
        update : bool, optional
            Whether to update this object in place.
        throw : bool, optional
            Whether to return generated point objects.
        mydecatlen2NUM : str, optional
            Data-type handling mode used by concrete implementations.

        Returns
        -------
        object
            Return value defined by the concrete point class.
        """
        pass

    @abstractmethod
    def __mul__(self, f, update=True, throw=False):
        """
        Multiple f to point coord & update self or return new point objects.

        All descriptions in parameters below, naturally extend to 3D.

        Parameters
        ----------
        f: list of multiplication factors. Depending on d, functionaliy changes
        as below.
            * [1, 2, 3, 4]: Each entry is multipled to both x and y. 4 new
            point objects gets created.
            * [[1, 2], [3, 4]]: [1, 2] denote first set of x and y distances.
            They get multipled with self.x and self.y to make a new point.
            Similar operation extewnds to [3, 4]. Two new points are created.
            * [[1, 2, 3, 4], [5, 6, 7, 8]]: These are X and Y arrays. Each x
            and y in X and Y, gets multipled with self.x and self.y to make n
            points, where n = len(d[0]).
            * [po1, po2, po3]: List of point objects. Point objects could be
            2D or 3D. UPXO, GMSH, VTK, PyVista, Shapely types are allowed.

        update: If True and if f is either K or Iterable(P, Q), where, K, P and
            Q are dth.dt.NUMBERS, self will be updated as self.x*K and self.y*K
            or self.x*P and self.y*Q.

        throw: If True and if additional conditions provided in update are
            atisfied, then the deepcopy of the point will be returned. If,
            however, update is False, a new point with coordiates self.x*K and
            self.y*K or self.x*P and self.y*Q, shall be created and returned.
        """
        pass

    @abstractmethod
    def distance(self, plist=None):
        """
        Calculate Euclidean distances from this point to other points.

        Parameters
        ----------
        plist : object or list, optional
            Point or point collection to evaluate.

        Returns
        -------
        float or list of float
            Distance result supplied by the concrete point class.
        """
        pass

    @abstractmethod
    def distance(self, plist=None):
        """
        Calculate Euclidean distances from this point to other points.

        Parameters
        ----------
        plist : object or list, optional
            Point or point collection to evaluate.

        Returns
        -------
        float or list of float
            Distance result supplied by the concrete point class.
        """
        pass

class UPXO_Edge(ABC):
    """
    Abstract base class for UPXO edge (line segment) entities.

    Defines the common interface contract for all concrete 2D and 3D edge
    implementations.  Subclasses store their start and end point references
    as ``i`` and ``j``.

    Attributes
    ----------
    i : object
        Start point of the edge.
    j : object
        End point of the edge.

    Notes
    -----
    Concrete subclasses are responsible for deciding whether calculations are
    performed in 2D, 3D, or projected coordinates.
    """

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        """
        Initialise the edge.

        Notes
        -----
        Concrete subclasses define the endpoint arguments and any construction
        options required by the specific edge representation.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a developer-readable string representation.

        Returns
        -------
        str
            Representation supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def __eq__(self, elist):
        """
        Test equality against one or more edges.

        Parameters
        ----------
        elist : object or list
            Edge or edge collection to compare against.

        Returns
        -------
        bool or list of bool
            Equality result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def __ne__(self, elist):
        """
        Test inequality against one or more edges.

        Parameters
        ----------
        elist : object or list
            Edge or edge collection to compare against.

        Returns
        -------
        bool or list of bool
            Inequality result supplied by the concrete edge class.
        """
        pass

    @property
    @abstractmethod
    def mid(self):
        """
        Unique identifier of this edge object.

        Returns
        -------
        object
            Identifier supplied by the concrete edge class.
        """
        pass

    @property
    @abstractmethod
    def ang(self):
        """
        Orientation angle of the edge.

        Returns
        -------
        float
            Edge orientation angle in degrees.
        """
        pass

    @property
    @abstractmethod
    def length(self):
        """
        Euclidean length of the edge.

        Returns
        -------
        float
            Edge length.
        """
        pass

    @classmethod
    def by_coord(cls, start_point, end_point):
        """
        Construct an edge from two coordinate pairs or point objects.

        Parameters
        ----------
        start_point : object
            Start point or coordinate specification.
        end_point : object
            End point or coordinate specification.

        Returns
        -------
        UPXO_Edge
            Concrete edge instance.
        """
        pass

    @classmethod
    def by_loc_len_ang(cls, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        """
        Construct an edge from a reference point, length, and angle.

        Parameters
        ----------
        ref : str, optional
            Reference endpoint identifier.
        loc : array-like, optional
            Reference endpoint location.
        length : float, optional
            Edge length.
        ang : float, optional
            Edge orientation angle.
        degree : bool, optional
            Whether ``ang`` is specified in degrees.

        Returns
        -------
        UPXO_Edge
            Concrete edge instance.
        """
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        """
        Calculate distances from this edge to a list of points.

        Parameters
        ----------
        plist : list, optional
            Points to evaluate.

        Returns
        -------
        object
            Distance result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        """
        Calculate distances from this edge to a list of other edges.

        Parameters
        ----------
        elist : list, optional
            Edges to evaluate.
        method : str, optional
            Distance method used by the concrete implementation.
        refi : str, optional
            Reference location on this edge.
        refj : str, optional
            Reference location on candidate edges.

        Returns
        -------
        object
            Distance result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        """
        Translate this edge by a displacement vector or scalar distance.

        Parameters
        ----------
        vector : array-like, optional
            Translation vector.
        dist : float, optional
            Scalar translation distance.
        update : bool, optional
            Whether to update this edge in place.
        throw : bool, optional
            Whether to return a generated edge object.

        Returns
        -------
        object
            Return value supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        """
        Translate this edge so the reference endpoint lands on ``point``.

        Parameters
        ----------
        ref : str, optional
            Reference endpoint identifier.
        point : object, optional
            Target point or coordinate specification.
        update : bool, optional
            Whether to update this edge in place.
        throw : bool, optional
            Whether to return a generated edge object.

        Returns
        -------
        object
            Return value supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        """
        Rotate this edge about an axis by the given angle.

        Parameters
        ----------
        axis : object, optional
            Rotation axis specification.
        angle : float, optional
            Rotation angle.
        degree : bool, optional
            Whether ``angle`` is specified in degrees.
        update : bool, optional
            Whether to update this edge in place.
        throw : bool, optional
            Whether to return a generated edge object.

        Returns
        -------
        object
            Return value supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def attach_mp(self, *, mp=None, name=None):
        """
        Attach a material-property object to this edge under ``name``.

        Parameters
        ----------
        mp : object, optional
            Material-property object to attach.
        name : str, optional
            Storage name for the material-property object.
        """
        self.mp[name] = mp

    @abstractmethod
    def attach_xtal(self, *, xtals=None):
        """
        Associate crystal objects with this edge.

        Parameters
        ----------
        xtals : object or list, optional
            Crystal object or collection to associate with this edge.
        """
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0):
        """
        Find all points within radius ``r`` of this edge.

        Parameters
        ----------
        plist : list, optional
            Candidate points.
        plane : str, optional
            Coordinate plane used for projected distance checks.
        r : float, optional
            Search radius.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        """
        Find the ``n`` nearest points to this edge.

        Parameters
        ----------
        plist : list, optional
            Candidate points.
        n : int, optional
            Number of nearest points to return.
        plane : str, optional
            Coordinate plane used for projected distance checks.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        """
        Find all mulpoint objects within radius ``r`` of this edge.

        Parameters
        ----------
        mplist : list, optional
            Candidate multi-point objects.
        plane : str, optional
            Coordinate plane used for projected distance checks.
        r : float, optional
            Search radius.
        tolf : float, optional
            Tolerance factor used by concrete implementations.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find all edges whose reference location is within radius ``r``.

        Parameters
        ----------
        elist : list, optional
            Candidate edges.
        plane : str, optional
            Coordinate plane used for projected distance checks.
        refloc : str, optional
            Reference location on candidate edges.
        r : float, optional
            Search radius.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find all muledge objects within radius ``r``.

        Parameters
        ----------
        melist : list, optional
            Candidate multi-edge objects.
        plane : str, optional
            Coordinate plane used for projected distance checks.
        refloc : str, optional
            Reference location on candidate multi-edge objects.
        r : float, optional
            Search radius.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find all crystal objects within radius ``r`` of this edge.

        Parameters
        ----------
        xlist : list, optional
            Candidate crystal objects.
        plane : str, optional
            Coordinate plane used for projected distance checks.
        refloc : str, optional
            Reference location on candidate crystal objects.
        r : float, optional
            Search radius.

        Returns
        -------
        object
            Neighbor result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        """
        Attach GMSH mesh properties to this edge.

        Parameters
        ----------
        prop_dict : dict
            GMSH property dictionary.
        """
        pass

    @abstractmethod
    def make_shapely(self):
        """
        Return a Shapely geometry object for this edge.

        Returns
        -------
        object
            Shapely geometry supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def make_vtk(self):
        """
        Return a VTK geometry object for this edge.

        Returns
        -------
        object
            VTK geometry supplied by the concrete edge class.
        """
        pass

    @property
    @abstractmethod
    def coords(self):
        """
        Endpoint coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Endpoint coordinate array supplied by the concrete edge class.
        """
        return np.array([self.x, self.y])

    @abstractmethod
    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        """
        Generate an array of translated copies of this edge.

        Parameters
        ----------
        ncopies : int, optional
            Number of translated copies to generate.
        vector : array-like, optional
            Translation vector specification.
        spacing : str, optional
            Spacing mode for translated copies.

        Returns
        -------
        object
            Translated edge collection supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        """
        Determine which edges from ``elist`` this edge lies on.

        Parameters
        ----------
        elist : list, optional
            Candidate edges.
        consider_ends : bool, optional
            Whether endpoint coincidence is considered part of the test.

        Returns
        -------
        object
            Containment result supplied by the concrete edge class.
        """
        pass

    @abstractmethod
    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """
        Determine which crystal from ``xlist`` contains this edge.

        Parameters
        ----------
        xlist : list, optional
            Candidate crystal objects.
        cosider_boundary : bool, optional
            Whether crystal boundaries are considered part of the test.
        consider_boundary_ends : bool, optional
            Whether edge endpoints on boundaries are considered.

        Returns
        -------
        object
            Containment result supplied by the concrete edge class.
        """
        pass
