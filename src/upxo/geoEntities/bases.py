"""
Abstract base classes for UPXO geometric entities.

Import
------
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
    """

    __slots__ = ('x', 'y', 'pln', 'f')

    @abstractmethod
    def __init__(self, x=.0, y=.0, pln='ij'):
        """Initialise the point at coordinates (x, y)."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return a developer-readable string representation."""
        pass

    @abstractmethod
    def __eq__(self, plist, *, use_tol=True):
        """Check if the two points are coincident."""
        pass

    @abstractmethod
    def __ne__(self, plist, *, use_tol=True):
        """Check if the two points are not coincident."""
        pass

    @abstractmethod
    def add(self, distances, update=True, throw=False,
            mydecatlen2NUM='taxx'):
        """Translate this point by the given distances."""
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
        """Calculate the EUclidean distance between self and list of points."""
        pass

    @abstractmethod
    def distance(self, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
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
    """

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        """Initialise the edge."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return a developer-readable string representation."""
        pass

    @abstractmethod
    def __eq__(self, elist):
        """Test equality against one or more edges."""
        pass

    @abstractmethod
    def __ne__(self, elist):
        """Test inequality against one or more edges."""
        pass

    @property
    @abstractmethod
    def mid(self):
        """Unique identifier (e.g. Python id) of this edge object."""
        pass

    @property
    @abstractmethod
    def ang(self):
        """Orientation angle of the edge in degrees."""
        pass

    @property
    @abstractmethod
    def length(self):
        """Euclidean length of the edge."""
        pass

    @classmethod
    def by_coord(cls, start_point, end_point):
        """Construct an edge from two coordinate pairs or point objects."""
        pass

    @classmethod
    def by_loc_len_ang(cls, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        """Construct an edge from a reference point, length, and angle."""
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        """Calculate distances from this edge to a list of points."""
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        """Calculate distances from this edge to a list of other edges."""
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        """Translate this edge by a displacement vector or scalar distance."""
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        """Translate this edge so that the reference endpoint lands on ``point``."""
        pass

    @abstractmethod
    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        """Rotate this edge about an axis by the given angle."""
        pass

    @abstractmethod
    def attach_mp(self, *, mp=None, name=None):
        """Attach a material-property object to this edge under ``name``."""
        self.mp[name] = mp

    @abstractmethod
    def attach_xtal(self, *, xtals=None):
        """Associate crystal objects with this edge."""
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0):
        """Find all points within radius ``r`` of this edge."""
        pass

    @abstractmethod
    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        """Find the ``n`` nearest points to this edge."""
        pass

    @abstractmethod
    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        """Find all mulpoint objects within radius ``r`` of this edge."""
        pass

    @abstractmethod
    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """Find all edges whose reference location is within radius ``r``."""
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """Find all muledge objects within radius ``r``."""
        pass

    @abstractmethod
    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """Find all crystal objects within radius ``r`` of this edge."""
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        """Attach GMSH mesh properties to this edge."""
        pass

    @abstractmethod
    def make_shapely(self):
        """Return a Shapely geometry object for this edge."""
        pass

    @abstractmethod
    def make_vtk(self):
        """Return a VTK geometry object for this edge."""
        pass

    @property
    @abstractmethod
    def coords(self):
        """Endpoint coordinates as a NumPy array."""
        return np.array([self.x, self.y])

    @abstractmethod
    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        """Generate an array of translated copies of this edge."""
        pass

    @abstractmethod
    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        """Determine which edges from ``elist`` this edge lies on."""
        pass

    @abstractmethod
    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """Determine which crystal from ``xlist`` contains this edge."""
        pass
