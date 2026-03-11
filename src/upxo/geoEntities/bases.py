from abc import ABC, abstractmethod

class UPXO_Point(ABC):
    """Template base class for point object. Expands to both 2D and 3D."""

    __slots__ = ('x', 'y', 'pln', 'f')

    @abstractmethod
    def __init__(self, x=.0, y=.0, pln='ij'):
        pass

    @abstractmethod
    def __repr__(self):
        """docstring."""
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

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __eq__(self, elist):
        pass

    @abstractmethod
    def __ne__(self, elist):
        pass

    @property
    @abstractmethod
    def mid(self):
        pass

    @property
    @abstractmethod
    def ang(self):
        pass

    @property
    @abstractmethod
    def length(self):
        """Calculate and return self length"""
        pass

    @classmethod
    def by_coord(cls, start_point, end_point):
        pass

    @classmethod
    def by_loc_len_ang(cls, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        pass

    @abstractmethod
    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        pass

    @abstractmethod
    def attach_mp(self, *, mp=None, name=None):
        self.mp[name] = mp

    @abstractmethod
    def attach_xtal(self, *, xtals=None):
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0):
        pass

    @abstractmethod
    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        pass

    @abstractmethod
    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        pass

    @abstractmethod
    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        pass

    @abstractmethod
    def make_shapely(self):
        pass

    @abstractmethod
    def make_vtk(self):
        pass

    @property
    @abstractmethod
    def coords(self):
        return np.array([self.x, self.y])

    @abstractmethod
    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        pass

    @abstractmethod
    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        pass

    @abstractmethod
    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        pass
