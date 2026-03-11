"""
Created on Sat Apr 20 20:07:35 2024

3D straight line.

Applications
------------
* Non-conformal geometry to conformal geometry conversion
* Heirarchical grain structure feature generation
* General geometry use

Classes
-------
* MSline2d
* ring2d
* mulring2d
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

__authors__ = ["Dr. Sunil Anandatheertha"]
__version__ = "1.0"
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
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.bases import UPXO_Point, UPXO_Edge
np.seterr(divide='ignore')
from upxo._sup.validation_values import isinstance_many
from upxo.geoEntities.sline2d import Sline2d as sl2d
from upxo.geoEntities.point2d import p2d_leanest
from shapely.geometry import Polygon, MultiPolygon
from upxo._sup.data_ops import mean_coordinates

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES

class MSline2d():
    """
    UPXO code class.

    Examples
    --------
    from upxo.geoEntities.mulsline2d import MSline2d as msl2d
    from upxo.geoEntities.sline2d import Sline2d as sl2d
    e0 = sl2d(0.0,0.0, 1.0,1.0)
    e1 = sl2d(1.0,1.0, 1.5,1.5)
    e2 = sl2d(1.5,1.5, 2.5,2.5)
    e3 = sl2d(2.5,2.5, 4.0,4.0)
    e4 = sl2d(4.0,4.0, 4.0,6.0)

    me = msl2d([e0, e1, e2, e3, e4])
    """
    __slots__ = ('lines', 'nodes', 'features', 'closed')

    EPS_coord_coincide = 1E-8

    def __init__(self, nodes=None, llist=None, closed=None):
        self.lines = llist
        self.nodes = nodes
        self.closed = closed
        self.features = {'neigh_gids': None}

    def __repr__(self):
        return f"MSL2. nln={len(self.lines)}. ID: {id(self)}: {self.nodes[0]}, {self.nodes[-1]}"

    def __iter__(self):
        """
        Return an iterable of point coordsinates in self.

        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d
        from upxo.geoEntities.sline2d import Sline2d
        lines = [Sline2d(0.0,0.0, 1.0,1.0), Sline2d(1.0,1.0, 1.5,1.5),
                 Sline2d(1.5,1.5, 2.5,2.5), Sline2d(2.5,2.5, 4.0,4.0),
                 Sline2d(4.0,4.0, 4.0,6.0)]
        MULLINE = MSline2d.from_lines(lines, close=True)
        lines = [line for line in MULLINE.lines]
        print(lines)
        """
        return iter(self.lines)

    def __getitem__(self, i):
        """
        Make self indexable. i: index location.

        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d
        from upxo.geoEntities.sline2d import Sline2d
        lines = [Sline2d(0.0,0.0, 1.0,1.0), Sline2d(1.0,1.0, 1.5,1.5),
                 Sline2d(1.5,1.5, 2.5,2.5), Sline2d(2.5,2.5, 4.0,4.0),
                 Sline2d(4.0,4.0, 4.0,6.0)]
        MULLINE = MSline2d.from_lines(lines, close=True)
        MULLINE[4]
        MULLINE[5]
        """
        if i >= self.nlines:
            raise ValueError('Index exceeds maximum number of lines.')
        return self.lines[i]

    @classmethod
    def from_lines(cls, llist, close=True):
        """
        Make a multi-straight line in 2d.

        Make mul-straight-line-2d using multiple lines, strt and connectvity
        specification. Recommended way to make the msl2d.

        Development phases
        ------------------
        Phase-1: Basic working codes with basic validations. DONE
        Phase-2: Include additional validations.

        Examples
        --------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        e0 = sl2d(0.0,0.0, 1.0,1.0)
        e1 = sl2d(1.0,1.0, 1.5,1.5)
        e2 = sl2d(1.5,1.5, 2.5,2.5)
        e3 = sl2d(2.5,2.5, 4.0,4.0)
        e4 = sl2d(4.0,4.0, 4.0,6.0)

        lines = [e0,e1,e2,e3,e4]
        me = msl2d.from_lines(lines, close=True)
        me.lines
        me.nodes

        lines = [e0,e1,e2,e3,e4]
        me = msl2d.from_lines(lines, close=False)
        me.lines
        """
        llist, closed = llist, False
        nodes = [line.pnta for line in llist]
        nodes.append(llist[-1].pntb)
        if close:
            fl = llist[0]  # First line
            ll = llist[-1]  # Last line
            llist.append(sl2d(ll.x1, ll.y1, fl.x0, fl.y0))
            nodes.append(nodes[0])
            closed = True
        return cls(nodes=nodes, llist=llist, closed=closed)

    @classmethod
    def by_nodes(cls, nodes, close=True):
        """
        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d
        from upxo.geoEntities.point2d import Point2d
        nodes = [Point2d(0,0), Point2d(1,1), Point2d(2,2), Point2d(3,3), Point2d(5,5)]
        MSline2d.by_nodes(nodes).lines

        Import
        ------
        from upxo.geoEntities.mulsline2d import MSline2d
        """
        if type(nodes) not in ITERABLES:
            raise ValueError('Invalid nodes input.')
        if len(nodes) < 2:
            raise ValueError('Invalid nodes iterable length. Must be >= 2.')
        if type(close) != bool:
            raise ValueError('Invalid close input.')
        # ----------------------------------------
        llist = [sl2d(nodes[i].x, nodes[i].y, nodes[i+1].x, nodes[i+1].y)
                 for i in range(len(nodes)-1)]
        nodes = [line.pnta for line in llist]
        # print(llist[-1].pntb)
        # print(nodes)
        nodes.append(llist[-1].pntb)
        if close:
            llist.append(sl2d(nodes[-1].x, nodes[-1].y,
                              nodes[0].x, nodes[0].y))
            nodes.append(nodes[0])
        return cls(nodes=nodes, llist=llist, closed=close)

    @classmethod
    def by_coords(cls, coords, close=True):
        # from upxo.geoEntities.mulsline2d import MSline2d
        """
        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d
        from upxo.geoEntities.point2d import Point2d
        coords = np.array([(0,0), (1,1), (2,2), (3,3), (5,5)])
        MSline2d.by_coords(coords).lines
        MSline2d.by_coords(coords).nodes
        """
        if type(coords) not in ITERABLES:
            raise ValueError('Invalid nodes input.')
        if len(coords) < 2:
            raise ValueError('Invalid nodes iterable length. Must be >= 2.')
        # ----------------------------------------
        llist = [sl2d(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
                 for i in range(len(coords)-1)]
        nodes = [line.pnta for line in llist]
        if close:
            llist.append(sl2d(nodes[-1].x, nodes[-1].y,
                              nodes[0].x, nodes[0].y))
            nodes.append(nodes[0])
        return cls(nodes=nodes, llist=llist, closed=close)

    @classmethod
    def by_walk(self, var_l='constant', var_ang='constant',
                specs = {'n': 5,
                         'max_total_length': 10,
                         'min_total_length': 8,
                         'mean_length': 1,}
                ):
        pass

    @property
    def nlines(self):
        """Return number of lines."""
        return len(self.lines)

    @property
    def centroid(self):
        return self.get_node_coords().mean(axis=0)

    @property
    def centroid_p2dl(self):
        return p2d_leanest(*self.centroid)

    @property
    def length(self):
        return sum([line.length for line in self.lines])

    @property
    def lengths(self):
        return [line.length for line in self.lines]

    @property
    def nnodes(self):
        """Return number of lines."""
        return len(self.nodes[:-1]) if self.closed else len(self.nodes)

    @property
    def coords(self):
        return np.array([[p.x, p.y] for p in self.nodes])
    
    @property
    def tree(self):
        return cKDTree(self.coords)

    @property
    def lengths(self):
        """
        Rerturns lgnths of individual lines in regular order.

        Example
        -------
        me.lengths
        """
        return [line.length for line in self.lines]

    @property
    def length(self):
        """
        Rerturns total lgnth of all lines.

        Example
        -------
        me.lengths
        """
        return sum(self.lengths)

    @property
    def length_mean(self):
        """
        Rerturns mean length.

        Example
        -------
        me.lengths
        """
        return self.length/self.n

    @property
    def gradients(self):
        """
        Return gradient of every line.
        """
        return [line.gradient for line in self.lines]

    @property
    def get_nodes(self):
        """Return unique list of nodes in the multi-straight-line object."""
        nodes = [[line.x0, line.y0] for line in self.lines]
        nodes += [[line.x1, line.y1] for line in self.lines]
        return np.unique(nodes, axis=0)

    def get_node_coords(self):
        return np.array([[node.x, node.y] for node in self.extract_nodes()])

    @property
    def mid_nodes(self):
        """
        Return mid-sode nodes of all lines.

        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.mid_nodes
        MULLINE.centroid_p2dl
        """
        return [line.mid_point for line in self.lines]

    @property
    def line_ids(self):
        """Return memory id of each line."""
        return [id(line) for line in self.lines]

    def flip(self, saa=True, throw=False):
        """
        Return mid-sode nodes of all lines.

        Example
        -------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.mid_nodes
        MULLINE.centroid_p2dl
        MULLINE.lines
        MULLINE.nodes
        MULLINE.flip()
        MULLINE.lines
        MULLINE.nodes
        """
        if saa:
            self.lines = list(np.flip(self.lines, axis=0))
            for line in self.lines:
                line.flip()
            self.update_nodes()
            if throw:
                return self
        if not saa:
            mulslines2d = deepcopy(self)
            mulslines2d.lines = list(np.flip(mulslines2d.lines, axis=0))
            for line in mulslines2d.lines:
                line.flip()
            mulslines2d.update_nodes()
            if throw:
                return mulslines2d

    def do_i_precede(self, multisline2d):
        """
        Check if self spatially precedes the input multisline2d.

        Return
        ------
        i_precede:
            True if self is spatially immediately behind multisline2d.
            False if self is spatially immediately bnehind multisline2d.
        flip_needed:
            False if i_precede is False
            True if user supplied multisline2d needs to be flipped in order to
            ensure nodal connectivity between node orderning in self and
            multisline2d.
            False if user supplied multisline2d is already in order to
            ensure nodal connectivity between node orderning in self and
            multisline2d.
        """
        condition_a = self.nodes[-1].eq_fast(multisline2d.nodes[0])[0]
        condition_b = self.nodes[-1].eq_fast(multisline2d.nodes[-1])[0]
        if condition_a:
            i_precede, flip_needed = True, False
        if condition_b:
            i_precede, flip_needed = True, True
        if not condition_a and not condition_b:
            i_precede, flip_needed = False, False
        return i_precede, flip_needed

    def do_i_proceed(self, multisline2d):
        """
        Check if self spatially comes after the input multisline2d.

        Return
        ------
        i_precede:
            True if self is spatially immediately behind multisline2d.
            False if self is spatially immediately bnehind multisline2d.
        flip_needed:
            False if i_precede is False
            True if user supplied multisline2d needs to be flipped in order to
            ensure nodal connectivity between node orderning in self and
            multisline2d.
            False if user supplied multisline2d is already in order to
            ensure nodal connectivity between node orderning in self and
            multisline2d.
        """
        condition_a = self.nodes[0].eq_fast(multisline2d.nodes[0])[0]
        condition_b = self.nodes[0].eq_fast(multisline2d.nodes[-1])[0]
        if condition_a:
            i_precede, flip_needed = True, False
        if condition_b:
            i_precede, flip_needed = True, True
        if not condition_a and not condition_b:
            i_precede, flip_needed = False, False
        return i_precede, flip_needed

    def is_adjacent(self, multisline2d):
        left = self.do_i_proceed(multisline2d)  # msl2d is to the left of self
        right = self.do_i_precede(multisline2d)  # msl2d is to the rght of self
        return any(left[0], right[0]), left, right

    def find_spatially_next_multisline2d(self, multislines2d):
        """
        From a list of multisline2d objects, multislines2d, find the ones which
        come spatially immediately after self.
        """
        precedes = []
        for msl in multislines2d:
            precedes.append(self.do_i_precede(msl))
        return precedes

    def has_coord(self, coord, return_flags=False):
        # Validations: coord must be np.array 2D coordfinate
        _coords_ = self.get_node_coords()
        flags = (_coords_[:, 0] == coord[0]) & (_coords_[:, 1] == coord[1])
        if return_flags:
            return any(flags), flags
        else:
            return any(flags)

    def find_coord_location(self, coord):
        # Validations: coord must be np.array 2D coordfinate
        exists, flags = self.has_coord(coord, return_flags=True)
        if exists:
            return np.argwhere(flags)
        else:
            return None

    def extract_nodes(self):
        nodes = [line.pnta for line in self.lines]
        if self.closed:
            nodes.append(self.nodes[0])
        else:
            nodes.append(self.lines[-1].pntb)
        return nodes

    def update_nodes(self):
        self.nodes = [line.pnta for line in self.lines]
        if self.closed:
            self.nodes.append(self.nodes[0])
        else:
            self.nodes.append(self.lines[-1].pntb)

    def close(self, reclose=False):
        """
        Close the self multi-straight line object.

        Examples
        --------
        a.
        """
        if not self.closed:
            fl, ll = self[0], self[-1]
            self.lines.append(sl2d(ll.x1, ll.y1, fl.x0, fl.y0))
        else:
            if not reclose:
                # Nothing to do here as self already closed and recvlose is
                # False
                pass
        if reclose:
            fl, ll = self[0], self[-1]
            self.lines.append(sl2d(ll.x1, ll.y1, fl.x0, fl.y0))

    def unclose(self):
        """Remove the closing line."""
        del self.lines[-1]

    def distances_nodes(self, points):
        """
        Find distances between all nodes to the goiven points.

        Parameters
        ----------
        points: List of points

        Example-1
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        points = np.random.random((2,2))
        MULLINE.distances_nodes(points)
        """
        # Validation
        # -------------------------------------
        nodes = np.array(self.nodes)
        """
        nodes = np.random.random((4, 2))
        points = np.random.random((10,2))
        distances = np.sqrt(np.sum((points[:, np.newaxis] - nodes) ** 2,
                                   axis=2))

        points[:, np.newaxis] broadcasts the points array to have shape
        (10, 1, 3), effectively creating a third dimension for broadcasting.

        points[:, np.newaxis] - nodes broadcasts nodes to shape (10, 4, 3) and
        subtracts each node from each point, resulting in an array of
        differences.

        np.sum(... ** 2, axis=2) squares each difference element-wise, sums
        along the third axis (axis=2), and then takes the square root to get
        the Euclidean distances.
        """
        distances = np.sqrt(np.sum((points[:, np.newaxis] - nodes) ** 2,
                                   axis=2)).T
        return distances

    def find_closest_nodes(self, point):
        """
        Find closest nodes to the given point.

        Parameters
        ----------
        points: List of points

        Example-1
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5)]
        MULLINE = msl2d.from_lines(lines, close=True)
        point = np.random.random(2)*np.random.randint(10)
        MULLINE.find_closest_nodes(point)

        Example-2
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5)]
        MULLINE = msl2d.from_lines(lines, close=True)
        point = lines[0].mid
        MULLINE.find_closest_nodes(point)
        """
        # Validation
        # -------------------------------------
        distances = self.distances_nodes(np.array(point)[:, np.newaxis].T).T.squeeze()
        closest_points = np.argwhere(distances == distances.min()).T.squeeze().tolist()
        return closest_points

    def add_nodes(self, nodes):
        """
        Example-1
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        from upxo.geoEntities.point2d import Point2d
        e0 = sl2d(0.0,0.0, 1.0,1.0)
        e1 = sl2d(1.0,1.0, 1.5,1.5)
        e2 = sl2d(1.5,1.5, 2.5,2.5)
        e3 = sl2d(2.5,2.5, 4.0,4.0)
        e4 = sl2d(4.0,4.0, 4.0,6.0)
        lines = [e0,e1,e2,e3,e4]
        mulline = msl2d.from_lines(lines, close=True)
        print(mulline)
        nodes_to_add = [Point2d(0.5, 0.5), Point2d(2.0, 2.0)]
        mulline.add_nodes(nodes_to_add)
        print(mulline)
        mulline.lines
        mulline.nodes
        mulline.closed
        mulline.extract_nodes()

        Example-2
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        from upxo.geoEntities.point2d import Point2d
        e0 = sl2d(0.0,0.0, 1.0,1.0)
        e1 = sl2d(1.0,1.0, 1.5,1.5)
        e2 = sl2d(1.5,1.5, 2.5,2.5)
        e3 = sl2d(2.5,2.5, 4.0,4.0)
        e4 = sl2d(4.0,4.0, 5.0,4.0)
        e5 = sl2d(5.0,4.0, 5.0,0.0)
        lines = [e0,e1,e2,e3,e4,e5]
        mulline = msl2d.from_lines(lines, close=True)
        print(mulline)
        mulline.lines
        nodes_to_add = [Point2d(0.5, 0.5), Point2d(2.0, 2.0), Point2d(1.75, 0)]
        mulline.add_nodes(nodes_to_add)
        print(mulline)
        mulline.lines
        mulline.nodes
        mulline.closed
        mulline.extract_nodes()

        @dev: raw codes used in development
        -----------------------------------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        from upxo.geoEntities.point2d import Point2d
        e0 = sl2d(0.0,0.0, 1.0,1.0)
        e1 = sl2d(1.0,1.0, 1.5,1.5)
        e2 = sl2d(1.5,1.5, 2.5,2.5)
        e3 = sl2d(2.5,2.5, 4.0,4.0)
        e4 = sl2d(4.0,4.0, 4.0,6.0)

        lines = [e0,e1,e2,e3,e4]
        mulline = msl2d.from_lines(lines, close=False)

        mulline.lines
        mulline.nodes
        mulline.closed
        mulline.extract_nodes()

        nodes_to_add = [Point2d(0.5, 0.5), Point2d(2.0, 2.0)]
        for node_to_add in nodes_to_add:
            # node_to_add = nodes_to_add[1]
            line_indices = []
            for i, line in enumerate(mulline.lines, start=0):
                if line.fully_contains_point(p2d=node_to_add, method='through'):
                    line_indices.append(i)

            if len(line_indices) != 0:
                line = mulline.lines[line_indices[0]]

                new_line = line.split(method='p2d', divider=node_to_add, saa=True, throw=True, update='pntb')[1]
                mulline.lines.insert(line_indices[0]+1, new_line)
                mulline.update_nodes()

        id(node_to_add)

        mulline.lines
        mulline.nodes
        mulline.extract_nodes()

        mulline.update_nodes()
        mulline.nodes
        mulline.extract_nodes()

        [id(ml) for ml in mulline.nodes]
        [id(ml) for ml in mulline.extract_nodes()]
        [id(nta) for nta in nodes_to_add]
        [id(line.pnta) for line in mulline.lines]
        """
        for node in nodes:
            line_indices = []
            for i, line in enumerate(self.lines, start=0):
                if line.fully_contains_point(p2d=node, method='through'):
                    line_indices.append(i)

            if len(line_indices) != 0:
                line = self.lines[line_indices[0]]
                new_line = line.split(method='p2d', divider=node,
                                      saa=True, throw=True, update='pntb')[1]
                self.lines.insert(line_indices[0]+1, new_line)
                self.update_nodes()

    def splice_nodes_and_lines(self, method='points', points=None, perform_checks=True):
        pass

    def roll(self, roll_distance):
        self.lines = list(np.roll(self.lines, -roll_distance))
        self.update_nodes()

    def sub_divide(self, line_number=0, f=0.5):
        """
        Sub-divide a single line in self.lines.

        Parameters
        ----------
        line_numbers: indeix in self.lines. Starts from 0.
        f: factor in (0, 1), indicating location where the line shall
            be divided.

        Examples
        --------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]

        me = msl2d.from_lines(lines, close=True)
        me.lines

        me = msl2d.from_lines(lines, close=False)
        me.lines

        me.sub_divide(line_number=0, f=0.25)
        me.lines

        me.sub_divide(line_number=len(me.lines), f=0.25)
        me.lines

        me.sub_divide(line_number=3, f=0.50)
        me.lines

        me.sub_divide(line_number=0, f=0.50)
        me.lines

        for i in range(10):
            me.sub_divide(line_number=0, f=0.50)
        me.lines

        for i in range(10):
            me.sub_divide(line_number=i, f=0.50)
        me.lines
        """
        if not isinstance(line_number, int):
            raise TypeError('Invalid line number type.')
        if line_number > len(self.lines):
            raise ValueError('Invalid line number spacification.')
        if line_number == 0:
            new_lines = self.lines[0].split(f=f, saa=False, throw=True)
            self.lines = new_lines + self.lines[1:]
        elif line_number == len(self.lines):
            new_lines = self.lines[-1].split(f=f, saa=False, throw=True)
            self.lines = self.lines[:-1] + new_lines
        else:
            # end result: left--line0--line1--right
            left = [line for line in self.lines[:line_number]]
            # Divisiozn operation stzrts
            line01 = self.lines[line_number].split(f=f, saa=False, throw=True)
            # Division operation ends
            right = [line for line in self.lines[line_number+1:]]
            self.lines = left + line01 + right

    def remove_point_by_index(self, index=2, remove='previous_line'):
        """
        Example-1
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines

        for line in MULLINE.lines: print(id(line))
        MULLINE.remove_point_by_index(index=2, remove='previous_line')
        for line in MULLINE.lines: print(id(line))

        MULLINE.lines

        Example-2
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines

        for line in MULLINE.lines: print(id(line))
        MULLINE.remove_point_by_index(index=2, remove='next_line')
        for line in MULLINE.lines: print(id(line))

        MULLINE.lines

        Example-3
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines

        for line in MULLINE.lines: print(id(line))
        MULLINE.remove_point_by_index(index=2, remove='both')
        for line in MULLINE.lines: print(id(line))

        MULLINE.lines
        """
        # Validations
        # -------------------------------------
        previous_line, next_line = index-1, index
        # -------------------------------------
        if remove == 'previous_line':
            # Next line will be updayed and previous line will be removed
            self.lines[next_line].move_i(self.lines[previous_line].coord_i)
            del self.lines[previous_line]
        elif remove == 'next_line':
            # Next line will be removed and previous line will be updated
            self.lines[previous_line].move_j(self.lines[next_line].coord_j)
            del self.lines[next_line]
        elif remove == 'both':
            # Next line and previous line will be removed and a new line will
            # be made in its place
            x0, y0 = self.lines[previous_line].coord_i
            x1, y1 = self.lines[next_line].coord_j
            new_line = sl2d(x0, y0, x1, y1)
            self.lines = self.lines[:previous_line] + [new_line] + self.lines[next_line+1:]
        else:
            raise ValueError('Invalid update specirfication.')

    def remove_point_by_location(self, location=(None,None,None),
                                 remove='previous_line'):
        """
        Example-1
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines
        MULLINE.line_ids
        MULLINE.remove_point_by_location(location=lines[0].coord_i, remove='previous_line')
        MULLINE.lines
        MULLINE.line_ids

        Example-2
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines
        MULLINE.line_ids
        MULLINE.remove_point_by_location(location=lines[0].mid, remove='previous_line')
        MULLINE.lines
        MULLINE.line_ids

        Example-3
        ---------
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        from upxo.geoEntities.sline2d import Sline2d as sl2d
        lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5),
                 sl2d(1.5,1.5,2.5,2.5), sl2d(2.5,2.5,4.0,4.0),
                 sl2d(4.0,4.0,4.0,6.0)]
        MULLINE = msl2d.from_lines(lines, close=True)
        MULLINE.lines
        MULLINE.plot()
        location = np.random.random(2)*np.random.randint(10)
        MULLINE.remove_point_by_location(location=location, remove='previous_line')
        MULLINE.lines
        """
        # Validations
        # -------------------------------------
        if self.n == 1:
            """
            When there is a single line in the multi-line, no point can
            be removed. Function call will exit.
            """
            return
        # -------------------------------------
        indices = self.find_closest_nodes(location)
        # -------------------------------------
        if isinstance(indices, int):
            self.remove_point_by_index(index=indices, remove=remove)
        elif isinstance(indices, list) and len(indices) > 1:
            self.remove_point_by_index(index=indices[0], remove=remove)
            for i, index in enumerate(indices[1:]):
                self.remove_point_by_location(location=location, remove=remove)
            # print('Multiple nodes were removed.')
        # -------------------------------------
        if self.n == 2:
            """
            In some cases, only 2 lines remain, of which the second one will
            usually be the closing line. So, its just the closing line sitting
            on top of original lines. As the closing line, in this case, jhas
            the same 'end'-points but is just flipped in direction, they both
            are essentially the same lines. In this case, the closing line
            will be removed after the chack has confirmed the existence of
            such a closing line.
            """
            # Check for equal length
            point0, point1 = self.lines[1].coord_list
            same_endpoints = [self.lines[0].is_point_endpoint(point0),
                              self.lines[0].is_point_endpoint(point1),
                              ]
            if all(same_endpoints):
                """Retain the original lines and trim the closing line."""
                del self.lines[1]

    def plot(self, ax=None, connect_ends=False):
        coords = self.get_node_coords()
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(coords[:, 0], coords[:, 1], '-o')
        if connect_ends:
            if self.closed and len(coords) > 2:
                ax.plot(coords[0, :], coords[-2, :], '-o')
            elif not self.closed and len(coords) > 1:
                ax.plot(coords[0, :], coords[-1, :], '-o')
        return ax

    def check_overlaping_points(self, tolerance=1E-8):
        coords = self.get_node_coords()
        d = cdist(coords, coords)
        if np.argwhere(d <= tolerance).shape[0] > coords.shape[0]:
            return True
        else:
            return False

    def check_overlaping_lines(self):
        midpoints = [line.mid_point for line in self.lines]
        gradients = [line.gradient for line in self.lines]

    def smooth(self, max_smooth_level=2):
        """
        from upxo.geoEntities.mulsline2d import MSline2d as msl2d
        nodes = [Point2d(0.0,0.0),
                 Point2d(1.0,0.0),
                 Point2d(1.0,1.0),
                 Point2d(2.5,2.0),
                 Point2d(4.0,2.0),
                 Point2d(4.0,6.0),
                 Point2d(4.0,8.0),
                 Point2d(2.0,8.0)]
        ml = msl2d.by_nodes(nodes, close=False)
        ml.nodes
        ml.lines
        ax = ml.plot()
        ml.smooth()

        ml.nnodes
        max_smooth_level = 3
        coords = mean_coordinates(ml.coords, max_smooth_level)
        new_nodes = [Point2d(c[0], c[1]) for c in coords[1:-1]]
        new_nodes_full = [ml.nodes[0]] + new_nodes + [ml.nodes[-1]]

        new_lines = [sl2d(new_nodes_full[i].x, new_nodes_full[i].y,
                          new_nodes_full[i+1].x, new_nodes_full[i+1].y)
                     for i in range(len(new_nodes_full)-1)]

        ml.lines = new_lines
        ml.nodes = new_nodes_full


        ax.plot(coords[:, 0], coords[:, 1])
        """
        # ---------------------------------------------------
        smoothing_carried_out = True
        if self.nnodes > 4:
            coords = mean_coordinates(self.coords, max_smooth_level)
        elif self.nnodes == 4:
            coords = mean_coordinates(self.coords, 3)
        elif self.nnodes == 3:
            coords = mean_coordinates(self.coords, 2)
        else:
            smoothing_carried_out = False
        # ---------------------------------------------------
        if smoothing_carried_out:
            new_nodes = [Point2d(c[0], c[1]) for c in coords[1:-1]]
            new_nodes_full = [self.nodes[0]] + new_nodes + [self.nodes[-1]]

            new_lines = [sl2d(new_nodes_full[i].x, new_nodes_full[i].y,
                              new_nodes_full[i+1].x, new_nodes_full[i+1].y)
                         for i in range(len(new_nodes_full)-1)]
            # ---------------------------------------------------
            self.lines = new_lines
            self.nodes = new_nodes_full


class ring2d():
    """
    Import
    ------
    from upxo.geoEntities.mulsline2d import ring2d

    Example-1
    ---------
    from upxo.geoEntities.mulsline2d import MSline2d
    from upxo.geoEntities.sline2d import Sline2d as sl2d
    lines = [sl2d(0.0,0.0,1.0,1.0), sl2d(1.0,1.0,1.5,1.5), sl2d(1.5,1.5,2.5,2.5)]
    msl1 = MSline2d.from_lines(lines, close=False)
    lines = [sl2d(2.5,2.5,4.0,4.0), sl2d(4.0,4.0,4.0,6.0)]
    msl2 = MSline2d.from_lines(lines, close=False)
    lines = [sl2d(4.0,6.0,4.0,8.0), sl2d(4.0,8.0,10.0,10.0)]
    msl3 = MSline2d.from_lines(lines, close=False)
    lines = [sl2d(0,0,20,10), sl2d(20,10,10,10)]
    msl4 = MSline2d.from_lines(lines, close=False)
    # -----------------------------------------
    segs = [msl1, msl2, msl3, msl4]
    # -----------------------------------------
    segs[0].get_node_coords(), segs[0].lines, segs[0].nodes, segs[0].closed
    segs[1].get_node_coords(), segs[1].lines, segs[1].nodes, segs[1].closed
    segs[2].get_node_coords(), segs[2].lines, segs[2].nodes, segs[2].closed
    segs[3].get_node_coords(), segs[3].lines, segs[3].nodes, segs[3].closed
    # -----------------------------------------
    from upxo.geoEntities.mulsline2d import ring2d
    # -----------------------------------------
    R = ring2d(segs)
    R.conn0
    R.conn1
    R.assess_spatial_continuity()


    coords = R.segments[0].get_node_coords()
    d = distance_matrix(coords, coords)
    if np.argwhere(d<=1E-8).shape[0] > coords.shape[0]
    # -----------------------------------------
    """
    __slots__ = ('segments', 'segids', 'segflips', 'nsegs',
                 'coords', 'closed', 'conn0', 'conn1')

    EPS_coord_coincide = 1E-8

    def __init__(self, segments=None, segids=None, segflips=None):
        self.segments = segments
        self.segids = segids
        self.segflips = segflips
        self.nsegs = len(self.segments)
        #self.set_coords()
        # -----------------------
        #_closure_level_0_ = self.connectivity0()
        #self.conn0, last_seg_flipped, flip_possible = _closure_level_0_
        ## -----------------------
        #if self.conn0 or flip_possible:
        #    self.connectivity1()
        # -----------------------
        # . . . . . . .
        '''
        overlaps = self.assess_segment_point_overlaps(segments)  # False: Proceed
        continuities, flips = self.assess_spatial_continuity(segments)
        if not all(continuities):
            if self.assess_possibility_of_continuity(segments):
                self.get_continuity_enforcement_indices()
                self.set_spatial_continuity(enforce_indices)
            else:
                raise ValueError('Invalid segments morphology passed.')'''
        # --------------------------------------
        # NOw that validations have been done, we will proceed.

    def __repr__(self):
        return f'UPXO ring. nseg={len(self.segments)}. MID: {id(self)}'

    def add_segment_unsafe(self, segment):
        self.segments.append(segment)

    def add_segid(self, segid):
        self.segids.append(segid)

    def add_segflip(self, segflip):
        self.segflips.append(segflip)

    def check_closed(self):
        if self.segflips[self.segids.index(max(self.segids))]:
            START = self.segments[0].nodes[0]
            END = self.segments[max(self.segids)].nodes[-1]
        else:
            START = self.segments[0].nodes[0]
            END = self.segments[max(self.segids)].nodes[0]
        return START.eq_fast(END)[0]

        '''startseg, endseg = min(self.segids), max(self.segids)
        segflip = self.segflips[self.segids.index(endseg)]
        # ------------------------
        startnode = self.segments[startseg].nodes[0]
        if segflip:
            endnode = self.segments[endseg].nodes[-1]
        else:
            endnode = self.segments[endseg].nodes[0]
        return startnode.eq_fast(endnode)'''

    def close(self):
        self.segments[-1].nodes.append(self.segments[0].nodes[0])
        self.segments[-1].update_nodes()
        self.closed = True

    def connectivity0(self, flip_if_possible=True):
        '''
        Assess closure of 1st & last segment with option to close if possible.

        Return
        ------
        closed: True, if closd or has been closed.
        last_seg_flipped: True, if seg was originally found open but closed.
        flip_possible: Whether flipped or not, if True, indicates that
            the segments can be closed between first and last ones.
        '''
        closed, last_seg_flipped, flip_possible = False, False, False
        if self.segments[0].nodes[0].eq_fast(self.segments[-1].nodes[-1])[0]:
            closed = True
        elif self.segments[0].nodes[0].eq_fast(self.segments[-1].nodes[0])[0]:
            flip_possible = True
            if flip_if_possible:
                self.segments[-1].flip()
                closed, last_seg_flipped = True, True
        return closed, last_seg_flipped, flip_possible

    def connectivity1(self):
        '''
        Assess closure of all intermediate segments.
        '''
        self.conn1 = {}
        for i in range(self.nsegs):
            if i < self.nsegs-1:
                c = self.segments[i].nodes[-1].eq_fast(self.segments[i+1].nodes[0])[0]
                self.conn1[(i, i+1)] = c
            else:
                c = self.segments[i].nodes[-1].eq_fast(self.segments[0].nodes[0])[0]
                self.conn1[(i, 0)] = c

    def assess_segment_point_overlaps(self, line_check=False):
        ol_points = [seg.check_overlaping_points() for seg in self.segments]
        # ---------------------------
        line_check=False  # Currently not operational.
        if line_check:
            ol_lines = [seg.check_overlaping_lines() for seg in self.segments]

    def assess_reorder_requirement(self):
        if self.check_closed():
            self.close()
            # NO re-ordering needed
            self.check_sorted()
            pass
        else:
            # Flips may be needed
            pass

    def set_coords(self):
        self.coords = self.segments[0].get_node_coords()
        for seg in self.segments[1:]:
            self.coords = np.vstack((self.coords, seg.get_node_coords()[1:]))

    def get_coords(self):
        return self.create_coords_from_segments()

    @property
    def centroid(self):
        return np.mean(self.get_coords(), axis=0)

    def create_coords_from_segments(self, force_close=False):
        # segments = gbsegs
        coords = self.segments[0].get_node_coords()
        for i, seg in enumerate(self.segments[1:], start=1):
            if self.segflips[i]:
                thissegcoords = np.flip(seg.get_node_coords(), axis=0)
                coords = np.vstack((coords, thissegcoords[1:]))
            else:
                coords = np.vstack((coords, seg.get_node_coords()[1:]))
        if force_close:
            coords = self.force_close_coordinates(coords, assess_first=True)
        return coords

    def force_close_coordinates(self, coord, assess_first=True):
        """Unsafe. Not intended for user."""
        if assess_first:
            if abs((coord[0]-coord[-1]).sum()) > self.EPS_coord_coincide:
                print('Coord not closed. Force closing.')
                coord = np.vstack((coord, coord[0]))
            else:
                pass
        else:
            coord = np.vstack((coord, coord[0]))
        return coord

    def create_polygon_from_segments(self):
        coords = self.create_coords_from_segments()
        return Polygon(self.create_coords_from_segments())

    def create_polygon_from_coords(self):
        return Polygon(self.create_coords_from_segments())
    
    @property
    def area(self):
        return self.create_polygon_from_segments().area
    
    @property
    def perimeter(self):
        return self.create_polygon_from_segments().length
    
    @property
    def is_closed(self):
        return self.check_closed()
    
    @property
    def nsegments(self):
        return self.nsegs

    @property
    def ncoords(self):
        return self.create_coords_from_segments().shape[0]
    
    @property
    def tree(self):
        return cKDTree(self.create_coords_from_segments())

    def assess_spatial_continuity(self):
        """
        From list of multisline2d, multislines2d, do all (i+1)^th multisline2d
        follow i^th multisline2d? with or without the need for flips.

        Return
        ------
        continuity: final result, whethwr all make a chain or not, irrespective
            of the need for flip.
        flip_needed: True if any of the multisline2d in the list neede to be
            flipped to enure spatial continuity. The actual e,ements needing
            flips can be found in second element opf every
            subliest in i_precede_chain.
        i_precede_chain: list of do_i_precede results for every i:i+1 pair
        """
        i_precede_chain = []
        for i in range(len(self.segments)-1):
            if i == 0:
                i_precede_chain.append(self.segments[i].do_i_precede(self.segments[i+1]))
            else:
                thismsl = self.segments[i]
                nextmsl = self.segments[i+1]
                i_precede_chain.append(thismsl.do_i_precede(nextmsl))
        continuity = all([flag[0] for flag in i_precede_chain])
        flip_needed = any([flag[1] for flag in i_precede_chain])
        return continuity, flip_needed, i_precede_chain

    def plot_segs(self,
                  plot_centroid=False, centroid_text='',
                  plot_coord_order=False,
                  visualize_flip_req=False):
        fig, ax = plt.subplots()
        FS = 8
        for gbsegcount, gbseg in enumerate(self.segments, start=0):
            coords = gbseg.get_node_coords()
            color = np.random.random(3)
            if not visualize_flip_req:
                ax.plot(coords[:, 0], coords[:, 1], color=color,
                        linewidth=3)
            if visualize_flip_req:
                if self.segflips[gbsegcount]:
                    ax.plot(coords[:, 0], coords[:, 1], color=color,
                            linewidth=3, linestyle='--')
                else:
                    ax.plot(coords[:, 0], coords[:, 1], color=color,
                            linewidth=3)
            centroid = gbseg.centroid
            ax.plot(centroid[0], centroid[1], 'kx')
            ax.text(centroid[0], centroid[1], gbsegcount, color='red',
                    fontsize=12, fontweight='bold')
            fs = FS + (gbsegcount % 2)*2
            offset = 0.1*(gbsegcount % 2)
            for coord_count, coord in enumerate(coords, start=0):
                ax.text(coord[0]+offset, coord[1]+offset,
                        coord_count, color=color, fontsize=fs)
        if plot_centroid:
            centroid = self.centroid
            plt.plot(centroid[0], centroid[1], 'k+',
                     ms=15, mfc='black', mew=2, alpha=1.0)
            plt.text(centroid[0], centroid[1], str(centroid_text),
                     fontsize=15, color='orange', fontweight='bold')
        if plot_coord_order:
            C = self.create_coords_from_segments()
            ax.plot(C[:,0], C[:,1], '-.k', linewidth=0.75)
        return ax

    def get_coords_newdef(self):
        coordinates = []
        for gbsegcount, gbseg in enumerate(self.segments, start=0):
            coords = gbseg.get_node_coords()
            coordinates.append(coords)
        C = self.create_coords_from_segments()
        return coordinates, C

class mulring2d():
    __slots__ = ('rings', 'jp', 'ip')

    """
    Explanations
    ------------
    'rings': individual ring objects
    'jp': Junction points
    'ip': All interface points
    'ippure': Pure interface points = {ip} - {jp}

    # Requirements
    1. Provide a self-sustained
    """

    def __init__(self, rings):
        self.rings = rings

    def build_points_list(self):
        pass

    def set_coords(self):
        pass

    def export_abaqus_for_meshing(self):
        pass

    def mesh(self):
        pass
