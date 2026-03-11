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
from upxo.geoEntities.sline3d import Sline3d as sl3d

class MSline3d():
    """
    UPXO code class.

    Examples
    --------
    from upxo.geoEntities.mulsline3d import MSline3d as msl3d
    from upxo.geoEntities.sline3d import Sline3d as sl3d
    e0 = sl3d(0.0,0.0,0.0, 1.0,1.0,1.0)
    e1 = sl3d(1.0,1.0,1.0, 1.5,1.5,0.0)
    e2 = sl3d(1.5,1.5,0.0, 2.5,2.5,3.0)
    e3 = sl3d(2.5,2.5,3.0, 4.0,4.0,3.5)
    e4 = sl3d(4.0,4.0,3.5, 4.0,6.0,3.5)

    me = msl3d([e0, e1, e2, e3, e4])
    """
    __slots__ = ('lines', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'f', 'closed')

    def __init__(self, llist):
        self.lines = llist

    def __repr__(self):
        return f"UPXO MSline3d. n={len(self.lines)}. ID: {id(self)}"

    @classmethod
    def from_lines(cls, llist, close=True):
        """
        Make a multi-straight line in 3d.

        Make mul-straight-line-3d using multiple lines, strt and connectvity
        specification. Recommended way to make the msl3d.

        Development phases
        ------------------
        Phase-1: Basic working codes with basic validations. DONE
        Phase-2: Include additional validations.

        Examples
        --------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        e0 = sl3d(0.0,0.0,0.0, 1.0,1.0,1.0)
        e1 = sl3d(1.0,1.0,1.0, 1.5,1.5,0.0)
        e2 = sl3d(1.5,1.5,0.0, 2.5,2.5,3.0)
        e3 = sl3d(2.5,2.5,3.0, 4.0,4.0,3.5)
        e4 = sl3d(4.0,4.0,3.5, 4.0,6.0,3.5)

        lines = [e0,e1,e2,e3,e4]
        me = msl3d.from_lines(lines, close=True)
        me.lines

        lines = [e0,e1,e2,e3,e4]
        me = msl3d.from_lines(lines, close=False)
        me.lines
        """
        lines = llist
        if close:
            fl = llist[0]  # First line
            ll = llist[-1]  # Last line
            lines.append(sl3d(ll.x1, ll.y1, ll.z1, fl.x0, fl.y0, fl.z0))
        return cls(lines)

    @classmethod
    def by_walk(self, var_l='constant', var_ang='constant',
                specs = {'n': 5,
                         'max_total_length': 10,
                         'min_total_length': 8,
                         'mean_length': 1,}
                ):
        pass

    @property
    def n(self):
        """Return number of lines."""
        return len(self.lines)

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
    def nodes(self):
        """Return unique list of nodes in the multi-straight-line object."""
        nodes = [[line.x0, line.y0, line.z0] for line in self.lines]
        nodes += [[line.x1, line.y1, line.z1] for line in self.lines]
        return np.unique(nodes, axis=0)

    @property
    def mid_nodes(self):
        """
        Return mid-sode nodes of all lines.

        Example
        -------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0), sl3d(2.5,2.5,3.0,4.0,4.0,3.5),
                 sl3d(4.0,4.0,3.5,4.0,6.0,3.5)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.mid_nodes
        """
        return [line.mid for line in self.lines]

    @property
    def line_ids(self):
        """Return memory id of each line."""
        return [id(line) for line in self.lines]

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
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0), sl3d(2.5,2.5,3.0,4.0,4.0,3.5),
                 sl3d(4.0,4.0,3.5,4.0,6.0,3.5)]
        MULLINE = msl3d.from_lines(lines, close=True)
        points = np.random.random((2,3))
        MULLINE.distances_nodes(points)
        """
        # Validation
        # -------------------------------------
        nodes = np.array(self.nodes)
        """
        points = np.random.random((10,3))
        nodes = np.random.random((4, 3))
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
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0)]
        MULLINE = msl3d.from_lines(lines, close=True)
        point = np.random.random(3)*np.random.randint(10)
        MULLINE.find_closest_nodes(point)

        Example-2
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0)]
        MULLINE = msl3d.from_lines(lines, close=True)
        point = lines[0].mid
        MULLINE.find_closest_nodes(point)
        """
        # Validation
        # -------------------------------------
        distances = self.distances_nodes(np.array(point)[:, np.newaxis].T).T.squeeze()
        closest_points = np.argwhere(distances == distances.min()).T.squeeze().tolist()
        return closest_points

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
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        e0 = sl3d(0.0,0.0,0.0, 1.0,1.0,1.0)
        e1 = sl3d(1.0,1.0,1.0, 1.5,1.5,0.0)
        e2 = sl3d(1.5,1.5,0.0, 2.5,2.5,3.0)
        e3 = sl3d(2.5,2.5,3.0, 4.0,4.0,3.5)
        e4 = sl3d(4.0,4.0,3.5, 4.0,6.0,3.5)

        lines = [e0,e1,e2,e3,e4]

        me = msl3d.from_lines(lines, close=True)
        me.lines

        me = msl3d.from_lines(lines, close=False)
        me.lines

        me.sub_divide(line_number=0, f=0.25)
        me.lines

        me.sub_divide(line_number=len(me.lines), f=0.25)
        me.lines

        me.sub_divide(line_number=3, f=0.50)
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
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0), sl3d(2.5,2.5,3.0,4.0,4.0,3.5),
                 sl3d(4.0,4.0,3.5,4.0,6.0,3.5)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.lines

        for line in MULLINE.lines: print(id(line))
        MULLINE.remove_point_by_index(index=2, remove='previous_line')
        for line in MULLINE.lines: print(id(line))

        MULLINE.lines

        Example-2
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0), sl3d(2.5,2.5,3.0,4.0,4.0,3.5),
                 sl3d(4.0,4.0,3.5,4.0,6.0,3.5)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.lines

        for line in MULLINE.lines: print(id(line))
        MULLINE.remove_point_by_index(index=2, remove='next_line')
        for line in MULLINE.lines: print(id(line))

        MULLINE.lines

        Example-3
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0), sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0), sl3d(2.5,2.5,3.0,4.0,4.0,3.5),
                 sl3d(4.0,4.0,3.5,4.0,6.0,3.5)]
        MULLINE = msl3d.from_lines(lines, close=True)
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
            x0, y0, z0 = self.lines[previous_line].coord_i
            x1, y1, z1 = self.lines[next_line].coord_j
            new_line = sl3d(x0, y0, z0, x1, y1, z1)
            self.lines = self.lines[:previous_line] + [new_line] + self.lines[next_line+1:]
        else:
            raise ValueError('Invalid update specirfication.')

    def remove_point_by_location(self, location=(None,None,None),
                                 remove='previous_line'):
        """
        Example-1
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0),
                 sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.lines
        MULLINE.line_ids
        MULLINE.remove_point_by_location(location=lines[0].coord_i, remove='previous_line')
        MULLINE.lines
        MULLINE.line_ids

        Example-2
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0),
                 sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.lines
        MULLINE.line_ids
        MULLINE.remove_point_by_location(location=lines[0].mid, remove='previous_line')
        MULLINE.lines
        MULLINE.line_ids

        Example-3
        ---------
        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        lines = [sl3d(0.0,0.0,0.0,1.0,1.0,1.0),
                 sl3d(1.0,1.0,1.0,1.5,1.5,0.0),
                 sl3d(1.5,1.5,0.0,2.5,2.5,3.0)]
        MULLINE = msl3d.from_lines(lines, close=True)
        MULLINE.lines
        location = np.random.random(3)*np.random.randint(10)
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
