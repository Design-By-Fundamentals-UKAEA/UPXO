"""
Multi-straight-line 3D geometric entity module for UPXO.

Provides ``MSline3d``, a collection of ordered ``Sline3d`` segments
representing an open or closed 3-D polyline.  Supports construction from
line lists, node/length queries, spatial distance calculations, subdivision,
and node removal operations.

Applications
------------
- Non-conformal to conformal geometry conversion.
- Hierarchical grain structure feature generation.
- General 3-D polyline geometry operations.

Classes
-------
MSline3d
    Ordered collection of ``Sline3d`` segments forming a 3-D polyline.

Coordinate system
-----------------
::

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

Usage
-----
::

    from upxo.geoEntities.mulsline3d import MSline3d as msl3d
    from upxo.geoEntities.sline3d import Sline3d as sl3d

@author: Dr. Sunil Anandatheertha
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
    """Ordered collection of ``Sline3d`` segments forming a 3-D polyline.

    Wraps a list of connected ``Sline3d`` objects and provides aggregate
    geometric properties (total length, mean length, node positions) and
    editing operations (subdivision, node removal).

    Attributes
    ----------
    lines : list of Sline3d
        Ordered sequence of 3-D line segments.

    Import
    ------
    ::

        from upxo.geoEntities.mulsline3d import MSline3d as msl3d

    Examples
    --------
    .. code-block:: python

        from upxo.geoEntities.mulsline3d import MSline3d as msl3d
        from upxo.geoEntities.sline3d import Sline3d as sl3d
        e0 = sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        e1 = sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0)
        e2 = sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)
        e3 = sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5)
        e4 = sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)
        me = msl3d([e0, e1, e2, e3, e4])
        print(me.n, me.length)
    """
    __slots__ = ('lines', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'f', 'closed')

    def __init__(self, llist):
        """Initialise from a list of ``Sline3d`` objects."""
        self.lines = llist

    def __repr__(self):
        """Return ``UPXO MSline3d. n=<N>. ID: <id>`` summary string."""
        return f"UPXO MSline3d. n={len(self.lines)}. ID: {id(self)}"

    @classmethod
    def from_lines(cls, llist, close=True):
        """Construct a ``MSline3d`` from a list of ``Sline3d`` objects.

        Parameters
        ----------
        llist : list of Sline3d
            Ordered sequence of 3-D line segments.
        close : bool, optional
            If ``True``, append a closing segment from the last endpoint back
            to the first.  Default is ``True``.

        Returns
        -------
        MSline3d
            New multi-line with optional closing segment appended.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0, 0, 0, 1, 1, 1), sl3d(1, 1, 1, 2, 2, 2)]
            me = msl3d.from_lines(lines, close=True)
            print(me.n)
        """
        lines = llist
        if close:
            fl = llist[0]
            ll = llist[-1]
            lines.append(sl3d(ll.x1, ll.y1, ll.z1, fl.x0, fl.y0, fl.z0))
        return cls(lines)

    @classmethod
    def by_walk(self, var_l='constant', var_ang='constant',
                specs={'n': 5,
                       'max_total_length': 10,
                       'min_total_length': 8,
                       'mean_length': 1}):
        """Construct a multi-line by random walk with length/angle variation. Not yet implemented."""
        raise NotImplementedError("by_walk is not yet implemented.")

    @property
    def n(self):
        """Number of line segments in the collection."""
        return len(self.lines)

    @property
    def lengths(self):
        """Lengths of individual segments in order.

        Returns
        -------
        list of float
            Length of each ``Sline3d`` in ``self.lines``.

        Examples
        --------
        .. code-block:: python

            me.lengths
        """
        return [line.length for line in self.lines]

    @property
    def length(self):
        """Total length of all segments combined.

        Returns
        -------
        float
            Sum of all individual segment lengths.

        Examples
        --------
        .. code-block:: python

            me.length
        """
        return sum(self.lengths)

    @property
    def length_mean(self):
        """Mean segment length.

        Returns
        -------
        float
            ``total_length / n``.

        Examples
        --------
        .. code-block:: python

            me.length_mean
        """
        return self.length/self.n

    @property
    def gradients(self):
        """Gradient ``[dx/dz, dy/dz]`` of every segment.

        Returns
        -------
        list of list of float
            Per-segment gradient pairs.
        """
        return [line.gradient for line in self.lines]

    @property
    def nodes(self):
        """Unique list of nodes (start and end points) across all segments.

        Returns
        -------
        numpy.ndarray, shape (M, 3)
            Unique ``[x, y, z]`` node coordinates.
        """
        nodes = [[line.x0, line.y0, line.z0] for line in self.lines]
        nodes += [[line.x1, line.y1, line.z1] for line in self.lines]
        return np.unique(nodes, axis=0)

    @property
    def mid_nodes(self):
        """Midpoint coordinates of all segments.

        Returns
        -------
        list of tuple
            Midpoint ``(xmid, ymid, zmid)`` for each segment in order.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0),
                     sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5),
                     sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)]
            MULLINE = msl3d.from_lines(lines, close=True)
            print(MULLINE.mid_nodes)
        """
        return [line.mid for line in self.lines]

    @property
    def line_ids(self):
        """Memory id of each segment.

        Returns
        -------
        list of int
            ``id(line)`` for each segment in ``self.lines``.
        """
        return [id(line) for line in self.lines]

    def unclose(self):
        """Remove the closing segment (last element of ``self.lines``) in place."""
        del self.lines[-1]

    def distances_nodes(self, points):
        """Compute distances from every node to each of the given points.

        Parameters
        ----------
        points : numpy.ndarray, shape (M, 3)
            Query points to measure distances from.

        Returns
        -------
        numpy.ndarray, shape (N_nodes, M)
            Distance from each node (row) to each query point (column).

        Notes
        -----
        Uses vectorised broadcasting:
        ``points[:, np.newaxis]`` shapes to ``(M, 1, 3)`` so that
        subtraction with ``nodes`` (shape ``(N, 3)``) yields ``(M, N, 3)``
        differences, squared-summed over axis 2, then transposed to
        ``(N_nodes, M)``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0),
                     sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5),
                     sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)]
            MULLINE = msl3d.from_lines(lines, close=True)
            points = np.random.random((2, 3))
            MULLINE.distances_nodes(points)
        """
        nodes = np.array(self.nodes)
        distances = np.sqrt(np.sum((points[:, np.newaxis] - nodes) ** 2,
                                   axis=2)).T
        return distances

    def find_closest_nodes(self, point):
        """Find the index (or indices) of the node(s) closest to ``point``.

        Parameters
        ----------
        point : array-like, shape (3,)
            Query coordinate ``[x, y, z]``.

        Returns
        -------
        int or list of int
            Index (or indices) into ``self.nodes`` of the nearest node(s).

        Examples
        --------
        **Example 1** — random query point:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)]
            MULLINE = msl3d.from_lines(lines, close=True)
            point = np.random.random(3) * np.random.randint(10)
            MULLINE.find_closest_nodes(point)

        **Example 2** — midpoint of the first segment:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)]
            MULLINE = msl3d.from_lines(lines, close=True)
            point = lines[0].mid
            MULLINE.find_closest_nodes(point)
        """
        distances = self.distances_nodes(np.array(point)[:, np.newaxis].T).T.squeeze()
        closest_points = np.argwhere(distances == distances.min()).T.squeeze().tolist()
        return closest_points

    def sub_divide(self, line_number=0, f=0.5):
        """Sub-divide a single segment at a fractional position.

        Replaces the segment at ``line_number`` with two shorter segments
        split at the fractional position ``f``.

        Parameters
        ----------
        line_number : int, optional
            Zero-based index of the segment to split.  Default is 0.
        f : float, optional
            Fractional split position in (0, 1).  Default is 0.5.

        Returns
        -------
        None
            Modifies ``self.lines`` in place.

        Raises
        ------
        TypeError
            If ``line_number`` is not an integer.
        ValueError
            If ``line_number`` exceeds the number of segments.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            e0 = sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            e1 = sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0)
            e2 = sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)
            e3 = sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5)
            e4 = sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)
            me = msl3d.from_lines([e0, e1, e2, e3, e4], close=False)
            me.sub_divide(line_number=0, f=0.25)
            me.sub_divide(line_number=len(me.lines), f=0.25)
            me.sub_divide(line_number=3, f=0.50)
        """
        if not isinstance(line_number, int):
            raise TypeError('Invalid line number type.')
        if line_number > len(self.lines):
            raise ValueError('Invalid line number specification.')
        if line_number == 0:
            new_lines = self.lines[0].split(f=f, saa=False, throw=True)
            self.lines = new_lines + self.lines[1:]
        elif line_number == len(self.lines):
            new_lines = self.lines[-1].split(f=f, saa=False, throw=True)
            self.lines = self.lines[:-1] + new_lines
        else:
            left = [line for line in self.lines[:line_number]]
            line01 = self.lines[line_number].split(f=f, saa=False, throw=True)
            right = [line for line in self.lines[line_number+1:]]
            self.lines = left + line01 + right

    def remove_point_by_index(self, index=2, remove='previous_line'):
        """Remove a node by index, merging the adjacent segments.

        Parameters
        ----------
        index : int, optional
            Zero-based index of the node (shared endpoint) to remove.
            Default is 2.
        remove : {'previous_line', 'next_line', 'both'}, optional
            How to handle the adjacent segments:

            * ``'previous_line'`` — extend the next segment back to the
              previous segment's start point and delete the previous segment.
            * ``'next_line'`` — extend the previous segment forward to the
              next segment's end point and delete the next segment.
            * ``'both'`` — replace both adjacent segments with a single new
              segment connecting the previous segment's start to the next
              segment's end.

            Default is ``'previous_line'``.

        Returns
        -------
        None
            Modifies ``self.lines`` in place.

        Examples
        --------
        **Example 1** — remove previous segment:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0),
                     sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5),
                     sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)]
            MULLINE = msl3d.from_lines(lines, close=True)
            MULLINE.remove_point_by_index(index=2, remove='previous_line')
            print(MULLINE.n)

        **Example 2** — remove next segment:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0),
                     sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5),
                     sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)]
            MULLINE = msl3d.from_lines(lines, close=True)
            MULLINE.remove_point_by_index(index=2, remove='next_line')

        **Example 3** — replace both adjacent segments with a single new segment:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0),
                     sl3d(2.5, 2.5, 3.0, 4.0, 4.0, 3.5),
                     sl3d(4.0, 4.0, 3.5, 4.0, 6.0, 3.5)]
            MULLINE = msl3d.from_lines(lines, close=True)
            MULLINE.remove_point_by_index(index=2, remove='both')
        """
        previous_line, next_line = index-1, index
        if remove == 'previous_line':
            self.lines[next_line].move_i(self.lines[previous_line].coord_i)
            del self.lines[previous_line]
        elif remove == 'next_line':
            self.lines[previous_line].move_j(self.lines[next_line].coord_j)
            del self.lines[next_line]
        elif remove == 'both':
            x0, y0, z0 = self.lines[previous_line].coord_i
            x1, y1, z1 = self.lines[next_line].coord_j
            new_line = sl3d(x0, y0, z0, x1, y1, z1)
            self.lines = self.lines[:previous_line] + [new_line] + self.lines[next_line+1:]
        else:
            raise ValueError('Invalid update specification.')

    def remove_point_by_location(self, location=(None, None, None),
                                 remove='previous_line'):
        """Remove a node nearest to ``location``, merging the adjacent segments.

        Finds the closest node to ``location`` using :meth:`find_closest_nodes`
        and delegates to :meth:`remove_point_by_index`.

        Parameters
        ----------
        location : array-like, shape (3,), optional
            Target ``[x, y, z]`` coordinate.  The nearest node is found and
            removed.  Default is ``(None, None, None)``.
        remove : {'previous_line', 'next_line', 'both'}, optional
            See :meth:`remove_point_by_index`.  Default is
            ``'previous_line'``.

        Returns
        -------
        None
            Modifies ``self.lines`` in place.

        Notes
        -----
        When only 2 segments remain after removal and the second is a
        redundant closing segment (same endpoints as the first but reversed),
        the closing segment is automatically deleted.

        Examples
        --------
        **Example 1** — remove by a known endpoint:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)]
            MULLINE = msl3d.from_lines(lines, close=True)
            MULLINE.remove_point_by_location(location=lines[0].coord_i,
                                             remove='previous_line')

        **Example 2** — remove by a segment midpoint:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)]
            MULLINE = msl3d.from_lines(lines, close=True)
            MULLINE.remove_point_by_location(location=lines[0].mid,
                                             remove='previous_line')

        **Example 3** — remove by a random location:

        .. code-block:: python

            from upxo.geoEntities.mulsline3d import MSline3d as msl3d
            from upxo.geoEntities.sline3d import Sline3d as sl3d
            import numpy as np
            lines = [sl3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                     sl3d(1.0, 1.0, 1.0, 1.5, 1.5, 0.0),
                     sl3d(1.5, 1.5, 0.0, 2.5, 2.5, 3.0)]
            MULLINE = msl3d.from_lines(lines, close=True)
            location = np.random.random(3) * np.random.randint(10)
            MULLINE.remove_point_by_location(location=location,
                                             remove='previous_line')
        """
        if self.n == 1:
            return
        indices = self.find_closest_nodes(location)
        if isinstance(indices, int):
            self.remove_point_by_index(index=indices, remove=remove)
        elif isinstance(indices, list) and len(indices) > 1:
            self.remove_point_by_index(index=indices[0], remove=remove)
            for i, index in enumerate(indices[1:]):
                self.remove_point_by_location(location=location, remove=remove)
        if self.n == 2:
            point0, point1 = self.lines[1].coord_list
            same_endpoints = [self.lines[0].is_point_endpoint(point0),
                              self.lines[0].is_point_endpoint(point1)]
            if all(same_endpoints):
                del self.lines[1]
