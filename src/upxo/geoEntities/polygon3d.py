# -*- coding: utf-8 -*-
"""
3D polygon geometric entity for UPXO.

Usage
-----
    from upxo.geoEntities.polygon3d import Polygon3d

Classes
-------
Polygon3d : 3D planar polygon with projection support.

Limitations
-----------
- ``project`` method is a stub awaiting implementation.
"""
import numpy as np
import shapely


class Polygon3d():
    """
    Represents a planar polygon in 3D space.

    Parameters
    ----------
    points : array-like
        Ordered list of 3D vertex coordinates defining the polygon boundary.
    be : object
        Boundary entity (e.g., list of edge objects) associated with this polygon.

    Attributes
    ----------
    points : array-like
    be : object
    """

    __slots__ = ('points', 'be')

    def __init__(self, points, be):
        """
        Initialise a 3D polygon from vertices and boundary entity.

        Parameters
        ----------
        points : array-like
            Ordered 3D vertex coordinates.
        be : object
            Boundary entity.
        """
        self.points = points
        self.be = be

    def project(self, plane='xy'):
        """
        Project this 3D polygon onto a 2D plane.

        Parameters
        ----------
        plane : str, optional
            Target projection plane: ``'xy'``, ``'yz'``, or ``'xz'``.
            Default is ``'xy'``.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        raise NotImplementedError("project is not yet implemented.")
