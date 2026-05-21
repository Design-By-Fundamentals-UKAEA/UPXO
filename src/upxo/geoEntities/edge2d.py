"""
2D (and 3D) edge geometric entity classes for UPXO.

Import
------
    from upxo.geoEntities.edge2d import edge2d
    from upxo.geoEntities.edge2d import edge2d_lean_highest
    from upxo.geoEntities.edge2d import edge2d_from_point2d

Classes
-------
edge2d
    Core 2D edge entity with full geometric and comparison operations.
edge2d_lean_highest
    Lean-mode 2D edge with reduced memory footprint.
edge2d_from_point2d
    2D edge constructed from ``Point2d`` endpoint objects.
edge3d
    3D edge entity (stub; development in progress).

Dependencies
------------
    numpy, matplotlib, math, copy,
    upxo._sup.dataTypeHandlers,
    upxo.geoEntities.mulpoint2d,
    upxo.geoEntities.point2d

Notes
-----
This is a UPXO core module. The ``edge2d`` class uses secondary class-level
epsilon constants derived from ``dataTypeHandlers.constants``; primary
global values must be changed through that module, not through
instance attributes.

Limitations
-----------
- ``edge3d`` is not yet implemented.
- Shapely, VTK, PyVista, and SciPy construction branches are stubs.
"""
from numpy.random import uniform as npru
from numpy import inf as INFINITY
from upxo._sup import dataTypeHandlers as dth
from upxo._sup.dataTypeHandlers import constants as k
from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d
# from mulpoint2d import mulpoint2d
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
from upxo.geoEntities.point2d import Point2d as point2d
from math import ceil, floor, nan
from itertools import islice
import numpy as np
import types
import math

np.seterr(divide='ignore')
# Script information for the file.
__name__ = "UPXO-edge"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.5: from.111222", "0.6: from.151222", "0.7: from.181222",
               "0.8.from20012023.git.yes"]
__license__ = "GPL v3"


class edge2d():
    """
    Core 2D edge entity in UPXO.

    Represents a straight line segment in 2D Cartesian space defined by
    two endpoint ``point2d`` objects. Provides geometric operations
    (length, slope, angle, centroid, intersection), comparison utilities,
    and construction routes from various input formats.

    Import
    ------
        from upxo.geoEntities.edge2d import edge2d

    Parameters
    ----------
    method : str, optional
        Construction route. Accepted values:

        ========================  =====================================
        ``'up2d'`` (default)      Two ``point2d`` objects via ``pnta`` / ``pntb``.
        ``'xy_list'``             Two lists ``xlist``, ``ylist``.
        ``'cpairs'``              Coordinate pairs ``[[x0,y0],[x1,y1]]``.
        ========================  =====================================

    pnta, pntb : point2d, optional
        Start and end point objects. Required when ``method='up2d'``.
    xlist, ylist : list of float, optional
        X and Y coordinate lists. Required when ``method='xy_list'``.
    cpairs : list of list, optional
        Coordinate pairs. Required when ``method='cpairs'``.
    edge_lean : {'leanest', 'ignore', 'no', 'notlean'}, optional
        Lean mode for the edge object. Default ``'leanest'``.
    points_lean : str, optional
        Lean mode for the endpoint objects. Default ``'ignore'``.
    tlen : float, optional
        Tolerance for length comparisons. Default ``0.0``.
    calc_slope : bool, optional
        If ``True``, compute slope on construction. Default ``True``.

    Attributes
    ----------
    pnta, pntb : point2d
        Start and end endpoint objects.
    length : float
        Euclidean length of the edge.
    slope : float
        Geometric slope (dy/dx). ``inf`` for vertical edges.
    angrad : float
        Angle with x-axis in radians.
    angdeg : float
        Angle with x-axis in degrees.
    xycen : array-like
        Centroid coordinates ``[x, y]``.
    edge_lean : str
        Active lean mode.
    tlen : float
        Length tolerance.

    Notes
    -----
    Common method arguments across the class API:

    - ``saa`` — save as attribute (only if the slot exists).
    - ``uto`` / ``update`` — update the object in-place.
    - ``throw`` — return the computation result.
    - ``edge_lean`` / ``point_lean`` — leanness of edge/point objects.
    - ``pnta`` / ``pntb`` — start and end points.
    - ``method`` — construction/computation branch selector.

    Limitations
    -----------
    - Shapely, VTK, PyVista, SciPy, and vedo construction branches are
      stubs and not yet implemented.
    - 3D edge support (``edge3d``) is a separate, incomplete class.

    Examples
    --------
    Construct from coordinate lists:

    >>> from upxo.geoEntities.edge2d import edge2d
    >>> e = edge2d(method='xy_list', xlist=[0.0, 1.0], ylist=[0.0, 0.0])
    >>> e.length
    1.0

    Construct from coordinate pairs:

    >>> e = edge2d(method='cpairs', cpairs=[[0, 0], [1, 1]])
    >>> e.slope
    1.0
    """
    # Secondary global EPS-e2d-length assignment from primary source
    # METHODS EXIT TO CHANGE LOCALLY ONLY AND NOT GLOBALLY
    # PRIMARY GLOBAL VALUE CHANGE MUST HAPPEN THROUGH 'constant' class
    # FROM WITHIN datatype_handlers MODULE. CHANGES ALLOWED FROM WITHIN
    # THE PRESENT CLASS APPLY TO SECONDZRY GLOBAL VALUES, WHICH IS
    # EPS_e2dl_low and EPS_e2dl_high AND NOT TO constants.EPS_e2dl_low
    # and constants.EPS_e2dl_high, WHICH ARE PRIMARY GLOBAL VALUES
    # EPS_e2dl_low MUST BE USED AS LOCALS AND NOT AS GLOBALS, I.E.
    # AS edge2d.EPS_e2dl_low and edge2d.EPS_e2dl_high AND NOT AS
    # self.EPS_e2dl_low and self.EPS_e2dl_high
    EPS_e2dl_low = k.EPS_e2dl_low
    EPS_e2dl_high = k.EPS_e2dl_high

    # Secondary global EPS-e2d-slope-extremes assignment from primary source
    # METHODS EXIT TO CHANGE LOCALLY ONLY AND NOT GLOBALLY
    # PRIMARY GLOBAL VALUE CHANGE MUST HAPPEN THROUGH 'constant' class
    # FROM WITHIN datatype_handlers MODULE. CHANGES ALLOWED FROM WITHIN
    # THE PRESENT CLASS APPLY TO SECONDZRY GLOBAL VALUES, WHICH IS
    # EPS_e2ds_lowest and EPS_e2ds_highest AND NOT TO constants.EPS_e2ds_lowest
    # and constants.EPS_e2ds_highest, WHICH ARE PRIMARY GLOBAL VALUES
    # EPS_e2ds_lowest MUST BE USED AS LOCALS AND NOT AS GLOBALS, I.E.
    # AS EPS_e2ds_lowest and EPS_e2ds_highest AND NOT AS
    # self.EPS_e2ds_low and self.EPS_e2ds_high
    EPS_e2ds_lowest = abs(k.EPS_e2ds_lowest)
    EPS_e2ds_highest = abs(k.EPS_e2ds_highest)

    # Primary global for slope limits. These define slope bounds of the
    # present edge2d. Slope of the currentedge are referred as RS in
    # comparision operations. NOTE: ONLY POSITIVE VALUE ALLOWED
    EPS_RS_low = 0.001  # MUST BE POSI5TIVE
    EPS_RS_high = 0.001  # MUST BE POSITIVE

    # Default slope low and high of the OTHER object, that is the one
    # being compared against. Will be used only when the OTHER object
    # does not contain it as attributes.
    # NOTE: THESE ARE NOT USED CURRENTLY.
    # TO BE RETAINED IN ALL DEVELOPMENTS and NOT TO BE DEPRECATED.
    # EPS_OS_low = 0.001  # MUST BE POSITIVE
    # EPS_OS_high = 0.001  # MUST BE POSITIVE

    # Secondary global EPS-e2d-delc assignment from primary source
    # 'delx': difference in coordinates. Applies seperately to x and y
    # METHODS EXIST TO CHANGE SECONDARY GLOBAL.
    EPS_e2d_delx = k.EPS_e2d_delx
    EPS_e2d_dely = k.EPS_e2d_dely

    __slots__ = ('mid',  # memory address id
                 'tlen',  # tolerance length
                 'tdist',  # tolerance distance
                 'tang',  # tolerance angle
                 'edge_lean',  # edge lean
                 'points_lean',  # points lean
                 'pnta',  # End point A
                 'pntb',  # End point B
                 'edges', # Stores a list of edges <---- entry to MULEDGE2D
                 '__x',  # x-coordinate of all the points
                 '__y',  # y-coordinate of all the points
                 'm',  # multi-point object of the two points
                 'xycen',  # x-y coordinates  of the centre point
                 'length',  # edge length
                 'edge_type',  # Type of the edge
                 'angrad',  # angle with x in rad
                 'angdeg',  # angle with x in deg
                 'slope',  # geometric slope of the edge object
                 'mscoord',  # mesh seed point coordinates
                 'sfv_mori',  # misorienetation details as a list
                 )

    def __init__(self,
                 method='up2d',
                 pnta=None, pntb=None,  # method = 'up2d'
                 xlist=[0.0, 1.0], ylist=[0.0, 1.0],  # method = 'xy_list'. DONE
                 xcoords=[0.0, 1.0], ycoords=[0.0, 1.0],  # method = ''
                 cpairs=[[0, 0], [1, 0]],  # method = ''
                 cpairs_list=[[[0, 0], [1, 0]],[[1, 0], [2, 2]]],  # method = ''
                 end_points=None,  # method = ''
                 edge_lean='leanest',
                 points_lean='ignore',
                 tlen=0.0,
                 calc_slope=True,
                 ):
        """
        Parameters
        ----------
        method : str, optional
            Construction strategy. ``'up2d'`` (default) builds from two
            ``Point2d`` objects; ``'xy_list'`` from ``xlist``/``ylist``;
            ``'cpairs'`` from a coordinate pair; ``'cpairs_list'`` from a
            list of pairs.
        pnta : Point2d, optional
            Start endpoint (used when ``method='up2d'``).
        pntb : Point2d, optional
            End endpoint (used when ``method='up2d'``).
        xlist : list of float, optional
            x-coordinates ``[xa, xb]`` (used when ``method='xy_list'``).
        ylist : list of float, optional
            y-coordinates ``[ya, yb]`` (used when ``method='xy_list'``).
        cpairs : list, optional
            Coordinate pair ``[[xa, ya], [xb, yb]]`` (used when
            ``method='cpairs'``).
        end_points : list, optional
            Alternate endpoint specification. Default ``None``.
        edge_lean : str, optional
            Lean of the edge object. Default ``'leanest'``.
        points_lean : str, optional
            Lean applied to the created endpoint objects. Default ``'ignore'``.
        tlen : float, optional
            Target length hint (informational). Default ``0.0``.
        calc_slope : bool, optional
            Compute slope during initialisation. Default ``True``.
        """
        # Associate a few attributes
        if edge_lean in ('ignore', 'leanest', 'no', 'notlean'):
            self.points_lean = points_lean
            self.edge_lean = edge_lean
            self.tlen = tlen
            self.mid = id(self)
            # Make representations of coordinates and end point objects
            if method.lower() in dth.opt.coord_list:
                '''
                BRANCH OPTIONS: any in ('xy_list', 'xylist',)
                preferred branch option: 'xy_list'
                DATA:
                    xlist, ylist = [0.57, 0.87], [0.69, 0.46]
                EXAMPLE CALL:
                    edge2d(method='xy_list', xlist=xlist, ylist=ylist)
                '''
                self.__x, self.__y = xlist, ylist
                self.pnta = point2d(x=xlist[0], y=ylist[0],
                                    lean=points_lean)
                self.pntb = point2d(x=xlist[1], y=ylist[1],
                                    lean=points_lean)
                self.calculate_level01_basics()
            if method.lower() in dth.opt.coord_pairs:
                """
                BRANCH: any in ('cpairs')
                preferred branch option: 'cpairs'
                DATA:
                    cpairs = [[0.1685, 0.674], [1.0385, 0.79846]]
                EXAMPLE CALL:
                    edge2d(method='cpairs', cpairs=cpairs)
                """
                _x, _y = np.array(cpairs).T
                self.__x, self.__y = _x, _y
                self.pnta = point2d(x=_x[0], y=_y[0],
                                    lean=points_lean)
                self.pntb = point2d(x=_x[1], y=_y[1],
                                    lean=points_lean)
                self.calculate_level01_basics()
            if method.lower() in dth.opt.upxo_point2d:
                '''
                BRANCH OPTIONS: any in ('upxo_point2d', 'point2d',
                                        'upoint2d', 'up2d', 'up2', 'p2d',
                                        'p2')
                preferred branch option: 'up2d'
                p1 = point2d(x=0.1685, y=0.674)
                p2 = point2d(x=1.0385, y=0.79846)
                edge2d(method='up2d', pnta=p1, pntb=p2)
                '''
                if pnta.lean != points_lean:
                    pnta.lean = points_lean
                if pntb.lean != points_lean:
                    pntb.lean = points_lean
                self.__x = [pnta.x, pntb.x]
                self.__y = [pnta.y, pntb.y]
                self.pnta, self.pntb = pnta, pntb
                self.calculate_level01_basics()
            if method.lower() in ('shpoints', 'shapely_points'):
                pass
            if method.lower() in ('shline', 'shapely_line'):
                pass
            if method.lower() in ('vtk', 'vtk_line'):
                pass
            if method.lower() in ('centre.slope.length',
                                    'slope.length.centre',
                                    'length.centre.slope'):
                # TODO: low priority: make codes for this branch
                pass
        # Associate with points
        # sunil

    def calculate_level01_basics(self, length_method='coords'):
        """Compute and store centroid, length, and slope immediately after construction."""
        # Calculate centre of the edge
        self.calc_centre(saa=True, throw=False)
        # Calculate the length of the edge
        # coords approach is used. It is faster than the point method
        # in "calc_length"
        self.calc_length(method=length_method, saa=True, throw=False)
        self.calc_slope()

    def update_point(self, index, obj, method='up2d'):
        """
        Replace one endpoint of the edge and recompute derived attributes.

        Parameters
        ----------
        index : {0, 1, 'a', 'b'}
            Which endpoint to replace. ``0`` / ``'a'`` → start; ``1`` / ``'b'`` → end.
        obj : point2d or list of float
            Replacement point. A ``point2d`` object when ``method='up2d'``;
            a ``[x, y]`` coordinate pair when ``method`` is ``'cpair'`` or
            ``'coord'``.
        method : {'up2d', 'cpair', 'coord'}, optional
            Specifies the type of ``obj``. Default ``'up2d'``.

        Notes
        -----
        Recomputes centroid, length, and slope after the update.

        Examples
        --------
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(method='xy_list', xlist=[0.0, 1.0], ylist=[0.0, 0.0])
        >>> e1.update_point(1, [1.0, 1.01], method='coord')
        >>> e1.length  # updated
        """
        if index in (0, 'a'):
            if method == 'up2d':
                self.pnta.x, self.pnta.y = obj.x, obj.y
            elif method in ('cpair', 'coord'):
                self.pnta.x, self.pnta.y = obj[0], obj[1]
        elif index in (1, 'b'):
            if method == 'up2d':
                self.pntb.x, self.pntb.y = obj.x, obj.y
            elif method in ('cpair', 'coord'):
                self.pntb.x, self.pntb.y = obj[0], obj[1]
        else:
            print('Please enter valid index')
        self.calculate_level01_basics(length_method='points')

    def compare_length(self, edges,
                       use_self_length_eps=True,
                       EPS_length_low=0.0, EPS_length_high=0.0):
        """
        Compare the length of self against one or more other edges.

        Returns a comparison symbol ('=', '>', '<') and a detail label for
        each edge in `edges`. The reference length is ``self.length``.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            One or more UPXO edge2d objects to compare against.
        use_self_length_eps : bool, optional
            If True, use the tolerance stored on self (``self.EPS_e2dl_low``
            and ``self.EPS_e2dl_high``). If False, use the explicit
            ``EPS_length_low`` / ``EPS_length_high`` arguments.
            Default is True.
        EPS_length_low : float, optional
            Lower tolerance for length equality (used when
            ``use_self_length_eps=False``). Default is 0.0.
        EPS_length_high : float, optional
            Upper tolerance for length equality (used when
            ``use_self_length_eps=False``). Default is 0.0.

        Returns
        -------
        LCOMP : list of str
            Comparison symbol per edge: '=' (equal), '>' (self longer),
            '<' (self shorter).
        LCOMP_details : list of str
            Human-readable label for each comparison (e.g. 'RL=OL').

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1, p2 = Point2d(x=0, y=0), Point2d(x=1, y=0)
        >>> p3, p4 = Point2d(x=1, y=1), Point2d(x=1, y=0.1)
        >>> e1 = edge2d(method='up2d', pnta=p1, pntb=p2)
        >>> e2 = edge2d(method='up2d', pnta=p2, pntb=p3)
        >>> e3 = edge2d(method='up2d', pnta=p3, pntb=p1)
        >>> e1.compare_length([e1, e2, e3])
        """
        # Test and make unique the input edges data
        edges = dth.make_list(edges, force_list=False)
        # Extract length (as RL) and length EPS of self
        # RL: Reference length value
        RL = self.length
        # Extract EPS values to use for self
        if use_self_length_eps:
            EPS_e2dl_low = self.EPS_e2dl_low
            EPS_e2dl_high = self.EPS_e2dl_high
        else:
            EPS_e2dl_low = EPS_length_low
            EPS_e2dl_high = EPS_length_high
        # If all input edges are of the same datatype then proceed
        if len(dth.unique_of_datatypes(edges)) == 1:
            if edges[0].__class__.__name__ in dth.dt.UPXO_EDGES:
                # UPXO EDGE OBJECT
                LCOMP, LCOMP_details = [], []
                if use_self_length_eps:
                    EPS_e2dl_low = abs(EPS_e2dl_low)
                    EPS_e2dl_high = abs(EPS_e2dl_high)
                for OL in [e.length for e in edges]:
                    if OL >= RL-EPS_e2dl_low and OL <= RL+EPS_e2dl_high:
                        LCOMP.append('=')
                        LCOMP_details.append('RL=OL')
                    elif OL < RL-EPS_e2dl_low:
                        # R can contain O
                        LCOMP.append('>')
                        LCOMP_details.append('RL>OL')
                    elif OL > RL+EPS_e2dl_high:
                        # R cannot contain O as OL > RL
                        LCOMP.append('<')
                        LCOMP_details.append('RL<OL')
                return LCOMP, LCOMP_details
            elif edges[0].__class__.__name__ in dth.dt.SH_LSTRING_2D:
                # SHAPELY LINESTRING OBJECT
                pass
        else:
            # A mix of many dataypes of lines. May never be needed!!
            pass

    def __eql__(self, edges, use_self_length_eps=True,
                EPS_length_low=0.0, EPS_length_high=0.0):
        """
        Test length equality of self against one or more edges.

        Delegates to :meth:`compare_length` and returns a boolean list.
        Two lengths are considered equal when the other edge's length falls
        within ``[self.length - EPS_low, self.length + EPS_high]``.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            One or more UPXO edge2d objects to test.
        use_self_length_eps : bool, optional
            If True, use tolerances stored on self (``EPS_e2dl_low`` /
            ``EPS_e2dl_high``). If False, use the explicit arguments below.
            Default is True.
        EPS_length_low : float, optional
            Lower tolerance (used when ``use_self_length_eps=False``).
            Default is 0.0.
        EPS_length_high : float, optional
            Upper tolerance (used when ``use_self_length_eps=False``).
            Default is 0.0.

        Returns
        -------
        list of bool
            True for each edge in `edges` whose length equals self.length
            within tolerance; False otherwise.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1 = Point2d(x=0.0, y=0.0)
        >>> p2 = Point2d(x=1.0, y=0.0)
        >>> p3 = Point2d(x=0.0, y=1.0)
        >>> p11 = Point2d(x=2.0, y=0.0)
        >>> e1 = edge2d(pnta=p1, pntb=p2)
        >>> e2 = edge2d(pnta=p1, pntb=p3)
        >>> e10 = edge2d(pnta=p1, pntb=p11)
        >>> e1.__eql__([e1, e2, e10])
        """
        LCOMP, _ = self.compare_length(edges,
                                       use_self_length_eps=use_self_length_eps,
                                       EPS_length_low=EPS_length_low,
                                       EPS_length_high=EPS_length_high)
        return [True if lcomp == '=' else False for lcomp in LCOMP]

    def compare_slope(self, edges,
                      use_global_slope_thresholds=True,
                      EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
                      use_self_eps=True,
                      EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Compare the slope of self against one or more other edges.

        Uses case-based slope comparisons covering horizontal, inclined, and
        vertical edge orientations across all quadrant combinations.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            One or more UPXO edge2d objects to compare against.
        use_global_slope_thresholds : bool, optional
            If True, use class-level ``EPS_e2ds_lowest`` / ``EPS_e2ds_highest``
            to classify horizontal, inclined, and vertical slopes.
            Default is True.
        EPS_e2ds_lowest : float, optional
            Lower slope threshold (used when
            ``use_global_slope_thresholds=False``). Default is -10**-3.
        EPS_e2ds_highest : float, optional
            Upper slope threshold (used when
            ``use_global_slope_thresholds=False``). Default is -10**-3.
        use_self_eps : bool, optional
            If True, use tolerances from self (``self.EPS_RS_low`` /
            ``self.EPS_RS_high``) for the reference-to-other comparison.
            Default is True.
        EPS_RS_low : float, optional
            Lower tolerance for slope equality (used when
            ``use_self_eps=False``). Default is 10**-3.
        EPS_RS_high : float, optional
            Upper tolerance for slope equality (used when
            ``use_self_eps=False``). Default is 10**-3.

        Returns
        -------
        SCOM : list of str
            Comparison symbol per edge: '=' (equal slope), '>' (self steeper),
            '<' (self shallower).
        SCOM_details : list of str
            Case label string (e.g. 'RS=OS.RH.OH') for each comparison.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1, p2 = Point2d(x=0, y=0), Point2d(x=1, y=0)
        >>> p3, p4 = Point2d(x=1, y=1), Point2d(x=1, y=0.1)
        >>> e1 = edge2d(method='up2d', pnta=p1, pntb=p2)
        >>> e2 = edge2d(method='up2d', pnta=p2, pntb=p3)
        >>> e1.compare_slope([e1, e2])
        """
        # Test and make unique the input edges data
        edges = dth.make_list(edges, force_list=False)
        # Extract slope and slope EPS values of self
        RS = self.slope  # Slope of reference UPXO edge object
        if use_self_eps:
            EPS_RS_low = abs(self.EPS_RS_low)
            EPS_RS_high = abs(self.EPS_RS_high)
        else:
            EPS_RS_low, EPS_RS_high = EPS_RS_low, EPS_RS_high
        # Extract threshold values for edge2d slopes
        if use_global_slope_thresholds:
            EPS_e2ds_lowest = edge2d.EPS_e2ds_lowest
            EPS_e2ds_highest = edge2d.EPS_e2ds_highest
        else:
            EPS_e2ds_lowest = EPS_e2ds_lowest
            EPS_e2ds_highest = EPS_e2ds_highest
        # If all input edges are of the same datatype then proceed
        if len(dth.unique_of_datatypes(edges)) == 1:
            # If all input edges belong to UPXO edge, then proceed
            if edges[0].__class__.__name__ in dth.dt.UPXO_EDGES:
                # Iterate through each edge and compare slopes
                SCOM = []  # Slope comparison results
                SCOM_details = []  # Detailed slope comparison results
                for e in edges:
                    # Extract slope. Its EPS values will not be used
                    OS = e.slope
                    # Calculate difference between RS and OS
                    if abs(RS) != INFINITY or abs(OS) != INFINITY:
                        ΔRO = RS-OS
                    else:
                        # DEALT with this on a case to case basis
                        pass
                    # CASE 1
                    # Refrence (R). RS: Slope of ref. UPXO edge
                    # Other (O) HORIZONTAL. OS: Slope of other UPXO edge
                    # R: HORIZONTAL TO LEFT OR RIGHT
                    # O: HORIZONTAL TO LEFT OR RIGHT
                    if (RS >= -EPS_e2ds_lowest and RS <= EPS_e2ds_lowest) or RS == 0:
                        if (OS >= -EPS_e2ds_lowest and OS <= EPS_e2ds_lowest) or OS == 0:
                            print('siadfbvi37y5gr')
                            SCOM.append('=')
                            SCOM_details.append('RS=OS.RH.OH')
                    # CASE 2
                    # R HORIZONTAL TO LEFT OR RIGHT
                    # O INCLINED IN FIRST QUADRANT or THIRD QUADRANT. OS: +VE
                    if (RS >= -EPS_e2ds_lowest and RS <= EPS_e2ds_lowest) or RS == 0:
                        if OS > EPS_e2ds_lowest and OS < EPS_e2ds_highest:
                            print('sdivjkj34t1d')
                            SCOM.append('<')
                            SCOM_details.append('RS<OS.RH.OQ14')
                    # CASE 3
                    # R HORIZONTAL TO LEFT OR RIGHT
                    # O VERTICAL UP OR VERTICAL DO

                    if (RS >= -EPS_e2ds_lowest and RS <= EPS_e2ds_lowest) or RS == 0:
                        if (OS <= -EPS_e2ds_highest and OS >= EPS_e2ds_highest) or abs(OS) == INFINITY:
                            print('w4987y3iehbvjhb35i')
                            SCOM.append('<')
                            SCOM_details.append('RS<OS.RH.OV')
                    # CASE 4
                    # R HORIZONTAL TO LEFT OR RIGHT. RS: 0 (+-EPS)
                    # O INCLINED IN 2nd or 4th QUADRANT. OS: -VE (+-EPS)
                    if (RS >= -EPS_e2ds_lowest and RS >= EPS_e2ds_highest) or RS == 0:
                        if OS <= -EPS_e2ds_lowest and OS >= -EPS_e2ds_highest:
                            print('q490t8hkjbvk2wt')
                            SCOM.append('<')
                            SCOM_details.append('RS<OS.RH.OQ23')
                    # ---------------------------------------------------------
                    # CASE 5
                    # R INCLINED IN Q14 (QUADRANT 1 OR 4). RS: +VE
                    # O HORIZONTAL TO LEFT OR RIGHT.
                    if RS > EPS_e2ds_lowest and RS < EPS_e2ds_highest:
                        if (OS >= -EPS_e2ds_lowest and OS <= EPS_e2ds_lowest) or OS == 0:
                            SCOM.append('>')
                            SCOM_details.append('RS>OS.RQ14.OH')
                    # CASE 6
                    # R INCLINED IN Q14. RS: +VE
                    # O INCLINED IN Q14. OS: +VE
                    if RS > EPS_e2ds_lowest and RS < EPS_e2ds_highest:
                        if OS > EPS_e2ds_lowest and OS < EPS_e2ds_highest:
                            # CASE 6A:  O IS BELOW R -- i.e. R IS ABOVE O
                            if OS > EPS_e2ds_lowest and OS < RS-EPS_RS_low:
                                print('pijm89321dvfb')
                                SCOM.append('>')
                                SCOM_details.append('RS>OS.RQ14.OQ14')
                            # CASE 6B:  O OVERLAPS WITH R
                            if OS >= RS-EPS_RS_low and OS <= RS+EPS_RS_high:
                                print('6sv54rhryt98yh')
                                SCOM.append('=')
                                SCOM_details.append('RS>OS.RQ14.OQ14')
                            # CASE 6C:  O IS ABOVE R -- i.e. R IS BELOW O
                            if OS > RS+EPS_RS_high and OS < EPS_e2ds_highest:
                                print('iuh9873485h3hbf')
                                SCOM.append('<')
                                SCOM_details.append('RS>OS.RQ14.OQ14')
                    # CASE 7
                    # R INCLINED IN Q14. RS: +VE
                    # O VERTICAL UP OR DOWN.
                    if RS > EPS_e2ds_lowest and RS < EPS_e2ds_highest:
                        if (OS <= -EPS_e2ds_highest and OS >= EPS_e2ds_highest) or abs(OS) == INFINITY:
                            print('6afe637432gsgvsdg')
                            SCOM.append('<')
                            SCOM_details.append('RS<OS.RQ12.OV')
                    # CASE 8
                    # R INCLINED IN Q14: +VE
                    # O INCLINED IN Q23. OS: -VE
                    if RS > EPS_e2ds_lowest and RS < EPS_e2ds_highest:
                        if OS <= -EPS_e2ds_lowest and OS >= -EPS_e2ds_highest:
                            print('w948tyhijnbdvkj24t')
                            SCOM.append('>')
                    # ---------------------------------------------------------
                    # CASE 9
                    # R VERTICAL UP OR DOWN
                    # O HORIZONTAL LEFT OR RIGHT
                    if (RS <= -EPS_e2ds_highest and RS >= EPS_e2ds_highest) or abs(RS) == INFINITY:
                        if (OS >= -EPS_e2ds_lowest and OS <= EPS_e2ds_lowest) or OS == 0:
                            print('wo4ut908u43w5rhb4sgr')
                            SCOM.append('>')
                            SCOM_details.append('RS>OS.RV.OH')
                    # CASE 10
                    # R VERTICAL UP OR DOWN
                    # O INCLINED IN Q14
                    if (RS <= -EPS_e2ds_highest
                            and RS >= EPS_e2ds_highest) or abs(RS) == INFINITY:
                        if OS > EPS_e2ds_lowest and OS < EPS_e2ds_highest:
                            print('kihjgbisfe0834outj3q4k')
                            SCOM.append('>')
                            SCOM_details.append('RS>OS.RV.OQ14')
                    # CASE 11
                    # R VERTICAL UP OR DOWN
                    # O VERTICAL UP OR DOWN
                    if (RS <= -EPS_e2ds_highest
                            and RS >= EPS_e2ds_highest) or abs(RS) == INFINITY:
                        if (OS <= -EPS_e2ds_highest and OS >= EPS_e2ds_highest) or abs(OS) == INFINITY:
                            SCOM.append('=')
                            if RS > 0 and OS > 0:
                                print('pijocu0q394ujgt')
                                SCOM_details.append('RS=OS.RY+.OY+.{∞-∞}')
                            elif RS > 0 and OS < 0:
                                print('asklfjdb0982045h3')
                                SCOM_details.append(
                                    'RS=OS.RY+.OY-.{INF-(-INF)}')
                            elif RS < 0 and OS > 0:
                                print('hcg97945vkjgf6421v')
                                SCOM_details.append('RS=OS.RY-.OY+.{-INF+INF}')
                            elif RS < 0 and OS < 0:
                                print('nw04398098usbdf')
                                SCOM_details.append('RS=OS.RY-.OY-.{-INF-INF}')
                    # CASE 12
                    # R VERTICAL UP OR DOWN
                    # O INCLINED IN Q23
                    if (RS <= -EPS_e2ds_highest and RS >= EPS_e2ds_highest) or abs(RS) == INFINITY:
                        if OS < -EPS_e2ds_lowest and OS > -EPS_e2ds_highest:
                            SCOM.append('>')
                            if RS > 0:
                                print('kjb9q82409tjo3rgv')
                                SCOM_details.append(f'RS=OS.RY+.OQ23.{ΔRO}')
                            elif RS < 0:
                                print('63avs143521brekhygbsdf')
                                SCOM_details.append(f'RS=OS.RY-.OQ23.{ΔRO}')
                    # ---------------------------------------------------------
                    # CASE 13
                    # R INCLINED IN Q23
                    # O HORIZONTAL LEFT OR RIGHT
                    if RS < -EPS_e2ds_lowest and RS > -EPS_e2ds_highest:
                        if OS >= -EPS_e2ds_lowest and OS < 0:
                            print('aiwsfyg97w49tygbwejv')
                            SCOM.append('<')
                            SCOM_details.append(f'RS<OS.RQ23.OX-.{ΔRO}')
                        if OS >= 0 and OS <= EPS_e2ds_lowest:
                            print('khbsd98350yuhkjhbjb')
                            SCOM.append('<')
                            SCOM_details.append(f'RS<OS.RQ23.OX+.{ΔRO}')
                    # CASE 14
                    # R INCLINED IN Q23
                    # O INCLINED IN Q12
                    if RS < -EPS_e2ds_lowest and RS > -EPS_e2ds_highest:
                        if OS > EPS_e2ds_lowest and OS < EPS_e2ds_highest:
                            print('asv6854521321rhw3t')
                            SCOM.append('<')
                            SCOM_details.append('RS<OS.RQ23.OQ12.{ΔRO}')
                    # CASE 15
                    # R INCLINED IN Q23
                    # O VERTICAL UP OR DOWN
                    if RS < -EPS_e2ds_lowest and RS > -EPS_e2ds_highest:
                        if OS <= -EPS_e2ds_highest and OS >= -INFINITY:
                            print('hybq326t09uysbvdc')
                            SCOM.append('>')
                            SCOM_details.append(f'RS>OS.RQ23.OY-.{ΔRO}')
                        if OS >= EPS_e2ds_highest and OS <= INFINITY:
                            print('akhfb9u40t935y4y')
                            SCOM.append('>')
                            SCOM_details.append(f'RS>OS.RQ23.OY+.{ΔRO}')
                    # CASE 16
                    # R INCLINED IN Q23
                    # O INCLINED IN Q23
                    if RS < -EPS_e2ds_lowest and RS > -EPS_e2ds_highest:
                        if OS < -EPS_e2ds_lowest and OS > -EPS_e2ds_highest:
                            # EPS_RS_low, EPS_RS_high, EPS_OS_low, EPS_OS_high
                            if OS > RS-EPS_RS_low:
                                # CASE 16A: O BELOW R
                                print('adkvjb304958t043u5hy564')
                                SCOM.append('>')
                                SCOM_details.append(f'RS>OS.RQ23.OQ23.{ΔRO}')
                            elif OS <= RS+EPS_RS_high and OS >= RS-EPS_RS_low:
                                # CASE 16B: O OVERLAPS WITH R (contained in R)
                                print('jhgvjhbjw4939ut3j')
                                SCOM.append('=')
                                SCOM_details.append(f'RS=OS.RQ23.OQ23.{ΔRO}')
                            elif OS < RS-EPS_RS_high:
                                # CASE 16C: O ABOVE R
                                print('asebf98798793q45knj')
                                SCOM.append('<')
                                SCOM_details.append('RS<OS.RQ23.OQ23.{ΔRO}')
                    # ---------------------------------------------------------
        return SCOM, SCOM_details

    def __eqs__(self, edges,
                use_global_slope_thresholds=True,
                EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
                use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Test slope equality of self against one or more edges.

        Delegates to :meth:`compare_slope` and returns a boolean tuple.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            One or more UPXO edge2d objects to test.
        use_global_slope_thresholds : bool, optional
            If True, use class-level slope threshold constants.
            Default is True.
        EPS_e2ds_lowest : float, optional
            Lower slope threshold (used when
            ``use_global_slope_thresholds=False``). Default is -10**-3.
        EPS_e2ds_highest : float, optional
            Upper slope threshold (used when
            ``use_global_slope_thresholds=False``). Default is -10**-3.
        use_self_eps : bool, optional
            If True, use self's ``EPS_RS_low`` / ``EPS_RS_high``.
            Default is True.
        EPS_RS_low : float, optional
            Lower slope equality tolerance (used when
            ``use_self_eps=False``). Default is 10**-3.
        EPS_RS_high : float, optional
            Upper slope equality tolerance (used when
            ``use_self_eps=False``). Default is 10**-3.

        Returns
        -------
        tuple of bool
            True for each edge in `edges` whose slope equals self.slope
            within tolerance; False otherwise.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1, p2 = Point2d(x=0.0, y=0.0), Point2d(x=1.0, y=0.0)
        >>> p3 = Point2d(x=0.0, y=1.0)
        >>> e1 = edge2d(pnta=p1, pntb=p2)
        >>> e2 = edge2d(pnta=p1, pntb=p3)
        >>> e1.__eqs__([e1, e2])
        """
        _ugst = use_global_slope_thresholds
        SCOMP, _ = self.compare_slope(edges,
                                      use_global_slope_thresholds=_ugst,
                                      EPS_e2ds_lowest=EPS_e2ds_lowest,
                                      EPS_e2ds_highest=EPS_e2ds_highest,
                                      use_self_eps=use_self_eps,
                                      EPS_RS_low=EPS_RS_low,
                                      EPS_RS_high=EPS_RS_high)
        return tuple(True if scomp == '=' else False for scomp in SCOMP)

    def __eq__(self, edges, equality_tests=('l', 's'), comparator='=',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Combined equality test over length and/or slope against other edges.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            Edges to compare against self.
        equality_tests : tuple of str, optional
            Which attributes to test. 'l' for length, 's' for slope.
            Default is ('l', 's').
        comparator : str, optional
            Comparison operator to apply: '=', '>', '<', '>=', '<=', '!='.
            Default is '='.
        use_self_length_eps : bool, optional
            Use self's stored length tolerances. Default is True.
        EPS_length_low : float, optional
            Lower length tolerance. Default is 0.0.
        EPS_length_high : float, optional
            Upper length tolerance. Default is 0.0.
        slope_test_method : str, optional
            Reserved for future slope method selection. Default is 'slope'.
        use_global_slope_thresholds : bool, optional
            Use class-level slope thresholds. Default is True.
        EPS_e2ds_lowest : float, optional
            Lower slope range threshold. Default is -10**-3.
        EPS_e2ds_highest : float, optional
            Upper slope range threshold. Default is -10**-3.
        use_self_eps : bool, optional
            Use self's stored slope tolerances. Default is True.
        EPS_RS_low : float, optional
            Lower slope equality tolerance. Default is 10**-3.
        EPS_RS_high : float, optional
            Upper slope equality tolerance. Default is 10**-3.

        Returns
        -------
        list of bool
            Per-edge truth values of the requested comparison.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1, p2 = Point2d(x=0, y=0), Point2d(x=1, y=0)
        >>> p3 = Point2d(x=0, y=1)
        >>> e1 = edge2d(pnta=p1, pntb=p2)
        >>> e2 = edge2d(pnta=p1, pntb=p3)
        >>> e1.__eq__([e1, e2])
        """
        if comparator not in dth.dt.ALL_COMPARATORS:
            print('UNKNOWN COMPARATOR INPUT')
        else:
            length_test, slope_test = False, False
            if any(dth.strip_str(e) in dth.opt.upxo_edge_length
                   for e in equality_tests):
                # CARRY OUT THE LENGTH EQUALITY TEST. Store TVs in LEQ
                LEQ = self.__eql__(edges,
                                   use_self_length_eps=use_self_length_eps,
                                   EPS_length_low=EPS_length_low,
                                   EPS_length_high=EPS_length_high)
                length_test = True
            if any(dth.strip_str(e) in dth.opt.upxo_edge_slope
                   for e in equality_tests):
                # CARRY OUT THE SLOPE EQUALITY TEST. Store TVs in SEQ
                _ugst = use_global_slope_thresholds
                SEQ = self.__eqs__(edges,
                                   use_global_slope_thresholds=_ugst,
                                   EPS_e2ds_lowest=EPS_e2ds_lowest,
                                   EPS_e2ds_highest=EPS_e2ds_highest,
                                   use_self_eps=use_self_eps,
                                   EPS_RS_low=EPS_RS_low,
                                   EPS_RS_high=EPS_RS_high)
                slope_test = True
            if len(LEQ) != len(SEQ):
                print('No. of elems. in length equality & slope equality')
                print('  arrays ARE NOT SAME.')
            else:
                if length_test and not slope_test:
                    LEQ = [True if lcomp == comparator else False
                           for lcomp in LEQ]
                    return LEQ
                if not length_test and slope_test:
                    SEQ = [True if scomp == comparator else False
                           for scomp in SEQ]
                    return SEQ
                if length_test and slope_test:
                    LEQ_SEQ = [(lcomp == comparator and scomp == comparator)
                               for lcomp, scomp in zip(LEQ, SEQ)]
                    return LEQ_SEQ

    def __ne__(self, edges, equality_tests=('l', 's'), comparison='!=',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Inequality test (``!=``) delegating to :meth:`__eq__` with '!='.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            Edges to test against.
        equality_tests : tuple of str, optional
            Attributes to test: 'l' (length), 's' (slope). Default ('l','s').
        comparison : str, optional
            Fixed to '!=' for this operator. Default is '!='.

        Returns
        -------
        list of bool
            True where self != edge for the selected criteria.
        """
        _ugst = use_global_slope_thresholds
        NE = self.__eq__(edges,
                         equality_tests=equality_tests,
                         comparison=comparison,
                         use_self_length_eps=use_self_length_eps,
                         EPS_length_low=EPS_length_low,
                         EPS_length_high=EPS_length_high,
                         slope_test_method=slope_test_method,
                         use_global_slope_thresholds=_ugst,
                         EPS_e2ds_lowest=EPS_e2ds_lowest,
                         EPS_e2ds_highest=EPS_e2ds_highest,
                         use_self_eps=use_self_eps,
                         EPS_RS_low=EPS_RS_low,
                         EPS_RS_high=EPS_RS_high)
        return NE

    def __lt__(self, edges, equality_tests=('l', 's'), comparison='<',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Less-than test (``<``) delegating to :meth:`__eq__` with '<'.

        Returns
        -------
        list of bool
            True where self < edge for the selected criteria.
        """
        _ugst = use_global_slope_thresholds
        LT = self.__eq__(edges,
                         equality_tests=equality_tests,
                         comparison=comparison,
                         use_self_length_eps=use_self_length_eps,
                         EPS_length_low=EPS_length_low,
                         EPS_length_high=EPS_length_high,
                         slope_test_method=slope_test_method,
                         use_global_slope_thresholds=_ugst,
                         EPS_e2ds_lowest=EPS_e2ds_lowest,
                         EPS_e2ds_highest=EPS_e2ds_highest,
                         use_self_eps=use_self_eps,
                         EPS_RS_low=EPS_RS_low,
                         EPS_RS_high=EPS_RS_high)
        return LT

    def __le__(self, edges, equality_tests=('l', 's'), comparison='<=',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Less-than-or-equal test (``<=``) delegating to :meth:`__eq__` with '<='.

        Returns
        -------
        list of bool
            True where self <= edge for the selected criteria.
        """
        _ugst = use_global_slope_thresholds
        LE = self.__eq__(edges,
                         equality_tests=equality_tests,
                         comparison=comparison,
                         use_self_length_eps=use_self_length_eps,
                         EPS_length_low=EPS_length_low,
                         EPS_length_high=EPS_length_high,
                         slope_test_method=slope_test_method,
                         use_global_slope_thresholds=_ugst,
                         EPS_e2ds_lowest=EPS_e2ds_lowest,
                         EPS_e2ds_highest=EPS_e2ds_highest,
                         use_self_eps=use_self_eps,
                         EPS_RS_low=EPS_RS_low,
                         EPS_RS_high=EPS_RS_high)
        return LE

    def __gt__(self, edges, equality_tests=('l', 's'), comparison='>',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Greater-than test (``>``) delegating to :meth:`__eq__` with '>'.

        Returns
        -------
        list of bool
            True where self > edge for the selected criteria.
        """
        _ugst = use_global_slope_thresholds
        GT = self.__eq__(edges,
                         equality_tests=equality_tests,
                         comparison=comparison,
                         use_self_length_eps=use_self_length_eps,
                         EPS_length_low=EPS_length_low,
                         EPS_length_high=EPS_length_high,
                         slope_test_method=slope_test_method,
                         use_global_slope_thresholds=_ugst,
                         EPS_e2ds_lowest=EPS_e2ds_lowest,
                         EPS_e2ds_highest=EPS_e2ds_highest,
                         use_self_eps=use_self_eps,
                         EPS_RS_low=EPS_RS_low,
                         EPS_RS_high=EPS_RS_high)
        return GT

    def __ge__(self, edges, equality_tests=('l', 's'), comparison='>=',
               use_self_length_eps=True,
               EPS_length_low=0.0, EPS_length_high=0.0,
               slope_test_method='slope', use_global_slope_thresholds=True,
               EPS_e2ds_lowest=-10**-3, EPS_e2ds_highest=-10**-3,
               use_self_eps=True, EPS_RS_low=10**-3, EPS_RS_high=10**-3):
        """
        Greater-than-or-equal test (``>=``) delegating to :meth:`__eq__` with '>='.

        Returns
        -------
        list of bool
            True where self >= edge for the selected criteria.
        """
        _ugst = use_global_slope_thresholds
        GE = self.__eq__(edges,
                         equality_tests=equality_tests,
                         comparison=comparison,
                         use_self_length_eps=use_self_length_eps,
                         EPS_length_low=EPS_length_low,
                         EPS_length_high=EPS_length_high,
                         slope_test_method=slope_test_method,
                         use_global_slope_thresholds=_ugst,
                         EPS_e2ds_lowest=EPS_e2ds_lowest,
                         EPS_e2ds_highest=EPS_e2ds_highest,
                         use_self_eps=use_self_eps,
                         EPS_RS_low=EPS_RS_low,
                         EPS_RS_high=EPS_RS_high)
        return GE

    def __add__(self, toadd, saa=True, throw=False, update_toadd=True,
                addto='bothpoints', addtocoord='xy'):
        """
        Add a scalar, coordinate pair, or another edge to this edge's endpoints.

        Parameters
        ----------
        toadd : int, float, list, or edge2d
            Value(s) to add. Accepts a single number, a 2-element coordinate
            pair, or a list of such values.
        saa : bool, optional
            Save-and-apply: if True, update self in-place with the first
            element of `toadd`. Default is True.
        throw : bool, optional
            If True, also create and return new edge objects for each element
            in `toadd`. Default is False.
        update_toadd : bool, optional
            Reserved for future use. Default is True.
        addto : str, optional
            Which endpoints to modify: 'bothpoints', 'pnta', or 'pntb'.
            Default is 'bothpoints'.
        addtocoord : str, optional
            Which coordinate(s) to modify: 'xy', 'x', or 'y'.
            Default is 'xy'.

        Returns
        -------
        list of edge2d or None
            New edge objects when ``throw=True``; otherwise None.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> p1, p2 = Point2d(x=0, y=0), Point2d(x=1, y=0)
        >>> e = edge2d(pnta=p1, pntb=p2)
        >>> e.__add__([1.0], saa=True, throw=False)
        """
        addto, addtocoord = str(addto).lower(), str(addtocoord).lower()
        if toadd in dth.dt.NUM_UPXOE2D_SHLS:
            # A single object has been entered
            # This object is one of the following three types:
            #    1. number
            #    2. UPXO edge2d
            #    3. SHAPELY LineString
            toadd = dth.make_list(toadd, force_list=False)
        if toadd in dth.dt.ITERABLES:
            if len(dth.unique_of_datatypes(toadd)) == 1:
                if toadd[0] in dth.dt.NUMBERS and len(toadd[0]) == 1:
                    # A list of single numbers have been entered.
                    pntax, pntay = dcp(self.pnta.x), dcp(self.pnta.y)
                    pntbx, pntby = dcp(self.pntb.x), dcp(self.pntb.y)
                    _x, _y = toadd[0], toadd[0]
                    if saa and throw:
                        # Add 1st number to both x and y coordinates of self,
                        # then create new edges for all toadd values.
                        # Then return the entire list.
                        # NOTE: 1st element in the returned should NOT be self
                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                            # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                            new_edges = [edge2d(edge_lean=self.edge_lean,
                                                points_lean=self.points_lean,
                                                method='coord',
                                                x=[pntax+_x, pntbx+_x],
                                                y=[pntay+_y, pntby+_y]
                                                ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+_x, pntbx],
                                                        y=[pntay+_y, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+_x, pntbx],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay+_y, pntby]
                                                        ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.__add__(toadd[0], make_new=False)
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+_x],
                                                        y=[pntay, pntby+_y]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+_x],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay, pntby+_y]
                                                        ) for a in toadd]
                        # Perform update operations
                        self.calc_centre(saa=True, throw=False)
                        self.calc_length(method='coords',
                                         saa=True, throw=False)
                        self.calc_slope()
                    if not saa and throw:
                        # Make a new edge object for each toadd[i]
                        # and return the list
                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            # ==> STEP : DEALING WITH MAKING NEW_EDGES
                            new_edges = [edge2d(edge_lean=self.edge_lean,
                                                points_lean=self.points_lean,
                                                method='coord',
                                                x=[pntax+_x, pntbx+_x],
                                                y=[pntay+_y, pntby+_y]
                                                ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+_x, pntbx],
                                                        y=[pntay+_y, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+_x, pntbx],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay+_y, pntby]
                                                        ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+_x],
                                                        y=[pntay, pntby+_y]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+_x],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay, pntby+_y]
                                                        ) for a in toadd]
                    if not saa and not throw:
                        # Do nothing
                        new_edges = None
                        pass
                    if saa and not throw:
                        # Add 1st number to both x and y coordinates and
                        # update self. Ignore the remaining numbers in toadd
                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pnta.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pntb.x += _x
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pntb.y += _y
                        # Perform update operations
                        self.calc_centre(saa=True, throw=False)
                        self.calc_length(method='coords',
                                         saa=True, throw=False)
                        self.calc_slope()
                        new_edges = None
                elif toadd[0] in dth.dt.NUMBERS and len(toadd[0]) == 2:
                    # A list of 2D or 3D coordinate pairs have been entered.
                    pntax, pntay = dcp(self.pnta.x), dcp(self.pnta.y)
                    pntbx, pntby = dcp(self.pntb.x), dcp(self.pntb.y)
                    _x, _y = toadd[0], toadd[1]
                    if saa and throw:
                        # Add [$--0,1--$] of 1st toadd[0] to both x and y
                        # coordinates of self, then create new edges for all
                        # toadd values. Then return the entire list.
                        # NOTE: 1st element in the returned should NOT be self

                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                            # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                            new_edges = [edge2d(edge_lean=self.edge_lean,
                                                points_lean=self.points_lean,
                                                method='coord',
                                                x=[pntax+a[0], pntbx+a[0]],
                                                y=[pntay+a[1], pntby+a[1]]
                                                ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+a[0], pntbx],
                                                        y=[pntay+a[1], pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+a[0], pntbx],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay+a[1], pntby]
                                                        ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.__add__(toadd[0], make_new=False)
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                        # Perform update operations
                        self.calc_centre(saa=True, throw=False)
                        self.calc_length(method='coords',
                                         saa=True, throw=False)
                        self.calc_slope()
                    if not saa and throw:
                        # Make a new edge object for each toadd[i][$--0,1--$]
                        # and return the list
                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            # ==> STEP : DEALING WITH MAKING NEW_EDGES
                            new_edges = [edge2d(edge_lean=self.edge_lean,
                                                points_lean=self.points_lean,
                                                method='coord',
                                                x=[pntax+a[0], pntbx+a[0]],
                                                y=[pntay+a[1], pntby+a[1]]
                                                ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+a[0], pntbx],
                                                        y=[pntay+a[1], pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+a[0], pntbx],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay+a[1], pntby]
                                                        ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                        pass
                    if not saa and not throw:
                        # Do nothing
                        new_edges = None
                        pass
                    if saa and not throw:
                        # Add [$--0,1--$] of 1st toadd[0] to
                        # self.$--pnta,pntb--$.$--x,y--$
                        # Ignore toadd[1:-1]

                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    self.pntb.y += _y
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _x
                                    self.pnta.y += _y
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pntb.x += _x
                                    self.pntb.y += _y
                                elif addtocoord in ('x'):
                                    self.pntb.x += _x
                                elif addtocoord in ('y'):
                                    self.pntb.y += _y
                        # Perform update operations
                        self.calc_centre(saa=True, throw=False)
                        self.calc_length(method='coords',
                                         saa=True, throw=False)
                        self.calc_slope()
                        new_edges = None
                elif toadd[0] in dth.dt.UPXO_EDGES:
                    # A list of UPXO edges have been entered.
                    pntax, pntay = dcp(self.pnta.x), dcp(self.pnta.y)
                    pntbx, pntby = dcp(self.pntb.x), dcp(self.pntb.y)
                    _oepax, _oepay = toadd[0].pnta.x, toadd[0].pnta.y
                    _oepbx, _oepby = toadd[0].pntb.x, toadd[0].pntb.y
                    if saa and throw:
                        # Step 1
                        # Add toadd[0].pnta.x and toadd[0].pnta.y
                        #       to self.pnta.x and self.pnta.y
                        # Add toadd[0].pntb.x and toadd[0].pntb.y
                        #       to self.pntb.x and self.pntb.y
                        # Step 2 if update_toadd is True
                        # Add self.$--pnta,pntb--$.$--x,y--$ to
                        # toadd[i].$--pnta,pntb--$.$--x,y--$, for
                        # i ∈ [0, len(toadd)] and return updated toadd array
                        # Step 3 if update_toadd is False
                        # Add self.$--pnta,pntb--$.$--x,y--$ to
                        # $--x0,y0--$ and $--x1,y1--$ which are deepcopies
                        # of toadd[i].$--pnta,pntb--$.$--x,y--$, for
                        # i ∈ [0, len(toadd)]. For each toadd[i],
                        # make a new edge and build toreturn array
                        # and return the toreturn array

                        # Perform update operations
                        if dth.strip_str(addto) in dth.opt.e2d_addto_both_pnts:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _oepax
                                    self.pntb.x += _oepbx
                                    self.pnta.y += _oepay
                                    self.pntb.y += _oepby
                                elif addtocoord in ('x'):
                                    self.pnta.x += _oepax
                                    self.pntb.x += _oepbx
                                elif addtocoord in ('y'):
                                    self.pnta.y += _oepay
                                    self.pntb.y += _oepby
                            # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                            # e.pnta.x, e.pnta.y, e.pntb.x, e.pntb.y
                            new_edges = [edge2d(edge_lean=self.edge_lean,
                                                points_lean=self.points_lean,
                                                method='coord',
                                                x=[pntax+e.pnta.x,
                                                   pntbx+e.pntb.x],
                                                y=[pntay+e.pnta.y,
                                                   pntby+e.pntb.y]
                                                ) for e in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pnta:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.x += _oepax
                                    self.pnta.y += _oepay
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+e.pnta.x,
                                                           pntbx+e.pntb.x],
                                                        y=[pntay+e.pnta.y,
                                                           pntby+e.pntb.y]
                                                        ) for e in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax+a[0], pntbx],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay+a[1], pntby]
                                                        ) for a in toadd]
                        if dth.strip_str(addto) in dth.opt.e2d_addto_pntb:
                            # ==> STEP - 1: DEALING WITH SAA=TRUE
                            if addtocoord in dth.opt.addtocoord:
                                if addtocoord in ('xy', 'both'):
                                    self.pnta.__add__(toadd[0], make_new=False)
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                                elif addtocoord in ('x'):
                                    self.pnta.x += _x
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx+a[0]],
                                                        y=[pntay, pntby]
                                                        ) for a in toadd]
                                elif addtocoord in ('y'):
                                    self.pnta.y += _y
                                    # ==> STEP - 2: DEALING WITH MAKING NEW_EDGES
                                    new_edges = [edge2d(edge_lean=self.edge_lean,
                                                        points_lean=self.points_lean,
                                                        method='coord',
                                                        x=[pntax, pntbx],
                                                        y=[pntay, pntby+a[1]]
                                                        ) for a in toadd]
                        # Perform update operations
                        self.calc_centre(saa=True, throw=False)
                        self.calc_length(method='coords',
                                         saa=True, throw=False)
                        self.calc_slope()
                    if not saa and throw:
                        # Step 1 if update_toadd is True
                        # Add self.$--pnta,pntb--$.$--x,y--$ to
                        # toadd[i].$--pnta,pntb--$.$--x,y--$, for
                        # i ∈ [0, len(toadd)] and return updated toadd array
                        # Step 2 if update_toadd is False
                        # Add self.$--pnta,pntb--$.$--x,y--$ to
                        # $--x0,y0--$ and $--x1,y1--$ which are deepcopies
                        # of toadd[i].$--pnta,pntb--$.$--x,y--$, for
                        # i ∈ [0, len(toadd)]. For each toadd[i],
                        # make a new edge and build toreturn array
                        # and return the toreturn array
                        pass
                    if not saa and not throw:
                        # Do nothing
                        new_edges = None
                        pass
                    if saa and not throw:
                        # Update self
                        # Add toadd[0].pnta.x and toadd[0].pnta.y
                        #       to self.pnta.x and self.pnta.y
                        # Add toadd[0].pntb.x and toadd[0].pntb.y
                        #       to self.pntb.x and self.pntb.y
                        new_edges = None
                        pass
                elif toadd[0] in dth.dt.SH_LSTRING_2D:
                    # A list of Shapely LineStrings LS have been enetered.
                    if saa and throw:
                        # STEP-1
                        # Add LS[i].xy[0][$--0,1--$] to slf.$--pnta.x,pntb.x--$
                        # Add LS[i].xy[1][$--0,1--$] to slf.$--pnta.y,pntb.y--$
                        # STEP-2
                        # Add slf.$--pnta,pntb--$.$--x,y--$ to
                        # LS[i].xy[$--0,1--$][$--0,1--$] for all
                        # i ∈ [0, len(toadd)]. For each toadd[i],
                        # make a new edge and build toreturn array
                        # and return the toreturn array

                        # Perform update operations
                        pass
                    if not saa and throw:
                        # Add slf.$--pnta,pntb--$.$--x,y--$ to
                        # LS[i].xy[$--0,1--$][$--0,1--$] for all
                        # i ∈ [0, len(toadd)]. For each toadd[i],
                        # make a new edge and build toreturn array
                        # and return the toreturn array

                        pass
                    if not saa and not throw:
                        # Do nothing
                        new_edges = None
                        pass
                    if saa and not throw:
                        # Add LS[i].xy[0][$--0,1--$] to slf.$--pnta.x,pntb.x--$
                        # Add LS[i].xy[1][$--0,1--$] to slf.$--pnta.y,pntb.y--$

                        # Perform update operations
                        new_edges = None
                        pass
                return new_edges
    def __sub__(self, edge=None, edge_type='edge2d', value=None):
        """
        Subtract a value or edge from this edge's endpoints.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __mul__(self, value=1.0):
        """
        Scale both endpoint coordinates of this edge by a scalar.

        Parameters
        ----------
        value : float, optional
            Scalar multiplier applied to both endpoint coordinates.
            Default is 1.0.

        Returns
        -------
        edge2d
            New edge2d with scaled endpoint coordinates.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(pnta=Point2d(1, 0), pntb=Point2d(2, 0))
        >>> e2 = e * 3.0
        """
        return edge2d(edge_lean=self.edge_lean,
                      points_lean=self.points_lean,
                      method='points2d',
                      pnta=self.pnta*value,
                      pntb=self.pntb*value,
                      )

    def __truediv__(self, value=1.0):
        """
        Divide both endpoint coordinates of this edge by a scalar.

        Parameters
        ----------
        value : float, optional
            Divisor applied to both endpoint coordinates. Default is 1.0.

        Returns
        -------
        edge2d
            New edge2d with divided endpoint coordinates.

        Raises
        ------
        ZeroDivisionError
            If ``value`` is 0.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(pnta=Point2d(2, 0), pntb=Point2d(4, 0))
        >>> e2 = e / 2.0
        """
        return edge2d(edge_lean=self.edge_lean,
                      points_lean=self.points_lean,
                      method='points2d',
                      pnta=self.pnta/value,
                      pntb=self.pntb/value,
                      )

    def __bool__(self):
        """
        Return True if edge length is within the tolerance length of self.

        Returns
        -------
        bool
            True when ``self.length <= self.tlen``.
        """
        return self.length <= self.tlen

    def __len__(self):
        """
        Return the Euclidean length of the edge.

        Returns
        -------
        float
            Same as ``self.length``.
        """
        return self.length

    def __abs__(self):
        """
        Replace endpoint coordinates with their absolute values in-place.

        Returns
        -------
        None
        """
        self.pnta = abs(self.pnta)
        self.pntb = abs(self.pntb)

    def __int__(self, saa=True, throw=True):
        """
        Return a new edge with endpoint coordinates cast to integers.

        Parameters
        ----------
        saa : bool, optional
            Passed to the constituent point __int__ call. Default is True.
        throw : bool, optional
            Passed to the constituent point __int__ call. Default is True.

        Returns
        -------
        edge2d
            New edge2d with integer endpoint coordinates.
        """
        e = edge2d(edge_lean=self.edge_lean,
                   points_lean=self.points_lean,
                   method='points2d',
                   pnta=self.pnta.__int__(saa=False, throw=True),
                   pntb=self.pntb.__int__(saa=False, throw=True),
                   )
        return e

    def __float__(self):
        """
        Float conversion of edge endpoint coordinates.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __complex__(self):
        """
        Complex conversion of the edge object.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __trunc__(self):
        """
        Truncate endpoint coordinates toward zero.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __round__(self, nDigits):
        """
        Return a new edge with endpoint coordinates rounded to `nDigits`.

        Parameters
        ----------
        nDigits : int
            Number of decimal places to round to.

        Returns
        -------
        edge2d
            New edge2d with rounded endpoint coordinates.
        """
        return edge2d(edge_lean=self.edge_lean,
                      points_lean=self.points_lean,
                      method='points2d',
                      pnta=self.pnta.round_round(saa=True, throw=True),
                      pntb=self.pntb.round_round(saa=True, throw=True)
                      )

    def __ceil__(self, saa=True, throw=True):
        """
        Return a new edge with endpoint coordinates rounded up (ceiling).

        Parameters
        ----------
        saa : bool, optional
            Passed to constituent point ceil call. Default is True.
        throw : bool, optional
            Passed to constituent point ceil call. Default is True.

        Returns
        -------
        edge2d
            New edge2d with ceiling-rounded endpoint coordinates.
        """
        return edge2d(edge_lean=self.edge_lean,
                      points_lean=self.points_lean,
                      method='points2d',
                      pnta=self.pnta.round_ceil(saa=True, throw=True),
                      pntb=self.pntb.round_ceil(saa=True, throw=True)
                      )

    def __iter__(self):
        """
        Iterate over constituent point objects.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __next__(self):
        """
        Advance the iterator over constituent points.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def __floor__(self):
        """
        Return a new edge with endpoint coordinates rounded down (floor).

        Returns
        -------
        edge2d
            New edge2d with floor-rounded endpoint coordinates.
        """
        return edge2d(edge_lean=self.edge_lean,
                      points_lean=self.points_lean,
                      method='points2d',
                      pnta=self.pnta.round_floor(saa=True, throw=True),
                      pntb=self.pntb.round_floor(saa=True, throw=True)
                      )

    def __repr__(self):
        """
        Return a human-readable string representation of this edge.

        Returns
        -------
        str
            Format: ``upxo.e2d[(xa, ya)⚯(xb, yb)⊢⊣length]``
        """
        str1 = f'upxo.e2d[({round(self.pnta.x,4)}, {round(self.pnta.y,4)})'
        str2 = f'⚯({round(self.pntb.x,4)}, {round(self.pntb.y,4)})⊢⊣'
        str3 = f'{round(self.length, 4)}]'
        return str1 + str2 + str3

    @ property
    def _status_(self):
        """
        Status update to developer. Not aimed at users

        Returns
        -------
        None.

        """
        print(f'edge2d object lean value = {self.edge_lean}')
        print(f'    pnta object lean value = {self.pnta.lean}')
        print(f'    pntb object lean value = {self.pntb.lean}')
        print(f'Edge2d object lean value = {self.edge_lean}')
        if self.has('mid'):
            print(f'mid exists. Length: {len(self.rid)}')
        else:
            print(f'mid does not exist. It will be made now.')
            self.mid = id(self)
        if hasattr(self, 'length'):
            print('    edge2d length = {self.length}')

    def has(self, suffix):
        """
        Check whether this edge object has the given attribute.

        Accepts several aliases for common attribute names (e.g. 'point1' maps
        to 'pnta', 'e2dl' maps to 'length').

        Parameters
        ----------
        suffix : str
            Attribute name or recognised alias to look up.

        Returns
        -------
        has_truth : bool
            True if the attribute (or its canonical name) exists on self.
        """
        suffix = suffix.lower()
        if suffix in ('edge_lean', 'edge2d_lean'):
            suffix = 'edge_lean'
        elif suffix in ('points_lean'):
            suffix = 'points_lean'
        elif suffix in ('__x'):
            suffix = '__x'
        elif suffix in ('__y'):
            suffix = '__y'
        elif suffix in ('pnta', 'point1', 'point.a', 'point_a'):
            suffix = 'pnta'
        elif suffix in ('pntb', 'point2', 'point.b', 'point_b'):
            suffix = 'pntb'
        elif suffix in ('xycen', 'cen_coords', 'cen_coord',
                        'cencoord', 'coordcen', 'coord_cen'):
            suffix = 'xycen'
        elif suffix in ('length', 'edge_length', 'e2d_length',
                        'e2dlength', 'e2dl'):
            suffix = 'length'

        has_truth = False
        if hasattr(self, suffix):
            has_truth = True
        return has_truth

    @property
    def centroid(self):
        """
        Midpoint of the edge.

        Returns
        -------
        Point2d
            Point at ``(pnta + pntb) / 2``.
        """
        return (self.pnta+self.pntb)*0.5

    def negx(self):
        """
        Make negative of the x-coordinates of constituent point objects
        This has the effect of mirroring the edge about the y-axis

        Returns
        -------
        None.

        """
        self.pnta.x *= -1
        self.pntb.x *= -1
        self.post_displacement_updates()

    def negy(self):
        """
        Make negative of the y-coordinates of constituent point objects
        This has the effect of mirroring the edge about the x-axis

        Returns
        -------
        None.

        """
        self.pnta.y *= -1
        self.pntb.y *= -1
        self.post_displacement_updates()

    def negxy(self):
        """
        Make negative of the x- and y- coordinates of constituent point objects

        Returns
        -------
        None.

        """
        self.pnta.x *= -1
        self.pntb.x *= -1
        self.pnta.y *= -1
        self.pntb.y *= -1
        self.post_displacement_updates()

    def mirror_x(self):
        """
        Mirror about the x-axis

        Returns
        -------
        None.

        """
        self.negy()

    def mirror_y(self):
        """
        Mirror about the y-axis

        Returns
        -------
        None.

        """
        self.negx()

    def displace(self, xyincr):
        """
        Displace the edge2d by xyincr = [delx, dely]

        Parameters
        ----------
        xyincr : list/tuple

        Returns
        -------
        None.

        """
        self.pnta.translate(method='xyincr', xyincr=xyincr,
                            saa=True, make_new=False, throw=False)
        self.pntb.translate(method='xyincr', xyincr=xyincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_a(self, xyincr):
        """
        Displace pnta by an ``[dx, dy]`` increment in-place.

        Parameters
        ----------
        xyincr : list or tuple of float
            ``[dx, dy]`` displacement applied to pnta.

        Returns
        -------
        None
        """
        self.pnta.translate(method='xyincr', xyincr=xyincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_ax(self, xincr):
        """
        Displace the x-coordinate of pnta by ``xincr`` in-place.

        Parameters
        ----------
        xincr : float or int
            Displacement to add to pnta.x.

        Returns
        -------
        None
        """
        self.pnta.translate(method='xincr', xincr=xincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_ay(self, yincr):
        """
        Displace the y-coordinate of pnta by ``yincr`` in-place.

        Parameters
        ----------
        yincr : float or int
            Displacement to add to pnta.y.

        Returns
        -------
        None
        """
        self.pnta.translate(method='yincr', yincr=yincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_b(self, xyincr):
        """
        Displace pntb by an ``[dx, dy]`` increment in-place.

        Parameters
        ----------
        xyincr : list or tuple of float
            ``[dx, dy]`` displacement applied to pntb.

        Returns
        -------
        None
        """
        self.pntb.translate(method='xyincr', xyincr=xyincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_bx(self, xincr):
        """
        Displace the x-coordinate of pntb by ``xincr`` in-place.

        Parameters
        ----------
        xincr : float or int
            Displacement to add to pntb.x.

        Returns
        -------
        None
        """
        self.pntb.translate(method='xincr', xincr=xincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def displace_by(self, yincr):
        """
        Displace the y-coordinate of pntb by ``yincr`` in-place.

        Parameters
        ----------
        yincr : float or int
            Displacement to add to pntb.y.

        Returns
        -------
        None
        """
        self.pntb.translate(method='yincr', yincr=yincr,
                            saa=True, make_new=False, throw=False)
        self.post_displacement_updates()

    def stretch(self, method='strain', center='A', strain=None,):
        """
        Stretch the edge by a given strain about a center point.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def rotate(self, method='point2d', about=None, angle=0,):
        """
        Rotate the edge about a point by the given angle.

        Negative angles rotate clockwise; positive angles anti-clockwise.

        Parameters
        ----------
        method : str, optional
            Pivot specification: 'point2d'/'point' or 'coord'/'coords'.
            Default is 'point2d'.
        about : Point2d, list, or str, optional
            Pivot. Accepts a Point2d, a coordinate pair, or 'A', 'B',
            'center'. Default is None.
        angle : float, optional
            Rotation angle in degrees. Default is 0.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def move_a_to(self, obj, method='update'):
        """
        Move pnta to a new location.

        Parameters
        ----------
        obj : list, tuple, or Point2d
            Target position as a 2-element coordinate pair or a UPXO Point2d.
        method : str, optional
            'update' — overwrite existing pnta coordinates in-place.
            'replace' — swap pnta with a newly created point object.
            Default is 'update'.

        Returns
        -------
        None

        Notes
        -----
        'replace' changes the ``id(pnta)`` object identity; 'update' preserves it.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(pnta=Point2d(-1.0, 0.0), pntb=Point2d(1.0, 0.0))
        >>> e1.move_a_to([10, 10], method='update')
        >>> e1.move_a_to(Point2d(10, 10), method='replace')
        """
        if type(obj) in dth.dt.ITERABLES:
            # obj is a coordinate pair
            if method == 'replace':
                # a new pnta at the given location will be made and the
                # existing pnta will be replace with this.
                self.pnta = point2d(x=obj[0], y=obj[1], lean=self.pnta.lean)
            elif method == 'update':
                # existing pnta will be updated to the new coordinates
                self.update_point('a', obj, method='coord')
        elif str(type(obj)) == "<class 'UPXO-point.point2d'>":
            # obj is a UPXO point2d
            if method == 'replace':
                # a new pnta at the given location will be made and the
                # existing pnta will be replace with this.
                self.pnta = obj
            elif method == 'update':
                # existing pnta will be updated to the new coordinates
                self.update_point('a', obj, method='up2d')
        self.calculate_level01_basics(length_method='points')

    def move_b_to(self, obj, method='update'):
        """
        Move pntb to a new location.

        Parameters
        ----------
        obj : list, tuple, or Point2d
            Target position as a 2-element coordinate pair or a UPXO Point2d.
        method : str, optional
            'update' — overwrite existing pntb coordinates in-place.
            'replace' — swap pntb with a newly created point object.
            Default is 'update'.

        Returns
        -------
        None

        Notes
        -----
        'replace' changes the ``id(pntb)`` object identity; 'update' preserves it.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(pnta=Point2d(-1.0, 0.0), pntb=Point2d(1.0, 0.0))
        >>> e1.move_b_to([10, 10], method='update')
        >>> e1.move_b_to(Point2d(10, 10), method='replace')
        """
        if type(obj) in dth.dt.ITERABLES:
            # obj is a coordinate pair
            if method == 'replace':
                # a new pnta at the given location will be made and the
                # existing pnta will be replace with this.
                self.pntb = point2d(x=obj[0], y=obj[1], lean=self.pntb.lean)
            elif method == 'update':
                # existing pnta will be updated to the new coordinates
                self.update_point('b', obj, method='coord')
        elif str(type(obj)) == "<class 'UPXO-point.point2d'>":
            # obj is a UPXO point2d
            if method == 'replace':
                # a new pnta at the given location will be made and the
                # existing pnta will be replace with this.
                self.pntb = obj
            elif method == 'update':
                # existing pnta will be updated to the new coordinates
                self.update_point('b', obj, method='up2d')
        self.calculate_level01_basics(length_method='points')

    def geo_sweep(self, method='along_edge', edge=None,
                  length=0.0, angle=None, uself=None, throw=False,):
        """
        Sweep this edge to generate a 2-D face or surface patch.

        Parameters
        ----------
        method : str, optional
            Sweep direction specification:
            'along_edge'/'edge', 'by_length'/'length',
            'along_coord'/'coord', 'angle',
            'miller'/'mi'/'miller_indices'/'uvw'.
            Default is 'along_edge'.
        edge : edge2d, optional
            Reference edge defining the sweep direction. Default is None.
        length : float, optional
            Sweep distance when ``method='by_length'``. Default is 0.0.
        angle : float, optional
            Sweep angle in degrees when ``method='angle'``. Default is None.
        uself : float, optional
            Fraction of self's length to use as sweep distance. Default is None.
        throw : bool, optional
            If True, return the swept geometry. Default is False.

        Limitations
        -----------
        - Not yet implemented; returns None.
        """
        pass

    def overlaps(self, obj=None, method='parallelity', tdist=0.0, tang=0.0):
        """
        Check whether another edge spatially overlaps self.

        Both endpoints of ``obj`` must be contained within self (using
        :meth:`contains_point`) for overlap to be True.

        Parameters
        ----------
        obj : edge2d, optional
            Edge to test against self. Default is None.
        method : str, optional
            Method passed to :meth:`contains_point`. Currently only
            'parallelity' is supported. Default is 'parallelity'.
        tdist : float, optional
            Tolerance distance for the contains-point test. Default is 0.0.
        tang : float, optional
            Tolerance angle (reserved, not yet used). Default is 0.0.

        Returns
        -------
        overlaps : bool
            True if both endpoints of ``obj`` lie on self.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(pnta=Point2d(-1.0, 0.0), pntb=Point2d(1.0, 0.0))
        >>> e2 = edge2d(pnta=Point2d(-0.5, 0.0), pntb=Point2d(0.5, 0.0))
        >>> e1.overlaps(obj=e2)
        """
        if not str(type(obj)) == "<class 'UPXO-edge.edge2d'>":
            print('Please input correct geometry object')
        else:
            PNT_A, PNT_B = obj.pnta, obj.pntb
            decision_PNT_A = self.contains_point(obj=PNT_A,
                                                 method=method,
                                                 tdist=tdist
                                                 )
            decision_PNT_B = self.contains_point(obj=PNT_B,
                                                 method=method,
                                                 tdist=tdist
                                                 )
            decision_PNT_A_B = [decision_PNT_A[0], decision_PNT_B[0]]
            if decision_PNT_A_B[0] and decision_PNT_A_B[1]:
                overlaps = True
            else:
                overlaps = False
        return overlaps

    def swap_points(self):
        """
        Swap pnta and pntb in-place, including their pmids.

        Internally called by :meth:`reverse`.

        Returns
        -------
        None

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 2), pntb=Point2d(10, 12))
        >>> e.swap_points()
        """
        self.pnta, self.pntb = self.pntb, self.pnta

    def reverse(self, reverse_pmid=True, make_new_points=False):
        """
        Reverse the direction of this edge (swap pnta and pntb).

        Parameters
        ----------
        reverse_pmid : bool, optional
            If True, swap the full point objects (including pmids) via
            :meth:`swap_points`. If False, only coordinates are swapped
            while pmid identity is preserved. Default is True.
        make_new_points : bool, optional
            If True, create brand-new point objects with swapped coordinates;
            ``reverse_pmid`` has no effect in this case. Default is False.

        Returns
        -------
        None

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 2), pntb=Point2d(10, 12))
        >>> e.reverse()
        >>> e.reverse(reverse_pmid=False)
        >>> e.reverse(make_new_points=True)
        """
        if reverse_pmid:
            self.swap_points()
        else:
            _x_, _y_ = [self.pnta.x, self.pntb.x], [self.pnta.y, self.pntb.y]
            self.pnta.x, self.pnta.y = _x_[1], _y_[1]
            self.pntb.x, self.pntb.y = _x_[0], _y_[0]
        if make_new_points:
            _x_, _y_ = [self.pnta.x, self.pntb.x], [self.pnta.y, self.pntb.y]
            self.pnta = point2d(_x_[1], _y_[1], ignore=self.pnta.lean)
            self.pntb = point2d(_x_[0], _y_[0], ignore=self.pntb.lean)
        pass

    def _intersect_(self, a1, a2, b1, b2):
        """
        Return the intersection point of two infinite lines in homogeneous coordinates.

        Based on: stackoverflow.com/questions/3252194 by Norbu Tsering.

        Parameters
        ----------
        a1 : array-like of float
            ``[x, y]`` — first point on line A.
        a2 : array-like of float
            ``[x, y]`` — second point on line A.
        b1 : array-like of float
            ``[x, y]`` — first point on line B.
        b2 : array-like of float
            ``[x, y]`` — second point on line B.

        Returns
        -------
        tuple of float
            ``(x, y)`` intersection, or ``(inf, inf)`` when lines are parallel.

        Limitations
        -----------
        - Operates on infinite lines, not bounded edge segments.
        """
        _cross_ = np.cross
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = _cross_(h[0], h[1])  # get first line
        l2 = _cross_(h[2], h[3])  # get second line
        x, y, z = _cross_(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def intersect_with_edges2d(self, edges):
        """
        Compute intersection points of self's infinite line with each edge's infinite line.

        Uses :meth:`_intersect_` which works on infinite lines, not bounded segments.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            Edges to intersect self against.

        Returns
        -------
        list of tuple of float
            Intersection coordinates ``(x, y)`` per edge. Returns
            ``(inf, inf)`` for parallel pairs.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e12 = edge2d(pnta=Point2d(-1, 0), pntb=Point2d(0, 1))
        >>> e34 = edge2d(pnta=Point2d(0, -1), pntb=Point2d(0, 1))
        >>> e12.intersect_with_edges2d([e34])
        """
        if type(edges) not in dth.dt.ITERABLES:
            edges = [edges]

        return [self._intersect_((self.pnta.x, self.pnta.y),
                                 (self.pntb.x, self.pntb.y),
                                 (e.pnta.x, e.pnta.y),
                                 (e.pntb.x, e.pntb.y)
                                 ) for e in edges]

    def edge2d_intersection(self,
                            edge,
                            return_ratios=True,
                            sort=True,
                            print_=False,
                            ):
        """
        Find the intersection geometry between self and another edge2d.

        Handles collinear, overlapping, and crossing cases. Returns actual
        intersection points on the bounded segments, not just the infinite-line
        crossing point.

        Parameters
        ----------
        edge : edge2d
            The other edge to intersect with.
        return_ratios : bool, optional
            If True, also return the parametric ratios (distance from pnta /
            self.length) for each intersection point. Default is True.
        sort : bool, optional
            If True, sort intersection points by distance from ``self.pnta``.
            Default is True.
        print_ : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        tuple
            ``(intersection_point,)`` when ``return_ratios=False``, or
            ``(intersection_point, ratios)`` when ``return_ratios=True``.
            ``intersection_point`` is an ``(N, 2)`` ndarray; ``ratios`` is a
            1-D ndarray of parametric values in ``[0, 1]``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> e2 = edge2d(method='up2d', pnta=Point2d(0.2, 0), pntb=Point2d(0.8, 0))
        >>> pts, ratios = e1.edge2d_intersection(e2, return_ratios=True)
        >>> e3 = edge2d(method='up2d', pnta=Point2d(0.1, -1), pntb=Point2d(0.9, 5))
        >>> e1.edge2d_intersection(e3)
        """
        __PRINT_INTERSECTION_POINTS = False
        # -------------------------------------
        intersection_point = self.intersect_with_edges2d([edge])
        if intersection_point[0][0] == np.inf:
            intersection_point = []
            # edge is either coinciding, parallel or contained inside self
            # ---------------------
            # Check for coincinding points
            coincide_pnta = list(np.where(np.array(edge.pnta == [self.pnta,
                                                                 self.pntb]))[0])
            coincide_pntb = list(np.where(np.array(edge.pntb == [self.pnta,
                                                                 self.pntb]))[0])
            # ---------------------
            if coincide_pnta:
                if len(coincide_pnta) == 1:
                    if coincide_pnta[0] == 0:
                        if print_:
                            print('pnta of edge coincides with pnta of self')
                        intersection_point.append((self.pnta.x, self.pnta.y))
                    elif coincide_pnta[0] == 1:
                        if print_:
                            print('pnta of edge coincides with pntb of self')
                        intersection_point.append((self.pntb.x, self.pntb.y))
            if coincide_pntb:
                if len(coincide_pntb) == 1:
                    if coincide_pntb[0] == 0:
                        if print_:
                            print('pnta of edge coincides with pnta of self')
                        intersection_point.append((self.pnta.x, self.pnta.y))
                    elif coincide_pntb[0] == 1:
                        if print_:
                            print('pnta of edge coincides with pntb of self')
                        intersection_point.append((self.pntb.x, self.pntb.y))
            if len(coincide_pnta) == 2 and len(coincide_pntb) == 2:
                if print_:
                    print('edge has coinciding end points')
                intersection_point.append((self.pnta.x, self.pnta.y))
                intersection_point.append((self.pntb.x, self.pntb.y))
            # ---------------------
            # if no points are coinciding, check if they are in or out:
            if not coincide_pnta:
                #print('NUMBER 1')
                #print(edge.pnta)
                contain_pnta = self.contains_point(obj=edge.pnta,
                                                   method='parallelity',
                                                   tdist=0.0)
                #print(contain_pnta)
                if contain_pnta[0]:
                    # POINT A OF EDGE IS INSIDE SELF
                    intersection_point.append((edge.pnta.x, edge.pnta.y))
                else:
                    if contain_pnta[1]:
                        # POINT A OF EDGE FALLS ON EXTENDED SELF, i.e. OUT
                        # outside_pointa = True
                        if edge.contains_point(obj=self.pnta,
                                               method='parallelity',
                                               tdist=0.0)[0]:
                            intersection_point.append((self.pnta.x,
                                                       self.pnta.y))
                        if edge.contains_point(obj=self.pntb,
                                               method='parallelity',
                                               tdist=0.0)[0]:
                            intersection_point.append((self.pntb.x,
                                                       self.pntb.y))
                    else:
                        # POINT A OF EDGE FALLS ELSEWHERE, i.e. OUT
                        # nothing more to do here
                        # this branch will not even be encountered
                        pass
            if not coincide_pntb:
                #print('-------------')
                #print('NUMBER 2')
                #print(edge.pntb)
                contain_pntb = self.contains_point(obj=edge.pntb,
                                                   method='parallelity',
                                                   tdist=0.0)
                #print(contain_pntb)
                if contain_pntb[0]:
                    # POINT B OF EDGE IS INSIDE SELF
                    intersection_point.append((edge.pntb.x, edge.pntb.y))
                else:
                    if contain_pntb[1]:
                        # POINT B OF EDGE FALLS ON EXTENDED SELF, i.e. OUT
                        # outside_pointb = True
                        if edge.contains_point(obj=self.pnta,
                                               method='parallelity',
                                               tdist=0.0)[0]:
                            intersection_point.append((self.pnta.x,
                                                       self.pnta.y))
                        if edge.contains_point(obj=self.pntb,
                                               method='parallelity',
                                               tdist=0.0)[0]:
                            intersection_point.append((self.pntb.x,
                                                       self.pntb.y))
                    else:
                        # POINT B OF EDGE FALLS ELSEWHERE, i.e. OUT
                        # nothing more to do here
                        # this branch will not even be encountered
                        pass
        # -----------------------------------
        intersection_point = np.array(intersection_point)
        if __PRINT_INTERSECTION_POINTS:
            print('============================')
            print(intersection_point)
            print('============================')
        # -----------------------------------
        # Make unique of points
        if len(intersection_point) > 0:
            _mp_ = mulpoint2d(method='xy_pair_list',
                              coordxy=intersection_point)
            intersection_point = np.vstack((_mp_.locx, _mp_.locy)).T
            if sort:
                xy = intersection_point.T
                _d_ = np.sqrt((xy[0]-self.pnta.x)**2 + (xy[1]-self.pnta.y)**2)
                intersection_point = intersection_point[np.argsort(_d_)]
            to_return = (intersection_point, )
            if return_ratios:
                xy = intersection_point.T
                _d_ = np.sqrt((xy[0]-self.pnta.x)**2 + (xy[1]-self.pnta.y)**2)
                ratios = _d_/self.length
                to_return = (intersection_point, ratios)
        else:
            to_return = (intersection_point, )
        # -----------------------------------
        if print_:
            print(f'No. of intersections = {len(intersection_point)}')
        # -----------------------------------
        return to_return

    def edge2d_intersections(self,
                             edges,
                             return_ratios=False,
                             sort=True,
                             print_=False):
        """
        Find intersection geometry between self and a list of edges.

        Calls :meth:`edge2d_intersection` for each edge in ``edges``.

        Parameters
        ----------
        edges : edge2d or list of edge2d
            Edges to intersect self against.
        return_ratios : bool, optional
            Pass through to :meth:`edge2d_intersection`. Default is False.
        sort : bool, optional
            Pass through to :meth:`edge2d_intersection`. Default is True.
        print_ : bool, optional
            Pass through to :meth:`edge2d_intersection`. Default is False.

        Returns
        -------
        list of tuple
            One result tuple per edge (see :meth:`edge2d_intersection`).

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e1 = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> e2 = edge2d(method='up2d', pnta=Point2d(0.2, 0), pntb=Point2d(0.8, 0))
        >>> e3 = edge2d(method='up2d', pnta=Point2d(0.2, -2), pntb=Point2d(0.8, 2))
        >>> e1.edge2d_intersections([e2, e3], return_ratios=True)
        """
        # Make list if already not
        if type(edges) not in dth.dt.ITERABLES:
            edges = [edges]
        # -----------------------------------------
        # Find intersections and prepare to return
        intersections = []
        for edge in edges:
            intersections.append(self.edge2d_intersection(edge,
                                                          return_ratios=return_ratios,
                                                          sort=sort,
                                                          print_=print_))
        return intersections

    def muledge2d_intersections(self, me):
        """
        Find intersections of self with all edges in a muledge2d object.

        Delegates to :meth:`edge2d_intersections` with ``return_ratios=True``.

        Parameters
        ----------
        me : muledge2d
            Multi-edge container whose ``.edges`` list is intersected against self.

        Returns
        -------
        list of tuple
            One result tuple per edge in ``me.edges`` (see
            :meth:`edge2d_intersection`).

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> from upxo.geoEntities.muledge2d import muledge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0.8, 0.2), pntb=Point2d(0.2, 0.8))
        >>> me = muledge2d(method='clist', ordered=True, closed=False,
        ...                clist=[[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> e.muledge2d_intersections(me)
        """
        intersections = self.edge2d_intersections(me.edges, return_ratios=True)
        return intersections
        # self.muledge2d_intersections(me_obj.edges)

    def split_at_point(self, obj, new_edge_location=0):
        """
        Split this edge at a point, returning the new child edge.

        Self is modified in-place (its pntb or pnta is updated to ``obj``);
        a new edge2d covering the remainder is returned.

        Parameters
        ----------
        obj : Point2d or list/tuple of float
            Split point. Must lie on the bounded edge (not at the endpoints).
            A 2-element coordinate pair is automatically promoted to Point2d.
        new_edge_location : int, optional
            0 — new edge covers ``[obj, old_pntb]``, self covers ``[pnta, obj]``.
            1 — new edge covers ``[old_pnta, obj]``, self covers ``[obj, pntb]``.
            Default is 0.

        Returns
        -------
        new_edge : edge2d or list
            The new child edge, or an empty list when the split point does not
            lie on the bounded segment (including coincident-endpoint cases).

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(1, 0), pntb=Point2d(0, 0))
        >>> new_edge = e.split_at_point(Point2d(0.5, 0), new_edge_location=0)
        >>> new_edge = e.split_at_point(Point2d(0.1, 0), new_edge_location=0)
        """
        if type(obj) in dth.dt.ITERABLES:
            # If obj is entered as a coordinate pair
            obj = point2d(obj[0], obj[1], lean=self.pnta.lean)
        # If obj belongs to UPXO point2d, it will be directly used
        coincide_pnta = np.where(np.array(self.pnta == obj))[0]
        coincide_pntb = np.where(np.array(self.pntb == obj))[0]
        # Address the special case when self edge length is zero
        if list(coincide_pnta) and list(coincide_pntb):
            if np.prod(coincide_pnta == coincide_pnta) == 1:
                # point coincides with both points of self edge.
                # this is only possible when the edge has zero length
                # THIS BRANCH IGNORED
                print('Point coincides with both self edge points')
                print('Edge not split')
                new_edge = []
        # Address the case point coincides with pnta of self edge and not
        # with pntb of self edge
        if list(coincide_pnta) or list(coincide_pntb):
            # point coincides with one of the points of self edge
            print('Point coincides with a point of self edge. Edge not split')
            new_edge = []
        # Adress the remaining cases
        if not list(coincide_pnta) and not list(coincide_pntb):
            if self.contains_point(obj=obj,
                                   method='parallelity',
                                   tdist=0.0)[0]:
                # Address the case where point lies inbetween pnta and pntb of
                # self edge
                if new_edge_location == 0:
                    # print([id(self.pnta), id(self.pntb)])
                    # print(id(obj))
                    self.pntb, new_edge = obj, edge2d(method='up2d',
                                                      pnta=obj,
                                                      pntb=self.pntb)
                    # print([id(self.pnta), id(self.pntb)])
                    # print(id(obj))
                    # print([id(new_edge.pnta), id(new_edge.pntb)])
                else:
                    new_edge, self.pnta = edge2d(method='up2d',
                                                 pnta=self.pnta,
                                                 pntb=obj), obj
                # --------------------------------
                # Update self properties
                self.calculate_level01_basics(length_method='points')
            else:
                # Address the case where point has a non-zero normal to
                # self edge
                new_edge = []
        return new_edge

    def split_by_ratio(self, ratio, new_edge_location=0):
        """
        Split this edge at a fractional position along its length.

        Computes the split point as ``pnta + ratio * (pntb - pnta)`` then
        delegates to :meth:`split_at_point`.

        Parameters
        ----------
        ratio : float
            Parametric position in the open interval (0, 1).
        new_edge_location : int, optional
            Passed through to :meth:`split_at_point`. Default is 0.

        Returns
        -------
        new_edge : edge2d or list
            See :meth:`split_at_point`.

        Raises
        ------
        Prints a message (no exception raised) when ``ratio`` is not in (0, 1).

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> new_edge = e.split_by_ratio(0.5, new_edge_location=0)
        """
        if ratio > 0 and ratio < 1:
            point = point2d(self.pnta.x + ratio*(self.pntb.x - self.pnta.x),
                            self.pnta.y + ratio*(self.pntb.y - self.pnta.y),
                            lean=self.pnta.lean)
            new_edge = self.split_at_point(point,
                                           new_edge_location=new_edge_location)
        else:
            print('Please enter 0 < ratio < 1')
        return new_edge

    def split_at_outside_point(self, point):
        """
        Split this edge by inserting an off-edge point as the new pntb.

        Creates a new edge covering the original ``[point, old_pntb]`` range,
        and updates self's pntb to ``point``.

        Parameters
        ----------
        point : Point2d
            The new pntb. May lie off the original edge line.

        Returns
        -------
        new_edge : edge2d
            New edge covering ``[point, old_pntb]``.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> new_edge = e.split_at_outside_point(Point2d(0.5, 0.5))
        """
        new_edge, self.pntb = edge2d(pnta=point,
                                     pntb=self.pntb,
                                     edge_lean='ignore'), point
        return new_edge

    def gen_npoints(self,
                    n,
                    distribution='polynomial',
                    coeff=[0, 1, 2],
                    weights=[],
                    start='a',
                    function='0.5*exp(x) + x^2/2',
                    constraints=[0.1, 0.1, 0.05, 5, 'uniform'],
                    make_mulpoint=False
                    ):
        """
        Generate n number of points between end points of the self edge in a
        certain distribution.

        Parameters
        ----------
        n : int
            Number of points to create.
        distribution : str, optional
            Provides the type of point distribution.
            The default is 'polynomial'. Other options include:
                'function', 'exponential', 'numpypoly_bl'
                1. 'polynomial': Will use coeff in list format
        coeff : list, optional
            Case-1: When distribution is 'polynomial'
            coeff[0] will be constant term. It will be rendered zero
            internally, regardless of user value. coeff[1] will be the
            coefficient accompanying the first order term.
            Case-2: When distribution if 'exponential'
            EXAPLANTION TO BE WRITTEN ONCE FEATURE BECOMES AVAILABLE.
            The default is [0, 1, 2].
        weights : list, optional
            Provides unit normalized multiplier factors. It will be used to
            calculate x and y locations of points as below:
                x = e.pnta.x + weights*(e.pntb.x-e.pnta.x)
                y = e.pnta.y + weights*(e.pntb.y-e.pnta.y)
            The default is [].
        start : str, optional
            Dictates the direction of inter-point seperation increase.
            If 'a', then more points will be closer to pnta of self edge.
            If 'b', then more points will be closer to pntb of self edge.
            If 'c1' then more points will be closer to pnta and pntb of the
                edge than the centre point.
            If 'c2' then more points will be closer to center point of the edge
                then pnta and pntb of the edge
            if 'c1_k' then behaviour will like 'c1', but points will cluster
                near point which divides the self edge at k*100 % from pnta,
                along the edge length
            if 'c2_k' then behaviour will like 'c2', but points will cluster
                less near point which divides the self edge at k*100 %
                from pnta, along the edge length. Clustering will be at both
                end pnta and pntb.
            The default is 'a'.
        function : str, optional
            FEATURE NOT YET AVAILABLE.
            The default is '0.5*exp(x) + x^2/2'.

        Returns
        -------
        coords : tuple
            list of x-coordinate values and list of y-coordinate values, in
            a tuple.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(-2, -1), pntb=Point2d(1, 3))
        >>> xy = e.gen_npoints(25, distribution='polynomial', coeff=[0, 1, 1, 1], start='a')
        >>> xy = e.gen_npoints(25, distribution='polynomial', coeff=[0, 1, 1, 1], start='c1')
        >>> xy = e.gen_npoints(50, distribution='random',
        ...                    constraints=[0.05, 0.05, 0.5, -1, 'uniform'])
        """
        if type(n) == int:
            pass
        else:
            n = int(n)
        if n < 1:
            print('Please enter valid n: n>2')
            coords = None
        if n in (1, 2):
            coords = self.gen_npoints(3,
                                      distribution='random',
                                      constraints=[0.1, 0.1, 0.1, n, 'uniform']
                                      )
        if n > 2 and type(coeff) in dth.dt.ITERABLES and coeff and len(coeff) > 1:
            if distribution == 'random':
                # constraints=[0.1, 0.1, 0.1, 20, 'uniform']
                if constraints[-1] == 'uniform':
                    if 0 < abs(constraints[0]) < 1 and 0 < abs(constraints[1]) < 1:
                        if abs(constraints[0]) + abs(constraints[1]) <= self.length*0.5:
                            lx = self.pntb.x - self.pnta.x
                            ly = self.pntb.y - self.pnta.y

                            spacer_ax = constraints[0]*lx
                            spacer_ay = constraints[0]*ly
                            spacer_bx = constraints[1]*lx
                            spacer_by = constraints[1]*ly

                            startx = self.pnta.x + spacer_ax
                            endx = self.pntb.x - spacer_bx

                            starty = self.pnta.y + spacer_ay
                            endy = self.pntb.y - spacer_by

                            spacing_minx = lx*constraints[2]
                            spacing_miny = ly*constraints[2]

                            EPS = 0.000000000001

                            if abs(lx) <= EPS and abs(ly) <= EPS:
                                print('Edge not suitable for sub-division')
                            if abs(lx) <= EPS and abs(ly) > EPS:
                                from upxo_math import MATH_ruc as ruc
                                ny = ceil(ly/spacing_miny)
                                if starty < endy:
                                    _y = np.array(sorted(islice(ruc(starty,
                                                                    endy,
                                                                    spacing_miny),
                                                                ny)
                                                         )
                                                  )
                                if starty > endy:
                                    _y = np.array(sorted(islice(ruc(endy,
                                                                    starty,
                                                                    abs(spacing_miny)),
                                                                ny)
                                                         )
                                                  )
                                if n < len(_y):
                                    n_subsets = np.arange(n)
                                    random_indices = np.arange(len(_y))
                                    np.random.shuffle(random_indices)
                                    _y = _y[random_indices]
                                    _y = _y[n_subsets]
                                    sort_ind = np.argsort(_y)
                                    _y = _y[sort_ind]

                                if constraints[3] in (1, 2):
                                    n_subsets = np.arange(constraints[3])
                                    random_indices = np.arange(len(_y))
                                    np.random.shuffle(random_indices)
                                    _y = _y[random_indices]
                                    _y = _y[n_subsets]
                                    sort_ind = np.argsort(_y)
                                    _y = _y[sort_ind]

                                _y = np.hstack(([self.pnta.y],
                                                _y,
                                                [self.pntb.y])
                                               )
                                _x = np.linspace(self.pnta.x,
                                                 self.pntb.x,
                                                 len(_y)
                                                 )
                                coords = (_x, _y)

                            if abs(ly) <= EPS and abs(lx) > EPS:
                                from upxo_math import MATH_ruc as ruc
                                nx = ceil(lx/spacing_minx)
                                if startx < endx:
                                    _x = np.array(sorted(islice(ruc(startx,
                                                                    endx,
                                                                    spacing_minx),
                                                                nx)
                                                         )
                                                  )
                                if startx > endx:
                                    _x = np.array(sorted(islice(ruc(endx,
                                                                    startx,
                                                                    abs(spacing_minx)),
                                                                nx)
                                                         )
                                                  )
                                if n < len(_x):
                                    n_subsets = np.arange(n)
                                    random_indices = np.arange(len(_x))
                                    np.random.shuffle(random_indices)
                                    _x = _x[random_indices]
                                    _x = _x[n_subsets]
                                    sort_ind = np.argsort(_x)
                                    _x = _x[sort_ind]

                                if constraints[3] in (1, 2):
                                    n_subsets = np.arange(constraints[3])
                                    random_indices = np.arange(len(_x))
                                    np.random.shuffle(random_indices)
                                    _x = _x[random_indices]
                                    _x = _x[n_subsets]
                                    sort_ind = np.argsort(_x)
                                    _x = _x[sort_ind]

                                _x = np.hstack(([self.pnta.x],
                                                _x,
                                                [self.pntb.x])
                                               )
                                _y = np.linspace(self.pnta.y,
                                                 self.pntb.y,
                                                 len(_x)
                                                 )
                                coords = (_x, _y)

                            if abs(lx) >= EPS and abs(ly) >= EPS:
                                from upxo_math import MATH_ruc as ruc
                                nx = ceil(lx/spacing_minx)
                                if startx < endx:
                                    # no problem here
                                    _x = np.array(sorted(islice(ruc(startx,
                                                                    endx,
                                                                    spacing_minx),
                                                                nx)
                                                         )
                                                  )
                                if startx > endx:
                                    # make endx as startx and startx as endx
                                    _x = np.array(sorted(islice(ruc(endx,
                                                                    startx,
                                                                    abs(spacing_minx)),
                                                                nx)
                                                         )
                                                  )
                                    _x = np.flip(_x)
                                # ------------------------------
                                # y-y1 = ((y2-y1)/(x2-x1))(x-x1)
                                # y = ((y2-y1)/(x2-x1))(x-x1) + y1
                                m = ly/lx
                                _y = np.array([m*(__x - self.pnta.x)
                                               + self.pnta.y for __x in _x])
                                if n < len(_x):
                                    n_subsets = np.arange(n)
                                    random_indices = np.arange(len(_x))
                                    np.random.shuffle(random_indices)
                                    _x, _y = _x[random_indices], _y[random_indices]
                                    _x, _y = _x[n_subsets], _y[n_subsets]
                                    sort_ind = np.argsort(_x)
                                    _x, _y = _x[sort_ind], _y[sort_ind]

                                if constraints[3] in (1, 2):
                                    n_subsets = np.arange(constraints[3])
                                    random_indices = np.arange(len(_x))
                                    np.random.shuffle(random_indices)
                                    _x, _y = _x[random_indices], _y[random_indices]
                                    _x, _y = _x[n_subsets], _y[n_subsets]
                                    sort_ind = np.argsort(_x)
                                    _x, _y = _x[sort_ind], _y[sort_ind]

                                _x = np.hstack(([self.pnta.x],
                                                _x,
                                                [self.pntb.x])
                                               )
                                _y = np.hstack(([self.pnta.y],
                                                _y,
                                                [self.pntb.y])
                                               )
                                coords = (_x, _y)

            if distribution == 'polynomial':
                from numpy.polynomial import Polynomial as numpypoly
                # Make coeff valid
                coeff[0] = 0
                if not weights:
                    # Make numpy polynomial
                    numpy_poly_function = numpypoly(coeff)
                    # Evaluate numpy over unit space and normalize
                    factor = numpy_poly_function(np.linspace(0, 1, n))
                elif weights and len(weights)==n:
                    factor = np.array(weights)
                if factor.max() != 0:
                    factor /= factor.max()
                # Map variation to the edge space
                if start == 'a':
                    x = self.pnta.x + factor*(self.pntb.x-self.pnta.x)
                    y = self.pnta.y + factor*(self.pntb.y-self.pnta.y)
                    coords = (x, y)
                elif start == 'b':
                    x = np.flip(self.pntb.x - factor*(self.pntb.x-self.pnta.x))
                    y = np.flip(self.pntb.y - factor*(self.pntb.y-self.pnta.y))
                    coords = (x, y)
                elif start == 'c1':
                    coords = self.gen_npoints(n,
                                              distribution=distribution,
                                              coeff=coeff,
                                              start='c1_0.5')
                elif start == 'c2':
                    coords = self.gen_npoints(n,
                                              distribution=distribution,
                                              coeff=coeff,
                                              start='c2_0.5')
                elif start[:3] in ('c1_', 'c2_'):
                    if start[3:].replace('.','',1).isdigit():
                        if 0 < float(start[3:]) < 1:
                            _k = float(start[3:])
                            divp_x = self.pnta.x+(self.pntb.x-self.pnta.x)*_k
                            divp_y = self.pnta.y+(self.pntb.y-self.pnta.y)*_k
                            dividing_point = point2d(divp_x,
                                                     divp_y,
                                                     lean=self.pnta.lean)
                            _e1 = edge2d(method='up2d',
                                         pnta=self.pnta,
                                         pntb=dividing_point)
                            _e2 = edge2d(method='up2d',
                                         pnta=dividing_point,
                                         pntb=self.pntb)
                            if start[1] == '1':
                                _coords_1 = _e1.gen_npoints(n,
                                                            distribution=distribution,
                                                            coeff=coeff,
                                                            start='a')
                                _coords_2 = _e2.gen_npoints(n,
                                                            distribution=distribution,
                                                            coeff=coeff,
                                                            start='b')
                            else:
                                _coords_1 = _e1.gen_npoints(n,
                                                            distribution=distribution,
                                                            coeff=coeff,
                                                            start='b')
                                _coords_2 = _e2.gen_npoints(n,
                                                            distribution=distribution,
                                                            coeff=coeff,
                                                            start='a')
                            coords = (np.hstack((_coords_1[0],
                                                 _coords_2[0][1:])),
                                      np.hstack((_coords_1[1],
                                                 _coords_2[1][1:])))
                        else:
                            print('factor should be non-zero and non-unit')
                            coords = None
                    else:
                        print('Enter valid ratio 0<k<1')
                        coords = None
                else:
                    coords = None
            else:
                # Other type of distribution
                pass
        if len(coeff) < 2:
            print('Coeff requirement: len(coeff) >= 2')
            coords = None
        return coords

    def split_nparts(self,
                     n,
                     distribution='polynomial',
                     coeff=[0, 1, 2],
                     weights=[],
                     start='a',
                     function='0.5*exp(x) + x^2/2',
                     constraints=[0.05, 0.05, 0.02, -1, 'uniform'],
                     make_mul_edge=False,
                     throw_coords=True,
                     throw_points=True,
                     ):
        """
        Generate n number of edges between end points of the self edge in a
        certain distribution.

        Parameters
        ----------
        n : int
            Number of edges to create.
        distribution : str, optional
            Provides the type of point distribution.
            The default is 'polynomial'. Other options include:
                'function', 'exponential', 'numpypoly_bl'
                1. 'polynomial': Will use coeff in list format
        coeff : list, optional
            Case-1: When distribution is 'polynomial'
            coeff[0] will be constant term. It will be rendered zero
            internally, regardless of user value. coeff[1] will be the
            coefficient accompanying the first order term.
            Case-2: When distribution if 'exponential'
            EXAPLANTION TO BE WRITTEN ONCE FEATURE BECOMES AVAILABLE.
            The default is [0, 1, 2].
        weights : list, optional
            Provides unit normalized multiplier factors. It will be used to
            calculate x and y locations of points as below:
                x = e.pnta.x + weights*(e.pntb.x-e.pnta.x)
                y = e.pnta.y + weights*(e.pntb.y-e.pnta.y)
            The default is [].
        start : str, optional
            Dictates the direction of inter-point seperation increase.
            If 'a', then more points will be closer to pnta of self edge.
            If 'b', then more points will be closer to pntb of self edge.
            If 'c1' then more points will be closer to pnta and pntb of the
                edge than the centre point.
            If 'c2' then more points will be closer to center point of the edge
                then pnta and pntb of the edge
            if 'c1_k' then behaviour will like 'c1', but points will cluster
                near point which divides the self edge at k*100 % from pnta,
                along the edge length
            if 'c2_k' then behaviour will like 'c2', but points will cluster
                less near point which divides the self edge at k*100 %
                from pnta, along the edge length. Clustering will be at both
                end pnta and pntb.
            The default is 'a'.
        function : str, optional
            FEATURE NOT YET AVAILABLE.
            The default is '0.5*exp(x) + x^2/2'.
        make_mul_edge : bool, optional
            If True, a multi-edge object is assembled from the split edges
            (not yet fully operational). Default is False.
        throw_coords : bool, optional
            If True, the coordinate arrays are included in the returned dict
            under key ``'coords'``. Default is True.
        throw_points : bool, optional
            If True, the intermediate Point2d objects are included under key
            ``'points'``. Default is True.

        Returns
        -------
        edges_data : dict
            Dictionary with keys ``'coords'`` (tuple of x- and y-arrays),
            ``'points'`` (list of Point2d), and ``'edges'`` (list of edge2d).
            Keys are present only when the corresponding ``throw_*`` flag is
            True and computation succeeds.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(-2, -1), pntb=Point2d(1, 3))
        >>> edges = e.split_nparts(15, distribution='polynomial',
        ...                        coeff=[0, 1, 10, -10, 1], start='c1_0.5')
        >>> edges = e.split_nparts(10, distribution='random',
        ...                        constraints=[0.05, 0.05, 0.05, -1, 'uniform'])
        """
        # Make empty dictionary to store data
        edges_data = {}
        # Get coordinates of intermediate points
        coords = self.gen_npoints(n,
                                  distribution=distribution,
                                  coeff=coeff,
                                  weights=weights,
                                  start=start,
                                  function=function,
                                  constraints=constraints,
                                  )
        if throw_coords:
            edges_data['coords'] = coords
        if coords:
            # Make intermediate points
            points = [self.pnta]
            for _x, _y in zip(coords[0][1:-1], coords[1][1:-1]):
                points.append(point2d(_x, _y, lean=self.pnta.lean))
            points.append(self.pntb)
            if throw_points:
                edges_data['points'] = points
            # Make edges from the intermediate points
            edges = [edge2d(method='up2d', pnta=pa, pntb=pb, edge_lean='ignore')
                     for pa, pb in zip(points[:-1], points[1:])]
            edges_data['edges'] = edges
            if make_mul_edge:
                # MUL-EDGE CREATION FROM INDIVIDUAL EDGES - NOT YET FULLY OPERATIUONAL
                pass
        return edges_data

    def split_edge2d_intersection(self,
                                  edge,
                                  split_input_edge=True,
                                  print_=False,
                                  sort_points=True,
                                  parent_edges_sort_points_from='a',
                                  user_edges_sort_points_from='a'
                                  ):
        """
        Splits the edge where it intersects with user input edge.
        Will also split input edge if split_input_edge is True.

        Parameters
        ----------
        edge : UPXO edge2d or list
            User input edge object
        split_input_edge : bool, optional
            Flag to split user input edge (i.e. edge). The default is True.
        print_ : bool, optional
            Flag to print outputs and states. The default is True.
        sort_points : bool, optional
            If True, sorts points before splitting. The default is True.
            Recommended value: True
        parent_edges_sort_points_from : str, optional
            If self edge can be split, if sort_points is True, the
            identified split points will be first sorted against distance from
            pnta of self edge if value is 'a' and pntb of self edge if
            value is 'b'.
            The default is 'a'.
            Permitted values: 'a' and 'b'.
                              'a' implies sorting from pnta
                              'b' implies sorting from pntb
        user_edges_sort_points_from : str, optional
            If user input edge can be split, if sort_points is True, the
            identified split points will be first sorted against distance from
            pnta of user input edge if value is 'a' and pntb of user input
            edge if value is 'b'.
            The default is 'a'.
            Permitted values: 'a' and 'b'.
                              'a' implies sorting from pnta
                              'b' implies sorting from pntb

        Returns
        -------
        edge_splits : dict
            Provides two keyed dictionary. Two keys are 'primary' and
            'secondary'. Values of 'primary' key provides the split original
            edge and all split edges of the original edge. Values of
            'secondary' key provides the split user input edge and all
            split edges of the user input edge.
            > If self edge has been split:
                edge_splits['primary'] = (primary split self edge,
                                          list of other split edges)
            > If self edge has not been split:
                edge_splits['primary'] = (primary split self edge,
                                          None)
            > If user input edge has been split:
                edge_splits['secondary'] = (primary split user input edge,
                                            list of other split edges)
            > If self input edge has not been split:
                edge_splits['secondary'] = (primary split user input edge,
                                            None)

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> other = edge2d(method='up2d', pnta=Point2d(0.5, -1), pntb=Point2d(0.5, 1))
        >>> splits = e.split_edge2d_intersection(other)
        >>> print(splits['primary'])
        >>> print(splits['secondary'])
        >>> other2 = edge2d(method='up2d', pnta=Point2d(-1, 0), pntb=Point2d(0.1, 0))
        >>> splits2 = e.split_edge2d_intersection(other2)
        """
        # e = edge2d(method='up2d', pnta=point2d(0, 0), pntb=point2d(1, 0))
        # edge = edge2d(method='up2d', pnta=point2d(-0.8, 0), pntb=point2d(-0.2, 0))
        # split_input_edge=True
        # print_=True
        # throw=True
        # sort_points=True
        # parent_edges_sort_points_from='a'
        # user_edges_sort_points_from='a'
        # Flags for developer to validate point ordering in edges
        __VALIDATE_IDS__SELF__EDGE = False
        __VALIDATE_IDS__USER__EDGE = False
        # initiate edge_splits, where all edge pieces will be collected
        edge_splits = {}
        # FIND INTERSECTION POINTS
        intersections = self.edge2d_intersection(edge,
                                                 return_ratios=False,
                                                 sort=True,
                                                 print_=False)
        # IF THERE ARE INTERSECTIONS, TRY SPLITTING
        if len(intersections[0]) > 0:
            _npa_ = np.array
            _npw_ = np.where
            _npasort_ = np.argsort
            points = _npa_([point2d(intersection[0],
                                    intersection[1],
                                    lean='ignore')
                            for intersection in intersections[0]])
            # --------------------------------
            # IDENTIFY POINTS WHICH SPLIT SELF EDGE
            # find which points coincide with self.pnta
            self_coincide_pnta = _npw_(_npa_(self.pnta == points))[0]
            # Idetify which points to exclude based on self.pnta
            # These points are those which will not split self edge
            self_exclude = []
            for _self_coincide_pnta in self_coincide_pnta:
                self_exclude.append(_self_coincide_pnta)
            # find which points coincide with self.pntb
            self_coincide_pntb = _npw_(_npa_(self.pntb == points))[0]
            # Idetity which points to exclude based on self.pnta
            # These points are those which will not split self edge
            for _self_coincide_pntb in self_coincide_pntb:
                self_exclude.append(_self_coincide_pntb)
            # Identify points indices which split the self edge
            self_include = [i for i in list(range(len(points)))
                            if i not in self_exclude]
            # Identify points which split the self edge
            if self_include:
                self_include_points = _npa_([points[i] for i in self_include])
                # sort points as per distance
                if sort_points:
                    if parent_edges_sort_points_from == 'a':
                        # sort points in increasing distance from
                        # self edge pnta
                        d = self.pnta.distance(otype='up2d_list',
                                               obj=self_include_points)
                        self_include_points = self_include_points[_npasort_(d)]
                    elif parent_edges_sort_points_from == 'b':
                        # sort points in increasing distance from
                        # self edge pntb
                        d = self.pntb.distance(otype='up2d_list',
                                               obj=self_include_points)
                        self_include_points = self_include_points[_npasort_(d)]
                if len(self_include_points) != 0:
                    # Validation with IDs
                    if __VALIDATE_IDS__SELF__EDGE:
                        print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
                        print(f'parent edge point ids are {id(self.pnta)} and {id(self.pntb)}')
                        print(f'split point ids are {[id(point) for point in self_include_points]}')
                        print('. . . . . . .')
                    # Split self at self_include_points
                    new_edges = []
                    if len(self_include_points) <= 2:
                        new_edges.append(self.split_at_point(self_include_points[0]))
                    if len(self_include_points) == 2:
                        new_edges.append(new_edges[0].split_at_point(self_include_points[1]))
                    # Validation with IDs
                    if __VALIDATE_IDS__SELF__EDGE:
                        print(f'parent edge point ids are now {id(self.pnta)} and {id(self.pntb)}')
                        print('. . . . . . .')
                        print(new_edges)
                        for i, _edge in enumerate(new_edges):
                            print(f'edge[{i}] point ids are {id(_edge.pnta)} and {id(_edge.pntb)}')
                        print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
                    # COLLECT THE EDGE SPLITS in key 'primary'
                    edge_splits['primary'] = (self, new_edges)
                else:
                    if print_:
                        print('self edge not split')
                    edge_splits['primary'] = (self, None)
            else:
                edge_splits['primary'] = (self, None)
                if print_:
                    print('self edge not split')
            # --------------------------------
            # IDENTIFY POINTS WHICH SPLIT USER INPUT EDGE
            if split_input_edge:
                # find which points coincide with edge.pnta
                edge_coincide_pnta = _npw_(_npa_(edge.pnta == points))[0]
                # Idetify which points to exclude based on input edge.pnta
                # These points are those which will not split user input edge
                edge_exclude = []
                for _edge_coincide_pnta in edge_coincide_pnta:
                    edge_exclude.append(_edge_coincide_pnta)
                # find which points coincide with edge.pntb
                edge_coincide_pntb = _npw_(_npa_(edge.pntb == points))[0]
                # Idetify which points to exclude based on input edge.pntb
                # These points are those which will not split user input edge
                for _edge_coincide_pntb in edge_coincide_pntb:
                    edge_exclude.append(_edge_coincide_pntb)
                # Identify points indices which split the user input edge
                edge_include = [i for i in list(range(len(points)))
                                if i not in edge_exclude]
                # Identify points which split the user input edge
                if edge_include:
                    edge_include_points = _npa_([points[i] for i in edge_include])
                    # sort points as per distance
                    if len(edge_include_points) != 0:
                        if sort_points:
                            if user_edges_sort_points_from=='a':
                                # sort points in increasing distance from
                                # user input edge pnta
                                d = edge.pnta.distance(otype='up2d_list',
                                                       obj=edge_include_points)
                                edge_include_points = edge_include_points[_npasort_(d)]
                            elif user_edges_sort_points_from=='b':
                                # sort points in increasing distance from
                                # user input edge pntb
                                d = edge.pntb.distance(otype='up2d_list',
                                                       obj=edge_include_points)
                                edge_include_points = edge_include_points[_npasort_(d)]
                        # Validation with IDs
                        if __VALIDATE_IDS__USER__EDGE:
                            print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
                            print(f'user edge point ids are {id(edge.pnta)} and {id(edge.pntb)}')
                            print(f'split point ids are {[id(point) for point in edge_include_points]}')
                            print('. . . . . . .')
                        # Split self at edge_include_points
                        new_edges = []
                        if len(edge_include_points) <= 2:
                            new_edges.append(edge.split_at_point(edge_include_points[0]))
                        if len(edge_include_points) == 2:
                            new_edges.append(new_edges[0].split_at_point(edge_include_points[1]))
                        # Validation with IDs
                        if __VALIDATE_IDS__USER__EDGE:
                            print(f'parent edge point ids are now {id(edge.pnta)} and {id(edge.pntb)}')
                            print('. . . . . . .')
                            for i, _edge in enumerate(new_edges):
                                print(f'edge[{i}] point ids are {id(_edge.pnta)} and {id(_edge.pntb)}')
                            print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
                        # COLLECT THE EDGE SPLITS in key 'secondary'
                        edge_splits['secondary'] = (edge, new_edges)
                else:
                    if print_:
                        print('User edge not split')
                    edge_splits['secondary'] = (edge, None)
            else:
                edge_splits['secondary'] = (edge, None)
                if print_:
                    print('User edge not split')
        else:
            edge_splits['primary'] = (self, None)
            edge_splits['secondary'] = (edge, None)
            if print_:
                print('No internal intersections found. Edges not split')
        return edge_splits

    def split_at_muledge2d_intersection(self, edges):
        """Split this edge at all intersection points with a multi-edge chain. Not yet implemented."""
        pass

    def contains_point(self, obj=None, method='parallelity', tdist=0.0):
        """
        Assess relative positioning of a point with respect to self edge.
        Output helps determine whether the point:
            1. is fully contained inside the self edge
            2. is coincident with one of the points of the self edge
            3. is located on the extended part of the self edge
            4. none of the above. Relative position unknown.

        Parameters
        ----------
        obj : list of float or Point2d
            Represents a point in space as ``[x, y]`` or a Point2d object.
            Default is None.
        method : str, optional
            Intersection detection method. Default is ``'parallelity'``.
        tdist : float, optional
            Tolerance distance. Default is 0.0.

        Returns
        -------
        intersection : list of bool
            Three-element list ``[inside, on_extension, at_endpoint]``:

            * ``[True, False, True]``  — point coincides with an endpoint.
            * ``[True, False, False]`` — point lies strictly inside the edge.
            * ``[False, True, False]`` — point lies on the extended line only.
            * ``[False, False, False]`` — relative position unknown.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(-1, 0), pntb=Point2d(1, 0))
        >>> e.contains_point([-0.5, 0])
        >>> e.contains_point(Point2d(-1.1, 0))
        """
        SQRT = math.sqrt
        if dth.IS_CPAIR(obj):
            # Entered obj is a coordinate pair
            if self.pnta.y == self.pntb.y:
                # When edge has zero slope
                pdist = abs(obj[1]-self.pnta.y)
            elif self.pnta.x == self.pntb.x:
                # When edge has infinite slope
                pdist = abs(obj[0]-self.pnta.x)
            else:
                m = self.slope
                # Calculate the y-intercept of the line
                yintercept = self.pnta.y - m * self.pnta.x
                # Calculate the perpendicular distance from the point to the line
                pdist = abs(m*obj[0] - obj[1] + yintercept) / SQRT(m**2 + 1)
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.pnta.x-obj[0])**2 +
                                           (self.pnta.y-obj[1])**2),
                                      SQRT((self.pntb.x-obj[0])**2 +
                                           (self.pntb.y-obj[1])**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]
        elif str(type(obj)) == "<class 'UPXO-point.point2d'>":
            if self.pnta.y == self.pntb.y:
                # When edge has zero slope
                pdist = abs(obj.y-self.pnta.y)
            elif self.pnta.x == self.pntb.x:
                # When edge has infinite slope
                pdist = abs(obj.x-self.pnta.x)
            else:
                m = self.slope
                # Calculate the y-intercept of the line
                yintercept = self.pnta.y - m * self.pnta.x
                # Calculate the perpendicular distance from the point to the line
                pdist = abs(m*obj.x - obj.y + yintercept) / math.sqrt(m**2 + 1)
            if pdist != 0:
                intersection = [False, False, False]
            else:
                distances = np.array([SQRT((self.pnta.x-obj.x)**2 +
                                           (self.pnta.y-obj.y)**2),
                                      SQRT((self.pntb.x-obj.x)**2 +
                                           (self.pntb.y-obj.y)**2)]
                                     )
                done = False
                if any(distances == self.length) or any(distances == 0):
                    # Point coincides with one of the edge points
                    intersection = [True, False, True]
                    done = True
                if not done:
                    if any(distances < self.length) and any(distances != self.length):
                        # Point is fully inside the edge
                        intersection = [True, False, False]
                if any(distances > self.length):
                    # Point is on the extended edge
                    intersection = [False, True, False]
        return intersection

    def contains_points(self, obj=None, otype=None, tdist=0.0):
        """
        Assess relative positioning of points with respect to self edge.
        Output helps determine whether each of the points:
            1. is fully contained inside the self edge
            2. is coincident with one of the points of the self edge
            3. is located on the extended part of the self edge
            4. none of the above. Relative position unknown.

        Parameters
        ----------
        obj : coord, UPXO point2d object
            Represents a point in space. The default is None.
        tdist : float, optional
            Tolerance distance. The default is 0.0.
        otype : str, optional
            Provide obj type to bypass typechecking and speed up

        Returns
        -------
        containment : [bool, bool, bool]
            Provides the relative position of point with resepect to self edge.

            1. Contains the point. It coincides with one of the edge points
               The truth values in 'containment' are [True, False, True]
            2. Contains the point. Point is fully inside the edge
               The truth values in 'containment' are [True, False, False]
            3. Point is on the extended edge
               The truth values in 'containment' are [False, True, False]
            4. Relative position of point unknown
               The truth values in 'containment' are [False, False, False]

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(4, -1))
        >>> e.contains_points([[-0.5, 0], [0, 0], [1, 0]], otype='cpair_list')
        >>> pts = [Point2d(-0.5, 0), Point2d(0, 0), Point2d(-1.1, 0)]
        >>> e.contains_points(pts, otype='up2d_list')
        """
        if dth.IS_ITER(obj):
            obj = np.array(obj)
        else:
            containment = []
        if otype:
            if otype in ('xy_list', 'xylist'):
                # Entered obj is a xy_list
                if self.pnta.y == self.pntb.y:
                    # When edge has zero slope
                    pdist = abs(obj[1] - self.pnta.y)
                elif self.pnta.x == self.pntb.x:
                    # When edge has infinite slope
                    pdist = abs(obj[0] - self.pnta.x)
                else:
                    SQRT = math.sqrt
                    m = self.slope
                    # Calculate the y-intercept of the line
                    yintercept = self.pnta.y - m * self.pnta.x
                    # Calc the perp dist from the point to the line
                    pdist = abs(m*obj[0] - obj[1] + yintercept) / SQRT(m**2+1)
                # Initiate containment and assign slope to elength
                containment, elength = [], self.length
                # Assess containment
                for _obj, pd in zip(obj.T, pdist):
                    if pd != 0:
                        containment.append([False, False, False])
                    else:
                        distances = np.array([SQRT((self.pnta.x-_obj[0])**2 +
                                                   (self.pnta.y-_obj[1])**2),
                                              SQRT((self.pntb.x-_obj[0])**2 +
                                                   (self.pntb.y-_obj[1])**2)]
                                             )
                        done = False
                        if any(distances == elength) or any(distances == 0):
                            # Point coincides with one of the edge points
                            containment.append([True, False, True])
                            done = True
                        if not done:
                            if any(distances < elength) and any(distances != elength):
                                # Point is fully inside the edge
                                containment.append([True, False, False])
                        if any(distances > elength):
                            # Point is on the extended edge
                            containment.append([False, True, False])
            elif otype == 'cpair_list':
                containment = self.contains_points(obj.T, otype='xy_list')
            elif otype == 'up2d_list':
                containment = self.contains_points(np.array([[_.x,
                                                              _.y]
                                                             for _ in obj]).T,
                                                   otype='xy_list')
            else:
                # type check and proceed
                # TODO 1: Build array of class names
                # TODO 2: Check if all are of same type
                # TODO 3: If yes, make xy_list and pass with otype = 'xy_list'
                # TODO 4: If not, build x and y list using each obj element
                #             and pass with otype = 'xy_list'
                pass
        else:
            if dth.DEEPCHECK_is_xy2d_list(obj):
                # Entered obj is a xy_list
                containment = self.contains_points(obj, otype='xy_list')
            elif dth.DEEPCHECK_is_coord2d_list(obj):
                # Entered obj is a list of coordinate pair
                containment = self.contains_points(obj.T, otype='xy_list')
            elif dth.ALL_UP2D(obj):
                containment = self.contains_points(obj, otype='up2d_list')
        return containment

    def contains_edge(self,
                      obj=None,
                      otype='ue2d',
                      tdist=0.0,
                      ):
        """
        Checks whether an edge is contained in self edge

        Parameters
        ----------
        obj : Multiple types
            Point object or coordinate pair OR line object or pair of
            coordinate pairs. Accepts following types:
                * Coordinate pair * Pair of coordinate pair
                * UPXO point2d * UPXO edge object
                * Shapely point * Shapely line object
                * VTK point * VTK line object
                * GMSH point * GMSH line object
        otype : str
            Specify type of the object

        tdist : float, optional
            Tolerance distance. The default is 0.0.

        Returns
        -------
        tuple
            ``(_pnta_, _pntb_, _edge_)`` where ``_pnta_`` and ``_pntb_`` are
            three-element containment lists (see :meth:`contains_point`) for
            each endpoint of the input edge, and ``_edge_`` is ``True`` when
            both endpoints are contained inside self.

        Examples
        --------
        >>> from upxo.geoEntities.point2d import Point2d
        >>> from upxo.geoEntities.edge2d import edge2d
        >>> e = edge2d(method='up2d', pnta=Point2d(0, 0), pntb=Point2d(1, 0))
        >>> k = e.contains_edge(obj=[[0.2, 0], [0.8, 0]], otype='clist')
        >>> k = e.contains_edge(obj=e, otype='ue2d')
        """
        if obj:
            if otype == 'clist':
                pnta = point2d(obj[0][0], obj[0][1], lean='ignore')
                pntb = point2d(obj[1][0], obj[1][1], lean='ignore')
            if otype == 'up2d':
                pnta, pntb = obj[0], obj[1]
            if otype == 'ue2d':
                pnta, pntb = obj.pnta, obj.pntb
            # Evaluate contains_point
            _pnta_ = self.contains_point(obj=pnta,
                                         method='parallelity',
                                         tdist=tdist
                                         )
            _pntb_ = self.contains_point(obj=pntb,
                                         method='parallelity',
                                         tdist=tdist
                                         )
            if _pnta_[0] and _pntb_[0]:
                _edge_ = True
            else:
                _edge_ = False
        else:
            _pnta_, _pntb_, _edge_ = None, None, None
            print('Please enter valid inputs')
        return (_pnta_, _pntb_, _edge_)

    def check_parallel(self, edge, tol_ang=0.0):
        """
        Check whether self and another edge are parallel.

        Parameters
        ----------
        edge : edge2d
            The other edge to compare against.
        tol_ang : float, optional
            Angular tolerance in degrees. Default is 0.0.

        Returns
        -------
        bool
            True if the edges are parallel within ``tol_ang``.
        """
        difference = abs(abs(self.angrad) - abs(edge.angrad))
        return True if difference <= tol_ang or difference <= 180+tol_ang else False

    def check_normal(self, edge, tol_ang=0.0):
        """
        Check whether self and another edge are perpendicular.

        Parameters
        ----------
        edge : edge2d
            The other edge to compare against.
        tol_ang : float, optional
            Angular tolerance in degrees. Default is 0.0.

        Returns
        -------
        bool
            True if the angular difference is 90° or 270°.
        """
        difference = abs(abs(self.angrad) - abs(edge.angrad))
        return True if difference in (90.0, 270) else False

    def calc_dot(self, edge):
        """
        Calculate the dot product of self with another edge.

        Parameters
        ----------
        edge : edge2d
            The other edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def calc_cross(self, edge):
        """
        Calculate the cross product of self with another edge.

        Parameters
        ----------
        edge : edge2d
            The other edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def calc_area(self, y=0):
        """
        Calculate the signed area under this edge down to a reference line.

        Parameters
        ----------
        y : float, optional
            Y-coordinate of the reference baseline. Default is 0 (x-axis).

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def set_edge_type(self, edge_type='ggb'):
        """
        Set the semantic type label of this edge.

        Parameters
        ----------
        edge_type : str, optional
            Label identifying the physical role of the edge. Default is
            ``'ggb'`` (generic grain boundary).

        Notes
        -----
        Suggested type labels:

        * ``'gbcb'`` — grain boundary zone - core boundary
        * ``'gtb'``  — grain-twin boundary
        * ``'gib'``  — grain inclusion boundary
        * ``'gicb'`` — grain inclusion cluster boundary
        """
        self.edge_type = edge_type

    @property
    def ends(self):
        """
        Endpoint coordinates as plain tuples.

        Returns
        -------
        tuple
            ``((xa, ya), (xb, yb))``
        """
        return (self.pnta.x, self.pnta.y), (self.pntb.x, self.pntb.y)

    @property
    def angle(self):
        """
        Orientation angle of the edge in degrees, measured from the positive
        x-axis, in the range [0, 360).

        Returns
        -------
        float
        """
        a, b = self.ends  # Call property method
        _angle_ = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
        if _angle_ < 0:
            _angle_ += 360.
        return _angle_

    def calc_slope(self):
        """
        Compute and store the slope of this edge in ``self.slope``.

        Sets ``self.slope`` to ``INFINITY`` when the edge length is at or
        below the threshold ``self.tlen`` (vertical or degenerate edge).

        Returns
        -------
        None
        """
        if self.length > self.tlen:
            self.slope = np.float64(self.pntb.y-self.pnta.y) / \
                (self.pntb.x-self.pnta.x)
        else:
            self.slope = INFINITY

    def _calc_slope_(self, x1, x2, y1, y2):
        """Return ``(y2 - y1) / (x2 - x1)`` as float64."""
        return np.float64(y2-y1)/(x2-x1)

    def calc_length(self, method='points', saa=True, throw=False):
        """
        Calculate the Euclidean length of this edge.

        Parameters
        ----------
        method : str, optional
            ``'points'`` uses the Point2d endpoint objects; ``'coords'`` uses
            the internal ``__x`` / ``__y`` arrays. Default is ``'points'``.
        saa : bool, optional
            If True, store the result in ``self.length``. Default is True.
        throw : bool, optional
            If True, return the computed length. Default is False.

        Returns
        -------
        float or None
            Computed length when ``throw`` is True; otherwise None.
        """
        if method.lower() in ('upxo_points', 'points'):
            _length = float(np.sqrt((self.pnta.x-self.pntb.x)**2 +
                                    (self.pnta.y-self.pntb.y)**2))
        elif method.lower() in ('coord', 'coords'):
            x, y = self.__x, self.__y
            _length = float(np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2))
        if saa:
            self.length = _length
        if throw:
            return _length

    def calc_centre(self,
                    saa=True,
                    throw=False
                    ):
        """
        Calculate the midpoint of this edge.

        Parameters
        ----------
        saa : bool, optional
            If True, store the result in ``self.xycen``. Default is True.
        throw : bool, optional
            If True, return the computed midpoint. Default is False.

        Returns
        -------
        list of float or None
            ``[x_centre, y_centre]`` when ``throw`` is True; otherwise None.
        """
        if self.edge_lean in ('lowest', 'leanest', 'no', 'low', 'medium'):
            _centerpoint = (self.pnta + self.pntb)*0.5
            _xycen = [_centerpoint.x, _centerpoint.y]
        else:
            _xycen = [0.5*(self._xa_+self._xb_), 0.5*(self._ya_+self._yb_)]
        if saa:
            self.xycen = _xycen
        if throw:
            return _xycen

    def update_m(self):
        """
        Rebuild ``self.m`` (the mulpoint2d representation) from current endpoints.

        Only operates when ``edge_lean`` is ``'leanest'`` or ``'lowest'``.

        Returns
        -------
        None
        """
        if self.edge_lean in ('leanest', 'lowest'):
            self.m = mulpoint2d(method='points',
                                point_objects=[self.pnta, self.pntb])
        else:
            print('Mulpoint creation restricted. Check "edge_lean".')

    def make_m(self):
        """
        Create the mulpoint2d object for this edge's endpoints.

        Delegates to :meth:`update_m`.

        Returns
        -------
        None
        """
        self.update_m()

    def make_bounding_box(self, return_format='coord_upxo'):
        """
        Construct the axis-aligned bounding box of this edge.

        Parameters
        ----------
        return_format : str, optional
            Format of the returned bounding box. Options:
            ``'coord_upxo'``, ``'coord'``, ``'coord_shapely'``,
            ``'upxo_partition'``, ``'upxo_muledge'``, ``'upxo_ring'``,
            ``'shapely_ring'``, ``'shapely_polygon'``, ``'vtk_polygon'``,
            ``'upxo_mulpoint'``, ``'shapely_mulpoint'``, ``'vedo_polygon'``.
            Default is ``'coord_upxo'``.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def make_bounding_box_and_extrude(self):
        """
        Construct the bounding box and extrude it into a volume.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def calc_center_from_m(self):
        """
        Update the centre of m, the mulpoint object representatopn of the
        edge's point objects

        Returns
        -------
        None.

        """
        self.m.recompute_centroid(accuracy='exact')
        self.xycen = self.m.centroid

    def calc_center(self, saa=True, throw=False):
        """
        Alias for :meth:`calc_centre` (US spelling).

        Parameters
        ----------
        saa : bool, optional
            If True, store the result in ``self.xycen``. Default is True.
        throw : bool, optional
            If True, return the computed midpoint. Default is False.

        Returns
        -------
        list of float or None
            ``[x_centre, y_centre]`` when ``throw`` is True; otherwise None.
        """
        if throw:
            return self.calc_centre(saa=saa, throw=throw)
        else:
            self.calc_centre(saa=saa, throw=throw)

    def update_xy(self):
        """
        Update the end point coordinates of the edge

        Returns
        -------
        None.

        """
        self.__x = [self.pnta.x, self.pntb.x]
        self.__y = [self.pnta.y, self.pntb.y]

    def update_centre(self):
        """
        Update centre of the edge object

        Returns
        -------
        None.

        """
        self.calc_centre(saa=True, throw=False)

    def update_center(self):
        """
        Update centre of the edge object

        Returns
        -------
        None.

        """
        self.update_centre()

    def update_end_points_from_x_and_y(self):
        """
        Using the coordinate values,update the endpoint objects

        Returns
        -------
        None.

        """
        self.pnta.x, self.pnta.y = self.__x[0], self.__y[0]  # Point A
        self.pntb.y, self.pntb.y = self.__x[1], self.__y[1]  # Point B

    def update_end_points_from_points(self, pnta=None, pntb=None):
        """
        Replace the endpoint objects of this edge.

        Parameters
        ----------
        pnta : Point2d, optional
            New start point. Default is None (no change intended).
        pntb : Point2d, optional
            New end point. Default is None (no change intended).

        Returns
        -------
        None
        """
        self.pnta, self.pntb = pnta, pntb

    def post_displacement_updates(self):
        """
        Set of operations and method calls to update attributes following a
        displacement operation

        Returns
        -------
        None.

        """
        self.update_xy()
        # Update centre
        self.calc_centre(saa=True, throw=False)
        # Update after rotation
        self.calc_slope()
        # Update mulpoint
        if hasattr(self, 'm'):
            self.update_m()

    def post_deformation_updates(self):
        """
        Set of operations and method calls to update attributes following a
        deformation operation

        Returns
        -------
        None.

        """
        self.update_xy()
        # Update length
        if hasattr(self, 'pnta') and hasattr(self, 'pntb'):
            self.calc_length(method='points', saa=True, throw=False)
        else:
            self.calc_length(method='coords', saa=True, throw=False)
        # Carry out the set of post - displacement updates
        self.post_displacement_updates()

    def make_vtk_line(self):
        """
        Construct a VTK line representation of this edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def make_pyvista_line(self):
        """
        Construct a PyVista line representation of this edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def make_shapely_line(self):
        """
        Construct a Shapely LineString from this edge.

        Returns
        -------
        shapely.geometry.LineString
            A two-point LineString from ``pnta`` to ``pntb``.
        """
        from shapely.geometry import LineString
        return LineString([(self.pnta.x, self.pnta.y),
                           (self.pntb.x, self.pntb.y)])

    def make_vedo_line(self):
        """
        Construct a Vedo line representation of this edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def make_gmsh_line(self):
        """
        Construct a GMSH line entity from this edge.

        Returns
        -------
        None

        Limitations
        -----------
        - Not yet implemented.
        """
        pass

    def plot(self,
             dpi=50,
             ):
        """
        Plot this edge with endpoint coordinate labels.

        Parameters
        ----------
        dpi : int, optional
            Resolution of the matplotlib figure. Default is 50.

        Returns
        -------
        None
        """
        x = [self.pnta.x, self.pntb.x]
        y = [self.pnta.y, self.pntb.y]
        plt.plot(x, y, 'bo', linestyle='-')
        for _x_, _y_ in zip(x, y):
            plt.text(_x_, _y_,
                     '(%4.2f, %4.2f)' % (_x_, _y_),
                     horizontalalignment='center',
                     verticalalignment='bottom'
                     )

    @property
    def _xa_(self):
        """x-coordinate of endpoint A (``pnta``). Setter triggers deformation updates."""
        return self.__x[0]

    @property
    def _xb_(self):
        """x-coordinate of endpoint B (``pntb``). Setter triggers deformation updates."""
        return self.__x[1]

    @property
    def _ya_(self):
        """y-coordinate of endpoint A (``pnta``). Setter triggers deformation updates."""
        return self.__y[0]

    @property
    def _yb_(self):
        """y-coordinate of endpoint B (``pntb``). Setter triggers deformation updates."""
        return self.__y[1]

    @ _xa_.setter
    def _xa_(self, x):
        """Set x of endpoint A and trigger deformation updates."""
        if self.__x[0] != x:
            self.__x[0] = x
            self.update_end_points_from_x_and_y()
            self.post_deformation_updates()

    @ _xb_.setter
    def _xb_(self, x):
        """Set x of endpoint B and trigger deformation updates."""
        if self.__x[1] != x:
            self.__x[1] = x
            self.update_end_points_from_x_and_y()
            self.post_deformation_updates()

    @ _ya_.setter
    def _ya_(self, y):
        """Set y of endpoint A and trigger deformation updates."""
        if self.__y[0] != y:
            self.__y[0] = y
            self.update_end_points_from_x_and_y()
            self.post_deformation_updates()

    @ _yb_.setter
    def _yb_(self, y):
        """Set y of endpoint B and trigger deformation updates."""
        if self.__y[1] != y:
            self.__y[1] = y
            self.update_end_points_from_x_and_y()
            self.post_deformation_updates()

class edge2d_lean_highest():
    """
    Minimal high-lean edge representation storing only raw xy coordinates.

    Uses ``__slots__`` to reduce per-instance memory overhead.

    Parameters
    ----------
    start : list of float, optional
        ``[x, y]`` of the start point. Default is ``[0.0, 0.0]``.
    end : list of float, optional
        ``[x, y]`` of the end point. Default is ``[1.0, 1.0]``.

    Attributes
    ----------
    xy : list
        ``[[xa, xb], [ya, yb]]`` coordinate storage.
    """
    __slots__ = 'xy'

    def __init__(self,
                 start=[0.0, 0.0],
                 end=[1.0, 1.0]
                 ):
        """Initialise with ``start`` and ``end`` endpoint coordinates."""
        self.xy = [[start[0], end[0]], [start[1], end[1]]]


class edge2d_from_point2d():
    """
    Lightweight edge built directly from two Point2d endpoint objects.

    Computes Euclidean length at construction time and stores it.

    Parameters
    ----------
    pnta : Point2d, optional
        Start endpoint. Default is None.
    pntb : Point2d, optional
        End endpoint. Default is None.

    Attributes
    ----------
    pnta : Point2d
    pntb : Point2d
    length : float
        Euclidean distance between ``pnta`` and ``pntb``.
    """
    __slots__ = ('pnta', 'pntb', 'length')

    def __init__(self,
                 pnta=None,
                 pntb=None
                 ):
        """Initialise from two ``Point2d`` endpoints and compute Euclidean length."""
        self.pnta = pnta
        self.pntb = pntb
        self.length = np.sqrt((pnta.x-pntb.x)**2+(pnta.y-pntb.y)**2)
        # self.length = pnta.distance(other_object_type = 'point2d',
        #                              point_data = pntb)
