"""
Pure point operations and point-to-entity comparison utilities for UPXO.

Import
------
    from upxo.geoEntities import pops

Functions
---------
Profiling
  PROFILE_up2d_INST

Point operations (single-point arithmetic / transforms)
  xadd, yadd, xmul, ymul, xdiv, ydiv, xabs, yabs,
  intize, floatize, roundround, roundceil, roundfloor,
  negxy, negx, negy, mirrorx, mirrory, translate, rotate

Point-point comparisons
  CMPEQ_points, CMPEQ_pnt_fast_exact, CMPEQ_pnt_fast_EPS,
  CMPEQ_pnt_fast_tdist

Point-edge operations
  CMPEQ_up2d_edge, CMPEQ_up2d_edges, DIST_point_edges

Point-point relative-position queries
  RELPOS_point_points_above, RELPOS_point_points_below,
  RELPOS_point_points_left, RELPOS_point_points_right
"""
import cProfile
import numpy as np
# import pops
from numpy import array as nparray
from math import sqrt, ceil, floor, radians, sin, cos
from upxo._sup import dataTypeHandlers as dth
"""
LIST OF PROFILER METHODS
------------------------
1. PROFILE_up2d_INST

LIST OF PURE POINT OPERATION METHODS
-------------------------------------
. xadd(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. yadd(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. xmul(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. ymul(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. xdiv(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. ydiv(point, k, saa=False, make_new=True, lean='ignore', throw=True)
. xabs(point, saa=False, make_new=True, lean='ignore', throw=True)
. yabs(point, saa=False, make_new=True, lean='ignore', throw=True)
. intize(point, saa=False, make_new=True, lean='ignore', throw=True)
. floatize(point, saa=False, make_new=True, lean='ignore', throw=True)
. roundround(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. xroundround(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. yroundround(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. roundceil(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. xroundceil(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. yroundceil(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. roundfloor(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. xroundfloor(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. yroundfloor(point, nd=4, saa=False, make_new=True, lean='ignore', throw=True)
. negxy(point, saa=False, make_new=True, lean='ignore', throw=True)
. negx(point, saa=False, make_new=True, lean='ignore', throw=True)
. negy(point, saa=False, make_new=True, lean='ignore', throw=True)
. mirrorx(point, saa=False, make_new=True, lean='ignore', throw=True)
. mirrory(point, saa=False, make_new=True, lean='ignore', throw=True)
. translate(point, xyincr=[0.0, 0.0], method='xyincr', xincr=0.0, yincr=0.0,
            xnew=0.0, ynew=0.0, xynew=[0.0, 0.0], saa=False, make_new=True,
            lean='ignore', throw=True)
. rotate(point, t=0.0, o=(0.0, 0.0), nd=12,
         saa=False, make_new=True, lean='ignore', throw=True)

LIST OF POINT-POINT OPERATION METHODS
-------------------------------------
. CMPEQ_points(p1, p2)
. CMPEQ_pnt_fast_exact(p1, p2)
. CMPEQ_pnt_fast_EPS(p1, p2)
. CMPEQ_pnt_fast_tdist(p1, p2, tdist=0.000000000001)

LIST OF POINT-MULTIPOINT OPERATION METHODS
------------------------------------------
1.

LIST OF POINT-EDGE OPERATION METHODS
-------------------------------------
1. CMPEQ_up2d_edge(point, edge)
2. CMPEQ_up2d_edges(point, edges)
3. DIST_point_edges(point, edges)

LIST OF POINT-MULTIEDGE OPERATION METHODS
-------------------------------------
1.

LIST OF POINT-RING OPERATION METHODS
-------------------------------------
1.

LIST OF POINT-XTAL OPERATION METHODS
-------------------------------------
1.

LIST OF POINT-PXTAL OPERATION METHODS
-------------------------------------
1.

"""
###############################################################################
# POINT TO POINT(S) COMPARISON OPERATIONS

def CMPEQ_points(p1, p2):
    """Return ``p1 == p2`` using the point equality operator."""
    return p1 == p2


def CMPEQ_pnt_fast_exact(p1, p2):
    """Return ``True`` if ``p1`` and ``p2`` have exactly equal x and y."""
    equality = False
    if p1.x == p2.x and p1.y == p2.y:
        equality = True
    return equality


def CMPEQ_pnt_fast_EPS(p1, p2):
    """Return ``True`` if the Euclidean distance between ``p1`` and ``p2`` is ≤ EPS."""
    EPS, equality = 0.000000000001, False
    if sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2) <= EPS:
        equality = True
    return equality


def CMPEQ_pnt_fast_tdist(p1, p2, tdist=0.000000000001):
    """Return ``True`` if the Euclidean distance between ``p1`` and ``p2`` is ≤ ``tdist``."""
    equality = False
    if sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2) <= tdist:
        equality = True
    return equality

###############################################################################
# POINT TO MULTI-POINT(S) COMPARISON OPERATIONS


###############################################################################
# POINT - TREE TRANSFORMATION


###############################################################################
def CMPEQ_up2d_edge(point, edge):
    """
    Check equality with points of the input UPXO edge

    Parameters
    ----------
    edge : UPXO edge object
        User input UPXO edge object

    Returns
    -------
    list
        List of two truth values. First for pnta of input edge and
        second for pntb of input edge. If distance <= point EPS,
        value is True, else False

    Examples
    --------
    >>> from upxo.geoEntities.point2d import point2d
    >>> from upxo.geoEntities.edge2d import edge2d
    >>> from upxo.geoEntities import pops
    >>> p = point2d(0, 0, lean='ignore')
    >>> e = edge2d(method='up2d', pnta=point2d(0, 0), pntb=point2d(1, 0))
    >>> pops.CMPEQ_up2d_edge(p, e)
    """
    # Check equality with a single edge

    x, y, _EPS_ = point.x, point.y, point.EPS
    pnta, pntb = edge.pnta, edge.pntb
    dist_masks = [sqrt((x-pnta.x)**2+(y-pnta.y)**2) <= _EPS_,
                  sqrt((x-pntb.x)**2+(y-pntb.y)**2) <= _EPS_
                  ]

    return dist_masks


def CMPEQ_up2d_edges(point, edges, data_set):
    """
    Check equality with points of the input UPXO edges

    Parameters
    ----------
    edges : dth.dt.ITERABLES
        Iterable having user input UPXO edge objects

    Returns
    -------
    equalities : np.array
        (N, 2) shaped numpy array of truth values, N: number of edges

    Examples
    --------
    >>> from upxo.geoEntities.point2d import point2d
    >>> from upxo.geoEntities.edge2d import edge2d
    >>> from upxo.geoEntities import pops
    >>> p = point2d(0, 0, lean='ignore')
    >>> edges = [edge2d(method='up2d', pnta=point2d(0, 0), pntb=point2d(1, 0))]
    >>> pops.CMPEQ_up2d_edges(p, edges, 'large_data_set')
    """
    np_array, np_sqrt = np.array, np.sqrt
    xp, yp, _EPS_ = point.x, point.y, point.EPS
    if data_set == 'very_large_data_set':
        pnts_a, pnts_b = [e.pnta for e in edges], [e.pntb for e in edges]
        xa, ya = np_array([[pa.x, pa.y] for pa in pnts_a]).T
        xb, yb = np_array([[pb.x, pb.y] for pb in pnts_b]).T
        equalities = np.vstack((np_sqrt((xa-xp)**2+(ya-yp)**2),
                                np_sqrt((xb-xp)**2+(yb-yp)**2))).T <= _EPS_
    elif data_set == 'large_data_set':
        equalities = []
        for edge in edges:
            pnta, pntb = edge.pnta, edge.pntb
            equalities.append([sqrt((xp-pnta.x)**2+(yp-pnta.y)**2) <= _EPS_,
                               sqrt((xp-pntb.x)**2+(yp-pntb.y)**2) <= _EPS_
                               ]
                              )
    return equalities


def DIST_point_edges(point, edges, data_set):
    """


    Parameters
    ----------
    edges : dth.dt.ITERABLES
        Iterable having user input UPXO edge objects

    Returns
    -------
    dist : np.array
        (N, 2) shaped numpy array of distances, N: number of edges
        dist[n, 0]: distance from point to pnta of nth edge
        dist[n, 1]: distance from point to pntb of nth edge

    Examples
    --------
    >>> from upxo.geoEntities.point2d import point2d
    >>> from upxo.geoEntities.edge2d import edge2d
    >>> from upxo.geoEntities import pops
    >>> p = point2d(0, 0, lean='ignore')
    >>> edges = [edge2d(method='up2d', pnta=point2d(0, 0), pntb=point2d(1, 0))]
    >>> pops.DIST_point_edges(p, edges, 'large_data_set')
    """
    np_array, np_sqrt = np.array, np.sqrt
    xp, yp = point.x, point.y
    if data_set == 'very_large_data_set':
        pnts_a, pnts_b = [e.pnta for e in edges], [e.pntb for e in edges]
        xa, ya = np_array([[pa.x, pa.y] for pa in pnts_a]).T
        xb, yb = np_array([[pb.x, pb.y] for pb in pnts_b]).T
        equalities = np.vstack((np_sqrt((xa-xp)**2+(ya-yp)**2),
                                np_sqrt((xb-xp)**2+(yb-yp)**2))).T
    elif data_set == 'large_data_set':
        equalities = []
        for edge in edges:
            pnta, pntb = edge.pnta, edge.pntb
            equalities.append([sqrt((xp-pnta.x)**2+(yp-pnta.y)**2),
                               sqrt((xp-pntb.x)**2+(yp-pntb.y)**2)
                               ]
                              )
    return equalities


def RELPOS_point_points_above(point, point_objects):
    """
    Returns Truth array as per elements in point_objects.
    True if other object is above, else False in any other case

    Parameters
    ----------
    point_objects : point2d / shapely point / coord_xy list
                   / coord_xy tuple
        Input set of points to compare against.

    Returns
    -------
    list of Boolean (truth values)
    """
    point_objects_coord_y = dth.point_list_to_coordxy(point_objects).T[1]
    point_y = point.y + 0.000000000001
    above_flags = point_objects_coord_y >= point_y
    return above_flags


def RELPOS_point_points_below(point, point_objects):
    """
    Returns Truth array as per elements in point_objects.
    True if other object is below, else False in any other case

    Parameters
    ----------
    point_objects : point2d / shapely point / coord_xy list
                   / coord_xy tuple
        Input set of points to compare against.

    Returns
    -------
    list of Boolean (truth values)
    """
    point_objects_coord_y = dth.point_list_to_coordxy(point_objects).T[1]
    point_y = point.y - 0.000000000001
    below_flags = point_objects_coord_y < point_y
    return below_flags


def RELPOS_point_points_left(point, point_objects):
    """
    Returns Truth array as per elements in point_objects.
    True if other object is to the left, else False in any other case

    Parameters
    ----------
    point_objects : point2d / shapely point / coord_xy list
                   / coord_xy tuple
        Input set of points to compare against.

    Returns
    -------
    list of Boolean (truth values)
    """
    point_objects_coord_x = dth.point_list_to_coordxy(point_objects).T[0]
    point_x = point.x - 0.000000000001
    left_flags = point_objects_coord_x < point_x
    return left_flags


def RELPOS_point_points_right(point, point_objects):
    """
    Returns Truth array as per elements in point_objects.
    True if other object is to the right, else False in any other case

    Parameters
    ----------
    point_objects : point2d / shapely point / coord_xy list
                   / coord_xy tuple
        Input set of points to compare against.

    Returns
    -------
    list of Boolean (truth values)
    """
    point_objects_coord_x = dth.point_list_to_coordxy(point_objects).T[0]
    point_x = point.x + 0.000000000001
    right_flags = point_objects_coord_x >= point_x
    return right_flags


def xadd(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Add ``k`` to the x-coordinate of ``point``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:
                point.x += k
                to_return = point.make_new(point.x, point.y, lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:
                point.x += k
                to_return = '[--parent updated--]'
            if not saa and make_new:
                to_return = point.make_new(point.x+k, point.y, lean=lean)
            if not saa and not make_new:
                to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        to_return += (point.make_new(point.x+k_,
                                                    point.y,
                                                    lean=lean),
                                      )
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                to_return_ += (point.make_new(point.x+k__,
                                                             point.y,
                                                             lean=lean),
                                               )
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def yadd(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Add ``k`` to the y-coordinate of ``point``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:
                point.y += k
                to_return = point.make_new(point.x, point.y, lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:
                point.y += k
                to_return = '[--parent updated--]'
            if not saa and make_new:
                to_return = point.make_new(point.x, point.y+k, lean=lean)
            if not saa and not make_new:
                to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        to_return += (point.make_new(point.x,
                                                    point.y+k_,
                                                    lean=lean),
                                      )
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                to_return_ += (point.make_new(point.x,
                                                             point.y+k__,
                                                             lean=lean),
                                               )
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def xmul(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Multiply the x-coordinate of ``point`` by ``k``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:
                point.x *= k
                to_return = point.make_new(point.x, point.y, lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:
                point.x *= k
                to_return = '[--parent updated--]'
            if not saa and make_new:
                to_return = point.make_new(point.x*k, point.y, lean=lean)
            if not saa and not make_new:
                to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        to_return += (point.make_new(point.x*k_,
                                                    point.y,
                                                    lean=lean),
                                      )
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                to_return_ += (point.make_new(point.x*k__,
                                                             point.y,
                                                             lean=lean),
                                               )
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def ymul(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Multiply the y-coordinate of ``point`` by ``k``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:
                point.y *= k
                to_return = point.make_new(point.x, point.y, lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:
                point.y *= k
                to_return = '[--parent updated--]'
            if not saa and make_new:
                to_return = point.make_new(point.x, point.y*k, lean=lean)
            if not saa and not make_new:
                to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        to_return += (point.make_new(point.x,
                                                    point.y*k_,
                                                    lean=lean),
                                      )
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                to_return_ += (point.make_new(point.x,
                                                             point.y*k__,
                                                             lean=lean),
                                               )
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def xdiv(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Divide the x-coordinate of ``point`` by ``k``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if k >= 0.000000000001:
                if saa and make_new:
                    point.x /= k
                    to_return = point.make_new(point.x, point.y, lean=lean)
                    to_return = (to_return, '[--parent also updated--]')
                if saa and not make_new:
                    point.x /= k
                    to_return = '[--parent updated--]'
                if not saa and make_new:
                    to_return = point.make_new(point.x/k, point.y, lean=lean)
                if not saa and not make_new:
                    to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        if abs(k_) >= 0.000000000001:
                            to_return += (point.make_new(point.x/k_,
                                                        point.y,
                                                        lean=lean),
                                          )
                        else:
                            to_return += ('[--Zero k ERR--]',)
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                if abs(k__) >= 0.000000000001:
                                    to_return_ += (point.make_new(point.x/k__,
                                                                 point.y,
                                                                 lean=lean),
                                                   )
                                else:
                                    to_return_ += ('[--Zero k ERR--]',)
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def ydiv(point, k, saa=False, make_new=True, lean='ignore', throw=True):
    """Divide the y-coordinate of ``point`` by ``k``, with saa/throw control."""
    if type(k) in dth.dt.NUMBERS + dth.dt.ITERABLES:
        if type(k) in dth.dt.NUMBERS:
            if k >= 0.000000000001:
                if saa and make_new:
                    point.y /= k
                    to_return = point.make_new(point.x, point.y, lean=lean)
                    to_return = (to_return, '[--parent also updated--]')
                if saa and not make_new:
                    point.y /= k
                    to_return = '[--parent updated--]'
                if not saa and make_new:
                    to_return = point.make_new(point.x, point.y/k, lean=lean)
                if not saa and not make_new:
                    to_return = '[--saa FALSE make_new FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            if make_new:
                to_return = ()
                for k_ in k:
                    if type(k_) in dth.dt.NUMBERS:
                        if abs(k_) >= 0.000000000001:
                            to_return += (point.make_new(point.x,
                                                        point.y/k_,
                                                        lean=lean),
                                          )
                        else:
                            to_return += ('[--Zero k ERR--]',)
                    elif type(k_) in dth.dt.ITERABLES:
                        to_return_ = ()
                        for k__ in k_:
                            if type(k__) in dth.dt.NUMBERS:
                                if abs(k__) >= 0.000000000001:
                                    to_return_ += (point.make_new(point.x,
                                                                 point.y/k__,
                                                                 lean=lean),
                                                   )
                                else:
                                    to_return_ += ('[--Zero k ERR--]',)
                            else:
                                to_return_ += ('[--invalid k__--]',)
                        to_return += (to_return_,)
                    else:
                        to_return += ('[--invalid k_--]',)
            else:
                to_return = '[--make_new FALSE--]'
    else:
        to_return = '[--invalid k--]'
    if throw:
        return to_return


def xabs(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Apply absolute value to the x-coordinate of ``point``, with saa/throw control."""
    if saa or make_new:
        _x = abs(point.x)
    if saa and make_new:
        point.x = _x
        to_return = (point.make_new(x=_x, y=point.y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=point.y, lean='ignore')
    if saa and not make_new:
        point.x = _x
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def yabs(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Apply absolute value to the y-coordinate of ``point``, with saa/throw control."""
    if saa or make_new:
        _y = abs(point.y)
    if saa and make_new:
        point.y = _y
        to_return = (point.make_new(x=point.x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=point.x, y=_y, lean='ignore')
    if saa and not make_new:
        point.y = _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def intize(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Convert both coordinates of ``point`` to ``int``, with saa/throw control."""
    if saa or make_new:
        _x, _y = int(point.x), int(point.y)
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def floatize(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Convert both coordinates of ``point`` to ``float``, with saa/throw control."""
    if saa or make_new:
        _x, _y = float(point.x), float(point.y)
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def roundround(point, nd=4, saa=False, make_new=True, lean='ignore',
               throw=True):
    """Round both coordinates of ``point`` to ``ndigits``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x, _y = round(point.x, nd), round(point.y, nd)
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def xroundround(point, nd=4, saa=False, make_new=True, lean='ignore',
                throw=True):
    """Round the x-coordinate of ``point`` to ``ndigits``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x = round(point.x, nd)
    if saa and make_new:
        point.x = _x
        to_return = (point.make_new(x=_x, y=point.y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=point.y, lean='ignore')
    if saa and not make_new:
        point.x = _x
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def yroundround(point, nd=4, saa=False, make_new=True, lean='ignore',
                throw=True):
    """Round the y-coordinate of ``point`` to ``ndigits``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _y = round(point.y, nd)
    if saa and make_new:
        point.y = _y
        to_return = (point.make_new(x=point.x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=point.x, y=_y, lean='ignore')
    if saa and not make_new:
        point.y = _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def roundceil(point, nd=4, saa=False, make_new=True, lean='ignore',
              throw=True):
    """Apply ``math.ceil`` to both coordinates of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x_, _y_ = point.x, point.y
        _x = np.sign(_x_)*ceil(abs(_x_*10**nd))/10**nd
        _y = np.sign(_y_)*ceil(abs(_y_*10**nd))/10**nd
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def xroundceil(point, nd=4, saa=False, make_new=True, lean='ignore',
               throw=True):
    """Apply ``math.ceil`` to the x-coordinate of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x_ = point.x
        _x = np.sign(_x_)*ceil(abs(_x_*10**nd))/10**nd
    if saa and make_new:
        point.x = _x
        to_return = (point.make_new(x=_x, y=point.y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=point.y, lean='ignore')
    if saa and not make_new:
        point.x = _x
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def yroundceil(point, nd=4, saa=False, make_new=True, lean='ignore',
               throw=True):
    """Apply ``math.ceil`` to the y-coordinate of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _y_ = point.y
        _y = np.sign(_y_)*ceil(abs(_y_*10**nd))/10**nd
    if saa and make_new:
        point.y = _y
        to_return = (point.make_new(x=point.x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=point.x, y=_y, lean='ignore')
    if saa and not make_new:
        point.y = _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def roundfloor(point, nd=4, saa=False, make_new=True, lean='ignore',
               throw=True):
    """Apply ``math.floor`` to both coordinates of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x_, _y_ = point.x, point.y
        _x = np.sign(_x_)*floor(abs(_x_*10**nd))/10**nd
        _y = np.sign(_y_)*floor(abs(_y_*10**nd))/10**nd
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def xroundfloor(point, nd=4, saa=False, make_new=True, lean='ignore',
                throw=True):
    """Apply ``math.floor`` to the x-coordinate of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _x_ = point.x
        _x = np.sign(_x_)*floor(abs(_x_*10**nd))/10**nd
    if saa and make_new:
        point.x = _x
        to_return = (point.make_new(x=_x, y=point.y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=point.y, lean='ignore')
    if saa and not make_new:
        point.x = _x
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def yroundfloor(point, nd=4, saa=False, make_new=True, lean='ignore',
                throw=True):
    """Apply ``math.floor`` to the y-coordinate of ``point``, with saa/throw control."""
    # nd: number of decimal places
    if saa or make_new:
        _y_ = point.y
        _y = np.sign(_y_)*floor(abs(_y_*10**nd))/10**nd
    if saa and make_new:
        point.y = _y
        to_return = (point.make_new(x=point.x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=point.x, y=_y, lean='ignore')
    if saa and not make_new:
        point.y = _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def negxy(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Negate both coordinates of ``point``, with saa/throw control."""
    if saa or make_new:
        _x, _y = -point.x, -point.y
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '#- saa FALSE make_new FALSE -#'
    if throw:
        return to_return

def negx(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Negate the x-coordinate of ``point``, with saa/throw control."""
    if saa or make_new:
        _x = -point.x
    if saa and make_new:
        point.x = _x
        to_return = (point.make_new(x=_x, y=point.y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=point.y, lean='ignore')
    if saa and not make_new:
        point.x = _x
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '[-- saa FALSE make_new FALSE --]'
    if throw:
        return to_return

def negy(point, saa=False, make_new=True, lean='ignore', throw=True):
    """Negate the y-coordinate of ``point``, with saa/throw control."""
    if saa or make_new:
        _y = -point.y
    if saa and make_new:
        point.y = _y
        to_return = (point.make_new(x=point.x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=point.x, y=_y, lean='ignore')
    if saa and not make_new:
        point.y = _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '[-- saa FALSE make_new FALSE --]'
    if throw:
        return to_return

def mirrorx(point, saa: bool = False, make_new: bool = True,
            lean: bool = 'ignore', throw: bool = True):
    """Mirror point about the x-axis."""
    return point.negy(saa=saa, make_new=make_new, lean=lean, throw=throw)

def mirrory(point, saa: bool = False, make_new: bool = True,
            lean: bool = 'ignore', throw: bool = True):
    """Mirror point about the y-axis."""
    return point.negx(saa=saa, make_new=make_new, lean=lean, throw=throw)

def translate(point, xyincr: list = [0.0, 0.0], method: str = 'xyincr',
              xincr: float = 0.0, yincr: float = 0.0,
              xnew: float = 0.0, ynew: float = 0.0,
              xynew: list = [0.0, 0.0], saa: bool = False,
              make_new: bool = True, lean: bool = 'ignore',
              throw: bool = True):
    """Translate ``point`` by a displacement vector, with saa/throw control."""
    if saa or make_new:
        if method in dth.opt.translate_xyincr:
            _x, _y = point.x+xyincr[0], point.y+xyincr[1]
        if method in dth.opt.translate_xincr:
            _x, _y = point.x+xincr, point.y
        if method in dth.opt.translate_yincr:
            _x, _y = point.x, point.y+yincr
        if method in dth.opt.translate_xynew:
            _x, _y = xynew[0], xynew[1]
        if method in dth.opt.translate_xnew:
            _x, _y = xnew, point.y
        if method in dth.opt.translate_ynew:
            _x, _y = point.x, ynew
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '[-- saa FALSE make_new FALSE --]'
    if throw:
        return to_return

def rotate(point, t=0.0, o=(0.0, 0.0), nd=12,
           saa=False, make_new=True, lean='ignore', throw=True):
    """
    Rotate counterclockwise.
    nd: number of decimal places to round off to
    """
    if saa or make_new:
        t, ox, oy, px, py = radians(t), o[0], o[1], point.x, point.y
        dx, dy = px-ox, py-oy
        _x = round(ox+cos(t)*dx-sin(t)*dy, 12)
        _y = round(oy+sin(t)*dx+cos(t)*dy, 12)
    if saa and make_new:
        point.x, point.y = _x, _y
        to_return = (point.make_new(x=_x, y=_y, lean='ignore'),
                     '[--parent also updated--]')
    if not saa and make_new:
        to_return = point.make_new(x=_x, y=_y, lean='ignore')
    if saa and not make_new:
        point.x, point.y = _x, _y
        to_return = '[--parent updated--]'
    if not saa and not make_new:
        to_return = '[-- saa FALSE make_new FALSE --]'
    if throw:
        return to_return
    return _x, _y

###############################################################################
def distance(self, otype='point2d', obj=None, cor=0.1, nworkers=1):
    """
    1. Single UPXO point2d object - DONE
    2. List of upxo point2d objects - DONE
    3. A single coordinate pair of point2d - DONE
    4. List of x coordinates and y coordinates - DONE
    """
    # ................
    # 1. Single UPXO point2d object - DONE
    # obj: UPXO point2d object
    if otype in dth.opt.upxo_point2d:
        '''
        Explanations:
            1. Use this when distances are to be computed against
            another point2d object
            2. INPUT TYPE of "obj": UPXO.point2d object

        Example 1: Point to point
            p1, p2 = point2d(x = 0, y = 0), point2d(x = 1, y = 1)
            p1.distance(otype = 'point2d', obj = p2)

        Example 2: Point to a list of points (method - 1)
            Not preferred, as there is a simpler method available under
            case 'point2d_list'
            x, y = list(range(0, 10, 2)), list(range(0, 20, 4))
            p = [point2d(x = _x, y = _y) for _x, _y in zip(x, y)]
            d = [p1.distance(otype = 'point2d',
                             obj = pi) for pi in p]
        '''
        return np.sqrt((self.x-obj.x)**2 + (self.y-obj.y)**2)
    # ................
    # 2. List of upxo point2d objects - DONE
    # obj: list of point2d objects
    if otype in dth.opt.upxo_point2d_list:
        '''
        Explanations:
            1. Use this when distances are to be computed against a list
            of point2d objects
            2. INPUT TYPE of "obj": list

        Example 1: Point to list of points
            p1 = point2d(x = 0, y = 0)
            p2 = point2d(x = 1, y = 1)
            p1.distance(otype = 'up2dlist',
                        obj = [p1, p2])

        Example 2: Point to list of points
        '''
        x, y = zip(*[(_.x, _.y) for _ in obj])
        return np.sqrt((self.x-np.array(x))**2 + (self.y-np.array(y))**2)
    # ................
    # 3. Coordinate pair of a single point2d - DONE
    # obj: coordinate
    if otype in dth.opt.coord_point2d or otype in dth.opt.coord_pair_point2d:
        '''
        # INPUT: a list / tuple of two float / int coordinates
        # EXAMPLE INPUT: (x0, y0)

        p = point2d(x = 0, y = 0)
        print(p.distance(otype = 'coord2d', obj = (10, 10)))
        '''
        return np.sqrt((self.x-obj[0])**2 + (self.y-obj[1])**2)
    # ................
    # 4. List of x coordinates and y coordinates - DONE
    # obj: list of lists of x and y coordinates
    if otype in dth.opt.coord_point2d_list:
        '''
        INPUT: a list/tuple of two lists/tuples. Each of the two
        inner lists/tuples contain the list of coordinate values
        EXAMPLE INPUT: ((x0, x1, x2,...), (y0, y1, y2,...))

        Example 1: Point to list of x and y coordinate list
            p = point2d(x = -2, y = 0)
            obj = [[-2, -1, -0, 1, 2], [0, 0, 0, 0, 0]]
            d = p.distance(otype = 'xy_list', obj = obj)
        '''
        return np.sqrt((self.x-np.array(obj[0]))**2
                       + (self.y-np.array(obj[1]))**2)
    # ................
    # 5. List of coordinate pairs of point2d objects - DONE
    # obj: list of lists of x-y coordinate pairs
    if otype in dth.opt.coord_pairs_point2d_list:
        # INPUT: a list/tuple of numerous lists/tuples. Each of the
        # many lists/tuples contain coordinates of a point
        # EXAMPLE INPUT: ((x0, y0), (x1, y1), (x2, y2),....)
        obj = np.array(obj).T
        return np.sqrt((self.x-obj[0])**2 + (self.y-obj[1])**2)
    # ................
    # 6. A list of cKDTree objects - DONE
    # obj: list of ckdtree objects
    if otype in dth.opt.ckdtree_lists:
        _, distances, _, _ = dth.find_neighdata_ckdt_list(self.x,
                                                          self.y,
                                                          obj,
                                                          cor,
                                                          nworkers)
        return distances
    # ................
    # 7. A list of shapely point objects
    # ................
    # 8. A list of vtk point objects
    # ................
    # 9. A list of pyvista point objects
    # ................
    # 10. A single OR list of UPXO multi-point2d objects
    if otype in dth.opt.upxo_mp2d_list:
        '''
        Explanations:
            1. Use this when computimng distance sgainst a set of point2d
               objects contained inside list of mulpoint2d objects
            2. INPUT TYPE of "obj": [upxo point object 1,
                                     upxo point object 2,
                                     ...,
                                     upxo point object n]

        Example 1:
            # Create the reference point2d object
                p0 = point2d()

            # Create a mulpoint2d object
                p1 = point2d(x = 2.0, y = 1.0)
                p2 = p1 + 1
                p3 = p2 * 0.6498
                p4 = p1*0.468 + p3/p2
                m1 = mulpoint2d(method = 'up2d_list',
                                point_objects = [p1, p2, p3, p1 + p4*p1])

            # Calculate distance
                d = p0.distance(otype = 'ump2d_list',
                                obj = [m1, m1])
        '''
        _depack_ = False
        if str(obj.__class__.__name__) == 'mulpoint2d':
            # If user enters a single mul-point object, make list
            obj, _depack_ = [obj], True
        _x, _y, distances = self.x, self.y, []
        for mp in obj:
            distances.append(np.sqrt((_x-mp.locx)**2
                                     + (_y-mp.locy)**2)
                             )
        if _depack_:
            distances = distances[0]

        return distances
    # ................
    # 11. A list of shapely mulobject objects (each made of points)
    # ...............
    # A single UPXO mulpoint3d objects
    if otype == 'upxo_mulpoint3d':
        pass
    # ................
    # A single edge2d object
    if otype == 'upxo_edge2d':
        pass
    # ................
    # A list of edge2d objects
    if otype == 'upxo_edge2d':
        pass
    # ................
    if otype in ('upxo_muledge2d', 'upxo_ring2d'):
        pass
    # ................
    if otype == 'upxo_edge3d':
        pass
    # ................
    if otype in ('upxo_muledge3d', 'upxo_ring3d'):
        pass
    # ................
    if otype == 'shapely_xtal2d_centroid':
        '''
        Explanations:
            1. Use this to find distance between self and centroid
               of the shapely polygon object
            2. INPUT TYPE of "obj": a valid shapely polygon object
            3. Centroidal x and y of polygon object will be used as
               obj = [[x], [y]]

        Example 1:
            from point2d_04 import point2d
            p0 = point2d()
            from shapely.geometry import Polygon
            shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

            p0.distance(otype = 'shapely_xtal2d_centroid',
                        obj = shapelypol)
        '''
        centroid = obj.centroid
        return self.distance(otype='coord_list',
                             obj=[[centroid.x], [centroid.y]])[0]
    if otype == 'shapely_xtal2dlist_centroid':
        '''
        Explanations:
            1. Use this to find distance between self and centroids of a
               list of shapely polygon objects
            2. INPUT TYPE of "obj": list of valid shapely polygon
               objects

        Example 1:
            from point2d_04 import point2d
            p0 = point2d()
            from shapely.geometry import Polygon
            shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
            shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

            obj = [shapelypol1, shapelypol2]
            p0.distance(otype = 'shapely_xtal2dlist_centroid',
                        obj = obj)
        '''
        centroids = [[_.centroid.x, _.centroid.y] for _ in obj]
        return self.distance(otype='coord_pairs',
                             obj=centroids)

    if otype == 'shapely_xtal2d_reppoint':
        '''
        Explanations:
            1. Use this to find distance between self and reppoint of the
               shapely polygon object
            2. INPUT TYPE of "obj": a valid shapely polygon object
            3. centroidal x and y of polygon object will be used as
               obj = [[x], [y]]

        Example 1:
            from point2d_04 import point2d
            p0 = point2d()
            from shapely.geometry import Polygon
            shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

            p0.distance(otype = 'shapely_xtal2d_reppoint',
                        obj = shapelypol)
        '''
        reppoint = obj.representative_point()
        return self.distance(otype='coord_list',
                             obj=[[reppoint.x], [reppoint.y]])[0]
    if otype == 'shapely_xtal2dlist_reppoint':
        '''
        Explanations:
            1. Use this to find distance between self and reppoints of a
               list of shapely polygon objects
            2. INPUT TYPE of "obj": list of valid shapely polygon
               objects

        Example 1:
            from point2d_04 import point2d
            p0 = point2d()
            from shapely.geometry import Polygon
            shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
            shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

            obj = [shapelypol1, shapelypol2]
            p0.distance(otype = 'shapely_xtal2dlist_reppoint',
                        obj = obj)
        '''
        reppoints = [[_.representative_point().x,
                      _.representative_point().y] for _ in obj]
        return self.distance(otype='coord_pairs',
                             obj=reppoints)
    # ................
    if otype == 'upxo_xtal2d_reppoint':
        # here obj will be the xtal containing the representative
        # point
        # representative point has to be UPXO point2d object
        return self.distance(otype='upxo_point2d',
                             obj=obj.reppoint)
    if otype == 'upxo_xtal2dlist_reppoint':
        # This is to call self.distance operating on case
        # 'upxo_xtal2d_reppoint'
        return [self.distance(otype='upxo_xtal2d_reppoint',
                              obj=_obj)
                for _obj in obj]
    # ................
    if otype == 'upxo_xtal2d_vertices':
        # This is to call self.distance operating on case
        # 'upxo_xtal2d_reppoint'
        pass
    # ................
    if otype == 'upxo_xtal3d_centroid':
        pass
    # ................
    if otype == 'upxo_xtal3d_reppoint':
        pass
    # ................
    if otype == 'upxo_xtal3d_vertices':
        pass
    # ................
    if otype == 'shapely_point':
        pass
    # ................
    if otype == 'vtk_point':
        pass
    # ................
    if obj is None:
        print('Need other object to compute distance(s)')
###############################################################################
# POINT - PIXEL TRANSFORMATION
def pixelize(point, dim=2, k1=0.5, k2=0.5, k3=0.5, z0=0.0, t=0.0,
             aunit='deg', avoidEPS=False, nd=12,
             saa=False,
             throw=False, throw_edges=False, throw_faces=False,
             make_midnode=False,
             makextal_upxo=False,
             makextal_shapely=False, makextal_vtk=False,
             makeelement_gmsh=False, make_abaqus_element=False,
             break_into_2tria=False, break_into_2trib=False,
             break_into_4tri=False):
    """
    Build a pixel (rectangular voxel) around a point and optionally subdivide it.

    Constructs a 2-D rectangular pixel centred on ``point`` with half-widths
    ``k1`` (x) and ``k2`` (y), optionally rotated by angle ``t``.  In 3-D mode
    (``dim=3``) the pixel is extruded to a hexahedral cell using half-depth
    ``k3``.  The cell can be returned as a whole or split into triangles /
    tetrahedra, and optionally converted to UPXO, Shapely, VTK, Gmsh, or
    Abaqus element representations.

    Parameters
    ----------
    point : Point2d
        Centre point of the pixel.
    dim : {2, 3}, optional
        Spatial dimension of the output cell.  Default is ``2``.
    k1 : float, optional
        Half-width along x.  Default is ``0.5``.
    k2 : float, optional
        Half-width along y.  Default is ``0.5``.
    k3 : float, optional
        Half-depth along z (3-D only).  Default is ``0.5``.
    z0 : float, optional
        z-coordinate of the bottom face (3-D only).  Default is ``0.0``.
    t : float, optional
        Rotation angle of the pixel about its centre.  Default is ``0.0``.
    aunit : {'deg', 'rad'}, optional
        Unit of ``t``.  Default is ``'deg'``.
    avoidEPS : bool, optional
        If ``True``, suppress epsilon-based adjustments.  Default is ``False``.
    nd : int, optional
        Number of decimal places for coordinate rounding.  Default is ``12``.
    saa : bool, optional
        If ``True``, store the result back on ``point`` (save-and-apply).
        Default is ``False``.
    throw : bool, optional
        If ``True``, return the pixel vertices.  Default is ``False``.
    throw_edges : bool, optional
        If ``True``, return edge objects.  Default is ``False``.
    throw_faces : bool, optional
        If ``True``, return face objects (3-D only).  Default is ``False``.
    make_midnode : bool, optional
        If ``True``, compute and include the mid-node.  Default is ``False``.
    makextal_upxo : bool, optional
        If ``True``, build a UPXO crystal from the pixel.  Default is ``False``.
    makextal_shapely : bool, optional
        If ``True``, build a Shapely polygon.  Default is ``False``.
    makextal_vtk : bool, optional
        If ``True``, build a VTK cell.  Default is ``False``.
    makeelement_gmsh : bool, optional
        If ``True``, build a Gmsh element.  Default is ``False``.
    make_abaqus_element : bool, optional
        If ``True``, build an Abaqus element.  Default is ``False``.
    break_into_2tria : bool, optional
        If ``True``, split the quad into two triangles (variant A).
        Default is ``False``.
    break_into_2trib : bool, optional
        If ``True``, split the quad into two triangles (variant B).
        Default is ``False``.
    break_into_4tri : bool, optional
        If ``True``, split the quad into four triangles about the centre.
        Default is ``False``.

    Returns
    -------
    None or tuple
        Returns pixel data when ``throw`` or related flags are ``True``;
        otherwise modifies ``point`` in-place (when ``saa=True``) and returns
        ``None``.

    Notes
    -----
    Corner-point naming convention (2-D, unrotated)::

        p2 ----------- p1      p1 = (x + k1,  y + k2)
        |               |      p2 = (x - k1,  y + k2)
        |    O(x, y)    |      p3 = (x - k1,  y - k2)
        |               |      p4 = (x + k1,  y - k2)
        p3 ----------- p4

    For 3-D hexahedral cells the bottom face duplicates the 2-D layout at
    ``z = z0`` and the top face at ``z = z0 + 2*k3``.
    """
    import matplotlib.pyplot as plt
    # k1, k2 = half of rectangle pixel length, width
    # t (deg): angle with positive x-axis, anti-clockwise positive
    _x, _y = point.x, point.y
    _O_ = (_x, _y)
    if aunit in dth.opt.angle_unit_deg:
        ang = radians(t)  # convert to radians
    elif aunit in dth.opt.angle_unit_rad:
        # nothing more to do here
        pass
    else:
        # Assume input angle is in degrees
        ang = radians(t)  # convert to radians
    # TWO-DIMENSIONS
    if dim == 2:
        if abs(t) <= 0.000000000001 or abs(t) in (0.0, 90.0, 180.0,
                                                  270.0, 360.0):
            p1, p2 = (_x+k1, _y+k2), (_x-k1, _y+k2)
            p3, p4 = (_x-k1, _y-k2), (_x+k1, _y-k2)
            if make_midnode:
                m12, m23 = (_x, _y+k2), (_x-k1, _y)
                m34, m41 = (_x, _y-k2), (_x+k1, _y)
                v = (p3, m34, p4, m41, p1, m12, p2, m23)
            else:
                v = (p3, p4, p1, p2)
        else:
            sint, cost = sin(ang), cos(ang)
            if avoidEPS:
                p1 = (round(_x+cost*k1-sint*k2, nd),
                      round(_y+sint*k1+cost*k2, nd))
                p2 = (round(_x-cost*k1-sint*k2, nd),
                      round(_y-sint*k1+cost*k2, nd))
                p3 = (round(_x-cost*k1+sint*k2, nd),
                      round(_y-sint*k1-cost*k2, nd))
                p4 = (round(_x+cost*k1+sint*k2, nd),
                      round(_y+sint*k1-cost*k2, nd))
                v = (p3, p4, p1, p2)
                if make_midnode:
                    print('FEATURE NOT AVAILABLE YET')
            else:
                p1 = (_x+cost*k1-sint*k2, _y+sint*k1+cost*k2)
                p2 = (_x+cost*-k1-sint*k2, _y+sint*-k1+cost*k2)
                p3 = (_x+cost*-k1-sint*-k2, _y+sint*-k1+cost*-k2)
                p4 = (_x+cost*k1-sint*-k2, _y+sint*k1+cost*-k2)
                v = (p3, p4, p1, p2)
                if make_midnode:
                    print('FEATURE NOT AVAILABLE YET')
        if break_into_2tria:
            if make_midnode:
                m12, m23 = (_x, _y+k2), (_x-k1, _y)
                m34, m41 = (_x, _y-k2), (_x+k1, _y)
                v = ((p3, m34, p4, _O_, p2, m23),
                     (p4, m41, p1, m12, p2, _O_))
            else:
                v = ((p3, p4, p2), (p4, p1, p2))
        if break_into_2trib:
            if make_midnode:
                m12, m23 = (_x, _y+k2), (_x-k1, _y)
                m34, m41 = (_x, _y-k2), (_x+k1, _y)
                v = ((p3, _O_, p1, m12, p2, m23),
                     (p3, m23, p4, m41, p1, _O_))
            else:
                v = ((p3, p1, p2), (p3, p4, p1))
        if break_into_4tri:
            if make_midnode:
                m12, m23 = (_x, _y+k2), (_x-k1, _y)
                m34, m41 = (_x, _y-k2), (_x+k1, _y)
                m1o, mo2 = (_x+0.5*k1, _y+0.5*k2), (_x-0.5*k1, _y+0.5*k2)
                m3o, m4o = (_x-0.5*k1, _y-0.5*k2), (_x+0.5*k1, _y-0.5*k2)
                v = ((p3, m3o, _O_, mo2, p2, m23),
                     (p3, m34, p4, m4o, _O_, m3o),
                     (_O_, m4o, p4, m41, p1, m1o),
                     (p2, mo2, _O_, m1o, p1, m12))
            else:
                v = ((p3, _O_, p2), (p3, p4, _O_),
                     (_O_, p4, p1), (p2, _O_, p1))
    # THREE-DIMENSIONS
    if dim == 3:
        p1, p2 = (_x+k1, _y+k2, z0+k3), (_x-k1, _y+k2, z0+k3)
        p3, p4 = (_x-k1, _y-k2, z0+k3), (_x+k1, _y-k2, z0+k3)
        p5, p6 = (_x+k1, _y+k2, z0-k3), (_x-k1, _y+k2, z0-k3)
        p7, p8 = (_x-k1, _y-k2, z0-k3), (_x+k1, _y-k2, z0-k3)
        v = (p1, p2, p3, p4, p5, p6, p7, p8)
        e = ((5, 6), (6, 7), (7, 4), (4, 5),
             (6, 2), (7, 3), (4, 0), (5, 1),
             (1, 2), (2, 3), (3, 0), (0, 1)
             )
        f = (((5, 6), (6, 7), (7, 4), (4, 5)),
             ((6, 2), (5, 6), (1, 2), (2, 3)),
             ((6, 2), (6, 7), (7, 3), (2, 3)),
             ((7, 3), (7, 4), (5, 1), (3, 0)),
             ((1, 2), (4, 5), (5, 1), (0, 1)),
             ((2, 3), (2, 3), (3, 0), (0, 1))
             )
        if throw:
            to_return = (v)
        if throw_edges:
            to_return += (e,)
        if throw_faces:
            to_return += (f,)
    if saa:
        if dim == 2:
            point.pixels = v
        elif dim == 3:
            point.pixels = to_return
    if throw:
        if dim == 2:
            return v
        elif dim == 3:
            return to_return


def make_polygon(point,
                 r,
                 nv,
                 t,
                 make_mpv=True,
                 make_edges=True,
                 mp_lean='ignore',
                 e_lean='ignore',
                 ep_lean='ignore',
                 fill_random=False,
                 fill_r_threshold=0.8,
                 make_mpfill=False,
                 n_random=20
                 ):
    """
    Makes a Polygon with self ~UPXO point 2d~ as the centre

    Parameters
    ----------
    r : float/int
        radius.
    nv : int
        Number of vertices on the cirle. Code uses n+1. User to input n.
    t : float
        Rotation angle. Degrees. ACW for positive t values.
    make_mpv : bool, optional
        bool to make UPXO mul-point object of the polygon's vertices.
        The default is True.
    make_edges : bool, optional
        bool to make UPXO edge objects. The default is True.
    mp_lean : str
        Leanness specification of the mul-point object
    e_lean : str
        Leanness specification of the edge objects
    ep_lean : str
        Leanness specification of the points in the mul-point object
    fill_random : bool
        Specifies whether to fill with points. The default is False
    fill_r_threshold : float
        Domain:[0, 1]
        Threshold radius factor for filling with points .
        For high nv, fill_r_threshold could be near 1
        For low nv, fill_r_threshold must be smaller
    make_mpv : bool, optional
        bool to make UPXO mul-point object of filler points.
        The default is False.
    n_random : int

    Returns
    -------
    polygon. Description of polygon is below:
        polygon['v']: vertices
        polygon['fill']: filler points
        polygon['e']: edges

    Example-1
    ---------
    p = point2d(x=0.0, y=0.0)
    polygon = p.make_polygon(0.5, 5, 0,
                             make_mpv=True,
                             mp_lean='ignore',
                             e_lean='ignore',
                             ep_lean='ignore',
                             fill_random=True,
                             fill_r_threshold=0.8,
                             n_random=1000,
                             make_edges=True,
                             make_mpfill=False,
                             )
    """
    xo, yo = point.x, point.y
    angles = np.linspace(0, 2*np.pi, nv+1)[:-1]
    xcoord, ycoord = r*np.cos(angles) + xo, r*np.sin(angles) + yo
    xcoord[abs(xcoord) < point.EPS] = 0
    ycoord[abs(ycoord) < point.EPS] = 0
    t = -t*3.141592653589793/180
    # ---------------------------------------
    if abs(t) >= 0.000000000001:
        '''
        Credit:
            https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
        Adopted with modifications by: Dr. Sunil Anandatheertha
        '''
        c, s = np.cos(t), np.sin(t)
        j = np.matrix([[c, s], [-s, c]])
        xycoord = np.dot(j, np.matrix([np.array(xcoord)-xo,
                                       np.array(ycoord)-xo]))
        xcoord = xo+np.array(xycoord[0])[0]
        ycoord = yo+np.array(xycoord[1])[0]
    # ---------------------------------------
    xcoord[abs(xcoord) < point.EPS] = 0
    ycoord[abs(ycoord) < point.EPS] = 0
    # import matplotlib.pyplot as plt
    # plt.plot(xcoord, ycoord, '-k.')
    # ---------------------------------------
    if make_mpv:
        from mulpoint2d import mulpoint2d
        mp = mulpoint2d(method='xy_list',
                        coordxy=[xcoord, ycoord],
                        lean=mp_lean,
                        name='circle')
        polygon = {'v': mp
                   }
    else:
        polygon = {'v': [xcoord, ycoord]
                   }
    if make_edges:
        cpairs_list = []
        for i in range(nv-1):
            cpairs_list.append([[xcoord[i], ycoord[i]],
                                [xcoord[i+1], ycoord[i+1]]]
                               )
        cpairs_list.append([[xcoord[i+1], ycoord[i+1]],
                            [xcoord[0], ycoord[0]]]
                           )
        from eops import make_edges
        polygon['e'] = make_edges(method='cpairs_list',
                                  cpairs_list=cpairs_list,
                                  points_lean=ep_lean,
                                  edge_lean=e_lean
                                  )
    # ---------------------------------------
    if fill_random:
        angles = np.pi*2*np.random.random(n_random)
        r = fill_r_threshold*r*np.random.random(n_random)
        xcoord_rand, ycoord_rand = xo+r*np.cos(angles), yo+r*np.sin(angles)

        if not make_mpfill:
            polygon['fill'] = [xcoord_rand, ycoord_rand]
        else:
            polygon['fill'] = mulpoint2d(method='xy_list',
                                         coordxy=[xcoord_rand,
                                                  ycoord_rand],
                                         lean=mp_lean,
                                         name='filling_circle'
                                         )
        # plt.plot(xcoord_rand, ycoord_rand, 'c.')
    # ---------------------------------------
    return polygon
