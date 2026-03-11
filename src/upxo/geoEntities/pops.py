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
    return p1 == p2


def CMPEQ_pnt_fast_exact(p1, p2):
    equality = False
    if p1.x == p2.x and p1.y == p2.y:
        equality = True
    return equality


def CMPEQ_pnt_fast_EPS(p1, p2):
    EPS, equality = 0.000000000001, False
    if sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2) <= EPS:
        equality = True
    return equality


def CMPEQ_pnt_fast_tdist(p1, p2, tdist=0.000000000001):
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

    PRE-REQUISITE DATA
    ------------------
    from edge2d import edge2d
    p = point2d(0, 0, lean='ignore')
    e = edge2d(method='up2d', pnta=point2d(0,0), pntb=point2d(1,0))

    ######################################
    EXAMPLE-1
    ---------
    pops.CMPEQ_up2d_edge(p, e)

    EXAMPLE-1: PROFILE
    ------------------
    import gops
    gops.PROFILE_this_method(pops.CMPEQ_up2d_edge, 10000, (p, e))
    ######################################
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

    PRE-REQUISITE DATA
    ------------------
    from edge2d import edge2d
    p = point2d(0, 0, lean='ignore')
    e1 = edge2d(method='up2d', pnta=point2d(0,0), pntb=point2d(1,0))
    e2 = edge2d(method='up2d', pnta=point2d(1,6), pntb=point2d(0,0))
    e3 = edge2d(method='up2d', pnta=point2d(3,2), pntb=point2d(0,0))
    edges = [e1, e2, e3]
    ######################################
    EXAMPLE-1
    ---------
    pops.CMPEQ_up2d_edges(p, edges)

    EXAMPLE-1: SPEED PROFILE
    ------------------------
    import gops
    gops.PROFILE_this_method(CMPEQ_up2d_edges, 10000, (p, edges))
    ######################################
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

    PRE-REQUISITE DATA
    ------------------
    from edge2d import edge2d
    p = point2d(0, 0, lean='ignore')
    e1 = edge2d(method='up2d', pnta=point2d(0,0), pntb=point2d(1,0))
    e2 = edge2d(method='up2d', pnta=point2d(1,6), pntb=point2d(0,0))
    e3 = edge2d(method='up2d', pnta=point2d(3,2), pntb=point2d(0,0))
    edges = [e1, e2, e3]

    EXAMPLE-1
    ---------
    pops.DIST_point_edges(p, edges)
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
        DESCRIPTION. An input set of points to compare against

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
        DESCRIPTION. An input set of points to compare against

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
        DESCRIPTION. An input set of points to compare against

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
        DESCRIPTION. An input set of points to compare against

    Returns
    -------
    list of Boolean (truth values)
    """
    point_objects_coord_x = dth.point_list_to_coordxy(point_objects).T[0]
    point_x = point.x + 0.000000000001
    right_flags = point_objects_coord_x >= point_x
    return right_flags


def xadd(point, k, saa=False, make_new=True, lean='ignore', throw=True):
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
    """
    Mirror about x-axis
    """
    return point.negy(saa=saa, make_new=make_new, lean=lean, throw=throw)

def mirrory(point, saa: bool = False, make_new: bool = True,
            lean: bool = 'ignore', throw: bool = True):
    """
    Mirror about y-axis
    """
    return point.negx(saa=saa, make_new=make_new, lean=lean, throw=throw)

def translate(point, xyincr: list = [0.0, 0.0], method: str = 'xyincr',
              xincr: float = 0.0, yincr: float = 0.0,
              xnew: float = 0.0, ynew: float = 0.0,
              xynew: list = [0.0, 0.0], saa: bool = False,
              make_new: bool = True, lean: bool = 'ignore',
              throw: bool = True):
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
            y+
            ^
            :
            :
    x-<-----O------> x+
            :
            :
            :
            y+

    p2.................................p1     /
    :              ^                   :    /
    :              :                   :  /
    :           k2 :                   :/  t: angle
    :              :  O(_x, _y)        :- - - - - - - -
    :                 |--------------->:
    :                         k1       :
    :                                  :
    p3.................................p4
        O = (_x, _y)
        p1 = (_x+k1, _y+k2)
        p2 = (_x-k1, _y+k2)
        p3 = (_x-k1, _y-k2)
        p4 = (_x+k1, _y-k2)
    p2 ----------------- p1   p2 ------- p1   p2 --------p1  p2 ------ p1
     :                   :    :  *    2   :   :        *  :   :  * 4 *  :
     :         O         :    :    *      :   :  1   *    :   :   * *   :
     :                   :    :  1   *    :   :    *  2   :   : 1  O  3 :
    p3 ----------------- p4   :        *  :   :  *        :   :   * *   :
                              p3 ------- p4   p3 --------p4   :  * 2 *  :
                                                             p3 ------ p4
             (A)                   (B)             (C)            (D)
    (A): v = (p3, p4, p1, p2)
    (B): v = ((p3, p4, p2), (p4, p1, p2))
    (C): v = ((p3, p1, p2), (p3, p4, p1))
    (D): v = ((p3, O, p2), (p3, p4, O), (O, p4, p1), (p2, O, p1))

    p2 ---m12--- p1
     :           :
    m23    O    m41
     :           :
    p3 ---m34--- p4
        m12 = (_x, _y+k2)
        m23 = (_x-k1, _y)
        m34 = (_x, _y-k2)
        m41 = (_x+k1, _y)

    p2 ---m12--- p1
     :           :
    m23    O    m41
     :           :
    p3 ---m34--- p4
          (A)
    p2 --m12-- p1    p2 ---m12--p1     [p2]-----m12-----[p1]
    :  *     2  :    : 1      *  :      :  *     4     *  :
    :    *      :    :      *    :      :   mo2     m1o   :
   m23    O   m41  m23     O   m41      :      *   *      :
    :       *   :    :   *       :     m23  1   [O]   3  m41
    :  1      * :    : *      2  :      :      *   *      :
    p3 --m34-- p4    p3 --m23---p4      :   m3o     m4o   :
                                        :  *     2     *  :
                                       [p3]-----m34-----[p4]
    1  --m12--  0    1  ---m12-- 0      1  -----m12-----  0
    :  *     2  :    : 1      *  :      :  *     4     *  :
    :    *      :    :      *    :      :   mo2     m1o   :
   m23    O   m41  m23     O   m41      :      *   *      :
    :       *   :    :   *       :     m23  1   [O]   3  m41
    :  1      * :    : *      2  :      :      *   *      :
    2  --m34--  3    2  --m23--- 3      :   m3o     m4o   :
                                        :  *     2     *  :
                                        2  -----m34-----  3
        (B)              (C)                   (D)
    m1o = (_x+0.5*k1, _y+0.5*k2)
    mo2 = (_x-0.5*k1, _y+0.5*k2)
    m3o = (_x-0.5*k1, _y-0.5*k2)
    m4o = (_x+0.5*k1, _y-0.5*k2)

    (A) v = (p3, m34, p4, m41, p1, m12, p2, m23)
    (B) v = ((p3, m34, p4, O, p2, m23), (p4, m41, p1, m12, p2, O))
    (C) v = ((p3, O, p1, m12, p2, m23), (p3, m23, p4, m41, p1, O))
    (D) v = ((p3, m3o, o, mo2, p2, m23), (p3, m34, p4, m4o, o, m3o),
             (o, m4o, p4, m41, p1, m1o), (p2, mo2, o, m1o, p1, m12))

    FOR 3D: Hexahedral only
        Top face same as above
             z+
             ^           y-
             |         /
             |       /
             |     /
             |   /
             | /
    x-<----- O------------> x+
            /|
          /  |
        /    |
      /      |
    y+       |
             |
             z-
    POINT DEFINITIONS:
        POINT INDEX: 0,  1,  2,  3,  4,  5,  6,  7
                    p1, p2, p3, p4, p5, p6, p7, p8
          p2 -------e12------- p1
         :|                  : |
       e9 |                e11 |
      :   e8              :    e7
    p3 -------e10------ p4     |
    |     |             |      |
    |     |       O     |      |
    e5    |             e6     |
    |     p6 -------e4--|----- p5
    |    :              |    :
    |   e1              |   e3
    | :                 |  :
    p7 -------e2-------- p8
    EDGE DEFINITIONS:
        EDGE INDEX: EDGENAME: POINTS: POINT INDICES
                 0: e1: p6-p7: (5, 6)
                 1: e2: p7-p8: (6, 7)
                 2: e3: p8-p5: (7, 4)
                 3: e4: p5-p6: (4, 5)
                 4: e5: p7-p3: (6, 2)
                 5: e6: p8-p4: (7, 3)
                 6: e7: p5-p1: (4, 0)
                 7: e8: p6-p2: (5, 1)
                 8: e9: p2-p3: (1, 2)
                 9: e10: p3-p4: (2, 3)
                 10: e11: p4-p1: (3, 0)
                 11: e12: p1-p2: (0, 1)
             -----------------                 p2 -------e12------- p1
         :|                  : |             :|                  : |
        : |       F6        :  |           e9 |       F6       e11 |
      :   |               :    |          :   e8              :    e7
       -------------F5-        |        p3 -------e10----F5 p4     |
    |     |             |      |        |     |             |      |
    |  F2 |             |  F4  |        |  F2 |             |   F4 |
    |     |             |      |        e5    |             e6     |
    |        -F3--------|-----          |     p6 -F3----e4--|----- p5
    |    :              |    :          |    :              |    :
    |   :        F1     |   :           |   e1       F1     |   e3
    | :                 |  :            | :                 |  :
       -----------------                p7 -------e2-------- p8
    FACE DEFINITIONS. corner points: generic face names
        p7, p8, p5, p6: BOTTOM FACE
        p7, p6, p2, p3: LEFT FACE
        p7, p8, p4, p3: FRONT FACE
        p8, p5, p1, p4: RIGHT FACE
        p6, p5, p1, p2: BACK FACE
        p3, p4, p1, p2: TOP FACE
    FACE DEFINITIONS. FACE INDEX: FACENAME: EDGES: EDGE-INDICES
        0: F1: (e1-e2-e3-e4): (0,1,2,3)
        1: F2: (e5-e1-e8-e9): (4,0,7,8)
        2: F3: (e5-e2-e6-e10): (4,1,5,9)
        3: F4: (e6-e3-e7-e11): (5,2,6,10)
        4: F5: (e8-e4-e7-e12): (7,3,6,11)
        5: F6: (e9-e10-e11-e12): (8,9,10,11)
    FACE DEFINITIONS. FACE INDEX: FACENAME: POINTS: POINT INDICES
        0: F1: ((p6-p7)-(p7-p8)-(p8-p5)-(p5-p6): ((5,6)-(6,7)-(7,4)-(4,5))
        1: F2: ((p7-p3)-(p6-p7)-(p6-p2)-(p2-p3): ((6,2)-(5,6)-(1,2)-(2,3))
        2: F3: ((p7-p3)-(p7-p8)-(p8-p4)-(p3-p4): ((6,2)-(6,7)-(7,3)-(2,3))
        3: F4: ((p8-p4)-(p8-p5)-(p5-p1)-(p4-p1): ((7,3)-(7,4)-(5,1)-(3,0))
        4: F5: ((p6-p2)-(p5-p6)-(p5-p1)-(p1-p2): ((1,2)-(4,5)-(5,1)-(0,1))
        5: F6: ((p2-p3)-(p3-p4)-(p4-p1)-(p1-p2): ((2,3)-(2,3)-(3,0)-(0,1))
          :--------------------:
         :|                  : |
       :  |                 :  |
      :   |               :    |
    :-------------------:      |
    |     |             |      |
    |     |             |      |
    |     |             |      |
    |     :-------------|------:
    |    :              |     :
    |  :                |   :
    | :                 |  :
    |-------------------|:
    CELL DEFINITION:
        CELL INDEX: CELLNAME: FACES
                 0: C1: (F1-F2-F3-F4-F5-F6)
        CELL INDEX: CELLNAME: EDGES
                 0: C1: (((p6, p7), (p7, p8), (p8, p5), (p5, p6)),
                         ((p7, p3), (p6, p7), (p6, p2), (p2, p3)),
                         ((p7, p3), (p7, p8), (p8, p4), (p3, p4)),
                         ((p8, p4), (p8, p5), (p5, p1), (p4, p1)),
                         ((p6, p2), (p5, p6), (p5, p1), (p1, p2)),
                         ((p2, p3), (p3, p4), (p4, p1), (p1, p2))
                         )
        CELL INDEX: CELLNAME: POINTS
                 0: C1: (((5,6), (6,7), (7,4), (4,5)),
                         ((6,2), (5,6), (1,2), (2,3)),
                         ((6,2), (6,7), (7,3), (2,3)),
                         ((7,3), (7,4), (5,1), (3,0)),
                         ((1,2), (4,5), (5,1), (0,1)),
                         ((2,3), (2,3), (3,0), (0,1))
                         )
        DATA EXTRACTION GUIDE:
            Vertices will be stored as:
                v = (p1, p2, p3, p4, p5, p6, p7, p8)
            Pixel, itself will may be expressed as:
                PXL = (((v[5],v[6]), v[6],v[7]), (v[7],v[4]), (v[4],v[5])),
                       ((v[6],v[2]), v[5],v[6]), (v[1],v[2]), (v[2],v[3])),
                       ((v[6],v[2]), v[6],v[7]), (v[7],v[3]), (v[2],v[3])),
                       ((v[7],v[3]), v[7],v[4]), (v[5],v[1]), (v[3],v[0])),
                       ((v[1],v[2]), v[4],v[5]), (v[5],v[1]), (v[0],v[1])),
                       ((v[2],v[3]), v[2],v[3]), (v[3],v[0]), (v[0],v[1]))
                      )
          p2 ----------------- p1
         :| .                : |
        : |  .    6        .:  |
      :   |   .         . :    |
    p3 -------------5-- p4     |
    |  '  2     .   .   |      |
    |     | .  .  O .  .| 4    |
    |     |  .  .  .    | .  . |
    |     p6'-3-----.---|----- p5
    |    :  .        .  |    :
    |   : .      1    . |   :
    | :.               .|  :
    p7 ----------------- p8
    IN THE ABOVE ILLUSTRATION:
        Tetrahedron 1: Bottom
        Tetrahedron 2: Left
        Tetrahedron 3: Front
        Tetrahedron 4: Right
        Tetrahedron 5: Back
        Tetrahedron 6: Top
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
