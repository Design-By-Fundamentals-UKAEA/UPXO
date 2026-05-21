"""
Factory functions for creating and converting UPXO geometric entities.

Import
------
    from upxo.geoEntities.featmake import make_p2d, make_p3d
    from upxo.geoEntities.featmake import intersect_slines2d

Functions
---------
make_p2d                          : Convert any point representation to a 2D UPXO point.
make_p3d                          : Convert any point representation to a 3D UPXO point.
intersect_slines2d                : Find intersection points between two Sline2d objects.
intersect_slines2d_collinear_one_way : One-directional collinear intersection helper.

Notes
-----
``make_p2d`` and ``make_p3d`` dispatch on the type string returned by
``find_spec_of_points`` and support many input formats: scalar coordinates,
coordinate lists, NumPy arrays, UPXO lean and full point objects.
"""
import math
import numpy as np
import numpy.matlib
from copy import deepcopy
import vtk
from shapely.geometry import Point as ShPnt, Polygon as ShPol
from functools import wraps
import matplotlib.pyplot as plt
from upxo._sup.dataTypeHandlers import opt as OPT, strip_str as SSTR
np.seterr(divide='ignore')
from upxo._sup.validation_values import isinstance_many
from upxo._sup.validation_values import find_spec_of_points


def make_p2d(points, return_type=None, plane='xy'):
    """
    Convert any point representation into a list of UPXO 2D point objects.

    Handles single points, lists of points, and NumPy arrays. The input may
    be 2D or 3D; the ``plane`` argument selects which two axes to project
    onto when the source is 3D.

    Parameters
    ----------
    points : object
        Input point(s) in any supported format: ``Point2d``, ``Point3d``,
        ``p2d_leanest``, ``p3d_leanest``, coordinate list ``[x, y]`` or
        ``[x, y, z]``, list of the above, or NumPy array with shape
        ``(n, 2)`` or ``(n, 3)``.
    return_type : str, optional
        Target output type. ``'leanest'`` / ``'p2dlean'`` → ``p2d_leanest``;
        ``'Point2d'`` / ``'p2d'`` or None → ``Point2d``. Default is None.
    plane : str, optional
        Which plane to project 3D coordinates onto. One of
        ``'xy'``, ``'yz'``, ``'xz'``, ``'yx'``, ``'zy'``, ``'zx'``.
        Default is ``'xy'``.

    Returns
    -------
    list of Point2d or p2d_leanest
        One element per input point in the requested type.

    Raises
    ------
    ValueError
        When the input point specification is not recognised.

    Examples
    --------
    >>> from upxo.geoEntities.point2d import Point2d as p2d
    >>> from upxo.geoEntities.featmake import make_p2d
    >>> make_p2d(p2d(1, 2), return_type='Point2d')
    >>> make_p2d([p2d(1, 2), p2d(3, 3)], return_type='leanest')
    >>> import numpy as np
    >>> make_p2d(np.random.random((10, 3)), return_type='p2d')
    """
    DEVMODE = False
    # -------------------------------------
    spec_found = False
    if SSTR(return_type) in OPT.name_point2d_leans:
        from upxo.geoEntities.point2d import p2d_leanest as _pnt_
    elif not return_type or SSTR(return_type) in OPT.name_point2d:
        from upxo.geoEntities.point2d import Point2d as _pnt_
    # -------------------------------------
    if find_spec_of_points(points) == 'p3d_leanest' and not spec_found:
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [points._x, points._y]
        elif plane in 'yz':
            coords = [points._y, points._z]
        elif plane in 'xz':
            coords = [points._x, points._z]
        elif plane in 'yx':
            coords = [points._y, points._x]
        elif plane in 'zy':
            coords = [points._z, points._y]
        elif plane in 'zx':
            coords = [points._z, points._x]
        spec_found, points = True, [_pnt_(coords[0], coords[1])]
    if find_spec_of_points(points) == 'p2d_leanest' and not spec_found:
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points._x, points._y)]
    # -------------------------------------
    if find_spec_of_points(points) == '[p3d_leanest]' and not spec_found:
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [[pnt._x, pnt._y] for pnt in points]
        elif plane in 'yz':
            coords = [[pnt._y, pnt._z] for pnt in points]
        elif plane in 'xz':
            coords = [[pnt._x, pnt._z] for pnt in points]
        elif plane in 'yx':
            coords = [[pnt._y, pnt._x] for pnt in points]
        elif plane in 'zy':
            coords = [[pnt._z, pnt._y] for pnt in points]
        elif plane in 'zx':
            coords = [[pnt._z, pnt._x] for pnt in points]
        spec_found, points = True, [_pnt_(co[0], co[1]) for co in coords]
    if find_spec_of_points(points) == '[p2d_leanest]' and not spec_found:
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt._x, pnt._y) for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'Point3d' and not spec_found:
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [points.x, points.y]
        elif plane in 'yz':
            coords = [points.y, points.z]
        elif plane in 'xz':
            coords = [points.x, points.z]
        elif plane in 'yx':
            coords = [points.y, points.x]
        elif plane in 'zy':
            coords = [points.z, points.y]
        elif plane in 'zx':
            coords = [points.z, points.x]
        spec_found, points = True, [_pnt_(coords[0], coords[1])]
    if find_spec_of_points(points) == 'Point2d' and not spec_found:
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points.x, points.y)]
    # -------------------------------------
    if find_spec_of_points(points) == '[Point3d]' and not spec_found:
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [[pnt.x, pnt.y] for pnt in points]
        elif plane in 'yz':
            coords = [[pnt.y, pnt.z] for pnt in points]
        elif plane in 'xz':
            coords = [[pnt.x, pnt.z] for pnt in points]
        elif plane in 'yx':
            coords = [[pnt.y, pnt.x] for pnt in points]
        elif plane in 'zy':
            coords = [[pnt.z, pnt.y] for pnt in points]
        elif plane in 'zx':
            coords = [[pnt.z, pnt.x] for pnt in points]
        # ic(), ic(coords)
        spec_found, points = True, [_pnt_(co[0], co[1]) for co in coords]
    if find_spec_of_points(points) == '[Point2d]' and not spec_found:
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt.x, pnt.y) for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[1,2,3]' and not spec_found:
        # p = [1,2,3]
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [points[0], points[1]]
        elif plane in 'yz':
            coords = [points[1], points[2]]
        elif plane in 'xz':
            coords = [points[2], points[0]]
        elif plane in 'yx':
            coords = [points[1], points[0]]
        elif plane in 'zy':
            coords = [points[2], points[1]]
        elif plane in 'zx':
            coords = [points[0], points[2]]
        spec_found, points = True, [_pnt_(coords[0], coords[1])]
    if find_spec_of_points(points) == 'type-[1,2]' and not spec_found:
        # p = [1,2]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0], points[1])]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3]]' and not spec_found:
        # p = [[1,2,3]]
        if DEVMODE:
            pass
        points = points[0]
        if plane in 'xy':
            coords = [points[0], points[1]]
        elif plane in 'yz':
            coords = [points[1], points[2]]
        elif plane in 'xz':
            coords = [points[0], points[2]]
        elif plane in 'yx':
            coords = [points[1], points[0]]
        elif plane in 'zy':
            coords = [points[2], points[1]]
        elif plane in 'zx':
            coords = [points[2], points[0]]
        spec_found, points = True, [_pnt_(coords[0], coords[1])]
    if find_spec_of_points(points) == 'type-[[1,2]]' and not spec_found:
        # p = [[1,2]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0][0], points[0][1])]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3],[4,5,6],[7,8,9]]' and not spec_found:
        # p = [[1,2,3],[4,5,6],[7,8,9]]
        if DEVMODE:
            pass
        if plane in 'xy':
            coords = [[pnt[0], pnt[1]] for pnt in points]
        elif plane in 'yz':
            coords = [[pnt[1], pnt[2]] for pnt in points]
        elif plane in 'xz':
            coords = [[pnt[0], pnt[2]] for pnt in points]
        elif plane in 'yx':
            coords = [[pnt[1], pnt[0]] for pnt in points]
        elif plane in 'zy':
            coords = [[pnt[2], pnt[1]] for pnt in points]
        elif plane in 'zx':
            coords = [[pnt[2], pnt[0]] for pnt in points]
        spec_found, points = True, [_pnt_(co[0], co[1]) for co in coords]
    if find_spec_of_points(points) == 'type-[[1,2],[3,4],[5,6]]' and not spec_found:
        # p = [[1,2],[3,4],[5,6]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt[0], pnt[1]) for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]' and not spec_found:
        if DEVMODE:
            pass
        points = np.array(points)
        if plane in 'xy':
            coords = [points[0], points[1]]
        elif plane in 'yz':
            coords = [points[1], points[2]]
        elif plane in 'xz':
            coords = [points[2], points[0]]
        elif plane in 'yx':
            coords = [points[1], points[0]]
        elif plane in 'zy':
            coords = [points[2], points[1]]
        elif plane in 'zx':
            coords = [points[0], points[2]]
        # p = [[2, 1, 1, 2], [3, 4, 5, 6]]
        spec_found, points = True, [_pnt_(x, y)
                                    for x, y in zip(coords[0], coords[1])]
    if find_spec_of_points(points) == 'type-[[1,2,3,4],[5,6,7,8]]' and not spec_found:
        if DEVMODE:
            pass
        points = np.array(points)
        spec_found, points = True, [_pnt_(x, y) for x, y in zip(points[0],
                                                                points[1])]
    # -------------------------------------
    if not spec_found:
        raise ValueError('Invalid point specification')
    if spec_found:
        return points

def make_p3d(points, return_type=None, zloc=0.0):
    """
    Convert any point representation into a list of UPXO 3D point objects.

    Parameters
    ----------
    points : object
        Input point(s) in any supported format: ``Point2d``, ``Point3d``,
        ``p2d_leanest``, ``p3d_leanest``, coordinate list ``[x, y, z]``,
        list of the above, or NumPy array with shape ``(n, 3)``.
    return_type : str, optional
        Target output type. ``'p3dlean'`` / ``'leanest'`` → ``p3d_leanest``;
        ``'Point3d'`` / ``'p3d'`` or None → ``Point3d``. Default is None.
    zloc : float, optional
        Z-coordinate to assign when promoting 2D points to 3D.
        Default is 0.0.

    Returns
    -------
    list of Point3d or p3d_leanest
        One element per input point in the requested type.

    Raises
    ------
    ValueError
        When the input point specification is not recognised.

    Examples
    --------
    >>> from upxo.geoEntities.point2d import Point2d as p2d
    >>> from upxo.geoEntities.featmake import make_p3d
    >>> make_p3d(p2d(1, 2), return_type='Point3d', zloc=1.0)
    >>> make_p3d([p2d(1, 2), p2d(3, 3)], return_type='leanest')
    >>> import numpy as np
    >>> make_p3d(np.random.random((10, 3)), return_type='p3d')
    """
    DEVMODE = False
    # -------------------------------------
    from upxo._sup.validation_values import find_spec_of_points
    spec_found = False
    if return_type in ('p3dlean', 'leanest'):
        from upxo.geoEntities.point3d import p3d_leanest as _pnt_
    elif not return_type or return_type in ('p3d', 'Point3d'):
        from upxo.geoEntities.point3d import Point3d as _pnt_
    # -------------------------------------
    if find_spec_of_points(points) == 'p3d_leanest':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points._x, points._y, points._z)]
    if find_spec_of_points(points) == 'p2d_leanest':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points._x, points._y, zloc)]
    # -------------------------------------
    if find_spec_of_points(points) == '[p3d_leanest]':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt._x, pnt._y, pnt._z)
                                    for pnt in points]
    if find_spec_of_points(points) == '[p2d_leanest]':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt._x, pnt._y, zloc)
                                    for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'Point3d':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points.x, points.y, points.z)]
    if find_spec_of_points(points) == 'Point2d':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points.x, points.y, zloc)]
    # -------------------------------------
    if find_spec_of_points(points) == '[Point3d]':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt.x, pnt.y, pnt.z)
                                    for pnt in points]
    if find_spec_of_points(points) == '[Point2d]':
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt.x, pnt.y, zloc)
                                    for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[1,2,3]':  # p = [1,2,3]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0], points[1], points[2])]
    if find_spec_of_points(points) == 'type-[1,2]':  # p = [1,2]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0], points[1], zloc)]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3]]':  # p = [[1,2,3]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0][0],
                                          points[0][1],
                                          points[0][2])]
    if find_spec_of_points(points) == 'type-[[1,2]]':  # p = [[1,2]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(points[0][0], points[0][1], zloc)]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
        # p = [[1,2,3],[4,5,6],[7,8,9]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt[0], pnt[1], pnt[2])
                                    for pnt in points]
    if find_spec_of_points(points) == 'type-[[1,2],[3,4],[5,6]]':
        # p = [[1,2],[3,4],[5,6]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(pnt[0], pnt[1], zloc)
                                    for pnt in points]
    # -------------------------------------
    if find_spec_of_points(points) == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
        # p = [[2, 1, 1, 2], [3, 4, 5, 6]]
        if DEVMODE:
            pass
        spec_found, points = True, [_pnt_(x, y, z)
                                    for x, y, z in zip(points[0],
                                                       points[1],
                                                       points[2])]
    # -------------------------------------
    if not spec_found:
        raise ValueError('Invalid point specification')
    if spec_found:
        return points


def intersect_slines2d(la, lb, p2d, return_type='upxo'):
    """
    Find all intersection points between two UPXO 2D straight lines.

    Checks both directions (``la`` with ``lb`` and ``lb`` with ``la``) to
    handle collinear overlaps, deduplicates the result, then falls back to
    the standard algebraic line–line intersection formula for non-parallel
    lines.

    Parameters
    ----------
    la : Sline2d
        First straight line.
    lb : Sline2d
        Second straight line.
    p2d : type
        The ``Point2d`` class used to construct output points.
    return_type : str, optional
        ``'upxo'`` returns ``Point2d`` objects; ``'coord'`` returns
        ``[x, y]`` pairs. Default is ``'upxo'``.

    Returns
    -------
    list
        List of intersection points in the format requested by
        ``return_type``.  Empty list when lines do not intersect.

    Examples
    --------
    >>> from upxo.geoEntities.sline2d import Sline2d as sl2d
    >>> from upxo.geoEntities.point2d import Point2d as p2d
    >>> import upxo.geoEntities.featmake as fmake
    >>> la = sl2d.by_coord([0, 0], [1, 1])
    >>> lb = sl2d.by_coord([0, 1], [1, 0])
    >>> fmake.intersect_slines2d(la, lb, p2d)
    >>> fmake.intersect_slines2d(la, lb, p2d, return_type='coord')
    """
    lbpnts_inside_la = intersect_slines2d_collinear_one_way(la, lb, p2d)
    lapnts_inside_lb = intersect_slines2d_collinear_one_way(lb, la, p2d)
    # print(lbpnts_inside_la, lapnts_inside_lb)

    intersection = lbpnts_inside_la + lapnts_inside_lb
    # Make this list unique
    if intersection:
        coords = np.unique([[p.x, p.y] for p in intersection], axis=0)
        if return_type == 'upxo':
            intersection = [p2d(xy[0], xy[1]) for xy in coords]
        elif return_type == 'coord':
            intersection = [list(coord) for coord in coords]
    # -------------------------------------------
    # Now we carry out the usual line intersection
    if la.gradient == lb.gradient:
        return intersection

    lineAx0, lineAy0, lineAx1, lineAy1 = la.x0, la.y0, la.x1, la.y1
    lineBx0, lineBy0, lineBx1, lineBy1 = lb.x0, lb.y0, lb.x1, lb.y1

    dx0, dy0 = lineAx1-lineAx0, lineAy1-lineAy0
    dx1, dy1 = lineBx1-lineBx0, lineBy1-lineBy0

    p0 = dy1*(lineBx1-lineAx0) - dx1*(lineBy1-lineAy0)
    p1 = dy1*(lineBx1-lineAx1) - dx1*(lineBy1-lineAy1)
    p2 = dy0*(lineAx1-lineBx0) - dx0*(lineAy1-lineBy0)
    p3 = dy0*(lineAx1-lineBx1) - dx0*(lineAy1-lineBy1)

    if (p0*p1 <= 0) and (p2*p3 <= 0):
        # Calculate slopes (handle division by zero for vertical lines)
        if lineAx1 - lineAx0 != 0:
            slope_a = (lineAy1 - lineAy0) / (lineAx1 - lineAx0)
        else:
            slope_a = float("inf")  # Vertical line
        # -----------------------------
        if lineBx1 - lineBx0 != 0:
            slope_b = (lineBy1 - lineBy0) / (lineBx1 - lineBx0)
        else:
            slope_b = float("inf")  # Vertical line
        # -----------------------------
        # Calculate y-intercepts
        intercept_a = lineAy0 - slope_a * lineAx0
        intercept_b = lineBy0 - slope_b * lineBx0
        # Solve for intersection point
        x = (intercept_b - intercept_a) / (slope_a - slope_b)
        y = slope_a * x + intercept_a
        if return_type == 'upxo':
            intersection.append(p2d(x, y))
        elif return_type == 'coord':
            intersection.append([x, y])

    return intersection


def intersect_slines2d_collinear_one_way(la, lb, p2d):
    """
    Find endpoints of ``lb`` that lie on ``la`` (one-directional collinear check).

    Tests whether each endpoint of ``lb`` is within the perpendicular distance
    tolerance of ``la`` and also within the chord length of ``la``.  Intended
    to be called from :func:`intersect_slines2d` for each direction.

    Parameters
    ----------
    la : Sline2d
        Reference line.
    lb : Sline2d
        Line whose endpoints are tested against ``la``.
    p2d : type
        The ``Point2d`` class used to build point objects for distance checks.

    Returns
    -------
    list of Point2d
        Endpoints of ``lb`` that lie on ``la``.  Empty list if none qualify.

    Examples
    --------
    >>> from upxo.geoEntities.sline2d import Sline2d as sl2d
    >>> from upxo.geoEntities.point2d import Point2d as p2d
    >>> import upxo.geoEntities.featmake as fmake
    >>> la = sl2d.by_coord([0, 0], [1, 1])
    >>> lb = sl2d.by_coord([0.1, 0.1], [1.8, 1.8])
    >>> fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)
    """
    la_p0 = p2d(la.x0, la.y0)
    la_p1 = p2d(la.x1, la.y1)
    lb_p0 = p2d(lb.x0, lb.y0)
    lb_p1 = p2d(lb.x1, lb.y1)
    # W.R.T Line-A.
    # STEP 1: Let's check if lb_p0 and/or lb_p1 lies in/on la.
    # - Find perpendicular distance to la
    # --- Points on line A
    la_points = [la_p0, la_p1]
    # --- Points on line B
    lb_points = [lb_p0, lb_p1]
    # --- PERPENDICULAR-DIST from la to lb_points
    pdist_la = la.perp_distance(lb_points, ptype='p2d')
    # --- Which points have zero PDIST: indices
    zero_pdist_to_la = np.where(pdist_la < p2d.ε)[0]
    # --- Which points have zero PDIST: actual points
    zero_pdist_to_la = [lb_points[i] for i in zero_pdist_to_la]
    # ---> NOTE: zero_pdist_to_la has points belonmging to lb
    # --- Calculate distanced from points in zero_pdist_to_la to
    #     points on la
    D = [[] for _ in zero_pdist_to_la]
    for i, p in enumerate(zero_pdist_to_la):
        D[i] = p.distance(la_points)

    lbpnts_inside_la = []
    for i, inside in enumerate([all(d <= la.length) for d in D]):
        if inside:
            lbpnts_inside_la.append(lb_points[i])
    return lbpnts_inside_la
