"""
Import
------
from upxo.geoEntities.featmake import make_p2d, make_p3d
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
    Convert point to a UPXO point object.

    Development targets and progress
    --------------------------------
    PHASE - 1: Leanest and point2d

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d as p2d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import Point3d as p3d
    from upxo.geoEntities.point3d import p3d_leanest

    make_p2d(p2d(1,2), return_type='leanest')
    make_p2d(p2d(1,2), return_type='p2d')
    make_p2d([p2d(1,2)], return_type='p2d')
    make_p2d([p2d(1,2), p2d(3,3)], return_type='leanest')
    make_p2d(p2d_leanest(1,3), return_type='leanest')
    make_p2d([p2d_leanest(1,2), p2d_leanest(1,2)], return_type='leanest')

    make_p2d(p2d(1,2), return_type='Point2d')
    make_p2d(p2d(1,2))
    make_p2d(p2d(1,2), return_type='leanest')
    make_p2d([p2d(1,2), p2d(3,3)], return_type='leanest')
    make_p2d(p2d_leanest(1,3), return_type='leanest')
    make_p2d([p2d_leanest(1,2), p2d_leanest(1,2)], return_type='leanest')

    make_p2d(p3d(1,2,3), return_type='Point2d')
    make_p2d([p3d(1,2,3), p3d(3,3,3)], return_type='Point2d')
    make_p2d(p3d_leanest(1,2,3), return_type='p2dlean')
    make_p2d(p3d_leanest(1,2,3))
    make_p2d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p2dlean')
    make_p2d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p2d')
    make_p2d([1,2,3], return_type='p2dlean')
    make_p2d([[1,2,3]], return_type='p2dlean')
    make_p2d([[1,2,3],[4,5,6],[7,8,9]], return_type='p2dlean')
    make_p2d([[1,2,3],[4,5,6],[7,8,9]], return_type='Point2d')

    make_p2d([[1,2,3,4],[2,3,4,5],[3,4,5,6]], return_type='leanest')
    make_p2d([[1,2,3,4],[1,2,3,4],[1,2,3,4]], return_type='p2d')

    coords = np.random.random((10,3))
    make_p2d(coords, return_type='p2d')

    make_p2d(p3d(1,2,3), return_type='Point2d', plane='xy')
    make_p2d(p3d(1,2,3), return_type='Point2d', plane='yz')
    make_p2d(p3d(1,2,3), return_type='Point2d', plane='xz')
    make_p2d(p3d(1,2,3), return_type='Point2d', plane='yx')
    make_p2d(p3d(1,2,3), return_type='Point2d', plane='zy')
    make_p2d(p3d(1,2,3), return_type='Point2d', plane='zx')

    make_p2d([p3d(1,2,3), p3d(3,3,3)], return_type='Point2d')
    make_p2d(p3d_leanest(1,2,3), return_type='p2dlean')
    make_p2d(p3d_leanest(1,2,3))
    make_p2d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p2dlean')
    make_p2d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p2d')
    make_p2d([1,2,3], return_type='p2dlean')
    make_p2d([[1,2,3]], return_type='p2dlean')
    make_p2d([[1,2,3],[4,5,6],[7,8,9]], return_type='p2dlean')
    make_p2d([[1,2,3],[4,5,6],[7,8,9]], return_type='Point2d')

    make_p2d([1,2,3,4], return_type='p2dlean')  # Invalid. Must not work.
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
    Convert point to a UPXO point object.

    Development targets and progress
    --------------------------------
    PHASE - 1: LEANEST. DONE
    PHASE - 2: 3D ALTERNATIVE. DONE

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d as p2d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import Point3d as p3d
    from upxo.geoEntities.point3d import p3d_leanest

    make_p3d(p2d(1,2), return_type='leanest')
    make_p3d(p2d(1,2), return_type='leanest', zloc=1.12345)
    make_p3d([p2d(1,2), p2d(3,3)], return_type='leanest')
    make_p3d(p2d_leanest(1,3), return_type='leanest')
    make_p3d([p2d_leanest(1,2), p2d_leanest(1,2)], return_type='leanest')

    make_p3d(p2d(1,2), return_type='Point3d')
    make_p3d(p2d(1,2))
    make_p3d(p2d(1,2), return_type='leanest', zloc=1.12345)
    make_p3d([p2d(1,2), p2d(3,3)], return_type='leanest')
    make_p3d(p2d_leanest(1,3), return_type='leanest')
    make_p3d([p2d_leanest(1,2), p2d_leanest(1,2)], return_type='leanest')

    make_p3d(p3d(1,2,3), return_type='Point3d')
    make_p3d([p3d(1,2,3), p3d(3,3,3)], return_type='Point3d')
    make_p3d(p3d_leanest(1,2,3), return_type='p3dlean')
    make_p3d(p3d_leanest(1,2,3))

    make_p3d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p3dlean')
    make_p3d([p3d_leanest(1,2,3), p3d_leanest(1,2,3)], return_type='p3d')

    make_p3d([1,2,3], return_type='p3dlean')
    make_p3d([1,2,3,4], return_type='p3dlean')  # Invalid: Produces no output !
    make_p3d([[1,2,3]], return_type='p3dlean')
    make_p3d([[1,2,3],[4,5,6],[7,8,9]], return_type='p3dlean')
    make_p3d([[1,2,3],[4,5,6],[7,8,9]], return_type='Point3d')

    make_p3d([[1,2,3,4],[1,2,3,4],[1,2,3,4]], return_type='leanest')
    make_p3d([[1,2,3,4],[1,2,3,4],[1,2,3,4]], return_type='p3d')

    make_p3d(np.random.random((10,3)), return_type='p3d')
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
    Find points of intersection between two UPXO stringht 2D lines.

    First checks intersection of la with lb.
    Then checks intersection of lb with la.
    Combines the two results and returns the unique set of points.

    Alternatively, if you want to check only the intewrsection of la with lb,
    then use the definition intersect_slines2d_one_way(la, lb).

    from upxo.geoEntities.sline2d import Sline2d as sl2d
    from upxo.geoEntities.point2d import Point2d as p2d

    Import and suggested use
    ------------------------
    import upxo.geoEntities.featmake as fmake
    from upxo.geoEntities.sline2d import Sline2d
    # fmake.intersect_slines2d(...)

    # Example-1: Collinear lines Case 1
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0.1, 0.1], [1.8, 1.8])
    fmake.intersect_slines2d(la, lb, p2d)
    fmake.intersect_slines2d(la, lb, p2d, return_type='upxo')
    fmake.intersect_slines2d(la, lb, p2d, return_type='coord')

    # Example-2: Collinear lines Case 2
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0.1, 0.1], [0.8, 0.8])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-3: Collinear lines Case 3
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([-0.1, -0.1], [1.8, 1.8])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-4: Collinear lines Case 4
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([1.8, 1.8], [0.1, 0.1])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-5: non-collinear lines Case 5
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0, 1], [1, 0])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-6: Collinear lines Case 6
    # ----------------------------------
    la = sl2d.by_coord([0.1, 0.1], [0.8, 0.8])
    lb = sl2d.by_coord([0, 0], [1, 1])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-7: Non-Collinear lines Case 7a
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0, 0], [1, 0])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-8: Non-Collinear lines Case 7b
    # ----------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([-0,-0], [1, 0])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-9: Collinear lines Case 8
    # ----------------------------------
    la = sl2d.by_coord([0,0], [1,1])
    lb = sl2d.by_coord([0,0], [1,1])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-10: Collinear lines Case 9
    # ----------------------------------
    la = sl2d.by_coord([0,0], [1,1])
    lb = sl2d.by_coord([0,0], [-1,-1])
    fmake.intersect_slines2d(la, lb, p2d)

    # Example-11: Collinear lines Case 10
    # -----------------------------------
    la = sl2d.by_coord([0,0], [1,0])
    lb = sl2d.by_coord([0,1], [1,1])
    fmake.intersect_slines2d(la, lb, p2d)
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
    The lines la and lb must be upxo sline2d type.

    from upxo.geoEntities.sline2d import Sline2d as sl2d
    from upxo.geoEntities.point2d import Point2d as p2d

    Import and suggested use
    ------------------------
    import upxo.geoEntities.featmake as fmake
    # fmake.intersect_slines2d_collinear_one_way(...)

    # Example-1: Collinear lines Case 1
    # ---------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0.1, 0.1], [1.8, 1.8])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-2: Collinear lines Case 2
    # ---------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0.1, 0.1], [0.8, 0.8])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-3: Collinear lines Case 3
    # ---------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([-0.1, -0.1], [1.8, 1.8])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-4: Collinear lines Case 4
    # ---------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([1.8, 1.8], [0.1, 0.1])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-5: non-collinear lines Case 5
    # -------------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0, 1], [1, 0])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-6: Collinear lines Case 6
    # ---------------------------------
    la = sl2d.by_coord([0.1, 0.1], [0.8, 0.8])
    lb = sl2d.by_coord([0, 0], [1, 1])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-7: Non-Collinear lines Case 7a
    # --------------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([0, 0], [1, 0])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-8: Non-Collinear lines Case 7b
    # --------------------------------------
    la = sl2d.by_coord([0, 0], [1, 1])
    lb = sl2d.by_coord([-0,-0], [1, 0])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-9: Collinear lines Case 8
    # ---------------------------------
    la = sl2d.by_coord([0,0], [1,1])
    lb = sl2d.by_coord([0,0], [1,1])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-10: Collinear lines Case 9
    # ----------------------------------
    la = sl2d.by_coord([0,0], [1,1])
    lb = sl2d.by_coord([0,0], [-1,-1])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)

    # Example-11: Collinear lines Case 10
    # -----------------------------------
    la = sl2d.by_coord([0,0], [1,0])
    lb = sl2d.by_coord([0,1], [1,1])
    fmake.intersect_slines2d_collinear_one_way(la, lb, p2d)
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
