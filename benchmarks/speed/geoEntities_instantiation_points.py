from timeit import timeit

direct = '''
from upxo.geoEntities.point2d import p2d_leanest, Point2d
from upxo.geoEntities.point3d import p3d_leanest, Point3d
# .........................
def init_point2d_leanest_direct(): a=p2d_leanest(0,0)
def init_point2d_direct(): a=Point2d(0,0)
def init_point3d_leanest_direct(): a=p3d_leanest(1,2,0)
def init_point3d_direct_1(): a=Point3d(0,0,0)
def init_point3d_direct_2(): a=Point3d(0,0)
'''
# ---------------------------------
nruns = 100*100
# ---------------------------------
extime = timeit(stmt='init_point2d_leanest_direct()',
                setup=direct,
                number=nruns)
print(f"point2d_leanest_direct: {nruns} instants: {extime} sec.")
extime = timeit(stmt='init_point2d_direct()',
                setup=direct,
                number=nruns)
print(f"point2d_direct: {nruns} instants: {extime} sec.")
# ---------------------------------
print(50*'-')
# ---------------------------------
extime = timeit(stmt='init_point3d_leanest_direct()',
                setup=direct,
                number=nruns)
print(f"point3d_leanest_direct: {nruns} instants: {extime} sec.")
extime = timeit(stmt='init_point3d_direct_1()',
                setup=direct,
                number=nruns)
print(f"point3d_direct_1: {nruns} instants: {extime} sec.")
extime = timeit(stmt='init_point3d_direct_2()',
                setup=direct,
                number=nruns)
print(f"point3d_direct_2: {nruns} instants: {extime} sec.")
# ---------------------------------
by_op_from_intersection_two_lines = """
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.sline2d import Sline2d as sl2d

def Collinear_lines_Case_1(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0.1,0.1],[1.8,1.8])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_2(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0.1,0.1],[0.8,0.8])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_3(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([-0.1,-0.1], [1.8,1.8])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_4(nrunsinternal):
    la,lb=sl2d.by_coord([0,0], [1,1]), sl2d.by_coord([1.8,1.8],[0.1,0.1])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Non_collinear_lines_Case_5(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0,1],[1,0])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_6(nrunsinternal):
    la,lb=sl2d.by_coord([0.1,0.1], [0.8,0.8]), sl2d.by_coord([0,0],[1,1])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Non_Collinear_lines_Case_7a(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0,0],[1,0])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Non_Collinear_lines_Case_7b(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([-0,-0],[1,0])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_8(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0,0],[1,1])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_9(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,1]), sl2d.by_coord([0,0],[-1,-1])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')

def Collinear_lines_Case_10(nrunsinternal):
    la,lb=sl2d.by_coord([0,0],[1,0]), sl2d.by_coord([0,1],[1,1])
    for n in range(nrunsinternal):
        p2d.from_intersection_two_lines(la, lb, tool='upxo')
"""
tool = 'upxo'
extime = timeit(stmt=f"Collinear_lines_Case_1({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_1: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_2({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_2: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_3({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_3: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_4({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_4: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Non_collinear_lines_Case_5({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Non_collinear_lines_Case_5: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_6({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_6: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Non_Collinear_lines_Case_7a({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Non_Collinear_lines_Case_7a: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Non_Collinear_lines_Case_7b({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Non_Collinear_lines_Case_7b: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_8({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_8: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_9({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_9: {nruns} instants: {extime} sec.")
extime = timeit(stmt=f"Collinear_lines_Case_10({nruns})",
                setup=by_op_from_intersection_two_lines,
                number=1)
print(f"point_by_op_Collinear_lines_Case_10: {nruns} instants: {extime} sec.")
# ---------------------------------
print(50*'-')
# ---------------------------------
