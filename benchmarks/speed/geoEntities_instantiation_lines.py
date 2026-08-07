from timeit import timeit


direct = '''
from upxo.geoEntities.sline2d import Sline2d_leanest, Sline2d
from upxo.geoEntities.sline3d import Sline3d_leanest, Sline3d
# .........................
def init_Sline2d_leanest_direct_1(): a=Sline2d_leanest(-2,3,4,5)
def init_Sline2d_leanest_direct_2(): a=Sline2d_leanest()
def init_Sline2d_direct_1(): a=Sline2d(0,0,1,1)
def init_Sline2d_direct_2(): a=Sline2d()
# .........................
def init_Sline3d_leanest_direct_1(): a=Sline3d_leanest(-2,3,4,5,1,2)
def init_Sline3d_leanest_direct_2(): a=Sline3d_leanest()
def init_Sline3d_direct_1(): a=Sline3d(0,0,0,1,1,1)
def init_Sline3d_direct_2(): a=Sline3d()
'''
# ---------------------------------
nruns = 100*100
# ---------------------------------
extime = timeit(stmt='init_Sline2d_leanest_direct_1()', setup=direct, number=nruns)
print(f"Sline2d_leanest_direct_1: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline2d_leanest_direct_2()', setup=direct, number=nruns)
print(f"Sline2d_leanest_direct_2: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline2d_direct_1()', setup=direct, number=nruns)
print(f"Sline2d_direct_1: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline2d_direct_2()', setup=direct, number=nruns)
print(f"Sline2d_direct_2: {nruns} instants: {extime} seconds")
# ---------------------------------
print(25*'-')
# ---------------------------------
extime = timeit(stmt='init_Sline3d_leanest_direct_1()', setup=direct, number=nruns)
print(f"Sline3d_leanest_direct_1: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline3d_leanest_direct_2()', setup=direct, number=nruns)
print(f"Sline3d_leanest_direct_2: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline3d_direct_1()', setup=direct, number=nruns)
print(f"Sline3d_direct_1: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline3d_direct_2()', setup=direct, number=nruns)
print(f"Sline3d_direct_2: {nruns} instants: {extime} seconds")
# ---------------------------------
print(50*'-')
# ---------------------------------
by_coord = '''
from upxo.geoEntities.sline2d import Sline2d
from upxo.geoEntities.sline3d import Sline3d

def init_Sline2d(): a=Sline2d.by_coord([-1,2],[3,4])
def init_Sline3d(): a=Sline3d.by_coord([-1,2,0],[3,4,1])
'''
# ---------------------------------
extime = timeit(stmt='init_Sline2d()', setup=by_coord, number=nruns)
print(f"Sline2d.by_coord: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_Sline3d()', setup=by_coord, number=nruns)
print(f"Sline3d.by_coord: {nruns} instants: {extime} seconds")
# ---------------------------------
sl2d_by_LFAL = """
from upxo.geoEntities.sline2d import Sline2d as sl2d

def init_sl2d_by_LFAL_case1():
    sl2d.by_LFAL(location=[0,0], factor=0.0, angle=0, length=1, degree=True)

def init_sl2d_by_LFAL_case2():
    sl2d.by_LFAL(location=[0,0], factor=0.0, angle=90, length=1, degree=True)

def init_sl2d_by_LFAL_case3():
    sl2d.by_LFAL(location=[10,10], factor=0.0, angle=45, length=1, degree=True)

def init_sl2d_by_LFAL_case4():
    sl2d.by_LFAL(location=[10,10], factor=1.0, angle=45, length=1, degree=True)

def init_sl2d_by_LFAL_case5():
    sl2d.by_LFAL(location=[10,10], factor=0.5, angle=45, length=1, degree=True)

def init_sl2d_by_LFAL_case6():
    sl2d.by_LFAL(location=[10,10], factor=0.2, angle=45, length=1, degree=True)
"""
# ---------------------------------
extime = timeit(stmt='init_sl2d_by_LFAL_case1()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 1: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_sl2d_by_LFAL_case2()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 2: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_sl2d_by_LFAL_case3()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 3: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_sl2d_by_LFAL_case4()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 4: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_sl2d_by_LFAL_case5()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 5: {nruns} instants: {extime} seconds")
extime = timeit(stmt='init_sl2d_by_LFAL_case6()', setup=sl2d_by_LFAL, number=nruns)
print(f"sl2d.by_LFAL case 6: {nruns} instants: {extime} seconds")
# ---------------------------------
print(50*'-')
# ---------------------------------
