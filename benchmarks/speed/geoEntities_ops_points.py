from timeit import timeit
# ---------------------------------
su1_eq = '''
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import p3d_leanest
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point3d import Point3d as p3d
def bool_eq_p2d_01(): a = p2d(3,4)==p2d_leanest(3,4)
def bool_eq_p2d_02(): a = p2d(3,4)==p2d(3,4)
def bool_eq_p2d_03(): a = p2d(3,4)==[3,4]
def bool_eq_p2d_04(): a = p2d(3,4)==[[3,4]]
def bool_eq_p2d_05(): a = p2d(3,4)==[[3],[4]]
'''
ninst = 100*100
extime = timeit(stmt='bool_eq_p2d_01()', setup=su1_eq, number=ninst)
print(f"p2d(3, 4) == p2d_leanest(3, 4): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_02()', setup=su1_eq, number=ninst)
print(f"p2d(3, 4) == p2d(3, 4): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_03()', setup=su1_eq, number=ninst)
print(f"p2d(3, 4) == [3, 4]: {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_04()', setup=su1_eq, number=ninst)
print(f"p2d(3, 4) == [[3, 4]]: {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_05()', setup=su1_eq, number=ninst)
print(f"p2d(3, 4) == [[3], [4]]: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su2_eq = '''
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import p3d_leanest
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point3d import Point3d as p3d
def bool_eq_p2d_01(): a = p2d(3,4).eq(p2d_leanest(3,4))
def bool_eq_p2d_02(): a = p2d(3,4).eq(p2d(3,4))
def bool_eq_p2d_03(): a = p2d(3,4).eq([3,4])
def bool_eq_p2d_04(): a = p2d(3,4).eq([[3,4]])
def bool_eq_p2d_05(): a = p2d(3,4).eq([[3],[4]])
'''
extime = timeit(stmt='bool_eq_p2d_01()', setup=su2_eq, number=ninst)
print(f"p2d(3,4).eq(p2d_leanest(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_02()', setup=su2_eq, number=ninst)
print(f"p2d(3,4).eq(p2d(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_03()', setup=su2_eq, number=ninst)
print(f"p2d(3,4).eq([3,4]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_04()', setup=su2_eq, number=ninst)
print(f"p2d(3,4).eq([[3,4]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_05()', setup=su2_eq, number=ninst)
print(f"p2d(3,4).eq([[3],[4]]): {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su3_eq = '''
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import p3d_leanest
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point3d import Point3d as p3d
def bool_eq_p2d_01(): a = p2d(3,4).eq_fast(p2d_leanest(3,4), point_spec=3)
def bool_eq_p2d_02(): a = p2d(3,4).eq_fast(p2d(3,4), point_spec=1)
def bool_eq_p2d_03(): a = p2d(3,4).eq_fast([3,4], point_spec=5)
def bool_eq_p2d_04(): a = p2d(3,4).eq_fast([[3,4]], point_spec=6)
def bool_eq_p2d_05(): a = p2d(3,4).eq_fast([[3],[4]], point_spec=8)
'''
extime = timeit(stmt='bool_eq_p2d_01()', setup=su3_eq, number=ninst)
print(f"p2d(3,4).eq_fast(p2d_leanest(3,4), point_spec=3): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_02()', setup=su3_eq, number=ninst)
print(f"p2d(3,4).eq_fast(p2d(3,4), point_spec=1): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_03()', setup=su3_eq, number=ninst)
print(f"p2d(3,4).eq_fast([3,4], point_spec=5): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_04()', setup=su3_eq, number=ninst)
print(f"p2d(3,4).eq_fast([[3,4]], point_spec=6): {ninst} ops: {extime} seconds")
extime = timeit(stmt='bool_eq_p2d_04()', setup=su3_eq, number=ninst)
print(f"p2d(3,4).eq_fast([[3],[4]], point_spec=8): {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su4_sqdist = '''
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import p3d_leanest
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point3d import Point3d as p3d
def p2d_op_sqdist01(): p2d(0,0).squared_distance(p2d_leanest(3,4))
def p2d_op_sqdist02(): p2d(0,0).squared_distance(p2d(3,4))
def p2d_op_sqdist03(): p2d(0,0).squared_distance([p2d(1,2),p2d(3,4)])
def p2d_op_sqdist04(): p2d(0,0).squared_distance((p2d(1,4),p2d(3,4)))
def p2d_op_sqdist05(): p2d(0,0).squared_distance([1, 2])
def p2d_op_sqdist06(): p2d(0,0).squared_distance([[1,2]])
def p2d_op_sqdist07(): p2d(0,0).squared_distance([[1,2],[10,12]])
def p2d_op_sqdist08(): p2d(0,0).squared_distance([[1,2],[10,12],[0,-5]])
def p2d_op_sqdist09(): p2d(0,0).squared_distance([[1,2,-1,-3],[4,5,5,6]])
'''
extime = timeit(stmt='p2d_op_sqdist01()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance(p2d_leanest(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist02()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance(p2d(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist03()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([p2d(1,2),p2d(3,4)]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist04()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance((p2d(1,4),p2d(3,4))): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist05()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([1, 2]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist06()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([[1,2]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist07()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([[1,2],[10,12]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist08()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([[1,2],[10,12],[0,-5]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_sqdist09()', setup=su4_sqdist, number=ninst)
print(f"p2d(0,0).squared_distance([[1,2,-1,-3],[4,5,5,6]]): {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su5_dist = '''
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import p3d_leanest
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point3d import Point3d as p3d
def p2d_op_dist01(): p2d(0,0).distance(p2d_leanest(3,4))
def p2d_op_dist02(): p2d(0,0).distance(p2d(3,4))
def p2d_op_dist03(): p2d(0,0).distance([p2d(1,2),p2d(3,4)])
def p2d_op_dist04(): p2d(0,0).distance((p2d(1,4),p2d(3,4)))
def p2d_op_dist05(): p2d(0,0).distance([1, 2])
def p2d_op_dist06(): p2d(0,0).distance([[1,2]])
def p2d_op_dist07(): p2d(0,0).distance([[1,2],[10,12]])
def p2d_op_dist08(): p2d(0,0).distance([[1,2],[10,12],[0,-5]])
def p2d_op_dist09(): p2d(0,0).distance([[1,2,-1,-3],[4,5,5,6]])
'''
extime = timeit(stmt='p2d_op_dist01()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance(p2d_leanest(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist02()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance(p2d(3,4)): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist03()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([p2d(1,2),p2d(3,4)]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist04()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance((p2d(1,4),p2d(3,4))): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist05()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([1, 2]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist06()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([[1,2]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist07()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([[1,2],[10,12]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist08()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([[1,2],[10,12],[0,-5]]): {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_dist09()', setup=su5_dist, number=ninst)
print(f"p2d(0,0).distance([[1,2,-1,-3],[4,5,5,6]]): {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su6_translate = '''
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_translate_1():A.translate(vector=[1,1],dist=5,update=True, throw=False)
'''
extime = timeit(stmt='p2d_op_translate_1()', setup=su6_translate, number=ninst)
print(f"p2d(0,0).distance(p2d_leanest(3,4)): {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su7_rotate = '''
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_rotate_1():A.rotate_points(p2d(1,0),45,degree=True,dec=8)
def p2d_op_rotate_2():A.rotate_points((1,0),45,degree=True,dec=8)
def p2d_op_rotate_3():A.rotate_points(([1,0],[2,0],[3,0]),-45,degree=True,dec=8)
'''
extime = timeit(stmt='p2d_op_rotate_1()', setup=su7_rotate, number=ninst)
print(f"p2d_op_rotate_1: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_rotate_2()', setup=su7_rotate, number=ninst)
print(f"p2d_op_rotate_2: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_rotate_3()', setup=su7_rotate, number=ninst)
print(f"p2d_op_rotate_3: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su8_find_closest_point = '''
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_find_closest_point_1():A.find_closest_points([p2d(0,0),p2d(0,1),p2d(0,0)])
def p2d_op_find_closest_point_2():A.find_closest_points([[1,2],[2,3],[10,12]])
'''
extime = timeit(stmt='p2d_op_find_closest_point_1()', setup=su8_find_closest_point, number=ninst)
print(f"p2d_op_find_closest_point_1: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_find_closest_point_2()', setup=su8_find_closest_point, number=ninst)
print(f"p2d_op_find_closest_point_2: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su9_find_neigh_points_by_distance = '''
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_find_neigh_points_by_distance_1():A.find_neigh_points_by_distance([[1,2],[10,12],[0,-5]])
'''
extime = timeit(stmt='p2d_op_find_neigh_points_by_distance_1()', setup=su9_find_neigh_points_by_distance, number=ninst)
print(f"p2d_op_find_neigh_points_by_distance_1: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su10_find_neigh_points_by_count = '''
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_find_neigh_points_by_count_1():A.find_neigh_points_by_count([[1,2], [10,12], [0,-5], [0,0]], 2)
'''
extime = timeit(stmt='p2d_op_find_neigh_points_by_count_1()', setup=su10_find_neigh_points_by_count, number=ninst)
print(f"p2d_op_find_neigh_points_by_count_1: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
su11_array_by_clustering = """
from upxo.geoEntities.point2d import Point2d as p2d
A = p2d(0,0)
def p2d_op_array_by_clust_1():A.array_by_clustering(n=10,r=1)
def p2d_op_array_by_clust_2():A.array_by_clustering(n=10,r=1,return_type='coords_2d')
def p2d_op_array_by_clust_3():A.array_by_clustering(n=10,r=1,return_type='upxo_2d')
def p2d_op_array_by_clust_4():A.array_by_clustering(n=10,r=1,return_type='upxo_2d_leanest')
def p2d_op_array_by_clust_5():A.array_by_clustering(n=10,r=1,return_type='shapely')
def p2d_op_array_by_clust_6():A.array_by_clustering(n=10,r=1,return_type='gmsh')
def p2d_op_array_by_clust_7():A.array_by_clustering(n=10,r=1,return_type='pyvista')
def p2d_op_array_by_clust_8():A.array_by_clustering(n=10,r=1,return_type='coords_3d')
def p2d_op_array_by_clust_9():A.array_by_clustering(n=10,r=1,return_type='coord_list_3d')
def p2d_op_array_by_clust_10():A.array_by_clustering(n=10,r=1,return_type='mulpoint2d')
"""
extime = timeit(stmt='p2d_op_array_by_clust_1()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_1: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_2()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_2: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_3()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_3: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_4()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_4: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_5()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_5: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_6()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_6: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_7()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_7: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_8()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_8: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_9()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_9: {ninst} ops: {extime} seconds")
extime = timeit(stmt='p2d_op_array_by_clust_10()', setup=su11_array_by_clustering, number=ninst)
print(f"p2d_op_array_by_clust_10: {ninst} ops: {extime} seconds")
print(50*'-')
# ---------------------------------
