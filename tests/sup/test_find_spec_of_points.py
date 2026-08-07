import unittest
from upxo._sup.validation_values import find_spec_of_points
from upxo.geoEntities.point2d import Point2d as p2d
from upxo.geoEntities.point2d import p2d_leanest
from upxo.geoEntities.point3d import Point3d as p3d
from upxo.geoEntities.point3d import p3d_leanest


class Test_find_spec_of_points(unittest.TestCase):

    def test_point2d_1(self):
        self.assertEqual(find_spec_of_points(p2d_leanest(1, 2)),
                         'p2d_leanest')

    def test_point2d_2(self):
        self.assertEqual(find_spec_of_points([p2d_leanest(1, 2),
                                              p2d_leanest(1, 2)]),
                         '[p2d_leanest]')

    def test_point2d_3(self):
        self.assertEqual(find_spec_of_points(p2d(1, 2)), 'Point2d')

    def test_point2d_4(self):
        self.assertEqual(find_spec_of_points([p2d(1, 2), p2d(3, 3)]),
                         '[Point2d]')

    def test_point2d_5(self):
        self.assertEqual(find_spec_of_points([1, 2]),
                         'type-[1,2]')

    def test_point2d_6(self):
        self.assertEqual(find_spec_of_points([[1, 2]]),
                         'type-[[1,2]]')

    def test_point2d_7(self):
        self.assertEqual(find_spec_of_points([[1,2], [3,4], [5,6]]),
                         'type-[[1,2],[3,4],[5,6]]')

    def test_point2d_8(self):
        self.assertEqual(find_spec_of_points([[2,1,1,2], [3,4,5,6]]),
                         'type-[[1,2,3,4],[5,6,7,8]]')

    def test_point3d_1(self):
        self.assertEqual(find_spec_of_points([1, 2, 3]),
                         'type-[1,2,3]')

    def test_point3d_2(self):
        self.assertEqual(find_spec_of_points(p3d(1, 2, 1)),
                         'Point3d')

    def test_point3d_3(self):
        self.assertEqual(find_spec_of_points(p3d(1, 2)),
                         'Point3d')

    def test_point3d_4(self):
        self.assertEqual(find_spec_of_points([p3d(1, 2), p3d(3, 3)]),
                         '[Point3d]')

    def test_point3d_5(self):
        self.assertEqual(find_spec_of_points(p3d_leanest(1,2,3)),
                         'p3d_leanest')

    def test_point3d_6(self):
        self.assertEqual(find_spec_of_points([p3d_leanest(1,2,3),
                                              p3d_leanest(1,2,3)]),
                         '[p3d_leanest]')

    def test_point3d_7(self):
        self.assertEqual(find_spec_of_points([1,2,3]),
                         'type-[1,2,3]')

    def test_point3d_8(self):
        self.assertEqual(find_spec_of_points([[1,2,3]]),
                         'type-[[1,2,3]]')

    def test_point3d_9(self):
        self.assertEqual(find_spec_of_points([[1,2,3], [4,5,6], [7,8,9]]),
                         'type-[[1,2,3],[4,5,6],[7,8,9]]')

    def test_point3d_10(self):
        self.assertEqual(find_spec_of_points([[1,2,3,4],
                                              [1,2,3,4],
                                              [1,2,3,4]]),
                         'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]')

if __name__ == '__main__':
    unittest.main()
