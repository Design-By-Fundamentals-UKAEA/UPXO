import unittest
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE
from upxo.meshing.gbconformant.d3v2p0.surface_quality import improve_surface_quality, triangle_quality


class SurfaceQualityTests(unittest.TestCase):
    def source(self):
        points = np.array([[0., 0, 0], [2., 0, 0], [1., .001, 0], [0., -1, 0]])
        return ClosedRVE(points, np.array([[0, 1, 2], [1, 0, 3]]),
            np.array([[1, 2], [1, 2]]), np.zeros(2, bool), np.full(2, -1), {})

    def test_sliver_flip_preserves_coordinates_and_patch_boundary(self):
        source = self.source()
        result = improve_surface_quality(source)
        self.assertGreater(triangle_quality(result.points, result.triangles).min(), .4)
        np.testing.assert_array_equal(result.points, source.points)
        self.assertEqual(result.report['surface_quality_repair']['accepted_flips'], 1)
        def boundary(f):
            e, n = np.unique(np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1), axis=0, return_counts=True)
            return e[n == 1]
        np.testing.assert_array_equal(boundary(source.triangles), boundary(result.triangles))

    def test_grain_patch_boundary_does_not_flip(self):
        source = self.source(); source.grain_pairs[1] = [1, 3]
        np.testing.assert_array_equal(improve_surface_quality(source).triangles, source.triangles)

    def test_intersection_rolls_back_proposal(self):
        source = self.source()
        with patch('upxo.meshing.gbconformant.d3v2p0.surface_quality.find_surface_intersections', return_value=np.array([[0, 1]])):
            result = improve_surface_quality(source)
        np.testing.assert_array_equal(result.triangles, source.triangles)
        self.assertEqual(result.report['surface_quality_repair']['history'][0]['rejected_geometry'], 1)

    def test_disabled(self):
        source = self.source()
        np.testing.assert_array_equal(improve_surface_quality(source, enabled=False).triangles, source.triangles)

    def test_nearly_closed_wedge_rejects_flip(self):
        source=self.source()
        with patch('upxo.meshing.gbconformant.d3v2p0.facet_angles.small_facet_angles',return_value=(np.array([[0,1]]),np.array([.01]))):
            result=improve_surface_quality(source)
        np.testing.assert_array_equal(result.triangles,source.triangles)
        self.assertEqual(result.report['surface_quality_repair']['accepted_flips'],0)

    def test_closed_shell_volume_report_remains_current(self):
        source = self.source()
        points = np.vstack((source.points, [.5, -.2, -1.]))
        faces = np.vstack((source.triangles, [[2,1,4],[0,2,4],[3,0,4],[1,3,4]]))
        shell = ClosedRVE(points, faces, np.ones((6,2), int), np.ones(6, bool),
                          np.array([0,0,1,2,3,4]), {'enclosed_grain_volumes': {'1': -999}})
        result = improve_surface_quality(shell)
        xyz = result.points[result.triangles]
        actual = np.einsum('ij,ij->i', xyz[:,0], np.cross(xyz[:,1],xyz[:,2])).sum()/6
        self.assertGreater(result.report['surface_quality_repair']['accepted_flips'], 0)
        self.assertAlmostEqual(result.report['enclosed_grain_volumes']['1'], actual)
