"""d3v2p0's validation and intersection tests run against d3v2p1, plus
worker-count independence on inputs large enough to use worker processes."""
import unittest
import numpy as np
from tests.meshing.gbconformant.d3v2p0 import test_tet_validation as base_validation
from tests.meshing.gbconformant.d3v2p0 import test_surface_intersections as base_intersections
from upxo.meshing.gbconformant.d3v2p0.tet_validation import validate_tet_surfaces as reference_validate
from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections as reference_find
from upxo.meshing.gbconformant.d3v2p1.tet_validation import validate_tet_surfaces
from upxo.meshing.gbconformant.d3v2p1.surface_intersections import find_surface_intersections


class _Swap:
    module, name, replacement = None, None, None

    def setUp(self):
        self._saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.replacement)

    def tearDown(self):
        setattr(self.module, self.name, self._saved)


class D3v2p1ValidationTests(_Swap, base_validation.ValidationTests):
    module, name, replacement = base_validation, 'validate_tet_surfaces', staticmethod(validate_tet_surfaces)


class D3v2p1IntersectionTests(_Swap, base_intersections.IntersectionTests):
    module, name, replacement = base_intersections, 'find_surface_intersections', staticmethod(find_surface_intersections)


def closed_block_rve(n=8, block=2):
    from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
    from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
    i, j, k = np.indices((n, n, n)) // block
    m = n // block
    labels = 1 + i * m * m + j * m + k
    s = smooth_interfaces(labels, iterations=0)
    xyz = s.original_points[s.triangles]
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    s.triangles = s.triangles.copy()
    s.grain_pairs = s.grain_pairs.copy()
    flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
    s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
    s.grain_pairs = np.sort(s.grain_pairs, axis=1)
    return close_rve_faces(s, labels, mesh_size=1.)


class WorkerCountTests(unittest.TestCase):
    def test_many_grain_report_matches_reference(self):
        rve = closed_block_rve()
        self.assertEqual(len(np.unique(rve.grain_pairs)), 64)
        ref = reference_validate(rve, check_intersections=True)
        for workers in (1, 3):
            with self.subTest(workers=workers):
                out = validate_tet_surfaces(rve, check_intersections=True, n_workers=workers)
                self.assertEqual(out.pop('backend')['used'], 'numpy' if workers == 1 else 'parallel')
                self.assertEqual(out, ref)
        self.assertEqual(ref['status'], 'PRELIMINARY_CHECKS_PASSED')

    def test_large_triangle_soup_pairs_match_reference(self):
        rng = np.random.default_rng(3)
        centres = rng.uniform(0, 30, size=(9000, 3))
        points = (centres[:, None] + rng.normal(0, .4, size=(9000, 3, 3))).reshape(-1, 3)
        triangles = np.arange(len(points)).reshape(-1, 3)
        ref = reference_find(points, triangles)
        self.assertGreater(len(ref), 0)
        for workers in (1, 3):
            with self.subTest(workers=workers):
                np.testing.assert_array_equal(find_surface_intersections(points, triangles, n_workers=workers), ref)
        ids = np.arange(0, 9000, 7)
        np.testing.assert_array_equal(find_surface_intersections(points, triangles, triangle_ids=ids, n_workers=3),
                                      reference_find(points, triangles, triangle_ids=ids))


if __name__ == '__main__':
    unittest.main()
