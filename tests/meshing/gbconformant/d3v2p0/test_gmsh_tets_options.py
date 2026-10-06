import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces


def closed_rve():
    labels = np.ones((5, 5, 5), int)
    labels[1, 1, 1] = 2
    labels[3, 3, 3] = 2
    s = smooth_interfaces(labels, iterations=0)
    xyz = s.original_points[s.triangles]
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
    s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
    s.grain_pairs = np.sort(s.grain_pairs, axis=1)
    return close_rve_faces(s, labels, mesh_size=1.)


class TetOptimizationOptionTests(unittest.TestCase):
    def test_defaults_are_reported_unchanged(self):
        result = mesh_repaired_rve_gmsh(closed_rve(), mesh_size=1.2)
        self.assertFalse(result.report['netgen_all_volumes'])
        self.assertIsNone(result.report['optimize_threshold'])

    def test_netgen_on_all_volumes_and_threshold_keep_verification(self):
        closed = closed_rve()
        result = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, netgen_all_volumes=True, optimize_threshold=.5)
        self.assertTrue(result.report['netgen_all_volumes'])
        self.assertEqual(result.report['optimize_threshold'], .5)
        self.assertTrue(result.report['source_surface_conformity_verified'])
        self.assertAlmostEqual(result.report['total_volume'], 125.)
        self.assertEqual(set(result.grain_ids), {1, 2})
        np.testing.assert_array_equal(result.points[:len(closed.points)], closed.points)

    def test_dihedral_report_and_limit(self):
        closed = closed_rve()
        free = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)
        d = free.report['dihedral_angles']
        self.assertTrue(free.report['ready'])
        self.assertIsNone(free.report['minimum_dihedral_threshold'])
        self.assertGreater(d['minimum'], 0)
        strict = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0., minimum_dihedral=min(d['minimum'] + 1, 70.))
        self.assertFalse(strict.report['ready'])
        loose = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0., minimum_dihedral=d['minimum'] / 2)
        self.assertTrue(loose.report['ready'])

    def test_bad_options(self):
        closed = closed_rve()
        for kwargs in (dict(optimize_threshold=0.), dict(optimize_threshold=1.5),
                       dict(optimize_threshold=float('nan')), dict(netgen_all_volumes=1),
                       dict(minimum_dihedral=-1.), dict(minimum_dihedral=71.)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mesh_repaired_rve_gmsh(closed, **kwargs)


if __name__ == '__main__':
    unittest.main()


class RemeshOpeningTests(unittest.TestCase):
    def test_remesh_opening_check_runs_and_reports(self):
        from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles
        labels = np.ones((6, 6, 6), int)
        labels[1:4, 1:4, 1:4] = 2
        s = smooth_interfaces(labels, iterations=5)
        r = remesh_interfaces_gmsh(s, .6, check_intersections=True, minimum_remesh_opening=30.)
        self.assertEqual(r.report['minimum_remesh_opening'], 30.)
        src = len(small_facet_angles(s.points, s.triangles, 30.)[0])
        if src == 0:
            self.assertEqual(len(small_facet_angles(r.points, r.triangles, 30.)[0]), 0)

    def test_bad_remesh_opening(self):
        from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        labels = np.ones((4, 4, 4), int); labels[1:3, 1:3, 1:3] = 2
        with self.assertRaises(ValueError):
            remesh_interfaces_gmsh(smooth_interfaces(labels, iterations=1), .6, minimum_remesh_opening=0.)


class SaveTetsTests(unittest.TestCase):
    def test_save_round_trip(self):
        import tempfile, os, json
        import pyvista as pv
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import save_grain_tetrahedra
        tets = mesh_repaired_rve_gmsh(closed_rve(), mesh_size=1.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_grain_tetrahedra(tets, os.path.join(tmp, 't.vtu'))
            grid = pv.read(path)
            self.assertEqual(grid.n_cells, len(tets.tetrahedra))
            np.testing.assert_array_equal(grid.cell_data['grain_id'], tets.grain_ids)
            self.assertEqual(json.load(open(path.with_suffix('.json')))['tetrahedra'], len(tets.tetrahedra))
