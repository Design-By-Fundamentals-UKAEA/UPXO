import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0 import gmsh_tets as ref
from upxo.meshing.gbconformant.d3v2p1 import gmsh_tets as fast
from upxo.meshing.gbconformant.d3v2p1.parallel import resolve_workers, balanced_chunks


def closed_rve():
    labels = np.ones((5, 5, 5), int)
    labels[1, 1, 1] = 2                       # grain 2 in two pieces, both cavities in grain 1
    labels[3, 3, 3] = 2
    s = smooth_interfaces(labels, iterations=0)
    xyz = s.original_points[s.triangles]
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
    s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
    s.grain_pairs = np.sort(s.grain_pairs, axis=1)
    return close_rve_faces(s, labels, mesh_size=1.)


class ParallelHelperTests(unittest.TestCase):
    def test_workers_and_chunks(self):
        self.assertGreaterEqual(resolve_workers(None), 1)
        self.assertEqual(resolve_workers(1), 1)
        with self.assertRaises(ValueError):
            resolve_workers(-1)
        chunks = balanced_chunks([5, 1, 4, 2, 3], 2)
        self.assertEqual(sorted(i for c in chunks for i in c), [0, 1, 2, 3, 4])
        self.assertEqual(balanced_chunks([5, 1, 4, 2, 3], 2), chunks)        # deterministic


class ParallelTetMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closed = closed_rve()
        cls.ref = ref.mesh_repaired_rve_gmsh(cls.closed, mesh_size=1.2, minimum_quality=0.)
        cls.serial = fast.mesh_repaired_rve_gmsh(cls.closed, mesh_size=1.2, minimum_quality=0., n_workers=1)
        cls.parallel = fast.mesh_repaired_rve_gmsh(cls.closed, mesh_size=1.2, minimum_quality=0., n_workers=3)

    def test_same_contract_as_d3v2p0(self):
        for result in (self.serial, self.parallel):
            np.testing.assert_array_equal(result.points[:len(self.closed.points)], self.closed.points)
            self.assertTrue(result.report['source_surface_conformity_verified'])
            self.assertAlmostEqual(result.report['total_volume'], self.ref.report['total_volume'])
            self.assertEqual(set(result.grain_ids), set(self.ref.grain_ids))
            for gid, entry in self.ref.report['per_grain'].items():
                self.assertAlmostEqual(result.report['per_grain'][gid]['volume'], entry['volume'])

    def test_worker_count_does_not_change_the_mesh(self):
        np.testing.assert_array_equal(self.serial.tetrahedra, self.parallel.tetrahedra)
        np.testing.assert_array_equal(self.serial.points, self.parallel.points)
        np.testing.assert_array_equal(self.serial.grain_ids, self.parallel.grain_ids)

    def test_save_is_reexported(self):
        self.assertIs(fast.save_grain_tetrahedra, ref.save_grain_tetrahedra)


if __name__ == '__main__':
    unittest.main()
