import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles
from upxo.meshing.gbconformant.d3v2p0.tet_swaps import swap_tets, swap_grain_tetrahedra, _orient


def bipyramid(h):
    """Equilateral triangle a, b, c in z = 0 with apexes d (z = +h) and e (z = -h)."""
    tri = np.array([[0, 0, 0], [1, 0, 0], [.5, np.sqrt(3) / 2, 0]], float)
    centre = tri.mean(0)
    return np.vstack((tri, centre + [0, 0, h], centre - [0, 0, h]))


def hull(points):
    a, b, c, d, e = range(5)
    return np.array([[a, b, d], [b, c, d], [c, a, d], [a, b, e], [b, c, e], [c, a, e]])


def volume(points, tets):
    x = points[tets]
    return np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0])).sum() / 6


class SwapTests(unittest.TestCase):
    def test_3_2_swap_replaces_a_poor_edge_ring(self):
        p = bipyramid(2.)                                      # ring around d-e has a 164 deg angle; two tets do not
        three = np.array([_orient(p, [0, 1, 3, 4]), _orient(p, [1, 2, 3, 4]), _orient(p, [2, 0, 3, 4])])
        t, g, report = swap_tets(p, three, [7, 7, 7], hull(p))
        self.assertEqual(len(t), 2)
        self.assertEqual(report['swaps_3_2'], 1)
        self.assertTrue(np.all(g == 7))
        self.assertGreater(dihedral_angles(p, t).min(), dihedral_angles(p, three).min())
        self.assertAlmostEqual(volume(p, t), volume(p, three))

    def test_2_3_swap_replaces_flat_tets(self):
        p = bipyramid(.08)                                     # two flat tets; three around d-e are better
        two = np.array([_orient(p, [0, 1, 2, 3]), _orient(p, [0, 1, 2, 4])])
        t, g, report = swap_tets(p, two, [3, 3], hull(p))
        self.assertEqual(report['swaps_2_3'], 1)
        self.assertEqual(len(t), 3)
        self.assertGreater(dihedral_angles(p, t).min(), dihedral_angles(p, two).min())
        self.assertAlmostEqual(volume(p, t), volume(p, two))

    def test_surface_faces_edges_and_grain_boundaries_are_kept(self):
        p = bipyramid(.08)
        two = np.array([_orient(p, [0, 1, 2, 3]), _orient(p, [0, 1, 2, 4])])
        surface = np.vstack((hull(p), [[0, 1, 2]]))            # the shared face is a grain surface
        t, _, report = swap_tets(p, two, [3, 3], surface)
        np.testing.assert_array_equal(t, two)
        t, _, report = swap_tets(p, two, [3, 4], hull(p))      # different grains
        np.testing.assert_array_equal(t, two)
        p = bipyramid(.8)
        three = np.array([_orient(p, [0, 1, 3, 4]), _orient(p, [1, 2, 3, 4]), _orient(p, [2, 0, 3, 4])])
        t, _, _ = swap_tets(p, three, [7, 7, 7], np.vstack((hull(p), [[3, 4, 0]])))   # edge d-e on a surface
        np.testing.assert_array_equal(t, three)

    def test_no_swap_when_nothing_improves(self):
        p = bipyramid(.8)
        two = np.array([_orient(p, [0, 1, 2, 3]), _orient(p, [0, 1, 2, 4])])
        t, _, report = swap_tets(p, two, [1, 1], hull(p))
        np.testing.assert_array_equal(t, two)
        self.assertEqual(report['swaps_2_3'] + report['swaps_3_2'], 0)

    def test_disabled_and_bad_arguments(self):
        p = bipyramid(.08)
        two = np.array([_orient(p, [0, 1, 2, 3]), _orient(p, [0, 1, 2, 4])])
        t, _, report = swap_tets(p, two, [3, 3], hull(p), enabled=False)
        np.testing.assert_array_equal(t, two)
        for kwargs in (dict(target=0.), dict(max_angle=70.), dict(max_passes=-1), dict(enabled=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                swap_tets(p, two, [3, 3], hull(p), **kwargs)
        with self.assertRaises(ValueError):
            swap_tets(p, two, [3], hull(p))


class GrainTetrahedraSwapTests(unittest.TestCase):
    def test_swaps_reverify_a_real_mesh(self):
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        labels = np.ones((5, 5, 5), int)
        labels[1, 1, 1] = 2
        labels[3, 3, 3] = 2
        s = smooth_interfaces(labels, iterations=0)
        xyz = s.original_points[s.triangles]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
        s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
        s.grain_pairs = np.sort(s.grain_pairs, axis=1)
        closed = close_rve_faces(s, labels, mesh_size=1.)
        tets = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)
        result = swap_grain_tetrahedra(tets, closed, target=50., max_angle=110.)
        self.assertIn('tet_swaps', result.report)
        self.assertAlmostEqual(result.report['total_volume'], 125.)
        self.assertEqual(set(result.grain_ids), {1, 2})
        self.assertGreaterEqual(dihedral_angles(result.points, result.tetrahedra).min(),
                                dihedral_angles(tets.points, tets.tetrahedra).min() - 1e-9)
        off = swap_grain_tetrahedra(tets, closed, enabled=False)
        np.testing.assert_array_equal(off.tetrahedra, tets.tetrahedra)


if __name__ == '__main__':
    unittest.main()


class NodeInsertionTests(unittest.TestCase):
    def cube(self, centre):
        from scipy.spatial import Delaunay
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
        p = np.vstack((corners, centre))
        hull_tris = Delaunay(corners).convex_hull
        tets = np.array([_orient(p, list(f) + [8]) for f in hull_tris])
        return p, tets, hull_tris

    def boundary(self, t):
        f = np.sort(t[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
        u, c = np.unique(f, axis=0, return_counts=True)
        return u[c == 1]

    def test_insertion_improves_a_bad_star_and_keeps_the_surface(self):
        from upxo.meshing.gbconformant.d3v2p0.tet_swaps import insert_nodes
        p, t, hull_tris = self.cube([.5, .5, .97])
        before = dihedral_angles(p, t).min()
        P, T, G, report = insert_nodes(p, t, [4] * len(t), hull_tris, cavity_rings=(1, 2, 3))
        self.assertGreater(report['inserted_nodes'], 0)
        self.assertGreater(dihedral_angles(P, T).min(), before)
        np.testing.assert_array_equal(P[:8], p[:8])                 # surface nodes keep their numbers
        self.assertEqual(len(P), 9)                                  # the displaced node was replaced
        self.assertAlmostEqual(volume(P, T), 1.)
        np.testing.assert_array_equal(self.boundary(T), np.unique(np.sort(hull_tris, axis=1), axis=0))
        self.assertTrue(np.all(G == 4))

    def test_nothing_to_do_and_bad_arguments(self):
        from upxo.meshing.gbconformant.d3v2p0.tet_swaps import insert_nodes
        p, t, hull_tris = self.cube([.5, .5, .5])
        P, T, _, report = insert_nodes(p, t, [1] * len(t), hull_tris, target=20.)
        np.testing.assert_array_equal(T, t)
        self.assertEqual(report['inserted_nodes'], 0)
        for kwargs in (dict(target=0.), dict(cavity_rings=()), dict(cavity_rings=(-1,)), dict(max_passes=-1),
                       dict(search_iterations=0), dict(samples=0), dict(enabled=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                insert_nodes(p, t, [1] * len(t), hull_tris, **kwargs)


class GrainTetrahedraInsertionTests(unittest.TestCase):
    def test_insertion_reverifies_a_real_mesh(self):
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        from upxo.meshing.gbconformant.d3v2p0.tet_swaps import insert_grain_tetrahedra
        labels = np.ones((5, 5, 5), int)
        labels[1, 1, 1] = 2
        labels[3, 3, 3] = 2
        s = smooth_interfaces(labels, iterations=0)
        xyz = s.original_points[s.triangles]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
        s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
        s.grain_pairs = np.sort(s.grain_pairs, axis=1)
        closed = close_rve_faces(s, labels, mesh_size=1.)
        tets = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)
        result = insert_grain_tetrahedra(tets, closed, target=50., max_angle=110., max_passes=1)
        self.assertIn('node_insertion', result.report)
        np.testing.assert_array_equal(result.points[:len(closed.points)], closed.points)
        self.assertAlmostEqual(sum(v['volume'] for v in result.report['per_grain'].values()), 125.)
        self.assertGreaterEqual(dihedral_angles(result.points, result.tetrahedra).min(),
                                dihedral_angles(tets.points, tets.tetrahedra).min() - 1e-9)
