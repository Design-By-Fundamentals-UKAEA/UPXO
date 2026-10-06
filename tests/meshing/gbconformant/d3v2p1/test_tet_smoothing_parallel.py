import unittest
import numpy as np
from scipy.spatial import Delaunay
from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles
from upxo.meshing.gbconformant.d3v2p1 import backend, tet_smoothing as fast


def jittered_block(seed, n=7):
    rng = np.random.default_rng(seed)
    g = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float)
    inner = np.all((g > 0) & (g < n - 1), axis=1)
    g[inner] += rng.uniform(-.35, .35, size=(inner.sum(), 3))
    tets = Delaunay(g).simplices
    x = g[tets]
    v = np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    tets = tets[np.abs(v) > 1e-9]
    v = v[np.abs(v) > 1e-9]
    tets[v < 0] = tets[v < 0][:, [0, 2, 1, 3]]
    return g, tets, inner


class BlockSmoothingTests(unittest.TestCase):
    def test_rule_holds_and_worker_count_is_irrelevant(self):
        target, floor, max_angle = 30., 20., 150.
        for seed in range(3):
            with self.subTest(seed=seed):
                p, t, inner = jittered_block(seed)
                a0 = dihedral_angles(p, t)
                one, rep1 = fast.smooth_tet_dihedrals(p, t, inner, target=target, max_angle=max_angle,
                                                      neighbour_floor=floor, n_workers=1, block_size=4.)
                many, rep3 = fast.smooth_tet_dihedrals(p, t, inner, target=target, max_angle=max_angle,
                                                       neighbour_floor=floor, n_workers=3, block_size=4.)
                np.testing.assert_array_equal(one, many)
                a1 = dihedral_angles(one, t)
                mn0, mn1 = a0.min(1), a1.min(1)
                self.assertTrue(np.all(mn1 >= np.where(mn0 >= target, floor, np.maximum(mn0, floor)) - 1e-6)
                                or np.all(mn1 >= np.where(mn0 >= target, floor, mn0) - 1e-6))
                self.assertTrue(np.all(a1.max(1) <= np.maximum(a0.max(1), max_angle) + 1e-6))
                np.testing.assert_array_equal(one[~inner], p[~inner])
                self.assertGreater(rep1['moved_nodes'], 0)
                self.assertGreaterEqual(mn1.min(), mn0.min() - 1e-6)

    def test_bad_arguments(self):
        p, t, inner = jittered_block(0, n=4)
        for kwargs in (dict(block_size=0.), dict(minimum_gain=1.), dict(n_workers=-1), dict(target=0.)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fast.smooth_tet_dihedrals(p, t, inner, **kwargs)


class GrainTetrahedraBlockTests(unittest.TestCase):
    def test_surface_moves_reverify(self):
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        labels = np.ones((6, 6, 6), int)
        labels[:3] = 2
        labels[3:, :3] = 3
        s = smooth_interfaces(labels, iterations=0)
        xyz = s.original_points[s.triangles]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
        s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
        s.grain_pairs = np.sort(s.grain_pairs, axis=1)
        closed = close_rve_faces(s, labels, mesh_size=1.)
        tets = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)
        ns = len(closed.points)
        kw = dict(surface=closed, target=40., neighbour_floor=30., triangle_target=35., triangle_floor=25.,
                  wedge_limit=30., block_size=2.)
        one = fast.smooth_grain_tetrahedra(tets, ns, n_workers=1, **kw)
        many = fast.smooth_grain_tetrahedra(tets, ns, n_workers=3, **kw)
        np.testing.assert_array_equal(one.points, many.points)
        self.assertAlmostEqual(one.report['total_volume'], 216.)
        ext = closed.report['rve_dimensions']
        for face in range(6):
            axis, side = divmod(face, 2)
            nodes = np.unique(closed.triangles[closed.exterior & (closed.rve_face == face)])
            np.testing.assert_array_equal(one.points[nodes, axis], side * ext[axis])
        self.assertGreaterEqual(dihedral_angles(one.points, one.tetrahedra).min(),
                                min(dihedral_angles(tets.points, tets.tetrahedra).min(), 30.) - 1e-6)


def _block_rve_tets():
    from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
    from tests.meshing.gbconformant.d3v2p1.test_validation_parallel import closed_block_rve
    closed = closed_block_rve()
    return closed, mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)


@unittest.skipUnless(backend.numba_available(), 'numba not available')
class KernelSmoothingTests(unittest.TestCase):
    """numba kernels against the numpy path: same moved nodes, positions equal
    to rounding, and the rule guarantees."""

    def assert_same(self, a, b, tol=1e-9):
        moved_a = np.flatnonzero(np.any(a[0] != a[1], axis=1))
        moved_b = np.flatnonzero(np.any(b[0] != b[1], axis=1))
        np.testing.assert_array_equal(moved_a, moved_b)
        np.testing.assert_allclose(a[0], b[0], rtol=0, atol=tol)

    def test_interior_nodes(self):
        p, t, inner = jittered_block(2, n=9)
        kw = dict(target=30., max_angle=150., neighbour_floor=20., block_size=3.)
        ref, rep_ref = fast.smooth_tet_dihedrals(p, t, inner, backend='numpy', **kw)
        got, rep = fast.smooth_tet_dihedrals(p, t, inner, n_workers=2, **kw)
        self.assertEqual(rep['backend']['kernels'], 'numba')
        self.assertEqual(rep_ref['backend']['kernels'], 'numpy')
        self.assertGreater(rep['moved_nodes'], 0)
        self.assert_same((got, p), (ref, p))
        a0, a1 = dihedral_angles(p, t), dihedral_angles(got, t)
        mn0, mn1 = a0.min(1), a1.min(1)
        self.assertTrue(np.all(mn1 >= np.where(mn0 >= 30., 20., np.maximum(mn0, 20.)) - 1e-6)
                        or np.all(mn1 >= np.where(mn0 >= 30., 20., mn0) - 1e-6))

    def test_surface_nodes(self):
        closed, tets = _block_rve_tets()
        ns = len(closed.points)
        kw = dict(surface=closed, target=40., neighbour_floor=30., triangle_target=35., triangle_floor=25.,
                  wedge_limit=30., minimum_facet_angle=15., block_size=2.)
        ref = fast.smooth_grain_tetrahedra(tets, ns, backend='numpy', **kw)
        got = fast.smooth_grain_tetrahedra(tets, ns, n_workers=2, **kw)
        self.assertEqual(got.report['dihedral_smoothing']['backend']['kernels'], 'numba')
        self.assertGreater(got.report['dihedral_smoothing']['moved_surface_nodes'], 0)
        self.assert_same((got.points, tets.points), (ref.points, tets.points))
        self.assertAlmostEqual(got.report['total_volume'], 512.)


if __name__ == '__main__':
    unittest.main()
