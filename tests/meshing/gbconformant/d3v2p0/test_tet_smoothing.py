import unittest
import numpy as np
from scipy.spatial import Delaunay
from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles
from upxo.meshing.gbconformant.d3v2p0.tet_smoothing import (smooth_tet_dihedrals, smooth_grain_tetrahedra,
                                                            minimum_sicn_gmsh)


def oriented(points, tets):
    x = points[tets]
    v = np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    tets = tets.copy()
    tets[v < 0] = tets[v < 0][:, [0, 2, 1, 3]]
    return tets


def cube_star(centre):
    """Unit cube corners plus one interior node; 12 tets join it to the face triangles."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    points = np.vstack((corners, centre))
    hull = Delaunay(corners).convex_hull
    tets = np.column_stack((hull, np.full(len(hull), 8)))
    return points, oriented(points, tets)


def jittered_block(seed, n=4):
    rng = np.random.default_rng(seed)
    g = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float)
    inner = np.all((g > 0) & (g < n - 1), axis=1)
    g[inner] += rng.uniform(-.35, .35, size=(inner.sum(), 3))
    g[~inner] += rng.uniform(-1e-3, 1e-3, size=((~inner).sum(), 3)) * (g[~inner] % (n - 1) != 0)
    tets = Delaunay(g).simplices
    x = g[tets]
    v = np.abs(np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0])))
    tets = tets[v > 1e-9]
    return g, oriented(g, tets), inner


class TetSmoothingTests(unittest.TestCase):
    def test_displaced_interior_node_is_recentred(self):
        points, tets = cube_star([.5, .5, .97])
        before = dihedral_angles(points, tets).min()
        new, report = smooth_tet_dihedrals(points, tets, np.arange(9) == 8, max_passes=20)
        after = dihedral_angles(new, tets).min()
        self.assertGreater(after, before + 10)
        np.testing.assert_array_equal(new[:8], points[:8])
        self.assertEqual(report['moved_nodes'], 1)
        self.assertLess(report['after']['below_target'], report['before']['below_target'])

    def test_acceptance_rule_holds_for_every_tet(self):
        target, max_angle = 30., 150.
        moved = 0
        for seed in range(6):
            with self.subTest(seed=seed):
                points, tets, inner = jittered_block(seed)
                a0 = dihedral_angles(points, tets)
                new, report = smooth_tet_dihedrals(points, tets, inner, target=target, max_angle=max_angle)
                moved += report['moved_nodes']
                a1 = dihedral_angles(new, tets)
                mn0, mx0, mn1, mx1 = a0.min(1), a0.max(1), a1.min(1), a1.max(1)
                self.assertTrue(np.all(mn1 >= np.minimum(mn0, target) - 1e-6))
                self.assertTrue(np.all(mx1 <= np.maximum(mx0, max_angle) + 1e-6))
                self.assertLessEqual(report['after']['below_target'], report['before']['below_target'])
                self.assertLessEqual(report['after']['above_max_angle'], report['before']['above_max_angle'])
                np.testing.assert_array_equal(new[~inner], points[~inner])
                x = new[tets]
                v = np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
                self.assertTrue(np.all(v > 0))
                self.assertAlmostEqual(v.sum(), np.einsum('ij,ij->i', *(lambda y: (y[:, 1] - y[:, 0], np.cross(y[:, 2] - y[:, 0], y[:, 3] - y[:, 0])))(points[tets])).sum())

        self.assertGreater(moved, 0)

    def test_nothing_movable_or_disabled_is_identity(self):
        points, tets = cube_star([.5, .5, .97])
        for kwargs, mask in ((dict(), np.zeros(9, bool)), (dict(enabled=False), np.arange(9) == 8)):
            new, report = smooth_tet_dihedrals(points, tets, mask, **kwargs)
            np.testing.assert_array_equal(new, points)
            self.assertEqual(report['moved_nodes'], 0)

    def test_bad_arguments(self):
        points, tets = cube_star([.5, .5, .5])
        mask = np.arange(9) == 8
        for kwargs in (dict(target=0.), dict(target=71.), dict(max_angle=70.), dict(max_angle=180.),
                       dict(max_passes=-1), dict(node_iterations=0), dict(max_halvings=-1),
                       dict(initial_step=0.), dict(softness=0.), dict(enabled=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                smooth_tet_dihedrals(points, tets, mask, **kwargs)
        with self.assertRaises(ValueError):
            smooth_tet_dihedrals(points, tets, mask[:-1])
        with self.assertRaises(ValueError):
            smooth_tet_dihedrals(points, tets[:, [0, 2, 1, 3]], mask)    # inverted tets


def closed_rve(labels=None):
    from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
    from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
    if labels is None:
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


class GrainTetrahedraSmoothingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        cls.closed = closed_rve()
        cls.tets = mesh_repaired_rve_gmsh(cls.closed, mesh_size=1.2, minimum_quality=0.)

    def test_quality_metric_matches_the_mesher(self):
        np.testing.assert_allclose(minimum_sicn_gmsh(self.tets.points, self.tets.tetrahedra),
                                   self.tets.quality, rtol=1e-6, atol=1e-9)

    def test_smoothing_reverifies(self):
        ns = len(self.closed.points)
        result = smooth_grain_tetrahedra(self.tets, ns, target=40., max_angle=120.)
        np.testing.assert_array_equal(result.points[:ns], self.tets.points[:ns])
        self.assertIn('dihedral_smoothing', result.report)
        self.assertAlmostEqual(result.report['total_volume'], 125.)
        self.assertGreaterEqual(result.report['dihedral_angles']['minimum'],
                                min(self.tets.report['dihedral_angles']['minimum'], 40.) - 1e-6)
        np.testing.assert_allclose(result.quality, minimum_sicn_gmsh(result.points, result.tetrahedra))
        self.assertEqual(result.report['ready'], bool(result.quality.min() >= result.report['quality_threshold']))

    def test_disabled_keeps_mesh(self):
        result = smooth_grain_tetrahedra(self.tets, len(self.closed.points), enabled=False)
        np.testing.assert_array_equal(result.points, self.tets.points)
        self.assertFalse(result.report['dihedral_smoothing']['enabled'])

    def test_bad_surface_count(self):
        with self.assertRaises(ValueError):
            smooth_grain_tetrahedra(self.tets, len(self.tets.points) + 1)


if __name__ == '__main__':
    unittest.main()


class NeighbourFloorTests(unittest.TestCase):
    def test_floor_bounds_every_tet(self):
        target, floor, max_angle = 30., 20., 150.
        moved = 0
        for seed in range(6):
            with self.subTest(seed=seed):
                points, tets, inner = jittered_block(seed)
                a0 = dihedral_angles(points, tets)
                new, report = smooth_tet_dihedrals(points, tets, inner, target=target, max_angle=max_angle,
                                                   neighbour_floor=floor)
                a1 = dihedral_angles(new, tets)
                mn0, mn1 = a0.min(1), a1.min(1)
                self.assertTrue(np.all(mn1 >= np.where(mn0 >= target, floor, mn0) - 1e-6))
                self.assertTrue(np.all(a1.max(1) <= np.maximum(a0.max(1), max_angle) + 1e-6))
                self.assertGreaterEqual(mn1.min(), mn0.min() - 1e-9)
                moved += report['moved_nodes']
        self.assertGreater(moved, 0)

    def test_floor_validation(self):
        points, tets = cube_star([.5, .5, .5])
        for floor in (-1., 31.):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                smooth_tet_dihedrals(points, tets, np.arange(9) == 8, target=30., neighbour_floor=floor)


class SurfaceNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        cls.closed = closed_rve()
        cls.tets = mesh_repaired_rve_gmsh(cls.closed, mesh_size=1.2, minimum_quality=0.)

    def displaced(self):
        """Move one patch-interior surface node (here on an RVE cap) toward a neighbour, in its plane."""
        from upxo.meshing.gbconformant.d3v2p0.tet_smoothing import _SurfaceMoves
        from dataclasses import replace
        closed, tets = self.closed, self.tets
        moves = _SurfaceMoves(closed, tets.points, (), None, 0., 30., 30.)
        for n in range(len(closed.points)):
            mode = moves.mode(n)
            if mode is None or mode[0] != 'surface':
                continue
            ring = moves.topo.ring(n)
            x = closed.points[closed.triangles[ring]]
            normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
            if np.ptp(np.abs(normal / np.linalg.norm(normal, axis=1)[:, None]), axis=0).max() > 1e-9:
                continue
            nb = moves.topo.neighbours(n)
            target = closed.points[nb[0]]
            new = closed.points.copy()
            new[n] = new[n] + .6 * (target - new[n])
            points = tets.points.copy()
            points[n] = new[n]
            x = points[tets.tetrahedra]
            v = np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
            if np.all(v > 0):
                return n, replace(closed, points=new), replace(tets, points=points)
        self.skipTest('no flat interface node found')

    def test_surface_node_moves_back_in_its_plane(self):
        n, closed, tets = self.displaced()
        ns = len(closed.points)
        before = dihedral_angles(tets.points, tets.tetrahedra).min(1)
        result = smooth_grain_tetrahedra(tets, ns, surface=closed, max_deviation=0., target=40.)
        self.assertGreater(result.report['dihedral_smoothing']['moved_surface_nodes'], 0)
        after = dihedral_angles(result.points, result.tetrahedra).min(1)
        star = np.any(tets.tetrahedra == n, axis=1)
        self.assertGreater(after[star].min(), before[star].min())
        moved = np.flatnonzero(np.any(result.points[:ns] != tets.points[:ns], axis=1))
        self.assertIn(n, moved)
        self.assertAlmostEqual(result.report['total_volume'], 125.)

    def test_surface_invariants(self):
        from upxo.meshing.gbconformant.d3v2p0.tet_smoothing import surface_after_smoothing
        from upxo.meshing.gbconformant.d3v2p0.surface_angles import triangle_angles
        n, closed, tets = self.displaced()
        ns = len(closed.points)
        result = smooth_grain_tetrahedra(tets, ns, surface=closed, target=40., neighbour_floor=30.,
                                         triangle_target=35., triangle_floor=25., frozen_grain_ids=[2])
        moved = surface_after_smoothing(closed, result)
        # RVE-plane coordinates exact
        ext = np.asarray(closed.report['rve_dimensions'])
        for face in range(6):
            axis, side = divmod(face, 2)
            nodes = np.unique(moved.triangles[closed.exterior & (closed.rve_face == face)])
            np.testing.assert_array_equal(moved.points[nodes, axis], side * ext[axis])
        # frozen grain nodes fixed
        frozen_nodes = np.unique(closed.triangles[np.any(closed.grain_pairs == 2, axis=1)])
        np.testing.assert_array_equal(moved.points[frozen_nodes], closed.points[frozen_nodes])
        # triangle rule
        a0 = triangle_angles(closed.points, closed.triangles).min(1)
        a1 = triangle_angles(moved.points, moved.triangles).min(1)
        self.assertTrue(np.all(a1 >= np.where(a0 >= 35., 25., a0) - 1e-6))
        # grain volumes agree with the moved surface
        for gid, entry in result.report['per_grain'].items():
            self.assertAlmostEqual(entry['volume'], moved.report['enclosed_grain_volumes'][gid], places=8)

    def test_surface_nodes_must_not_be_in_the_movable_mask(self):
        ns = len(self.closed.points)
        with self.assertRaises(ValueError):
            smooth_tet_dihedrals(self.tets.points, self.tets.tetrahedra, np.ones(len(self.tets.points), bool),
                                 surface=self.closed)
        with self.assertRaises(ValueError):
            smooth_grain_tetrahedra(self.tets, ns - 1, surface=self.closed)


class CornerMoveTests(unittest.TestCase):
    def test_corners_move_only_when_enabled_and_keep_rve_planes(self):
        from upxo.meshing.gbconformant.d3v2p0.tet_smoothing import _SurfaceMoves
        labels = np.ones((5, 5, 5), int)
        labels[:2] = 2
        labels[2:, :2] = 3                                       # a junction line along z meets two RVE faces
        closed = closed_rve(labels)
        fixed = _SurfaceMoves(closed, closed.points, (), None, 0., 30., 30.)
        free = _SurfaceMoves(closed, closed.points, (), None, 0., 30., 30., corner_max_deviation=.2)
        corners = [n for n in range(len(closed.points)) if fixed.mode(n) is None and free.mode(n) is not None]
        self.assertGreater(len(corners), 0)
        for n in corners:
            mode = free.mode(n)
            self.assertEqual(mode[0], 'corner')
            self.assertLess(len(mode[1]), 3)                        # RVE corners never move
            y = free.projector(n, mode, closed.points)(closed.points[n][None] + .1)
            for axis, side in mode[1]:
                self.assertEqual(y[0, axis], side * 5.)
        frozen = _SurfaceMoves(closed, closed.points, (2,), None, 0., 30., 30., corner_max_deviation=.2)
        frozen_nodes = np.unique(closed.triangles[np.any(closed.grain_pairs == 2, axis=1)])
        self.assertTrue(all(frozen.mode(int(n)) is None for n in frozen_nodes))

    def test_corner_deviation_validation(self):
        points, tets = cube_star([.5, .5, .5])
        closed = closed_rve()
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        t = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0.)
        with self.assertRaises(ValueError):
            smooth_tet_dihedrals(t.points, t.tetrahedra, np.arange(len(t.points)) >= len(closed.points),
                                 surface=closed, corner_max_deviation=-1.)
