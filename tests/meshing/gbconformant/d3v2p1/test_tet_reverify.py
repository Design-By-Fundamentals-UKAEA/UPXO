"""d3v2p1 tet_swaps._reverify and insert_grain_tetrahedra: d3v2p0's result, error for error."""
import copy
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0 import tet_swaps as ref
from upxo.meshing.gbconformant.d3v2p1 import tet_swaps as fast

_CACHE = {}


def real_mesh():
    """Two grains in a 5 x 5 x 5 block, meshed with Gmsh (built once)."""
    if 'mesh' not in _CACHE:
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
        _CACHE['mesh'] = (tets, closed)
    return _CACHE['mesh']


def assertTetsEqual(testcase, a, b):
    for name in ('points', 'tetrahedra', 'grain_ids', 'quality'):
        x, y = getattr(a, name), getattr(b, name)
        testcase.assertEqual((x.shape, x.dtype), (y.shape, y.dtype), name)
        np.testing.assert_array_equal(x, y, err_msg=name)
    testcase.assertEqual(a.report, b.report)


class ReverifyTests(unittest.TestCase):
    def setUp(self):
        self.tets, self.closed = real_mesh()
        self.new_t, self.new_g, _ = ref.swap_tets(self.tets.points, self.tets.tetrahedra, self.tets.grain_ids,
                                                  self.closed.triangles, target=50., max_angle=110.)

    def call(self, module, tets=None, new_t=None, new_g=None):
        tets = self.tets if tets is None else tets
        return module._reverify(tets, tets.points, self.new_t if new_t is None else new_t,
                                self.new_g if new_g is None else new_g, self.closed, copy.deepcopy(tets.report),
                                'swaps')

    def test_same_result_after_swaps_and_on_the_unchanged_mesh(self):
        assertTetsEqual(self, self.call(fast), self.call(ref))
        unchanged = (self.tets.tetrahedra, self.tets.grain_ids)
        assertTetsEqual(self, self.call(fast, new_t=unchanged[0], new_g=unchanged[1]),
                        self.call(ref, new_t=unchanged[0], new_g=unchanged[1]))

    def assertSameError(self, **kwargs):
        with self.assertRaises(RuntimeError) as want:
            self.call(ref, **kwargs)
        with self.assertRaises(RuntimeError) as got:
            self.call(fast, **kwargs)
        self.assertEqual(str(got.exception), str(want.exception))
        return str(got.exception)

    def test_every_failure_raises_the_same_error(self):
        t, g = self.new_t.copy(), self.new_g.copy()
        flipped = t.copy()
        flipped[0] = flipped[0][[1, 0, 2, 3]]                                       # negative volume
        self.assertIn('nonpositive', self.assertSameError(new_t=flipped))
        duplicated = np.vstack((t, t[:1]))                                           # a face on three tets
        self.assertIn('Nonmanifold', self.assertSameError(new_t=duplicated, new_g=np.append(g, g[:1])))
        dropped = t[1:]                                                              # a hole: boundary differs
        self.assertIn('boundary differs', self.assertSameError(new_t=dropped, new_g=g[1:]))
        wrong = copy.deepcopy(self.tets)
        key = next(iter(wrong.report['per_grain']))
        wrong.report['per_grain'][key]['volume'] *= 1.1                             # grain volume changed
        with self.assertRaises(RuntimeError) as want:
            self.call(ref, tets=wrong)
        with self.assertRaises(RuntimeError) as got:
            self.call(fast, tets=wrong)
        self.assertEqual(str(got.exception), str(want.exception))
        self.assertIn('changed', str(got.exception))
        missing = copy.deepcopy(self.tets)
        missing.report['per_grain']['999'] = dict(volume=1., tetrahedra=1, minimum_quality=.5)   # a grain with no tets
        with self.assertRaises(RuntimeError) as want:
            self.call(ref, tets=missing)
        with self.assertRaises(RuntimeError) as got:
            self.call(fast, tets=missing)
        self.assertEqual(str(got.exception), str(want.exception))


class SwapAndInsertionTests(unittest.TestCase):
    def test_swap_grain_tetrahedra_matches_d3v2p0(self):
        tets, closed = real_mesh()
        want = ref.swap_grain_tetrahedra(tets, closed, target=50., max_angle=110.)
        for tier in ('numpy', 'auto'):
            with self.subTest(tier=tier):
                got = fast.swap_grain_tetrahedra(tets, closed, target=50., max_angle=110., backend=tier)
                for name in ('points', 'tetrahedra', 'grain_ids', 'quality'):
                    np.testing.assert_array_equal(getattr(got, name), getattr(want, name), err_msg=name)
                got_report = {k: v for k, v in got.report.items() if k != 'tet_swaps'}
                want_report = {k: v for k, v in want.report.items() if k != 'tet_swaps'}
                self.assertEqual(got_report, want_report)

    def test_insert_grain_tetrahedra_matches_d3v2p0(self):
        tets, closed = real_mesh()
        kw = dict(target=22., max_angle=150., max_passes=1)                         # a few nodes: quick, and verified
        want = ref.insert_grain_tetrahedra(tets, closed, **kw)
        got = fast.insert_grain_tetrahedra(tets, closed, **kw)
        self.assertGreater(got.report['node_insertion']['inserted_nodes'], 0)       # _reverify ran
        assertTetsEqual(self, got, want)
        off_want = ref.insert_grain_tetrahedra(tets, closed, enabled=False)
        off_got = fast.insert_grain_tetrahedra(tets, closed, enabled=False)
        assertTetsEqual(self, off_got, off_want)


if __name__ == '__main__':
    unittest.main()
