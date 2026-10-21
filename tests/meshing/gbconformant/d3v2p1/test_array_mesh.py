"""d3v2p1 tet_swaps._ArrayMesh answers as d3v2p0's _Mesh does, operation for operation."""
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.tet_swaps import _Mesh
from upxo.meshing.gbconformant.d3v2p1.tet_swaps import _ArrayMesh


def random_tets(seed, n_nodes=40, n_tets=300, dtype=np.int64):
    rng = np.random.default_rng(seed)
    t = np.array([rng.choice(n_nodes, 4, replace=False) for _ in range(n_tets)]).astype(dtype)
    return t, rng.integers(1, 5, size=n_tets), n_nodes


class ArrayMeshTests(unittest.TestCase):
    def assertSameState(self, a, b):
        self.assertEqual(len(a.t), len(b.t))
        for k in range(len(a.t)):
            self.assertEqual(a.t[k], b.t[k])
            self.assertIs(type(a.t[k]), list)
            self.assertTrue(all(type(v) is int for v in a.t[k]))
        self.assertEqual(a.g, b.g)
        self.assertEqual(a.alive, b.alive)
        np.testing.assert_array_equal(a.order, b.order)
        np.testing.assert_array_equal(a.offsets, b.offsets)

    def test_operations_match(self):
        for dtype in (np.int64, np.int32):
            t, g, n = random_tets(0, dtype=dtype)
            old, new = _Mesh(t, g, n + 10), _ArrayMesh(t, g, n + 10)
            self.assertSameState(old, new)
            rng = np.random.default_rng(1)
            for step in range(120):
                action = rng.integers(0, 3)
                if action == 0:                                          # kill a tet
                    k = int(rng.integers(0, len(old.t)))
                    old.alive[k] = new.alive[k] = False
                elif action == 1:                                        # add a tet
                    tet = rng.choice(n, 4, replace=False)
                    grain = int(rng.integers(1, 5))
                    self.assertEqual(old.add(list(tet.tolist()), grain), new.add(list(tet.tolist()), grain))
                else:                                                    # look a star up
                    node = int(rng.integers(0, n + 10))
                    self.assertEqual(old.star(node), new.star(node))
            self.assertSameState(old, new)
            alive = np.flatnonzero(old.alive)
            np.testing.assert_array_equal(np.asarray(old.t)[alive], new.array()[alive])
            self.assertEqual(new.array().dtype, np.asarray(old.t).dtype)
            np.testing.assert_array_equal(np.asarray(old.t, dtype=dtype)[alive], new.array(dtype)[alive])
            self.assertEqual(new.array(dtype).dtype, dtype)

    def test_original_array_is_not_modified(self):
        t, g, n = random_tets(2)
        before = t.copy()
        mesh = _ArrayMesh(t, g, n)
        mesh.add([0, 1, 2, 3], 1)
        mesh.alive[0] = False
        np.testing.assert_array_equal(t, before)

    def test_no_added_tets(self):
        t, g, n = random_tets(3)
        mesh = _ArrayMesh(t, g, n)
        np.testing.assert_array_equal(mesh.array(), t)
        self.assertEqual(len(mesh.t), len(t))


if __name__ == '__main__':
    unittest.main()


class SwapLoopTests(unittest.TestCase):
    """The swap loop on _ArrayMesh gives d3v2p0's tets, grains and report on larger jittered meshes."""

    def test_larger_meshes_every_tier(self):
        from .test_tet_smoothing_parallel import jittered_block
        from upxo.meshing.gbconformant.d3v2p0 import tet_swaps as reference
        from upxo.meshing.gbconformant.d3v2p1 import backend, tet_swaps as fast
        tiers = ['numpy'] + (['numba'] if backend.numba_available() else [])
        for seed in (1, 6):
            p, t, _ = jittered_block(seed, n=7)
            grains = (p[t].mean(axis=1)[:, 0] > 3.).astype(int) + 1
            tri = np.empty((0, 3), int)
            kw = dict(target=45., max_angle=135., max_passes=5)
            rt, rg, rr = reference.swap_tets(p, t, grains, tri, **kw)
            self.assertGreater(rr['swaps_2_3'] + rr['swaps_3_2'], 3)
            for tier in tiers:
                with self.subTest(seed=seed, tier=tier):
                    ft, fg, fr = fast.swap_tets(p, t, grains, tri, backend=tier, **kw)
                    np.testing.assert_array_equal(rt, ft)
                    np.testing.assert_array_equal(rg, fg)
                    self.assertEqual(rt.dtype, ft.dtype)
                    self.assertEqual(rr, {k: v for k, v in fr.items() if k != 'backend'})
