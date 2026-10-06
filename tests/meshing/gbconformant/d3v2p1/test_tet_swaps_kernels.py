"""d3v2p0's swap tests on d3v2p1 (numba and numpy tiers), and identical
results to d3v2p0 on a real mesh."""
import functools
import unittest
import numpy as np
from tests.meshing.gbconformant.d3v2p0 import test_tet_swaps as base
from upxo.meshing.gbconformant.d3v2p0 import tet_swaps as reference
from upxo.meshing.gbconformant.d3v2p1 import backend, tet_swaps as fast

HAVE_NUMBA = backend.numba_available()


class _Swap:
    tier = 'auto'

    def setUp(self):
        self._saved = base.swap_tets, base.swap_grain_tetrahedra
        base.swap_tets = functools.partial(fast.swap_tets, backend=self.tier)
        base.swap_grain_tetrahedra = functools.partial(fast.swap_grain_tetrahedra, backend=self.tier)

    def tearDown(self):
        base.swap_tets, base.swap_grain_tetrahedra = self._saved


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class NumbaSwapTests(_Swap, base.SwapTests):
    tier = 'numba'


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class NumbaGrainSwapTests(_Swap, base.GrainTetrahedraSwapTests):
    tier = 'numba'


class NumpySwapTests(_Swap, base.SwapTests):
    tier = 'numpy'


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class IdenticalToReferenceTests(unittest.TestCase):
    def test_jittered_mesh(self):
        from tests.meshing.gbconformant.d3v2p1.test_tet_smoothing_parallel import jittered_block
        p, t, inner = jittered_block(4, n=8)
        grains = (p[t].mean(axis=1)[:, 0] > 3.5).astype(int) + 1
        tri = np.empty((0, 3), int)                                  # no protected faces
        kw = dict(target=40., max_angle=140., max_passes=4)
        rt, rg, rr = reference.swap_tets(p, t, grains, tri, **kw)
        ft, fg, fr = fast.swap_tets(p, t, grains, tri, backend='numba', **kw)
        self.assertGreater(rr['swaps_2_3'] + rr['swaps_3_2'], 0)
        np.testing.assert_array_equal(rt, ft)
        np.testing.assert_array_equal(rg, fg)
        self.assertEqual({k: v for k, v in rr.items()}, {k: v for k, v in fr.items() if k != 'backend'})
        self.assertEqual(fr['backend']['used'], 'numba')

    def test_numpy_tier_is_reference(self):
        from tests.meshing.gbconformant.d3v2p1.test_tet_smoothing_parallel import jittered_block
        p, t, inner = jittered_block(5, n=6)
        grains = np.ones(len(t), int)
        tri = np.empty((0, 3), int)
        rt, _, rr = reference.swap_tets(p, t, grains, tri, target=40., max_angle=140.)
        ft, _, fr = fast.swap_tets(p, t, grains, tri, target=40., max_angle=140., backend='numpy')
        np.testing.assert_array_equal(rt, ft)
        self.assertEqual(fr['backend']['used'], 'numpy')


if __name__ == '__main__':
    unittest.main()
