"""d3v2p1.interfaces.smooth_interfaces: d3v2p0's Interfaces, field for field, for every option."""
import unittest
import numpy as np
from ..d3v2p0 import test_interfaces as base_interfaces
from ..d3v2p0 import test_differential_smoothing as base_smoothing
from upxo.meshing.gbconformant.d3v2p0 import interfaces as ref
from upxo.meshing.gbconformant.d3v2p1 import interfaces as fast

FIELDS = ('points', 'triangles', 'grain_pairs', 'original_points', 'node_kind', 'fixed_axes', 'junction_edges')


def voronoi(seed, n=14, grains=10):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, n, size=(grains, 3))
    x = np.stack(np.meshgrid(*[np.arange(n) + .5] * 3, indexing='ij'), -1)
    return (np.argmin(((x[..., None, :] - seeds) ** 2).sum(-1), -1) + 1).astype(int)


def volumes():
    out = {f'voronoi{seed}': voronoi(seed) for seed in (0, 1, 2)}
    out['dumbbell'] = base_smoothing.dumbbell()
    sparse = voronoi(3, n=12, grains=6) * 37 - 11                     # large, non-contiguous, negative IDs
    out['odd_ids'] = sparse.astype(np.int32)
    return out


OPTIONS = {
    'defaults': dict(),
    'plain': dict(iterations=20, relaxation=.5, junction_iterations=5, junction_relaxation=.25),
    'by_pair': dict(iterations=30, relaxation=.6, iterations_by_pair={(1, 2): 4, (2, 1): 4}, ramp_rings=3),
    'thickness': dict(iterations=20, relaxation=.6, thickness_cap_fraction=.5),
    'wedge': dict(iterations=20, relaxation=.9, minimum_wedge_angle=30.),
    'corner': dict(iterations=20, relaxation=.9, minimum_corner_angle=20.),
    'pipeline': dict(iterations=50, relaxation=.75, junction_iterations=10, junction_relaxation=.35,
                     junction_max_displacement=.75, ramp_rings=2, thickness_cap_fraction=.5,
                     minimum_wedge_angle=30., minimum_corner_angle=20.),
    'zero': dict(iterations=0),
}


class EquivalenceTests(unittest.TestCase):
    def assertSame(self, a, b):
        for name in FIELDS:
            x, y = getattr(a, name), getattr(b, name)
            self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype), name)
            np.testing.assert_array_equal(x, y, err_msg=name)
        self.assertEqual(a.accepted_iterations, b.accepted_iterations)
        self.assertEqual(a.smoothing_report, b.smoothing_report)

    def test_every_option_on_every_volume(self):
        for vname, labels in volumes().items():
            for oname, options in OPTIONS.items():
                kwargs = dict(options)
                if 'iterations_by_pair' in kwargs and vname != 'dumbbell' and not vname.startswith('voronoi'):
                    kwargs.pop('iterations_by_pair')
                with self.subTest(volume=vname, options=oname):
                    self.assertSame(ref.smooth_interfaces(labels, **kwargs), fast.smooth_interfaces(labels, **kwargs))

    def test_frozen_grains_and_spacing(self):
        labels = voronoi(4)
        kwargs = dict(spacing=(1., 1.5, 2.), iterations=15, relaxation=.7, frozen_grain_ids=(2, 5),
                      minimum_wedge_angle=30., thickness_cap_fraction=.5)
        self.assertSame(ref.smooth_interfaces(labels, **kwargs), fast.smooth_interfaces(labels, **kwargs))

    def test_no_interfaces_and_invalid_arguments(self):
        flat = np.ones((3, 3, 3), int)
        self.assertSame(ref.smooth_interfaces(flat), fast.smooth_interfaces(flat))
        labels = voronoi(0, n=6, grains=3)
        for kwargs in (dict(iterations=-1), dict(relaxation=0.), dict(junction_relaxation=2.), dict(ramp_rings=-1),
                       dict(minimum_wedge_angle=95.), dict(thickness_cap_fraction=0.), dict(frozen_grain_ids=(999,))):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as expected:
                    ref.smooth_interfaces(labels, **kwargs)
                with self.assertRaises(ValueError) as got:
                    fast.smooth_interfaces(labels, **kwargs)
                self.assertEqual(str(expected.exception), str(got.exception))

    def test_progress_messages_match(self):
        labels = voronoi(0, n=8, grains=4)
        calls = {'ref': [], 'fast': []}
        ref.smooth_interfaces(labels, iterations=5, progress=lambda *a: calls['ref'].append(a))
        fast.smooth_interfaces(labels, iterations=5, progress=lambda *a: calls['fast'].append(a))
        self.assertEqual(calls['ref'], calls['fast'])
        self.assertGreater(len(calls['ref']), 0)

    def test_classes_and_helpers_are_d3v2p0s(self):
        self.assertIs(fast.Interfaces, ref.Interfaces)
        self.assertIs(fast.upscale_labels, ref.upscale_labels)


def _swap(module, name, function):
    saved = getattr(module, name)
    setattr(module, name, function)
    return saved


class _Swapped:
    """d3v2p0's tests of smooth_interfaces, run against the d3v2p1 function."""
    module = None

    def setUp(self):
        self._saved = _swap(self.module, 'smooth_interfaces', fast.smooth_interfaces)

    def tearDown(self):
        setattr(self.module, 'smooth_interfaces', self._saved)


class JunctionOwnTests(_Swapped, base_interfaces.JunctionTests):
    module = base_interfaces


class UpscaleOwnTests(_Swapped, base_interfaces.UpscaleTests):
    module = base_interfaces


class DifferentialOwnTests(_Swapped, base_smoothing.DifferentialSmoothingTests):
    module = base_smoothing


class CornerGuardOwnTests(_Swapped, base_smoothing.CornerGuardTests):
    module = base_smoothing


if __name__ == '__main__':
    unittest.main()
