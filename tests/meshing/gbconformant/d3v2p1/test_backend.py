"""Core detection, tier choice, environment overrides and serial fallback."""
import multiprocessing
import os
import sys
import unittest
import warnings
from unittest import mock
import numpy as np
from upxo.meshing.gbconformant.d3v2p1 import backend


def _square(x):
    return x * x


def _die_in_worker(x):
    if multiprocessing.parent_process() is not None:
        os._exit(1)                                   # simulates a worker that cannot run
    return x + 1


def _raise_value_error(x):
    raise ValueError(f'bad task {x}')


_MARKER = {}


def _set_marker(value, other):
    _MARKER['value'] = value + other


def _read_marker(x):
    return _MARKER['value'] + x


class _Fresh(unittest.TestCase):
    """Clear detection caches and the UPXO environment for each test."""

    def setUp(self):
        self._cache = dict(backend._CACHE)
        backend._CACHE.clear()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ('UPXO_BACKEND', 'UPXO_N_WORKERS'):
            os.environ.pop(key, None)

    def tearDown(self):
        self._env.stop()
        backend._CACHE.clear()
        backend._CACHE.update(self._cache)


class CoreDetectionTests(_Fresh):
    def test_counts_are_consistent(self):
        c = backend.cores()
        self.assertGreaterEqual(c['logical'], c['usable'])
        self.assertGreaterEqual(c['usable'], 1)
        self.assertLessEqual(c['auto_workers'], c['usable'])
        if c['physical']:
            self.assertEqual(c['auto_workers'], min(c['physical'], c['usable']))

    def test_without_psutil_uses_usable_cores(self):
        with mock.patch.dict(sys.modules, {'psutil': None}), mock.patch('os.path.exists', return_value=False):
            self.assertIsNone(backend.physical_cores())
            self.assertEqual(backend.auto_workers(), backend.usable_cores())


class PlanTests(_Fresh):
    def test_tiers(self):
        self.assertEqual(backend.plan('numpy').used, 'numpy')
        self.assertEqual(backend.plan(n_workers=1).used, 'numpy')
        p = backend.plan(n_workers=2)
        self.assertEqual((p.used, p.workers), ('parallel', min(2, backend.usable_cores())) if backend.usable_cores() > 1
                         else ('numpy', 1))
        self.assertEqual(backend.plan(n_workers=8, items=10, min_items_per_worker=4).workers,
                         min(2, backend.usable_cores()) if backend.usable_cores() > 1 else 1)
        self.assertEqual(backend.plan(n_workers=8, items=3, min_items_per_worker=4).used, 'numpy')
        n = backend.plan('numba', n_workers=2)
        self.assertIn('using parallel', n.fallback)
        self.assertEqual(n.requested, 'numba')
        self.assertEqual(set(n.report()), {'requested', 'used', 'n_workers', 'fallback', 'cores'})

    def test_automatic_cap(self):
        auto = backend.auto_workers()
        self.assertEqual(backend.plan(max_auto_workers=2).workers, min(2, auto) if auto > 1 else 1)
        explicit = min(3, backend.usable_cores())
        self.assertEqual(backend.plan(n_workers=3, max_auto_workers=2).workers, explicit if explicit > 1 else 1)
        os.environ['UPXO_N_WORKERS'] = '3'
        self.assertEqual(backend.plan(max_auto_workers=2).workers, explicit if explicit > 1 else 1)

    def test_bad_arguments(self):
        for kwargs in (dict(backend='gpu'), dict(n_workers=-1), dict(n_workers=1.5), dict(n_workers=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend.plan(**kwargs)

    def test_environment_overrides(self):
        os.environ['UPXO_BACKEND'] = 'numpy'
        self.assertEqual(backend.plan(n_workers=4).used, 'numpy')
        self.assertEqual(backend.plan('parallel', n_workers=2).used,
                         'parallel' if backend.usable_cores() > 1 else 'numpy')     # explicit argument wins
        os.environ['UPXO_BACKEND'] = 'auto'
        os.environ['UPXO_N_WORKERS'] = '1'
        self.assertEqual(backend.plan().used, 'numpy')
        self.assertEqual(backend.resolve_workers(None), 1)
        os.environ['UPXO_N_WORKERS'] = 'many'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(backend.resolve_workers(None), backend.auto_workers())
        self.assertTrue(any('UPXO_N_WORKERS' in str(w.message) for w in caught))
        os.environ['UPXO_BACKEND'] = 'cuda'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            backend.plan()
        self.assertTrue(any('UPXO_BACKEND' in str(w.message) for w in caught))

    def test_requested_backend(self):
        self.assertEqual(backend.requested_backend(), 'auto')
        self.assertEqual(backend.requested_backend('numba'), 'numba')
        os.environ['UPXO_BACKEND'] = 'numba'
        self.assertEqual(backend.requested_backend('auto'), 'numba')
        self.assertEqual(backend.requested_backend('numpy'), 'numpy')          # an explicit argument wins
        os.environ['UPXO_BACKEND'] = 'cuda'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            self.assertEqual(backend.requested_backend('auto'), 'auto')       # an unknown value is ignored
        with self.assertRaises(ValueError):
            backend.requested_backend('gpu')


@unittest.skipIf(backend.usable_cores() < 2, 'needs two usable CPUs')
class WorkerPoolTests(_Fresh):
    def test_parallel_and_serial_agree(self):
        tasks = list(range(7))
        with backend.WorkerPool(backend.plan(n_workers=2)) as pool:
            self.assertEqual(pool.map(_square, tasks), [t * t for t in tasks])
        with backend.WorkerPool(backend.plan('numpy')) as pool:
            self.assertEqual(pool.map(_square, tasks), [t * t for t in tasks])

    def test_pool_start_failure_runs_serially(self):
        chosen = backend.plan(n_workers=2)
        with mock.patch.object(backend, 'ProcessPoolExecutor', side_effect=OSError('no processes here')), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with backend.WorkerPool(chosen) as pool:
                self.assertEqual(pool.map(_square, [1, 2, 3]), [1, 4, 9])
        self.assertEqual(chosen.used, 'numpy')
        self.assertIn('could not start worker processes', chosen.fallback)
        self.assertTrue(caught)

    def test_dead_worker_runs_serially(self):
        chosen = backend.plan(n_workers=2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            with backend.WorkerPool(chosen) as pool:
                self.assertEqual(pool.map(_die_in_worker, [1, 2, 3]), [2, 3, 4])
        self.assertEqual(chosen.used, 'numpy')
        self.assertIn('BrokenProcessPool', chosen.fallback)

    def test_task_errors_propagate_without_fallback(self):
        chosen = backend.plan(n_workers=2)
        with backend.WorkerPool(chosen) as pool, self.assertRaises(ValueError):
            pool.map(_raise_value_error, [1, 2])
        self.assertEqual(chosen.used, 'parallel')
        self.assertIsNone(chosen.fallback)


@unittest.skipIf(backend.usable_cores() < 2, 'needs two usable CPUs')
class InitializerTests(_Fresh):
    """Workers start through d3v2p1._pool_init, so the parent never waits on a worker's heavy imports."""

    def test_module_level_function_is_named_through_the_light_module(self):
        from upxo.meshing.gbconformant.d3v2p1 import _pool_init
        init, args = backend._light_initializer(_set_marker, (3, 4))
        self.assertIs(init, _pool_init.initialize)
        self.assertEqual(args, (_set_marker.__module__, '_set_marker', (3, 4)))

    def test_other_callables_and_no_initializer_pass_through(self):
        self.assertEqual(backend._light_initializer(None, ()), (None, ()))
        local = lambda a: None                                    # noqa: E731
        self.assertEqual(backend._light_initializer(local, (1,)), (local, (1,)))

        def nested(a):
            return a
        self.assertEqual(backend._light_initializer(nested, (1,)), (nested, (1,)))
        copy = type(_set_marker)(_set_marker.__code__, _set_marker.__globals__, '_set_marker')   # same name, not the module's
        self.assertEqual(backend._light_initializer(copy, (1, 2)), (copy, (1, 2)))

    def test_initializer_runs_in_every_worker_with_its_arguments(self):
        plan = backend.Plan('parallel', 'parallel', 2)
        with backend.WorkerPool(plan, initializer=_set_marker, initargs=(10, 5)) as pool:
            self.assertEqual(pool.map(_read_marker, range(8)), [15 + x for x in range(8)])
        self.assertTrue(plan.parallel)
        self.assertIsNone(plan.fallback)

    def test_light_module_imports_nothing_heavy(self):
        import subprocess
        code = ("import sys; import upxo.meshing.gbconformant.d3v2p1._pool_init; "
                "heavy = [m for m in ('numpy', 'scipy', 'pyvista', 'gmsh', 'numba') if m in sys.modules]; "
                "sys.exit(1 if heavy else 0)")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        done = subprocess.run([sys.executable, '-c', code], env=env)
        self.assertEqual(done.returncode, 0)


class StageFallbackTests(_Fresh):
    def test_tet_smoothing_same_result_after_pool_failure(self):
        from .test_tet_smoothing_parallel import jittered_block
        from upxo.meshing.gbconformant.d3v2p1 import tet_smoothing as fast
        p, t, inner = jittered_block(1, n=12)                # 8 blocks per colour: the pool is used
        kw = dict(target=30., max_angle=150., neighbour_floor=20., block_size=3.)
        # serial reference with the same kernels as the fallback run ('auto')
        serial, rep_serial = fast.smooth_tet_dihedrals(p, t, inner, n_workers=1, **kw)
        with mock.patch.object(backend, 'ProcessPoolExecutor', side_effect=OSError('blocked')), \
                warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            failed, rep_failed = fast.smooth_tet_dihedrals(p, t, inner, n_workers=2, **kw)
        np.testing.assert_array_equal(serial, failed)
        self.assertGreater(rep_serial['moved_nodes'], 0)
        self.assertEqual(rep_serial['backend']['used'], 'numpy')
        self.assertEqual(rep_serial['backend']['kernels'], rep_failed['backend']['kernels'])
        self.assertEqual(rep_failed['backend']['used'], 'numpy')
        self.assertIn('blocked', rep_failed['backend']['fallback'])

    def test_angle_repair_same_result_after_pool_failure(self):
        from .test_surface_angles_parallel import WorkerCountTests
        from upxo.meshing.gbconformant.d3v2p1 import surface_angles as fast
        s = WorkerCountTests.rough_grid(None, 0)
        serial = fast.improve_surface_angles(s, backend='numpy', wedge_limit=30.)
        with mock.patch.object(backend, 'ProcessPoolExecutor', side_effect=OSError('blocked')), \
                warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            failed = fast.improve_surface_angles(s, n_workers=2, wedge_limit=30.)
        np.testing.assert_array_equal(serial.points, failed.points)
        np.testing.assert_array_equal(serial.triangles, failed.triangles)
        self.assertIn('blocked', failed.report['surface_angle_repair']['backend']['fallback'])


if __name__ == '__main__':
    unittest.main()
