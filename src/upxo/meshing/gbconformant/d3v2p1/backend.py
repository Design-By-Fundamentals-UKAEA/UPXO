"""Execution backend for d3v2p1: core detection, worker count, tier choice
and a worker pool that falls back to serial execution.

Tiers:
- 'numpy': serial, pure numpy, in this process. Always available.
- 'parallel': worker processes. Used when more than one worker is allowed
  and processes start; if they cannot start or a worker dies, the work is
  redone serially and the reason is recorded.
- 'numba': compiled kernels. Detected here; no d3v2p1 stage has numba
  kernels yet, so a request for it falls back to 'parallel'.

Choice order: explicit function arguments, then the environment variables
UPXO_BACKEND ('auto', 'numpy', 'parallel', 'numba') and UPXO_N_WORKERS (a
positive integer), then automatic selection. The automatic worker count is
the number of physical cores this process may use, so hyper-threads are
not oversubscribed. Results of every d3v2p1 stage do not depend on the tier
or the worker count unless that stage's documentation says otherwise.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError

BACKENDS = ('auto', 'numpy', 'parallel', 'numba')
_CACHE = {}


def logical_cores():
    """Logical CPUs in the machine (at least 1)."""
    return max(1, os.cpu_count() or 1)


def usable_cores():
    """Logical CPUs this process may run on (affinity / container limits)."""
    if 'usable' not in _CACHE:
        count = None
        if hasattr(os, 'process_cpu_count'):                    # Python 3.13+
            count = os.process_cpu_count()
        elif hasattr(os, 'sched_getaffinity'):                  # Linux
            try:
                count = len(os.sched_getaffinity(0))
            except OSError:
                count = None
        _CACHE['usable'] = max(1, count or logical_cores())
    return _CACHE['usable']


def physical_cores():
    """Physical cores, or None when they cannot be determined.

    Uses psutil when installed, else /proc/cpuinfo on Linux.
    """
    if 'physical' not in _CACHE:
        count = None
        try:
            import psutil
            count = psutil.cpu_count(logical=False)
        except Exception:
            count = None
        if not count and os.path.exists('/proc/cpuinfo'):
            try:
                cores, physical_id = set(), None
                with open('/proc/cpuinfo', encoding='utf-8') as stream:
                    for line in stream:
                        key, _, value = line.partition(':')
                        key = key.strip()
                        if key == 'physical id':
                            physical_id = value.strip()
                        elif key == 'core id':
                            cores.add((physical_id, value.strip()))
                count = len(cores) or None
            except OSError:
                count = None
        _CACHE['physical'] = int(count) if count else None
    return _CACHE['physical']


def auto_workers():
    """Default worker count: physical cores, limited to the usable CPUs.
    Falls back to the usable logical CPUs when physical cores are unknown."""
    physical = physical_cores()
    usable = usable_cores()
    return max(1, min(physical, usable) if physical else usable)


def cores():
    """JSON-compatible summary of the detected CPUs."""
    return dict(logical=logical_cores(), usable=usable_cores(), physical=physical_cores(),
                auto_workers=auto_workers())


def numba_available():
    """True when numba imports and compiles a small function (checked once)."""
    if 'numba' not in _CACHE:
        try:
            import numba

            @numba.njit(cache=False)
            def _probe(x):
                return x + 1
            _CACHE['numba'] = _probe(1) == 2
        except Exception:
            _CACHE['numba'] = False
    return _CACHE['numba']


def _env_workers():
    value = os.environ.get('UPXO_N_WORKERS')
    if value in (None, ''):
        return None
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        warnings.warn(f'Ignoring UPXO_N_WORKERS={value!r}: expected a positive integer')
        return None
    return n


def _env_backend():
    value = os.environ.get('UPXO_BACKEND')
    if value in (None, ''):
        return None
    value = value.strip().lower()
    if value not in BACKENDS:
        warnings.warn(f'Ignoring UPXO_BACKEND={value!r}: expected one of {BACKENDS}')
        return None
    return value


def check_workers(n_workers):
    if n_workers is None:
        return
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 0:
        raise ValueError('n_workers must be None or a nonnegative integer')


def resolve_workers(n_workers=None):
    """Worker count: None or 0 -> UPXO_N_WORKERS or auto_workers(); a positive
    integer -> at most the usable CPUs."""
    check_workers(n_workers)
    if not n_workers:
        n_workers = _env_workers() or auto_workers()
    return max(1, min(int(n_workers), usable_cores()))


class Plan:
    """Chosen tier and worker count for one stage call, plus what happened.

    used: 'numpy' or 'parallel'. fallback: None, or the reason a requested
    tier was not used (missing kernels, pool start or worker failure).
    """

    def __init__(self, requested, used, workers, fallback=None):
        self.requested, self.used, self.workers, self.fallback = requested, used, workers, fallback

    @property
    def parallel(self):
        return self.used == 'parallel'

    def degrade(self, reason):
        """Switch to the serial numpy tier for the rest of the call."""
        if self.used != 'numpy':
            self.fallback = reason if self.fallback is None else f'{self.fallback}; {reason}'
            self.used, self.workers = 'numpy', 1

    def report(self):
        return dict(requested=self.requested, used=self.used, n_workers=int(self.workers),
                    fallback=self.fallback, cores=cores())


def plan(backend='auto', n_workers=None, items=None, min_items_per_worker=1):
    """Choose the tier and worker count for a stage call.

    backend: 'auto' (default), 'numpy', 'parallel' or 'numba'; 'auto' and
    the default n_workers=None defer to UPXO_BACKEND / UPXO_N_WORKERS.
    items: amount of independent work; with min_items_per_worker it caps the
    worker count, so small inputs run serially.
    """
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    check_workers(n_workers)
    requested = backend
    if backend == 'auto':
        backend = _env_backend() or 'auto'
    fallback = None
    if backend == 'numba':
        fallback = 'no numba kernels in this stage; using parallel' if numba_available() \
            else 'numba not available; using parallel'
        backend = 'parallel'
    if backend == 'numpy':
        return Plan(requested, 'numpy', 1, fallback)
    workers = resolve_workers(n_workers)
    if items is not None:
        workers = min(workers, max(1, int(items) // max(1, int(min_items_per_worker))))
    if workers <= 1:
        return Plan(requested, 'numpy', 1, fallback)
    return Plan(requested, 'parallel', workers, fallback)


# Failures of the process machinery itself. Errors raised by the stage code
# inside a worker are re-raised unchanged (a serial rerun would raise them too).
_START_ERRORS = (OSError, ValueError, NotImplementedError, ImportError, RuntimeError)
_RUN_ERRORS = (BrokenProcessPool, PicklingError)


class WorkerPool:
    """Process pool for one stage call that falls back to serial execution.

    map(function, tasks, serial=None) returns [function(t) for t in tasks]
    in task order. In the parallel tier it runs in worker processes (started
    on first use, kept for the call). If the pool cannot start or a worker
    fails, the plan is switched to 'numpy', the reason recorded, and the
    tasks are run by serial() (default: function in this process). serial
    must be given when function relies on worker-only state (an
    initializer). Use as a context manager.
    """

    def __init__(self, plan, initializer=None, initargs=()):
        self.plan = plan
        self.initializer, self.initargs = initializer, initargs
        self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def map(self, function, tasks, serial=None):
        tasks = list(tasks)
        run_serial = serial if serial is not None else (lambda: [function(t) for t in tasks])
        if not self.plan.parallel or len(tasks) <= 1 and self._pool is None:
            return run_serial()
        if self._pool is None:
            try:
                self._pool = ProcessPoolExecutor(max_workers=self.plan.workers, initializer=self.initializer,
                                                 initargs=self.initargs)
            except _START_ERRORS as error:
                return self._fall_back('could not start worker processes', error, run_serial)
        try:
            return list(self._pool.map(function, tasks))
        except _RUN_ERRORS as error:
            return self._fall_back('worker processes failed', error, run_serial)

    def _fall_back(self, what, error, run_serial):
        self.close()
        self.plan.degrade(f'{what} ({type(error).__name__}: {error}); ran serially')
        warnings.warn(f'd3v2p1: {self.plan.fallback}')
        return run_serial()
