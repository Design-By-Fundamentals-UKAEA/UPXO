"""Deterministic work splitting and chunked execution on a backend plan."""
from .backend import WorkerPool, resolve_workers  # noqa: F401  (resolve_workers re-exported)


def balanced_chunks(weights, n_chunks):
    """Split item indices into n_chunks with similar total weight (largest first).

    Deterministic: ties keep index order. Returns a list of index lists,
    empty chunks dropped.
    """
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    loads = [0.] * max(1, n_chunks)
    chunks = [[] for _ in loads]
    for i in order:
        k = min(range(len(loads)), key=lambda j: (loads[j], j))
        chunks[k].append(i)
        loads[k] += weights[i]
    return [sorted(c) for c in chunks if c]


def run_chunks(function, tasks, plan):
    """function(task) for every task, in task order, on the plan's tier
    (worker processes, or serially when the plan is 'numpy' or the pool
    fails; the plan records a fallback)."""
    with WorkerPool(plan) as pool:
        return pool.map(function, tasks)


def single_thread_worker():
    """Limit BLAS/OpenMP pools in a worker process to one thread, so many
    workers doing small array operations do not oversubscribe the cores.
    Uses threadpoolctl when installed; otherwise does nothing."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(1)
