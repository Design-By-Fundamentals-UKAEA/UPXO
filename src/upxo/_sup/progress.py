"""Optional progress reporting for long-running computations.

Computation code reports how far it has got; whoever launched the
computation decides what to do with that (print it, feed a progress bar,
or collect it in a test). With no listener installed every call here
is a near-free no-op, so library behaviour is unchanged by default.

Typical use inside a computation::

    from upxo._sup import progress

    for grain in progress.track(grains, message="Characterising grains"):
        ...

    with progress.stage("Cleaning", 0.5, 1.0):      # second half of the run
        for k in range(n_passes):
            progress.report(k + 1, n_passes, "pass")

and by whoever starts it::

    with progress.progress_scope(progress.StdoutSink()):
        run_the_computation()

The listener is held in a ``contextvars.ContextVar``, so it follows the call
stack without being passed through function signatures. A new thread starts
with an empty context; wrap its target with :func:`bind_context` to give it
the launching thread's listener.
"""
from __future__ import annotations

import contextvars
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "ProgressEvent", "ProgressCancelled", "progress_scope", "report", "track",
    "stage", "plan", "is_active", "check_cancel", "bind_context",
    "ListSink", "StdoutSink", "TqdmSink",
]


@dataclass(frozen=True)
class ProgressEvent:
    """One progress report.

    stage     "Outer > inner" label built from the active stages.
    current   Items done so far, or None when not counted.
    total     Items expected, or None when unknown (indeterminate).
    message   Free text for the current step.
    fraction  Overall completion in [0, 1], or None when it cannot be known
              (no total and no enclosing stage range).
    t         ``time.monotonic()`` when the event was created.
    """
    stage: str
    current: Optional[int]
    total: Optional[int]
    message: str
    fraction: Optional[float]
    t: float


class ProgressCancelled(Exception):
    """Raised inside a computation when its scope's cancel event is set."""


class _Scope:
    __slots__ = ("sink", "cancel", "stages")

    def __init__(self, sink, cancel):
        self.sink = sink
        self.cancel = cancel
        # Active stages, outermost first: (name, lo, hi), where lo and hi
        # give the stage's share of the overall [0, 1] range.
        self.stages: List[Tuple[str, float, float]] = []


_SCOPE: contextvars.ContextVar = contextvars.ContextVar("upxo_progress_scope", default=None)


@contextmanager
def progress_scope(sink: Callable[[ProgressEvent], None], cancel=None):
    """Install ``sink`` as the listener for everything run inside the block.

    sink    Callable taking a :class:`ProgressEvent`. It should return
            quickly and rate-limit itself. Exceptions it raises are
            swallowed (a broken display must not abort a computation),
            except :class:`ProgressCancelled`.
    cancel  Optional ``threading.Event``. When set, the next ``report`` /
            ``track`` / ``check_cancel`` call raises
            :class:`ProgressCancelled`.
    """
    token = _SCOPE.set(_Scope(sink, cancel))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def is_active() -> bool:
    """True when a listener is installed. Lets hot code skip building
    expensive messages."""
    return _SCOPE.get() is not None


def check_cancel() -> None:
    """Raise :class:`ProgressCancelled` if the active scope was cancelled."""
    scope = _SCOPE.get()
    if scope is not None and scope.cancel is not None and scope.cancel.is_set():
        raise ProgressCancelled("cancelled")


def _emit(scope: _Scope, stage_label: Optional[str], current, total, message) -> None:
    names = [s[0] for s in scope.stages]
    if stage_label:
        names.append(stage_label)
    local = None
    if total:
        local = max(0.0, min(1.0, current / total)) if current is not None else None
    fraction = None
    if local is not None:
        lo, hi = (scope.stages[-1][1], scope.stages[-1][2]) if scope.stages else (0.0, 1.0)
        fraction = lo + (hi - lo) * local
    event = ProgressEvent(" > ".join(names), current, total, message, fraction, time.monotonic())
    try:
        scope.sink(event)
    except ProgressCancelled:
        raise
    except Exception:
        pass


def report(current: Optional[int] = None, total: Optional[int] = None,
           message: str = "", stage: Optional[str] = None) -> None:
    """Report that ``current`` of ``total`` items are done.

    Pass ``total=None`` when the amount of work is unknown. Does nothing
    when no listener is installed. Raises :class:`ProgressCancelled` if the
    scope was cancelled.
    """
    scope = _SCOPE.get()
    if scope is None:
        return
    if scope.cancel is not None and scope.cancel.is_set():
        raise ProgressCancelled("cancelled")
    _emit(scope, stage, current, total, message)


def track(iterable: Iterable, total: Optional[int] = None, message: str = "",
          stage: Optional[str] = None, every: int = 1,
          min_interval_s: float = 0.1) -> Iterator:
    """Yield the items of ``iterable`` while reporting progress.

    total           Defaults to ``len(iterable)`` when available.
    every           Consider reporting only every this many items.
    min_interval_s  Minimum time between reports. The first report (0 done)
                    and the last one are always sent.

    Checks for cancellation before each item. With no listener installed it
    is a plain pass-through.
    """
    scope = _SCOPE.get()
    if scope is None:
        yield from iterable
        return
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    every = max(1, int(every))
    _emit(scope, stage, 0, total, message)
    last = time.monotonic()
    n = 0
    cancel = scope.cancel
    for item in iterable:
        if cancel is not None and cancel.is_set():
            raise ProgressCancelled("cancelled")
        yield item
        n += 1
        if n % every:
            continue
        now = time.monotonic()
        if now - last >= min_interval_s or (total is not None and n >= total):
            _emit(scope, stage, n, total, message)
            last = now
    if total is None or n < total or n % every:
        _emit(scope, stage, n, total if total is not None else n, message)


@contextmanager
def stage(name: str, start: float = 0.0, end: float = 1.0):
    """Name a phase of the work and give it a share of the overall range.

    ``start`` and ``end`` are fractions of the *enclosing* range, so stages
    nest: inside a stage covering 0.0-0.5 of the run, a child stage covering
    0.5-1.0 maps to 0.25-0.5 overall. Events inside the block carry the
    stage name and an overall ``fraction``. No-op without a listener.
    """
    scope = _SCOPE.get()
    if scope is None:
        yield
        return
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("stage requires 0 <= start <= end <= 1")
    p_lo, p_hi = (scope.stages[-1][1], scope.stages[-1][2]) if scope.stages else (0.0, 1.0)
    lo = p_lo + (p_hi - p_lo) * start
    hi = p_lo + (p_hi - p_lo) * end
    scope.stages.append((name, lo, hi))
    try:
        yield
    finally:
        scope.stages.pop()


def plan(*weights: float) -> List[Tuple[float, float]]:
    """Split [0, 1] into consecutive ``(start, end)`` ranges proportional to
    ``weights``, for use with :func:`stage`::

        a, b = progress.plan(1, 9)       # (0, 0.1), (0.1, 1.0)
    """
    total = float(sum(weights))
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative with a positive sum")
    out, acc = [], 0.0
    for w in weights:
        lo = acc / total
        acc += w
        out.append((lo, min(1.0, acc / total)))
    return out


def bind_context(fn: Callable) -> Callable:
    """Return ``fn`` wrapped so it runs in the context captured *now*.

    Use for a worker thread's target so it reports to the listener of the
    thread that started it::

        threading.Thread(target=progress.bind_context(worker)).start()
    """
    ctx = contextvars.copy_context()

    def wrapper(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)
    return wrapper


class ListSink:
    """Collects events in ``self.events`` (for tests)."""

    def __init__(self):
        self.events: List[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)


class StdoutSink:
    """Prints one line per event, at most one per ``min_interval_s``
    (the first and any 100 % event are always printed)."""

    def __init__(self, stream=None, min_interval_s: float = 0.5):
        self.stream = stream
        self.min_interval_s = min_interval_s
        self._last = None

    def __call__(self, event: ProgressEvent) -> None:
        done = event.fraction is not None and event.fraction >= 1.0
        if (self._last is not None and not done
                and event.t - self._last < self.min_interval_s):
            return
        self._last = event.t
        parts = []
        if event.stage:
            parts.append(f"[{event.stage}]")
        if event.current is not None:
            parts.append(f"{event.current}/{event.total}" if event.total else str(event.current))
        if event.fraction is not None:
            parts.append(f"({100 * event.fraction:.0f}%)")
        if event.message:
            parts.append(event.message)
        print(" ".join(parts), file=self.stream or sys.stdout, flush=True)


class TqdmSink:
    """Drives a ``tqdm`` bar. Requires ``tqdm`` (imported when constructed)."""

    def __init__(self, **tqdm_kwargs):
        from tqdm import tqdm
        self._bar = tqdm(total=None, **tqdm_kwargs)

    def __call__(self, event: ProgressEvent) -> None:
        bar = self._bar
        if event.total is not None and bar.total != event.total:
            bar.total = event.total
        desc = event.stage or event.message
        if desc:
            bar.set_description(desc, refresh=False)
        if event.current is not None:
            bar.n = event.current
        bar.refresh()

    def close(self) -> None:
        self._bar.close()
