import io
import threading

import pytest

from upxo._sup import progress


def _fractions(events):
    return [e.fraction for e in events]


def test_no_listener_is_a_noop():
    assert not progress.is_active()
    progress.report(1, 2, "x")
    progress.check_cancel()
    with progress.stage("s", 0.0, 0.5):
        pass
    assert list(progress.track(range(5))) == [0, 1, 2, 3, 4]


def test_track_reports_start_and_end_and_yields_everything():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        out = list(progress.track(range(10), message="work", min_interval_s=0.0))
    assert out == list(range(10))
    assert sink.events[0].current == 0 and sink.events[0].total == 10
    assert sink.events[-1].current == 10 and sink.events[-1].fraction == 1.0
    assert all(e.message == "work" for e in sink.events)
    currents = [e.current for e in sink.events]
    assert currents == sorted(currents)


def test_track_throttle_still_sends_first_and_last():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        list(progress.track(range(1000), min_interval_s=3600.0))
    assert [e.current for e in sink.events] == [0, 1000]


def test_track_every_skips_but_always_ends_on_total():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        list(progress.track(range(10), every=4, min_interval_s=0.0))
    assert [e.current for e in sink.events] == [0, 4, 8, 10]


def test_track_without_len_is_indeterminate_then_closes_with_count():
    sink = progress.ListSink()
    gen = (i for i in range(7))
    with progress.progress_scope(sink):
        assert len(list(progress.track(gen, min_interval_s=0.0))) == 7
    assert sink.events[0].total is None and sink.events[0].fraction is None
    assert sink.events[-1].current == 7 and sink.events[-1].total == 7


def test_report_fraction_and_unknown_total():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        progress.report(1, 4, "a")
        progress.report(3, None, "b")
        progress.report(None, None, "c")
    assert _fractions(sink.events) == [0.25, None, None]


def test_stage_names_nest_and_fractions_compose():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        with progress.stage("outer", 0.0, 0.5):
            progress.report(5, 10)
            with progress.stage("inner", 0.5, 1.0):
                progress.report(1, 2)
        progress.report(1, 2)
    assert [e.stage for e in sink.events] == ["outer", "outer > inner", ""]
    assert _fractions(sink.events) == [pytest.approx(0.25), pytest.approx(0.375), 0.5]


def test_stage_restores_after_exception():
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        with pytest.raises(RuntimeError):
            with progress.stage("boom", 0.0, 0.5):
                raise RuntimeError
        progress.report(1, 2)
    assert sink.events[0].stage == "" and sink.events[0].fraction == 0.5


def test_stage_rejects_bad_range():
    with progress.progress_scope(progress.ListSink()):
        with pytest.raises(ValueError):
            with progress.stage("bad", 0.6, 0.4):
                pass


def test_plan_splits_proportionally():
    (a0, a1), (b0, b1) = progress.plan(1, 9)
    assert (a0, a1) == (0.0, pytest.approx(0.1))
    assert (b0, b1) == (pytest.approx(0.1), 1.0)
    with pytest.raises(ValueError):
        progress.plan(0, 0)


def test_cancel_stops_track_and_report():
    cancel = threading.Event()
    seen = []
    with progress.progress_scope(progress.ListSink(), cancel=cancel):
        with pytest.raises(progress.ProgressCancelled):
            for i in progress.track(range(100), min_interval_s=0.0):
                seen.append(i)
                if i == 3:
                    cancel.set()
        with pytest.raises(progress.ProgressCancelled):
            progress.report(1, 2)
        with pytest.raises(progress.ProgressCancelled):
            progress.check_cancel()
    assert seen == [0, 1, 2, 3]


def test_sink_errors_are_swallowed_but_cancel_from_sink_propagates():
    def bad(_event):
        raise ValueError("display broke")

    def cancelling(_event):
        raise progress.ProgressCancelled("stop")

    with progress.progress_scope(bad):
        progress.report(1, 2)
        assert list(progress.track(range(3))) == [0, 1, 2]
    with progress.progress_scope(cancelling):
        with pytest.raises(progress.ProgressCancelled):
            progress.report(1, 2)


def test_scope_is_restored_and_inner_scope_replaces_outer():
    outer, inner = progress.ListSink(), progress.ListSink()
    with progress.progress_scope(outer):
        with progress.progress_scope(inner):
            progress.report(1, 2)
        progress.report(2, 2)
    assert not progress.is_active()
    assert len(inner.events) == 1 and len(outer.events) == 1


def test_thread_needs_bind_context_to_inherit_listener():
    sink = progress.ListSink()

    def work():
        progress.report(1, 1, "from thread")

    with progress.progress_scope(sink):
        plain = threading.Thread(target=work)
        bound = threading.Thread(target=progress.bind_context(work))
        plain.start(); plain.join()
        assert sink.events == []
        bound.start(); bound.join()
    assert [e.message for e in sink.events] == ["from thread"]


def test_stdout_sink_prints_and_rate_limits():
    buf = io.StringIO()
    sink = progress.StdoutSink(stream=buf, min_interval_s=3600.0)
    with progress.progress_scope(sink):
        with progress.stage("Run", 0.0, 1.0):
            list(progress.track(range(5), message="items", min_interval_s=0.0))
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[Run] 0/5") and "items" in lines[0]
    assert lines[1].startswith("[Run] 5/5") and "(100%)" in lines[1]


def test_tqdm_sink_updates_bar():
    pytest.importorskip("tqdm")
    sink = progress.TqdmSink(file=io.StringIO())
    with progress.progress_scope(sink):
        list(progress.track(range(4), message="m", min_interval_s=0.0))
    assert sink._bar.total == 4 and sink._bar.n == 4
    sink.close()
