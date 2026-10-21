"""Running stages: one stage at a time, or the whole sequence up to a chosen stage."""
from __future__ import annotations

import time
from typing import Callable, Optional

from upxo.meshing.gbconformant.pipeline.resources import peak_memory_mb
from upxo.meshing.gbconformant.pipeline.rolling import (
    ensure_skipped_optionals, mark_skipped,
)
from upxo.meshing.gbconformant.pipeline.stages import STAGES, has_stage, stage_def
from upxo.meshing.gbconformant.pipeline.state import PipelineState, StageStatus

Event = Callable[[dict], None]


class PipelineRunner:
    """Execute stages against a ``PipelineState``."""

    def __init__(self, state: PipelineState,
                 handlers: Optional[dict[str, Callable]] = None,
                 *, use_defaults: bool = True):
        self.state = state
        if handlers is not None:
            self.handlers = dict(handlers)
        elif use_defaults:
            from upxo.meshing.gbconformant.pipeline.handlers import default_handlers
            self.handlers = default_handlers()
        else:
            self.handlers = {}

    def run_stage(self, stage_id: str) -> StageStatus:
        if stage_id not in self.state.stage_ids:
            raise KeyError(stage_id)
        # Synthesize artefacts for optional stages that were switched off and never visited.
        ensure_skipped_optionals(self.state, stage_id)

        handler = self.handlers.get(stage_id)
        if handler is None:
            self.state.set_status(stage_id, StageStatus.OK)
            self.state.reports[stage_id] = {
                "placeholder": True,
                "message": "No handler registered yet.",
                "duration_s": 0.0,
            }
            self.state.append_log(f"placeholder: {stage_id}")
            return StageStatus.OK
        self.state.set_status(stage_id, StageStatus.RUNNING)
        self.state.append_log(f"running: {stage_id}")
        t0 = time.perf_counter()
        self.state.current_stage = stage_id
        before = dict(self.state.config)
        try:
            report = dict(handler(self.state) or {})
            report["duration_s"] = round(time.perf_counter() - t0, 3)
            memory = peak_memory_mb()
            if memory is not None:
                report["peak_memory_mb"] = memory
            self.state.reports[stage_id] = report
            self.state.set_status(stage_id, StageStatus.OK)
            self.state.append_log(
                f"ok: {stage_id} ({report['duration_s']}s)")
            self.state.current_stage = None
            from upxo.meshing.gbconformant.pipeline.saved_results import config_changes, save_after_run
            save_after_run(self.state, stage_id, config_changes(before, self.state.config))
            return StageStatus.OK
        except Exception as exc:
            from upxo.meshing.gbconformant.pipeline.cancel import PipelineCancelled
            duration = round(time.perf_counter() - t0, 3)
            failed_report = dict(self.state.reports.get(stage_id) or {})
            failed_report["duration_s"] = duration
            failed_report["error"] = str(exc)
            self.state.reports[stage_id] = failed_report
            self.state.set_status(stage_id, StageStatus.FAILED)
            self.state.errors[stage_id] = str(exc)
            if isinstance(exc, PipelineCancelled):
                self.state.append_log(f"cancelled: {stage_id} ({duration}s)")
            else:
                self.state.append_log(f"failed: {stage_id}: {exc} ({duration}s)")
            raise
        finally:
            self.state.current_stage = None


def _emit(on_event: Optional[Event], **event) -> None:
    if on_event is not None:
        on_event(event)


def run_pipeline(state: PipelineState, *, through: str = "smooth_interfaces",
                 skip_done: bool = False, on_event: Optional[Event] = None,
                 handlers: Optional[dict[str, Callable]] = None) -> list[str]:
    """Run the stages in order up to and including ``through``.

    A stage whose switch is off is recorded as skipped. With ``skip_done`` a stage that
    already holds results (status OK or Skipped) is not run again. ``on_event`` receives
    one dict per step: ``start``, ``done``, ``already_done``, ``skipped``, ``progress``
    and ``failed``, each naming its ``stage``. Returns the ids of the stages that ran;
    a failing stage raises.
    """
    if not has_stage(through):
        stage_def(through)                       # raises, listing the stages
    runner = PipelineRunner(state, handlers=handlers)
    ran = []
    for stage in STAGES:
        sid = stage.id
        if sid not in runner.handlers:
            if sid == through:
                break
            continue
        flag = stage.optional_flag
        if flag and not state.config.get(flag, True):
            mark_skipped(state, sid)
            _emit(on_event, event="skipped", stage=sid, title=stage.title,
                  reason=f"{flag} is off")
        elif skip_done and state.status.get(sid) in (StageStatus.OK, StageStatus.SKIPPED):
            _emit(on_event, event="already_done", stage=sid, title=stage.title,
                  status=state.status[sid].value)
        else:
            _run_with_events(state, runner, stage, on_event)
            ran.append(sid)
        if sid == through:
            break
    return ran


def _run_with_events(state: PipelineState, runner: PipelineRunner, stage,
                     on_event: Optional[Event]) -> None:
    sid = stage.id
    if on_event is not None:
        def relay(message, current=None, total=None):
            on_event({"event": "progress", "stage": sid, "message": message,
                      "current": current, "total": total})
        state.set_progress_handler(relay)
    _emit(on_event, event="start", stage=sid, title=stage.title)
    try:
        runner.run_stage(sid)
    except Exception as error:
        _emit(on_event, event="failed", stage=sid, title=stage.title, error=str(error),
              seconds=(state.reports.get(sid) or {}).get("duration_s"))
        raise
    finally:
        if on_event is not None:
            state.set_progress_handler(None)
    _emit(on_event, event="done", stage=sid, title=stage.title,
          seconds=(state.reports.get(sid) or {}).get("duration_s"))
