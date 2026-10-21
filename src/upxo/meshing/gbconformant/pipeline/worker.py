"""Child process that runs one stage from the saved results of its run folder.

Started by :mod:`.background` as ``python -m upxo.meshing.gbconformant.pipeline.worker JOB EVENTS``.
It restores the stages before the one to run, runs it (which saves its results), and appends
one JSON object per line to the EVENTS file: ``progress`` while it works, then ``finished`` or
``failed``.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

PROGRESS_INTERVAL_S = 0.1


class EventFile:
    """Appends one JSON line per event; the reader only consumes complete lines."""

    def __init__(self, path: Path):
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, **event) -> None:
        self._file.write(json.dumps(event, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _watch_invalidations(state) -> list:
    """Record the invalidations a stage asks for, so the process that asked for the stage can repeat them."""
    calls: list = []
    depth = [0]
    for name in ("invalidate_from", "invalidate_after"):
        original = getattr(state, name)

        def wrapper(stage_id, reason="", _original=original, _name=name):
            if depth[0] == 0:
                calls.append([_name, stage_id, reason])
            depth[0] += 1
            try:
                return _original(stage_id, reason)
            finally:
                depth[0] -= 1

        setattr(state, name, wrapper)
    return calls


def run_job(job: dict, events: EventFile) -> int:
    from upxo.meshing.gbconformant.pipeline import saved_results
    from upxo.meshing.gbconformant.pipeline.run import new_state
    from upxo.meshing.gbconformant.pipeline.runner import PipelineRunner
    from upxo.meshing.gbconformant.pipeline.settings import CONFIG_VERSION, load_versioned
    from upxo.meshing.gbconformant.pipeline.stages import stage_ids

    stage_id = job["stage"]
    config, _ = load_versioned(job["config"], CONFIG_VERSION)
    state = new_state(config)
    state.checkpoints_enabled = True
    summary = saved_results.restore(state, before=stage_id)
    missing = [s for s in job.get("done", []) if s not in summary["restored"]]
    if missing:
        events.write(event="failed", kind="prerequisites", stage=stage_id,
                     error="Saved results of earlier stages could not be restored: " + ", ".join(missing))
        return 3
    invalidations = _watch_invalidations(state)
    order = stage_ids()
    status_before = dict(state.status)
    last = [0.0]

    def relay(message, current=None, total=None):
        now = time.monotonic()
        if now - last[0] >= PROGRESS_INTERVAL_S:
            last[0] = now
            events.write(event="progress", stage=stage_id, message=message, current=current,
                         total=total)

    state.set_progress_handler(relay)
    logged = len(state.log_lines)
    try:
        PipelineRunner(state).run_stage(stage_id)
    except Exception as error:
        events.write(event="failed", kind="stage", stage=stage_id, error=str(error),
                     traceback=traceback.format_exc(), log=state.log_lines[logged:])
        return 1
    store = saved_results.store_for(state)
    meta = store.meta(stage_id) or {}
    also = [s for s in order[:order.index(stage_id)]
            if s in state.status and state.status[s] != status_before.get(s)]   # optional stages skipped on the way
    events.write(event="finished", stage=stage_id, invalidations=invalidations,
                 run_folder=str(store.root.parent), also=also,
                 log=state.log_lines[logged:], complete=not meta.get("skipped"),
                 not_stored=sorted(meta.get("skipped", {})),
                 seconds=(state.reports.get(stage_id) or {}).get("duration_s"),
                 status=state.status[stage_id].value)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m upxo.meshing.gbconformant.pipeline.worker JOB.json EVENTS.jsonl",
              file=sys.stderr)
        return 2
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    events = EventFile(Path(argv[1]))
    try:
        return run_job(job, events)
    except BaseException as error:                    # nothing may end the process without saying why
        events.write(event="failed", kind="worker", stage=job.get("stage"), error=str(error),
                     traceback=traceback.format_exc())
        return 4
    finally:
        events.close()


if __name__ == "__main__":
    sys.exit(main())
