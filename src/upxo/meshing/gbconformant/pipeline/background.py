"""Run one stage in a separate process, watch its progress, cancel it at any moment.

The stage runs in a fresh Python process that restores the saved results of the stages
before it, runs the stage (which saves its own results), and reports through a file. The
caller's :class:`PipelineState` is changed only when the stage finishes, by reading the
saved results of that stage; a cancelled or failed stage leaves it as it was.

    job = StageProcess(state, "tet_mesh")      # NotReady if the earlier results are not on disk
    job.start()
    while not job.done:
        for event in job.poll(): ...           # progress messages
        time.sleep(0.1)
    if job.outcome == "finished":
        job.apply()
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from upxo.meshing.gbconformant.pipeline import saved_results
from upxo.meshing.gbconformant.pipeline.runfiles import _json_default, ensure_run_dir
from upxo.meshing.gbconformant.pipeline.stages import has_stage, stage_def
from upxo.meshing.gbconformant.pipeline.state import PipelineState, StageStatus

WORKER_MODULE = "upxo.meshing.gbconformant.pipeline.worker"
JOB_FOLDER = ".worker"
LOG_TAIL_LINES = 12


class NotReady(Exception):
    """The stage cannot be run in a separate process; ``reasons`` say why."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


class IncompleteTransfer(Exception):
    """The stage ran but some of its results could not be saved, so the caller cannot take them over."""


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


class StageProcess:
    """One stage running in its own process."""

    def __init__(self, state: PipelineState, stage_id: str, *, python: Optional[str] = None):
        if not has_stage(stage_id):
            stage_def(stage_id)                                   # raises, listing the stages
        reasons = []
        if not getattr(state, "checkpoints_enabled", False):
            reasons.append("saved results are switched off")
        else:
            missing = saved_results.missing_prerequisites(state, stage_id)
            if missing:
                reasons.append("results of earlier stages are not saved: " + ", ".join(missing))
        if reasons:
            raise NotReady(reasons)
        self.state = state
        self.stage_id = stage_id
        self.python = python or sys.executable
        self.folder: Optional[Path] = None
        self._process: Optional[subprocess.Popen] = None
        self._offset = 0
        self._cancelled = False
        self._final: dict = {}
        self._exit_seen = False
        self.started_at: Optional[float] = None
        self.done = False
        self.outcome: Optional[str] = None
        """``finished``, ``failed``, ``cancelled`` or ``crashed`` once :attr:`done`."""
        self.error = ""

    # --- running -------------------------------------------------------------------------------
    def start(self) -> None:
        run_dir = ensure_run_dir(self.state.config)
        self.folder = run_dir / JOB_FOLDER / f"{self.stage_id}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.folder.mkdir(parents=True, exist_ok=True)
        order = saved_results.stage_order()
        job = {
            "stage": self.stage_id,
            "config": json.loads(json.dumps(self.state.config, default=_json_default)),
            "done": [s for s in order[:order.index(self.stage_id)]
                     if self.state.status.get(s) in (StageStatus.OK, StageStatus.SKIPPED)],
        }
        (self.folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
        self._events = self.folder / "events.jsonl"
        self._events.write_text("", encoding="utf-8")
        self._log_path = self.folder / "output.log"
        flags = {}
        if os.name == "nt":
            flags["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                      | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            flags["start_new_session"] = True
        with open(self._log_path, "wb") as log:
            self._process = subprocess.Popen(
                [self.python, "-m", WORKER_MODULE, str(self.folder / "job.json"), str(self._events)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ.copy(), **flags)
        self.started_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def _read_events(self) -> list[dict]:
        events = []
        try:
            with open(self._events, "rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
        except OSError:
            return events
        end = data.rfind(b"\n")
        if end < 0:
            return events
        self._offset += end + 1
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            events.append(event)
            if event.get("event") in ("finished", "failed"):
                self._final = event
        return events

    def poll(self) -> list[dict]:
        """New events since the last call (progress, then the final one); sets :attr:`done` when the process is over."""
        if self._process is None or self.done:
            return []
        events = self._read_events()
        if self._process.poll() is not None:
            if not self._exit_seen:
                self._exit_seen = True                         # one more read after the exit
                events += self._read_events()
            self._conclude()
        return events

    def _conclude(self) -> None:
        self.done = True
        code = self._process.returncode
        if self._cancelled:
            self.outcome, self.error = "cancelled", "Cancelled by user"
        elif self._final.get("event") == "finished" and code == 0:
            self.outcome = "finished"
        elif self._final.get("event") == "failed":
            self.outcome, self.error = "failed", str(self._final.get("error") or "")
        else:
            self.outcome = "crashed"
            self.error = (f"The stage process ended unexpectedly (exit code {code}). "
                          f"Last output:\n{self._log_tail()}")

    def _log_tail(self) -> str:
        try:
            lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-LOG_TAIL_LINES:])

    def cancel(self) -> None:
        """Stop the stage now (the process and anything it started). The caller's state stays as it was."""
        if self._process is None or self.done:
            return
        self._cancelled = True
        _kill_tree(self._process)
        self.poll()

    # --- taking the results over ---------------------------------------------------------------
    @property
    def traceback(self) -> str:
        return str(self._final.get("traceback") or "")

    def worker_log(self) -> list[str]:
        return list(self._final.get("log") or [])

    def apply(self) -> None:
        """Take the finished stage's results into the caller's state."""
        if self.outcome != "finished":
            raise RuntimeError(f"The stage did not finish ({self.outcome}).")
        if not self._final.get("complete", True):
            raise IncompleteTransfer("Results that could not be saved: "
                                     + ", ".join(self._final.get("not_stored") or []))
        for method, stage, *rest in self._final.get("invalidations") or []:
            getattr(self.state, method)(stage, *(rest[:1] or [""]))
        folder = self._final.get("run_folder")
        for stage in list(self._final.get("also") or []) + [self.stage_id]:
            saved_results.load_stage(self.state, stage, folder)
        self.state.errors.pop(self.stage_id, None)
        for line in self.worker_log():
            self.state.append_log(line)

    def record_failure(self) -> None:
        """Show the stage as failed in the caller's state: status, error text, report and log."""
        stage_id = self.stage_id
        report = dict(self.state.reports.get(stage_id) or {})
        report.update(error=self.error, duration_s=round(self.elapsed, 3))
        self.state.reports[stage_id] = report
        self.state.errors[stage_id] = self.error
        self.state.set_status(stage_id, StageStatus.FAILED)
        for line in self.worker_log():
            self.state.append_log(line)
        verb = "cancelled" if self.outcome == "cancelled" else "failed"
        self.state.append_log(f"{verb}: {stage_id}: {self.error} ({report['duration_s']}s)")

    def cleanup(self) -> None:
        """Remove the job folder (kept after a failure, for diagnosis)."""
        import shutil
        if self.folder is not None and self.outcome in ("finished", "cancelled"):
            shutil.rmtree(self.folder, ignore_errors=True)


def run_stage_separately(state: PipelineState, stage_id: str,
                         on_event: Optional[Callable[[dict], None]] = None,
                         should_cancel: Optional[Callable[[], bool]] = None,
                         interval: float = 0.1) -> StageProcess:
    """Run ``stage_id`` in a separate process and wait; the results are in ``state`` when the outcome is ``finished``."""
    job = StageProcess(state, stage_id)
    job.start()
    while not job.done:
        for event in job.poll():
            if on_event is not None:
                on_event(event)
        if should_cancel is not None and should_cancel():
            job.cancel()
            break
        if not job.done:
            time.sleep(interval)
    if job.outcome == "finished":
        job.apply()
    job.cleanup()
    return job
