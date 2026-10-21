"""Settings, artefacts, status and invalidation of one pipeline run.

A :class:`PipelineState` holds the settings, the result of every stage that has run
(the heavy arrays live here) and the status of each stage. Invalidating a stage drops
its results and those of every later stage.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum


class StageStatus(str, Enum):
    NOT_RUN = "Not run"
    RUNNING = "Running"
    OK = "OK"
    FAILED = "Failed"
    STALE = "Stale"
    SKIPPED = "Skipped"


class PipelineState:
    """Holds settings, artefacts, and per-stage status."""

    def __init__(self, config: dict, stage_ids: list[str]):
        self.config = config
        self.stage_ids = list(stage_ids)
        self.status: dict[str, StageStatus] = {
            sid: StageStatus.NOT_RUN for sid in self.stage_ids
        }
        self.reports: dict[str, dict] = {}
        self.artefacts: dict[str, object] = {}
        # Stage that last wrote each artefact; set while a stage runs.
        self.artefact_writer: dict[str, str] = {}
        self.current_stage: str | None = None
        # Results are saved to the run folder only when the caller turns this on.
        self.checkpoints_enabled = False
        self.on_invalidate = None           # called with the invalidated stage ids
        self.errors: dict[str, str] = {}
        # Why a stage's results are stale: ``{stage_id: {"reason", "source", "at"}}``.
        self.stale_reasons: dict[str, dict] = {}
        self._busy = False
        self.cancel_requested = False
        self.log_lines: list[str] = []
        self._progress_handler = None

    @property
    def busy(self) -> bool:
        return self._busy

    def try_begin_operation(self) -> bool:
        if self._busy:
            return False
        self._busy = True
        self.cancel_requested = False
        return True

    def end_operation(self) -> None:
        self._busy = False
        self._progress_handler = None
        self.cancel_requested = False

    def request_cancel(self) -> None:
        """Ask the current stage to stop at the next progress checkpoint."""
        self.cancel_requested = True

    def check_cancel(self) -> None:
        """Raise ``PipelineCancelled`` if a cancel was requested."""
        if self.cancel_requested:
            from upxo.meshing.gbconformant.pipeline.cancel import PipelineCancelled
            raise PipelineCancelled("Cancelled by user")

    def set_progress_handler(self, handler) -> None:
        """``handler(message, current=None, total=None)`` receives the progress of the running stage."""
        self._progress_handler = handler

    def report_progress(self, message: str, current=None, total=None) -> None:
        self.check_cancel()
        handler = self._progress_handler
        if handler is None:
            return
        try:
            handler(message, current, total)
        except TypeError:
            try:
                handler(message)
            except Exception:
                pass
        except Exception:
            pass
        # Re-check after the handler so a cancel requested meanwhile applies promptly.
        self.check_cancel()

    def append_log(self, message: str) -> None:
        self.log_lines.append(message)

    def set_status(self, stage_id: str, status: StageStatus) -> None:
        if stage_id not in self.status:
            raise KeyError(stage_id)
        self.status[stage_id] = status
        if status != StageStatus.STALE:
            self.stale_reasons.pop(stage_id, None)

    def store_artefact(self, key: str, value: object) -> None:
        self.artefacts[key] = value
        if self.current_stage is not None:
            self.artefact_writer[key] = self.current_stage
        else:
            self.artefact_writer.pop(key, None)

    def get_artefact(self, key: str, default=None):
        return self.artefacts.get(key, default)

    def invalidate_from(self, stage_id: str, reason: str = "") -> list[str]:
        """Mark ``stage_id`` and every later stage stale; drop their artefacts.

        ``reason`` says why (it is kept for each stage that held results). Returns the list of
        invalidated stage ids.
        """
        try:
            start = self.stage_ids.index(stage_id)
        except ValueError as exc:
            raise KeyError(stage_id) from exc
        hit = self.stage_ids[start:]
        stamp = datetime.now().isoformat(timespec="seconds")
        for sid in hit:
            if self.status[sid] in (
                StageStatus.OK, StageStatus.FAILED, StageStatus.SKIPPED,
                StageStatus.STALE,
            ):
                if self.status[sid] != StageStatus.STALE or sid not in self.stale_reasons:
                    self.stale_reasons[sid] = {"reason": reason or "an earlier stage changed",
                                               "source": stage_id, "at": stamp}
                self.status[sid] = StageStatus.STALE
            self.reports.pop(sid, None)
            self.errors.pop(sid, None)
        # Artefact keys are stage-scoped when prefixed with the stage id, and
        # also include shared names; clear anything tagged as downstream.
        snapshot_key = self._INPUT_SNAPSHOT.get(hit[0])
        snapshot = self.artefacts.get(snapshot_key) if snapshot_key else None
        drop = [k for k in self.artefacts if self._artefact_owned_by(k, hit)]
        for k in drop:
            del self.artefacts[k]
            self.artefact_writer.pop(k, None)
        if snapshot is not None and "labels" not in self.artefacts:
            # The invalidated cleaning stage can run again from its own input.
            self.artefacts["labels"] = snapshot.copy()
            if start > 0:
                self.artefact_writer["labels"] = self.stage_ids[start - 1]
        if self.on_invalidate is not None:
            self.on_invalidate(list(hit))
        return hit

    def invalidate_after(self, stage_id: str, reason: str = "") -> list[str]:
        """Invalidate every stage that comes after ``stage_id``; returns their ids."""
        try:
            start = self.stage_ids.index(stage_id) + 1
        except ValueError as exc:
            raise KeyError(stage_id) from exc
        if start >= len(self.stage_ids):
            return []
        return self.invalidate_from(self.stage_ids[start], reason)

    def invalidation_preview(self, stage_id: str) -> dict:
        """What invalidating from ``stage_id`` would throw away: the finished stages and the time they took."""
        start = self.stage_ids.index(stage_id)
        done = [sid for sid in self.stage_ids[start:] if self.status[sid] == StageStatus.OK]
        seconds = 0.0
        for sid in done:
            value = (self.reports.get(sid) or {}).get("duration_s")
            seconds += float(value) if isinstance(value, (int, float)) else 0.0
        return {"stages": done, "seconds": seconds}

    def stale_reason_text(self, stage_id: str) -> str:
        """``"<reason> (<time>)"`` for a stale stage, else an empty string."""
        info = self.stale_reasons.get(stage_id)
        if not info or self.status.get(stage_id) != StageStatus.STALE:
            return ""
        when = str(info.get("at", ""))[11:16]
        return f"{info.get('reason', '')}" + (f" ({when})" if when else "")

    # Input labels each cleaning stage stores before it runs.
    _INPUT_SNAPSHOT = {
        "subset_extract": "labels_before_subset",
        "small_grains": "labels_before_small_grains",
        "spikes": "labels_before_spikes",
        "pinches": "labels_before_pinches",
        "voxel_topology": "labels_before_topology",
    }

    # Rolling handles that stages overwrite in turn. When the writing stage is not
    # known (stored outside a stage run) they are dropped with any invalidation.
    _ROLLING = frozenset({
        "labels", "labels_before", "interfaces", "gmsh_result",
        "closed_rve", "remeshed_rve", "verified_tets", "frozen_geometry",
        "tet_validation", "abaqus_folder",
    })

    def _artefact_owned_by(self, key: str, stage_ids: list[str]) -> bool:
        """True when ``key`` was produced by one of ``stage_ids`` (the invalidated ones).

        Keys named after a stage (``<stage>``, ``<stage>.x``, ``<stage>:x``) belong to
        it; every other artefact belongs to the stage that last wrote it, so results of
        earlier, still-valid stages are kept.
        """
        for sid in stage_ids:
            if key == sid or key.startswith(sid + ".") or key.startswith(sid + ":"):
                return True
        writer = self.artefact_writer.get(key)
        if writer is not None:
            return writer in stage_ids
        return key in self._ROLLING and bool(stage_ids)
