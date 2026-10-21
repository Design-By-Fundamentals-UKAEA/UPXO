"""Run the pipeline from settings.

``run`` restores whatever a previous run of the same folder saved and still trusts,
runs the stages after it, and writes the run manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from upxo.meshing.gbconformant.pipeline import saved_results
from upxo.meshing.gbconformant.pipeline.manifest import write_run_manifest
from upxo.meshing.gbconformant.pipeline.runfiles import ensure_run_dir
from upxo.meshing.gbconformant.pipeline.runner import run_pipeline
from upxo.meshing.gbconformant.pipeline.settings import CONFIG_VERSION, build_config, load_versioned
from upxo.meshing.gbconformant.pipeline.stages import stage_def, stage_ids
from upxo.meshing.gbconformant.pipeline.state import PipelineState, StageStatus

DEFAULT_LAST_STAGE = "export_abaqus"


@dataclass
class RunResult:
    """What a run did."""

    state: PipelineState
    run_folder: Path
    through: str
    restored: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    manifest: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.failed_stage is None

    @property
    def reports(self) -> dict:
        return self.state.reports

    def export_folder(self) -> Optional[Path]:
        """Folder of the Abaqus export, when the run reached it."""
        folder = self.state.get_artefact("abaqus_folder")
        return None if folder is None else Path(folder)

    @property
    def suggestions(self) -> list:
        """Settings worth changing after the failure (empty when the run did not fail or nothing is known)."""
        if self.ok or self.failed_stage not in stage_ids():
            return []
        from upxo.meshing.gbconformant.pipeline.advice import suggest
        return suggest(self.failed_stage, self.error or "", self.state.config)

    def raise_for_failure(self) -> None:
        if not self.ok:
            raise RuntimeError(f"Stage {self.failed_stage} failed: {self.error}")


def new_state(config: dict) -> PipelineState:
    """A state for every stage, saving each stage's results as it finishes (unless the settings say not to)."""
    state = PipelineState(config, stage_ids())
    saved_results.enable(state)
    return state


def run(config: Optional[dict] = None, *, through: str = DEFAULT_LAST_STAGE, resume: bool = True,
        on_event: Optional[Callable[[dict], None]] = None, write_manifest: bool = True,
        **settings) -> RunResult:
    """Run the pipeline up to and including ``through``.

    ``config`` and keyword ``settings`` are settings by name (see
    :mod:`upxo.meshing.gbconformant.pipeline.settings`); unknown names and unusable values
    raise ``SettingsError``. With ``resume`` the results a previous run of the same run
    folder saved are restored first (each stage only while its settings and everything
    before it are unchanged) and only the remaining stages run. A failing stage does not
    raise: the result names it in ``failed_stage`` and ``error``.
    """
    stage_def(through)                                           # a typo fails before any work
    state = new_state(build_config(config, **settings))
    result = RunResult(state=state, run_folder=ensure_run_dir(state.config), through=through)
    events = on_event or (lambda event: None)
    try:
        if resume and state.checkpoints_enabled:
            summary = saved_results.restore(state)
            result.restored = list(summary["restored"])
            for stage in result.restored:
                events({"event": "restored", "stage": stage, "title": stage_def(stage).title})
            for stage in summary["stale"]:
                events({"event": "stale", "stage": stage, "title": stage_def(stage).title})
        result.ran = run_pipeline(state, through=through, skip_done=True, on_event=events)
    except Exception as error:
        failed = next((s for s in stage_ids() if state.status.get(s) == StageStatus.FAILED), None)
        result.failed_stage = failed or "unknown"
        result.error = str(error)
        result.ran = [s for s in stage_ids() if state.reports.get(s, {}).get("duration_s") is not None
                      and state.status.get(s) == StageStatus.OK and s not in result.restored]
    result.skipped = [s for s in stage_ids() if state.status.get(s) == StageStatus.SKIPPED]
    result.run_folder = ensure_run_dir(state.config)
    if write_manifest:
        try:
            result.manifest = write_run_manifest(state)
        except Exception as error:                               # the run itself is what matters
            state.append_log(f"manifest not written: {error}")
    return result


def open_run(run_folder, *, restore: bool = True) -> PipelineState:
    """A state holding the saved results of a finished run folder, with the settings it was made with.

    The settings come from the folder's ``run_manifest.json``. Stages whose input files or settings no
    longer match what they were made from are left stale, as in :func:`run`.
    """
    folder = Path(run_folder)
    manifest = folder / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{folder} has no run_manifest.json; it is not a run folder.")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    config, _ = load_versioned(dict(data.get("config") or {}), int(data.get("config_version", CONFIG_VERSION)))
    config["_OPEN_FOLDER"] = str(folder)
    state = new_state(config)
    if restore:
        saved_results.restore(state)
    return state
