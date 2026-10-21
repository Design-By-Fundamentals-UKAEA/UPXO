"""Connects a pipeline run to the checkpoint store of its run folder.

Settings that decide a stage's results (its own settings, its enable flag, the
input dataset, the subset box, the seeds, the backend) go into a signature that
chains through the stages, so a restore trusts a stage only while everything up
to it is unchanged.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from upxo.meshing.gbconformant.pipeline.checkpoints import (
    CheckpointStore, chain_signatures)
from upxo.meshing.gbconformant.pipeline.runfiles import ensure_run_dir
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, clean_path_text, coerce_value, settings_of_stage
from upxo.meshing.gbconformant.pipeline.stages import stage_def, stage_ids
from upxo.meshing.gbconformant.pipeline.state import StageStatus

FLAG = "CHECKPOINT_RESULTS"


def stage_order() -> list[str]:
    return stage_ids()


def input_signature(config: dict) -> list:
    """Name, size and modification time of the input files (not their contents)."""
    paths: list[Path] = []
    dataset = str(config.get("DATASET_TYPE", "fm_steel"))
    if dataset in ("fm_steel", "twinned_fcc"):
        folder = Path(clean_path_text(config.get("RAW_FOLDER" if dataset == "fm_steel" else "TWINNED_FOLDER")))
        if folder.is_dir():
            paths = sorted(p for p in folder.iterdir()
                           if p.suffix in (".npy", ".pkl", ".npz") and p.is_file())
    else:
        sample = Path(clean_path_text(config.get("SAMPLE_PATH")))
        paths = [sample] if sample.is_file() else []
    out = []
    for path in paths:
        info = path.stat()
        out.append([path.name, info.st_size, info.st_mtime_ns])
    return out


def effective_level(config: dict) -> str:
    """The hierarchy level that will be (or was) loaded: the chosen one, else the finest."""
    level = str(config.get("HIERARCHY_LEVEL") or "")
    if level:
        return level
    folder = Path(clean_path_text(config.get("RAW_FOLDER")))
    if not folder.is_dir():
        return ""
    from upxo.meshing.gbconformant.fm_steel_abaqus import detect_levels
    levels = detect_levels(folder)
    return levels[-1] if levels else ""


def own_part(config: dict, stage_id: str) -> dict:
    stage = stage_def(stage_id)
    if stage.optional_flag and not config.get(stage.optional_flag, True):
        return {stage.optional_flag: False}      # switched off: its other settings do not matter
    keys = set(settings_of_stage(stage_id))
    if stage.optional_flag:
        keys.add(stage.optional_flag)
    if stage.backend != "none":
        keys.add("BACKEND")
    part = {key: config.get(key) for key in sorted(keys)}
    for key in list(part):
        setting = SCHEMA.get(key)
        if setting is not None and setting.signed_when_changed and (part[key] is None or part[key] == setting.default):
            del part[key]                            # at its default: as if the setting did not exist
    if stage_id == "load_lgi":
        part["input_files"] = input_signature(config)
        if str(config.get("DATASET_TYPE", "fm_steel")) == "fm_steel":
            part["HIERARCHY_LEVEL"] = effective_level(config)   # the load stage fills it in
            part.pop("SPACING", None)                           # and fixes the spacing at 1
    return part


def signatures(config: dict) -> dict:
    return chain_signatures(stage_order(), lambda sid: own_part(config, sid))


def store_for(state) -> CheckpointStore:
    return CheckpointStore(ensure_run_dir(state.config),
                           keep_history=bool(state.config.get("CHECKPOINT_KEEP_HISTORY", True)))


def enable(state) -> None:
    """Switch checkpointing on for ``state`` (when its settings allow it)."""
    state.checkpoints_enabled = bool(state.config.get(FLAG, True))
    state.on_invalidate = lambda stages: discard(state, stages)


def discard(state, stages) -> None:
    if not getattr(state, "checkpoints_enabled", False):
        return
    order = stage_order()
    dropped = [s for s in stages if s in set(order)]
    try:
        store_for(state).discard(dropped)
        if dropped:
            rewind(state, min(dropped, key=order.index))
    except Exception as error:                      # never block an edit on disk trouble
        state.append_log(f"checkpoint cleanup failed: {error}")


def _status_text(state, stage_id: str) -> str:
    return "Skipped" if state.status.get(stage_id) == StageStatus.SKIPPED else "OK"


def config_changes(before: dict, after: dict) -> dict:
    """The settings whose value differs between ``before`` and ``after`` (a stage changed them)."""
    return {key: value for key, value in after.items()
            if key not in before or before[key] != value}


def apply_config_changes(config: dict, changes: dict) -> None:
    """Set the settings a stage had changed, each brought to its type."""
    for key, value in changes.items():
        setting = SCHEMA.get(key)
        if setting is not None:
            checked, problem = coerce_value(setting, value)
            if problem is None:
                value = checked
        config[key] = value


def save_after_run(state, stage_id: str, changes: dict | None = None) -> None:
    """Write the results ``stage_id`` just produced (and any finished upstream stage still unsaved).

    ``changes`` are the settings the stage changed while running.
    """
    if not getattr(state, "checkpoints_enabled", False) or stage_id not in stage_order():
        return
    try:
        sigs = signatures(state.config)
        store = store_for(state)
        for earlier in stage_order():
            if earlier == stage_id:
                break
            if store.meta(earlier) is None and state.status.get(earlier) in (
                    StageStatus.OK, StageStatus.SKIPPED):
                store.save_stage(state, earlier, sigs[earlier], _status_text(state, earlier))
        meta = store.save_stage(state, stage_id, sigs[stage_id], _status_text(state, stage_id),
                                config_changes=changes)
        if meta["skipped"]:
            state.append_log(f"checkpoint {stage_id}: not stored: {', '.join(meta['skipped'])}")
    except Exception as error:
        state.append_log(f"checkpoint failed for {stage_id}: {error}")


def save_after_skip(state, stage_id: str, changes: dict | None = None) -> None:
    save_after_run(state, stage_id, changes)


def restore(state, before: str | None = None) -> dict:
    """Load every still-valid stage of the state's run folder; returns the store's summary.

    With ``before`` only the stages ahead of that stage are loaded.
    """
    sigs = signatures(state.config)
    store = store_for(state)

    def set_status(stage_id, text):
        state.set_status(stage_id, {"OK": StageStatus.OK, "Skipped": StageStatus.SKIPPED,
                                    "Stale": StageStatus.STALE}[text])

    def set_config(stage_id, changes):
        apply_config_changes(state.config, changes)

    order = stage_order()
    if before is not None:
        order = order[:order.index(before)]
    summary = store.restore(state, order, sigs, set_status, set_config)
    stamp = datetime.now().isoformat(timespec="seconds")
    for line in summary["problems"]:
        state.append_log(f"restore: {line}")
        stage, _, reason = line.partition(": ")
        if stage in state.status and state.status[stage] == StageStatus.STALE:
            state.stale_reasons[stage] = {"reason": reason, "source": stage, "at": stamp}
    return summary


def load_stage(state, stage_id: str, run_folder=None) -> None:
    """Take the saved results of ``stage_id`` into ``state`` (its status, report, artefacts and settings changes).

    The saved signature must match the state's current settings. ``run_folder`` is where they
    were saved when that differs from the state's own run folder (a stage that crops the input
    starts a folder of its own).
    """
    store = store_for(state) if run_folder is None else CheckpointStore(
        run_folder, keep_history=bool(state.config.get("CHECKPOINT_KEEP_HISTORY", True)))
    meta = store.meta(stage_id)
    if meta is None:
        raise FileNotFoundError(f"No saved results for {stage_id}.")
    if meta.get("signature") != signatures(state.config)[stage_id]:
        raise ValueError(f"The saved results of {stage_id} were made with other settings.")
    artefacts, report = store._read_stage(stage_id, meta)
    for key, value in artefacts.items():
        state.artefacts[key] = value
        state.artefact_writer[key] = stage_id
    if report:
        state.reports[stage_id] = report
    apply_config_changes(state.config, meta.get("config") or {})
    state.set_status(stage_id, StageStatus.SKIPPED if meta.get("status") == "Skipped" else StageStatus.OK)


def missing_prerequisites(state, stage_id: str) -> list[str]:
    """Earlier stages whose saved results a fresh process could not take up from.

    A stage is listed when it has not been run (and is not switched off), or when it ran but
    its saved results are missing, out of date or incomplete. A fresh process can take up from
    ``stage_id`` only when this list is empty.
    """
    sigs = signatures(state.config)
    store = store_for(state)
    missing = []
    for earlier in stage_order():
        if earlier == stage_id:
            break
        stage = stage_def(earlier)
        if state.status.get(earlier) not in (StageStatus.OK, StageStatus.SKIPPED):
            if stage.optional_flag and not state.config.get(stage.optional_flag, True):
                continue                                  # switched off: the process skips it itself
            missing.append(earlier)
            continue
        meta = store.meta(earlier)
        if (meta is None or meta.get("signature") != sigs[earlier] or meta.get("skipped")
                or (not store.keep_history and meta.get("superseded"))):
            missing.append(earlier)
    return missing


def rewind(state, stage_id: str) -> list[str]:
    """Bring back what the stages before ``stage_id`` left, after ``stage_id`` and the later stages were dropped.

    Stages overwrite shared results in turn (the labels, the surface, the tetrahedra), so dropping a stage's
    results also drops the newest copy of those. The copy the stage before it saved is read back from the run
    folder, so ``stage_id`` can run again from where the earlier stages left off. Returns the names restored.
    """
    store = store_for(state)
    order = stage_order()
    sigs = signatures(state.config)
    owner: dict[str, str] = {}
    for earlier in order[:order.index(stage_id)]:
        meta = store.meta(earlier)
        if meta is None or meta.get("signature") != sigs[earlier]:
            break                                   # what follows was not built on these settings
        for key in meta.get("artefacts", {}):
            owner[key] = earlier
    wanted: dict[str, list[str]] = {}
    for key, earlier in owner.items():
        if key not in state.artefacts or state.artefact_writer.get(key) != earlier:
            wanted.setdefault(earlier, []).append(key)
    brought = []
    for earlier in order:
        if earlier not in wanted:
            continue
        for key, value in store.load_keys(earlier, wanted[earlier]).items():
            state.artefacts[key] = value
            state.artefact_writer[key] = earlier
            brought.append(key)
    return brought
