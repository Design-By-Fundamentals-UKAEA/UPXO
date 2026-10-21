"""Command line: run the conformal meshing pipeline, list its stages and settings, inspect a run.

    python -m upxo.meshing.gbconformant.pipeline stages
    python -m upxo.meshing.gbconformant.pipeline settings [--stage STAGE] [--json]
    python -m upxo.meshing.gbconformant.pipeline run --raw-folder FOLDER [--preset NAME] [--set KEY=VALUE ...]
    python -m upxo.meshing.gbconformant.pipeline presets [NAME] [--json]
    python -m upxo.meshing.gbconformant.pipeline estimate (--voxels N | --shape NX NY NZ) [--preset NAME] [--set ...]
    python -m upxo.meshing.gbconformant.pipeline status RUN_FOLDER
    python -m upxo.meshing.gbconformant.pipeline report RUN_FOLDER [--sections ...] [--no-figures]
    python -m upxo.meshing.gbconformant.pipeline reproduce DECK_OR_RUN_FOLDER
    python -m upxo.meshing.gbconformant.pipeline datacheck DECK_FOLDER [--abaqus COMMAND] [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from upxo.meshing.gbconformant.pipeline.settings import (
    SETTINGS, SettingsError, load_versioned, parse_assignment)
from upxo.meshing.gbconformant.pipeline.stages import STAGES, stage_ids

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _out(text: str = "") -> None:
    print(text, flush=True)


def _fmt(value) -> str:
    if value is None:
        return "auto"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value) if value else "(none)"
    return str(value)


# --- stages ------------------------------------------------------------------------------------------------
def cmd_stages(args) -> int:
    width = max(len(s.id) for s in STAGES)
    last_group = None
    for stage in STAGES:
        if stage.group != last_group:
            _out(f"\n{stage.group}")
            last_group = stage.group
        flag = f"  (switch: {stage.optional_flag})" if stage.optional_flag else ""
        _out(f"  {stage.id:<{width}}  {stage.title}{flag}")
    return EXIT_OK


# --- settings ------------------------------------------------------------------------------------------------
def _setting_row(setting) -> dict:
    return {"name": setting.key, "default": setting.default, "type": setting.kind,
            "minimum": setting.minimum, "maximum": setting.maximum,
            "choices": list(setting.choices), "auto_allowed": bool(setting.auto),
            "stages": list(setting.stages)}


def cmd_settings(args) -> int:
    rows = [s for s in SETTINGS if not args.stage or args.stage in s.stages]
    if args.stage and args.stage not in stage_ids():
        _out(f"{args.stage} is not a stage; see the list of stages.")
        return EXIT_USAGE
    if args.json:
        _out(json.dumps([_setting_row(s) for s in rows], indent=2, default=list))
        return EXIT_OK
    width = max((len(s.key) for s in rows), default=10)
    for s in rows:
        limits = ""
        if s.choices:
            limits = " | ".join(str(c) for c in s.choices)
        elif s.minimum is not None or s.maximum is not None:
            limits = f"[{'' if s.minimum is None else s.minimum}, {'' if s.maximum is None else s.maximum}]"
        auto = " (auto allowed)" if s.auto else ""
        _out(f"{s.key:<{width}}  {s.kind:<10} default {_fmt(s.default):<14} {limits}{auto}")
    return EXIT_OK


# --- run -------------------------------------------------------------------------------------------------------
def _read_settings_file(path: Path) -> dict:
    """Settings from a JSON file: a flat object, or one holding ``config`` (and ``config_version``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SettingsError(f"{path}: expected a JSON object of settings.")
    version = int(data.get("config_version", 2)) if "config" in data else 2
    loaded = dict(data.get("config", data))
    config, problems = load_versioned(loaded, version)
    for line in problems:
        print(f"settings file: {line}", file=sys.stderr)
    # Only what the file states is an override; the rest stays default.
    return {key: config[key] for key in loaded if key in config}


def _collect_settings(args) -> dict:
    settings: dict = {}
    if args.settings:
        settings.update(_read_settings_file(Path(args.settings)))
    if getattr(args, "preset", None):
        from upxo.meshing.gbconformant.pipeline import presets
        try:
            settings.update(presets.values_of(args.preset))
        except KeyError as error:
            raise SettingsError(error.args[0]) from None
    if getattr(args, "raw_folder", None):
        settings.update(DATASET_TYPE="fm_steel", RAW_FOLDER=str(Path(args.raw_folder).expanduser()))
    if getattr(args, "twinned_folder", None):
        settings.update(DATASET_TYPE="twinned_fcc", TWINNED_FOLDER=str(Path(args.twinned_folder).expanduser()))
    if getattr(args, "labels", None):
        settings.update(DATASET_TYPE="labels", SAMPLE_PATH=str(Path(args.labels).expanduser()))
    if getattr(args, "level", None):
        settings["HIERARCHY_LEVEL"] = args.level
    if getattr(args, "run_name", None):
        settings["RUN_NAME"] = args.run_name
    if getattr(args, "subset_box", None):
        mode, *numbers = args.subset_box
        if len(numbers) != 6:
            raise SettingsError("--subset-box needs MODE and six numbers: lo x y z, hi x y z.")
        settings.update(STAGE_SUBSET=True, SUBSET_MODE=mode,
                        **dict(zip(("SUBSET_LO", "SUBSET_HI"),
                                   (_numbers(numbers[:3]), _numbers(numbers[3:])))))
    for text in args.set or ():
        key, value = parse_assignment(text)
        settings[key] = value
    return settings


def _numbers(items) -> list:
    try:
        return [float(v) for v in items]
    except ValueError:
        raise SettingsError(f"{items} are not numbers.") from None


class _Printer:
    """Prints one line per stage event; progress messages only when verbose."""

    def __init__(self, verbose: bool, quiet: bool, silent: bool = False):
        self.verbose, self.quiet, self.silent = verbose, quiet, silent
        self.order = stage_ids()

    def __call__(self, event: dict) -> None:
        kind = event["event"]
        if self.silent or (self.quiet and kind != "failed"):
            return
        stage = event.get("stage")
        position = f"[{self.order.index(stage) + 1:>2}/{len(self.order)}]" if stage in self.order else ""
        title = event.get("title", stage)
        if kind == "start":
            _out(f"{position} {title} ...")
        elif kind == "done":
            seconds = event.get("seconds")
            _out(f"{position} {title}: done" + ("" if seconds is None else f" in {seconds:.1f} s"))
        elif kind == "restored":
            _out(f"{position} {title}: restored from saved results")
        elif kind == "stale":
            _out(f"{position} {title}: saved results out of date, will run again")
        elif kind == "skipped":
            _out(f"{position} {title}: skipped ({event.get('reason')})")
        elif kind == "failed":
            _out(f"{position} {title}: FAILED: {event.get('error')}")
        elif kind == "progress" and self.verbose:
            _out(f"        {event.get('message')}")


def cmd_run(args) -> int:
    if args.data_dir:
        os.environ["UPXO_CONFORMAL_DATA"] = str(Path(args.data_dir).expanduser())
    try:
        settings = _collect_settings(args)
        if args.run_folder:
            settings["_RUN_FOLDER"] = str(Path(args.run_folder).expanduser())
        from upxo.meshing.gbconformant.pipeline.run import run
        result = run(settings, through=args.through, resume=not args.no_resume,
                     on_event=_Printer(args.verbose, args.quiet, silent=args.json))
    except (SettingsError, KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return EXIT_USAGE
    summary = {
        "ok": result.ok, "run_folder": str(result.run_folder), "through": result.through,
        "restored": result.restored, "ran": result.ran, "skipped": result.skipped,
        "failed_stage": result.failed_stage, "error": result.error,
        "suggestions": [{"setting": s.setting, "value": s.value, "why": s.why} for s in result.suggestions],
        "export_folder": str(result.export_folder() or ""),
        "manifest": str(result.manifest or ""),
    }
    if args.json:
        _out(json.dumps(summary, indent=2))
    elif not args.quiet or not result.ok:
        _out()
        _out(f"run folder: {result.run_folder}")
        if result.export_folder():
            _out(f"export:     {result.export_folder()}")
        if result.manifest:
            _out(f"manifest:   {result.manifest}")
        if not result.ok:
            _out(f"stopped at {result.failed_stage}: {result.error}")
            if result.suggestions:
                _out("settings worth trying (add to the same command, the earlier stages are restored):")
                for item in result.suggestions:
                    _out(f"  {item.text()}" + ("" if item.setting is None else
                                                 f"    --set {item.setting}={'auto' if item.value is None else item.value}"))
    return EXIT_OK if result.ok else EXIT_FAILED


# --- status ------------------------------------------------------------------------------------------------------
def cmd_status(args) -> int:
    from upxo.meshing.gbconformant.pipeline.checkpoints import CheckpointStore

    folder = Path(args.run_folder).expanduser()
    if not folder.is_dir():
        print(f"error: {folder} is not a folder.", file=sys.stderr)
        return EXIT_USAGE
    store = CheckpointStore(folder)
    saved = store.stages()
    if not saved:
        _out(f"{folder}: no saved stage results.")
        return EXIT_OK
    total = 0
    _out(f"{folder}")
    for stage in stage_ids():
        if stage not in saved:
            continue
        meta = store.meta(stage) or {}
        size = int(meta.get("bytes") or 0)
        total += size
        _out(f"  {stage:<22} {meta.get('status', '?'):<8} {size / 1e6:8.2f} MB  {meta.get('written_utc', '')}")
    _out(f"  {'total':<22} {'':<8} {total / 1e6:8.2f} MB")
    return EXIT_OK


# --- report ------------------------------------------------------------------------------------------------------
def cmd_report(args) -> int:
    from upxo.meshing.gbconformant.pipeline import results
    from upxo.meshing.gbconformant.pipeline.run import open_run
    from upxo.reporting import render_html

    try:
        state = open_run(args.run_folder)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    sections = [s.strip() for s in args.sections.split(",") if s.strip()] if args.sections else list(results.SECTIONS)
    try:
        session = results.report_session(state)
        added = results.populate_report(session, state, sections, figures=not args.no_figures)
    except KeyError as error:
        print(f"error: {error.args[0]}", file=sys.stderr)
        return EXIT_USAGE
    session.save()
    path = render_html(session)
    _out(f"sections: {', '.join(added) or '(nothing to report yet)'}")
    _out(f"report:   {path}")
    return EXIT_OK


# --- presets and estimate ----------------------------------------------------------------------------------------
def cmd_presets(args) -> int:
    from upxo.meshing.gbconformant.pipeline import presets

    if args.name:
        try:
            found = [presets.get_preset(args.name)]
        except KeyError as error:
            print(f"error: {error.args[0]}", file=sys.stderr)
            return EXIT_USAGE
    else:
        found = list(presets.PRESETS.values())
    if args.json:
        _out(json.dumps([{"name": p.key, "title": p.title, "material": p.material, "size_class": p.size_class,
                          "summary": p.summary, "settings": p.settings, "notes": list(p.notes),
                          "source": p.source, "measured": p.measured} for p in found], indent=2))
        return EXIT_OK
    for preset in found:
        _out(f"{preset.key}  [{preset.material}, {preset.size_class}]")
        for line in preset.describe().splitlines():
            _out(f"    {line}")
        _out()
    return EXIT_OK


def cmd_estimate(args) -> int:
    from upxo.meshing.gbconformant.pipeline.estimate import forecast, load_observations
    from upxo.meshing.gbconformant.pipeline.settings import build_config

    if args.data_dir:
        os.environ["UPXO_CONFORMAL_DATA"] = str(Path(args.data_dir).expanduser())
    voxels = args.voxels if args.voxels else (args.shape[0] * args.shape[1] * args.shape[2] if args.shape else None)
    if not voxels or voxels < 1:
        print("error: give --voxels N or --shape NX NY NZ.", file=sys.stderr)
        return EXIT_USAGE
    try:
        config = build_config(_collect_settings(args))
    except (SettingsError, KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return EXIT_USAGE
    result = forecast(int(voxels), config, load_observations())
    if args.json:
        _out(json.dumps(result.as_dict(), indent=2))
    else:
        _out(f"{int(voxels):,} voxels")
        _out(result.text())
    return EXIT_OK


# --- reproduce and data check ----------------------------------------------------------------------------------------
def cmd_reproduce(args) -> int:
    from upxo.meshing.gbconformant.pipeline.provenance import command_from_folder

    try:
        _out(command_from_folder(args.folder))
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error.args[0] if isinstance(error, KeyError) else error}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def cmd_datacheck(args) -> int:
    """Run Abaqus's data check on a deck folder, with the job settings the deck was written with."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.jobs import run_datacheck
    from upxo.meshing.gbconformant.pipeline.settings import coerce_config
    from upxo.meshing.gbconformant.pipeline.stage_output import deck_settings

    folder = Path(args.folder).expanduser()
    if not (folder / "model_master.inp").is_file():
        print(f"error: {folder} holds no model_master.inp.", file=sys.stderr)
        return EXIT_USAGE
    try:
        info = json.loads((folder / "deck_info.json").read_text(encoding="utf8"))
        config, _ = coerce_config({})
        saved = info.get("settings") or {}
        settings = deck_settings(config)
        for name in ("abaqus_command", "umat_file", "job_name", "job_cpus", "job_memory_gb", "job_time_hours", "slurm_partition"):
            if name in saved:
                setattr(settings, name, saved[name])
        result = run_datacheck(folder, settings, command=args.abaqus, timeout=args.timeout)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    _out(f"data check: {'passed' if result['ok'] else 'FAILED'} ({result['seconds']} s)" if result["seconds"] is not None
         else f"data check not run: {result['problem']}")
    for line in result["errors"][:20]:
        _out(f"  {line}")
    if result["warnings"]:
        _out(f"  {len(result['warnings'])} warning(s); see abaqus_datacheck.json")
    return EXIT_OK if result["ok"] else EXIT_FAILED


# --- parser -------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m upxo.meshing.gbconformant.pipeline",
        description="Grain-conformal tetrahedral meshing: run it, list its stages and settings, inspect a run.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="list the stages in run order").set_defaults(func=cmd_stages)

    p = sub.add_parser("settings", help="list the settings with their defaults, types and limits")
    p.add_argument("--stage", help="only the settings of this stage")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_settings)

    p = sub.add_parser("run", help="run the pipeline")
    p.add_argument("--raw-folder", help="FM-steel raw export folder (sets the dataset type to fm_steel)")
    p.add_argument("--twinned-folder", help="twinned FCC structure folder: labels, orientations, twin roles and "
                   "parents (sets the dataset type to twinned_fcc; voxel size and units: --set TWINNED_VOXEL_SIZE=..., "
                   "--set TWINNED_UNITS=...)")
    p.add_argument("--labels", help="one labelled image, a 3-D integer .npy (sets the dataset type to labels)")
    p.add_argument("--level", help="hierarchy level to mesh: pag, pck, blk or sblk (default: the finest found)")
    p.add_argument("--settings", help="JSON file of settings")
    p.add_argument("--preset", help="start from a built-in preset (see the presets command); --set changes win")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="set one setting (repeatable); 'auto' leaves an automatic setting automatic")
    p.add_argument("--subset-box", nargs=7, metavar=("MODE", "X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="crop the input first: MODE is index, percent or physical")
    p.add_argument("--through", default="export_abaqus", help="last stage to run (default: export_abaqus)")
    p.add_argument("--run-name", help="name of the run folder")
    p.add_argument("--data-dir", help="folder that holds the runs (default: the data directory of the checkout)")
    p.add_argument("--run-folder", help="resume a subset run held in this folder")
    p.add_argument("--no-resume", action="store_true", help="ignore saved results and run every stage")
    p.add_argument("-v", "--verbose", action="store_true", help="print the progress messages of each stage")
    p.add_argument("-q", "--quiet", action="store_true", help="print only failures")
    p.add_argument("--json", action="store_true", help="print the outcome as JSON")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("presets", help="list the built-in presets, or describe one")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("estimate", help="forecast the tetrahedra, time and memory of a crop before running it")
    p.add_argument("--voxels", type=int, help="number of voxels of the crop")
    p.add_argument("--shape", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="shape of the crop in voxels")
    p.add_argument("--preset", help="forecast under a built-in preset")
    p.add_argument("--settings", help="JSON file of settings")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="set one setting (repeatable)")
    p.add_argument("--data-dir", help="folder that holds the earlier runs the forecast learns from")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("report", help="write the HTML report of a finished run folder")
    p.add_argument("run_folder")
    p.add_argument("--sections", help="comma-separated: summary, settings, quality, fidelity, deliverables (default: all)")
    p.add_argument("--no-figures", action="store_true", help="tables and text only")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("reproduce", help="print the command that rebuilds a deck (from its provenance.json) or a run")
    p.add_argument("folder", help="a deck folder or a run folder")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("datacheck", help="run the Abaqus data check on a deck folder (needs Abaqus)")
    p.add_argument("folder", help="the deck folder (it holds model_master.inp)")
    p.add_argument("--abaqus", help="the Abaqus command if it is not the one the deck was written with")
    p.add_argument("--timeout", type=int, default=3600, help="seconds to wait for the check (default 3600)")
    p.set_defaults(func=cmd_datacheck)

    p = sub.add_parser("status", help="list the saved stage results of a run folder")
    p.add_argument("run_folder")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
