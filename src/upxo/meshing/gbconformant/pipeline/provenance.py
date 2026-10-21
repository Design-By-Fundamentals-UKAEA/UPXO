"""The command that rebuilds a deck, and the record written beside it.

``reproduce_arguments(config)`` turns the settings of a run into the arguments of the ``run`` command: the input as
a flag, the crop as ``--subset-box`` and every setting that differs from its default as ``--set KEY=VALUE``.
Settings the stages fill in while running (the spacings) and settings this package does not know are left out.
``write_provenance`` puts ``provenance.json`` into a deck folder: the command, the settings that differ from the
defaults, the seeds, the input files with their sizes and times, the crop and the package versions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upxo.meshing.gbconformant.pipeline.manifest import _subset_record, collect_environment
from upxo.meshing.gbconformant.pipeline.saved_results import input_signature
from upxo.meshing.gbconformant.pipeline.settings import (
    CONFIG_VERSION, PATH_SETTINGS, SCHEMA, clean_path_text, default_config, differences)

FORMAT = "upxo.gbconformant.provenance"
VERSION = 1
PROVENANCE_NAME = "provenance.json"
MODULE = "upxo.meshing.gbconformant.pipeline"
DERIVED = ("SPACING", "EFFECTIVE_SPACING")      # filled in by the stages, except that a labelled image's spacing is an input
FLAG_KEYS = ("DATASET_TYPE", "RAW_FOLDER", "TWINNED_FOLDER", "SAMPLE_PATH", "HIERARCHY_LEVEL", "STAGE_SUBSET",
             "SUBSET_MODE", "SUBSET_LO", "SUBSET_HI", "RUN_NAME")
INPUT_FLAGS = {"fm_steel": ("--raw-folder", "RAW_FOLDER"), "twinned_fcc": ("--twinned-folder", "TWINNED_FOLDER"),
               "labels": ("--labels", "SAMPLE_PATH")}


def _absolute(text) -> str:
    """A path as written in the command: cleaned, and made absolute when it names something that exists relative to here."""
    cleaned = clean_path_text(text)
    path = Path(cleaned)
    return str(path.resolve()) if cleaned and not path.is_absolute() and path.exists() else cleaned


def format_value(setting, value: Any) -> str:
    """A setting's value as the text ``--set`` reads back."""
    if value is None:
        return "auto"
    if setting.key in PATH_SETTINGS:
        return _absolute(value)
    if setting.kind == "bool" or isinstance(value, bool):
        return "true" if value else "false"
    if setting.kind == "any":
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return ",".join(repr(float(v)) if isinstance(v, float) else str(v) for v in value) or "auto"
    return str(value)


def changed_settings(config: dict) -> list[dict]:
    """The settings of ``config`` that differ from the defaults and that a command can set."""
    derived = DERIVED[1:] if str(config.get("DATASET_TYPE")) == "labels" else DERIVED
    return [row for row in differences(default_config(), config)
            if row["key"] in SCHEMA and row["key"] not in derived and not str(row["key"]).startswith("_")]


def reproduce_arguments(config: dict) -> list[str]:
    """The arguments of ``run`` that rebuild what ``config`` describes (without the program name)."""
    defaults = default_config()
    dataset = str(config.get("DATASET_TYPE", defaults["DATASET_TYPE"]))
    flag, key = INPUT_FLAGS.get(dataset, INPUT_FLAGS["fm_steel"])
    tokens = ["run", flag, _absolute(config.get(key))]
    if config.get("HIERARCHY_LEVEL"):
        tokens += ["--level", str(config["HIERARCHY_LEVEL"])]
    high = config.get("SUBSET_HI")
    handled = set(FLAG_KEYS)
    if config.get("STAGE_SUBSET") and high is not None:
        low = config.get("SUBSET_LO") or (0, 0, 0)
        tokens += ["--subset-box", str(config.get("SUBSET_MODE", "index"))] + [repr(float(v)) for v in (*low, *high)]
    else:
        handled -= {"STAGE_SUBSET", "SUBSET_MODE", "SUBSET_LO", "SUBSET_HI"}
    run_name = config.get("RUN_NAME")
    if run_name and run_name != defaults.get("RUN_NAME"):
        tokens += ["--run-name", str(run_name)]
    for row in changed_settings(config):
        if row["key"] in handled:
            continue
        setting = SCHEMA[row["key"]]
        tokens += ["--set", f"{row['key']}={format_value(setting, row['current'])}"]
    return tokens


def _quoted(token: str) -> str:
    if token and not any(c in token for c in ' \t"\'&|<>^'):
        return token
    return '"' + token.replace('"', '\\"') + '"'


def format_command(tokens: list[str], python: str = "python") -> str:
    """The program and its arguments as one line to paste into a terminal."""
    return " ".join([python, "-m", MODULE] + [_quoted(t) for t in tokens])


def reproduce_command(config: dict, python: str = "python") -> str:
    return format_command(reproduce_arguments(config), python)


def build_provenance(config: dict) -> dict[str, Any]:
    dataset = str(config.get("DATASET_TYPE", "fm_steel"))
    _flag, key = INPUT_FLAGS.get(dataset, INPUT_FLAGS["fm_steel"])
    tokens = reproduce_arguments(config)
    return {
        "format": FORMAT,
        "version": VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "command": format_command(tokens),
        "arguments": tokens,
        "config_version": CONFIG_VERSION,
        "settings_changed_from_defaults": [
            {"key": r["key"], "default": r["default"], "value": r["current"]} for r in changed_settings(config)],
        "seeds": {"grain_id_seed": config.get("GRAIN_ID_SEED"), "identity_seed": config.get("IDENTITY_SEED")},
        "input": {"dataset_type": dataset, "folder_or_file": _absolute(config.get(key)),
                  "files": input_signature(config)},
        "subset": _subset_record(config),
        "environment": collect_environment(),
    }


def write_provenance(folder, config: dict) -> Path:
    path = Path(folder) / PROVENANCE_NAME
    path.write_text(json.dumps(build_provenance(config), indent=2, default=str), encoding="utf8")
    return path


def command_from_folder(path) -> str:
    """The command stored with a deck (``provenance.json``), or rebuilt from the settings in a run folder's manifest."""
    path = Path(path)
    stored = path / PROVENANCE_NAME
    if stored.is_file():
        return json.loads(stored.read_text(encoding="utf8"))["command"]
    manifest = path / "run_manifest.json"
    if manifest.is_file():
        from upxo.meshing.gbconformant.pipeline.settings import coerce_config
        config, _problems = coerce_config(json.loads(manifest.read_text(encoding="utf8")).get("config") or {})
        return reproduce_command(config)
    raise FileNotFoundError(f"{path} holds neither {PROVENANCE_NAME} nor run_manifest.json.")
