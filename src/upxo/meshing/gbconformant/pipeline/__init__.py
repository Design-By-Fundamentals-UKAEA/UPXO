"""Run the grain-conformal tetrahedral meshing pipeline, and save and restore its stage results.

    from upxo.meshing.gbconformant.pipeline import run
    result = run(RAW_FOLDER="path/to/raw_export", HIERARCHY_LEVEL="blk", through="export_abaqus")

``run`` applies the stages in order (see :data:`STAGES`), saving each stage's results in the
run folder in portable files; a second ``run`` of the same settings restores them and runs
only what changed. ``python -m upxo.meshing.gbconformant.pipeline --help`` lists the commands.
"""
from .checkpoints import CheckpointStore, canonical_signature, chain_signatures
from .run import RunResult, run
from .serialize import FORMAT, VERSION, SerializeError, read_value, write_value
from .settings import (
    CONFIG_VERSION, SCHEMA, SETTINGS, Setting, SettingsError, build_config, default_config,
    load_versioned, parse_assignment)
from .stages import STAGES, StageDef, stage_def, stage_ids
from .state import PipelineState, StageStatus

__all__ = [
    'CONFIG_VERSION', 'CheckpointStore', 'FORMAT', 'PipelineState', 'RunResult', 'SCHEMA',
    'SETTINGS', 'STAGES', 'SerializeError', 'Setting', 'SettingsError', 'StageDef',
    'StageStatus', 'VERSION', 'build_config', 'canonical_signature', 'chain_signatures',
    'default_config', 'load_versioned', 'parse_assignment', 'read_value', 'run', 'stage_def',
    'stage_ids', 'write_value',
]
