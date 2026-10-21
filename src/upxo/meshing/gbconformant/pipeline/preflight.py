"""What a stage needs before it can run: stored results and non-empty settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from upxo.meshing.gbconformant.pipeline.stages import has_stage, stage_def
from upxo.meshing.gbconformant.pipeline.state import PipelineState

# What to do about a missing artefact.
_ARTEFACT_HINTS = {
    "labels": "Load labelled grain image (Load LGI) first.",
    "interfaces": "Run Smooth interfaces first.",
    "interfaces_smoothed": "Run Smooth interfaces first.",
    "gmsh_result": "Run Gmsh remesh interfaces first.",
    "closed_rve": "Run Close RVE first.",
    "remeshed_rve": "Run Joint remesh first.",
    "repaired_rve": "Run surface repair / validation path first.",
    "verified_tets": "Run Tetrahedral meshing first.",
}


@dataclass(frozen=True)
class StageRequirements:
    """What must exist before a stage can run."""

    artefacts: tuple[str, ...] = ()
    """Artefact keys; alternatives of which any one will do are joined with ``|``."""
    config_nonempty: tuple[str, ...] = ()
    message: str = ""


# Stages not listed have no requirement beyond what their handler checks itself.
_REQUIREMENTS: dict[str, StageRequirements] = {
    "load_lgi": StageRequirements(
        config_nonempty=("SAMPLE_PATH",),
        message="Set a SAMPLE_PATH (.npy labelled grain image) before loading.",
    ),
    "subset_extract": StageRequirements(artefacts=("labels",)),
    "grain_id_map": StageRequirements(artefacts=("labels",)),
    "small_grains": StageRequirements(artefacts=("labels",)),
    "spikes": StageRequirements(artefacts=("labels",)),
    "pinches": StageRequirements(artefacts=("labels",)),
    "voxel_topology": StageRequirements(artefacts=("labels",)),
    "freeze_thin": StageRequirements(artefacts=("labels",)),
    "upscale": StageRequirements(artefacts=("labels",)),
    "smooth_interfaces": StageRequirements(artefacts=("labels",)),
    "geometry_guard": StageRequirements(
        artefacts=("interfaces|interfaces_smoothed",),
    ),
    "guard_relax": StageRequirements(artefacts=("interfaces",)),
    "unpinch": StageRequirements(artefacts=("interfaces",)),
    "gmsh_interfaces": StageRequirements(artefacts=("interfaces",)),
    "close_rve": StageRequirements(artefacts=("gmsh_result",)),
    "pre_joint_quality": StageRequirements(artefacts=("closed_rve",)),
    "joint_remesh": StageRequirements(artefacts=("closed_rve",)),
    "pre_tet_validation": StageRequirements(artefacts=("remeshed_rve",)),
    "contact_repair": StageRequirements(artefacts=("remeshed_rve|repaired_rve",)),
    "crossing_repair": StageRequirements(artefacts=("repaired_rve",)),
    "surface_repair": StageRequirements(artefacts=("repaired_rve",)),
    "surface_gate": StageRequirements(artefacts=("repaired_rve",)),
    "tet_mesh": StageRequirements(artefacts=("repaired_rve",)),
    "tet_smooth": StageRequirements(artefacts=("verified_tets", "repaired_rve")),
    "tet_swaps": StageRequirements(artefacts=("verified_tets", "repaired_rve")),
    "tet_smooth_after": StageRequirements(artefacts=("verified_tets", "repaired_rve")),
    "dihedral_gate": StageRequirements(artefacts=("verified_tets", "repaired_rve")),
    "insert_nodes": StageRequirements(artefacts=("verified_tets", "repaired_rve")),
    "region_quality": StageRequirements(artefacts=("verified_tets",)),
    "compare_before_after": StageRequirements(artefacts=("verified_tets",)),
    "export_surfaces": StageRequirements(),
    "export_volume": StageRequirements(artefacts=("verified_tets",)),
    "export_abaqus": StageRequirements(artefacts=("verified_tets",)),
}


@dataclass
class PreflightIssue:
    stage_id: str
    message: str
    missing_artefacts: list[str] = field(default_factory=list)
    missing_config: list[str] = field(default_factory=list)


def _alternatives(spec: str) -> list[str]:
    return [a.strip() for a in spec.split("|") if a.strip()]


def _artefact_present(state: PipelineState, spec: str) -> bool:
    """``spec`` is ``name`` or ``alt1|alt2`` (any present)."""
    return any(state.get_artefact(a) is not None for a in _alternatives(spec))


def _artefact_missing_label(spec: str) -> str:
    return " or ".join(_alternatives(spec))


def check_stage_preflight(
    state: PipelineState, stage_id: str,
) -> Optional[PreflightIssue]:
    """Return an issue if the stage should not run yet; else ``None``."""
    req = _REQUIREMENTS.get(stage_id)
    if req is None:
        return None
    dataset = state.config.get("DATASET_TYPE", "fm_steel")
    fm_input = stage_id == "load_lgi" and dataset == "fm_steel"
    twinned_input = stage_id == "load_lgi" and dataset == "twinned_fcc"

    missing_art = [_artefact_missing_label(spec) for spec in req.artefacts
                   if not _artefact_present(state, spec)]
    config_keys = ("RAW_FOLDER",) if fm_input else ("TWINNED_FOLDER",) if twinned_input else req.config_nonempty
    missing_cfg = []
    for key in config_keys:
        val = state.config.get(key)
        if val is None or (isinstance(val, str) and not str(val).strip()):
            missing_cfg.append(key)
    if not missing_art and not missing_cfg:
        return None

    message_text = ("Set RAW_FOLDER (an FM-steel raw export folder) before loading." if fm_input else
                    "Set TWINNED_FOLDER (a twinned FCC structure folder) before loading." if twinned_input
                    else req.message)
    parts: list[str] = []
    if message_text and missing_cfg:
        parts.append(message_text)
    for name in missing_art:
        alternatives = name.split(" or ")
        hint = next((text for key, text in _ARTEFACT_HINTS.items() if key in alternatives), None)
        parts.append(f"Need {name} — {hint}" if hint else f"Need artefact: {name}")
    if missing_cfg and not message_text:
        parts.append("Missing config: " + ", ".join(missing_cfg))
    elif missing_cfg and message_text and message_text not in parts:
        parts.append("Missing: " + ", ".join(missing_cfg))

    title = stage_def(stage_id).title if has_stage(stage_id) else stage_id
    return PreflightIssue(
        stage_id=stage_id,
        message=f"{title}: " + " ".join(parts),
        missing_artefacts=missing_art,
        missing_config=missing_cfg,
    )


def format_preflight_issue(issue: PreflightIssue) -> str:
    return issue.message
