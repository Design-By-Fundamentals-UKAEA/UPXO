"""The stages of the grain-conformal tetrahedral meshing pipeline, in run order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

COMPUTE = "compute"
EXPORT = "export"


@dataclass(frozen=True)
class StageDef:
    """One stage that computes or writes something."""

    id: str
    title: str
    group: str
    kind: str = COMPUTE
    optional_flag: Optional[str] = None
    """Setting that switches the stage off when False; the stage is then skipped."""
    backend: str = "d3v2p0"
    """``d3v2p0`` or ``d3v2p1`` (the algorithm generation); ``none`` when no backend choice applies."""


def _c(stage_id, title, group, flag=None, backend="d3v2p0", kind=COMPUTE) -> StageDef:
    return StageDef(stage_id, title, group, kind, flag, backend)


STAGES: tuple[StageDef, ...] = (
    # Input
    _c("load_lgi", "Load labelled grain image", "Input", backend="none"),
    _c("subset_extract", "Subset extract", "Input", "STAGE_SUBSET", "none"),
    _c("grain_id_map", "Grain ID mapping", "Input", "STAGE_GRAIN_ID_SHUFFLE", "none"),
    # Cleaning
    _c("small_grains", "Small-grain removal", "Cleaning", "STAGE_SMALL_GRAINS"),
    _c("spikes", "Spike removal", "Cleaning", "STAGE_SPIKES"),
    _c("pinches", "Pinch cleaning", "Cleaning", "STAGE_PINCHES"),
    _c("voxel_topology", "Voxel topology cleaning", "Cleaning", "STAGE_VOXEL_TOPOLOGY"),
    _c("freeze_thin", "Freeze thin grains", "Cleaning", "STAGE_FREEZE_THIN"),
    _c("upscale", "Resolution upscale", "Cleaning", "STAGE_UPSCALE"),
    # Surface
    _c("smooth_interfaces", "Extract + smooth interfaces", "Surface", backend="d3v2p1"),
    _c("geometry_guard", "Geometry guard", "Surface", backend="d3v2p1"),
    _c("guard_relax", "Guard relaxation", "Surface", "STAGE_GUARD_RELAX", "d3v2p1"),
    _c("unpinch", "Separate point contacts", "Surface", "STAGE_UNPINCH"),
    _c("gmsh_interfaces", "Gmsh remesh interfaces", "Surface", backend="d3v2p1"),
    _c("close_rve", "Close RVE faces", "Surface"),
    _c("pre_joint_quality", "Pre-joint surface quality", "Surface", "STAGE_PRE_JOINT_QUALITY", "d3v2p1"),
    _c("joint_remesh", "Joint remesh", "Surface", backend="d3v2p1"),
    # Validation and repair
    _c("pre_tet_validation", "Pre-tet surface validation", "Validation", backend="d3v2p1"),
    _c("contact_repair", "Contact repair", "Validation", "STAGE_CONTACT_REPAIR"),
    _c("crossing_repair", "Crossing repair", "Validation", "STAGE_CROSSING_REPAIR", "d3v2p1"),
    _c("surface_repair", "Surface quality / angle repair", "Validation",
       "STAGE_SURFACE_ANGLE_REPAIR", "d3v2p1"),
    _c("surface_gate", "Revalidate + surface gate", "Validation", backend="d3v2p1"),
    # Volume
    _c("tet_mesh", "Tetrahedral meshing", "Volume", backend="d3v2p1"),
    _c("tet_smooth", "Tet dihedral smoothing", "Volume", "STAGE_TET_SMOOTH", "d3v2p1"),
    _c("tet_swaps", "Tet swaps", "Volume", "STAGE_TET_SWAPS", "d3v2p1"),
    _c("tet_smooth_after", "Smooth after swaps", "Volume", "STAGE_SMOOTH_AFTER_SWAPS", "d3v2p1"),
    _c("dihedral_gate", "Dihedral gate + swaps", "Volume", "STAGE_DIHEDRAL_GATE", "d3v2p1"),
    _c("insert_nodes", "Node insertion", "Volume", "STAGE_INSERT_NODES", "d3v2p1"),
    # Quality by region (report only)
    _c("region_quality", "Region quality", "Quality", "STAGE_REGION_QUALITY", "none"),
    # Comparison (report only)
    _c("compare_before_after", "Before/after comparison", "Comparison", "STAGE_COMPARISON",
       "none"),
    # Export
    _c("export_surfaces", "Export surfaces", "Export", kind=EXPORT),
    _c("export_volume", "Export volume mesh", "Export", kind=EXPORT),
    _c("export_abaqus", "Abaqus INP export", "Export", kind=EXPORT),
)

_BY_ID = {s.id: s for s in STAGES}


def stage_ids() -> list[str]:
    return [s.id for s in STAGES]


def stage_def(stage_id: str) -> StageDef:
    try:
        return _BY_ID[stage_id]
    except KeyError:
        raise KeyError(f"{stage_id!r} is not a stage; the stages are {', '.join(_BY_ID)}.") from None


def has_stage(stage_id: str) -> bool:
    return stage_id in _BY_ID
