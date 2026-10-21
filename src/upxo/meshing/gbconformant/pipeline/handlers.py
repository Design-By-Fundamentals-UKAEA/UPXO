"""The handler of every stage, by stage id."""
from __future__ import annotations

from typing import Callable

from upxo.meshing.gbconformant.pipeline.stage_cleaning import (
    handle_freeze_thin, handle_pinches, handle_small_grains, handle_spikes, handle_upscale,
    handle_voxel_topology,
)
from upxo.meshing.gbconformant.pipeline.stage_input import (
    handle_grain_id_map, handle_load_lgi, handle_subset_extract,
)
from upxo.meshing.gbconformant.pipeline.stage_output import (
    handle_compare_before_after, handle_export_abaqus, handle_export_surfaces,
    handle_export_volume,
)
from upxo.meshing.gbconformant.pipeline.stage_quality import handle_region_quality
from upxo.meshing.gbconformant.pipeline.stage_surface import (
    handle_close_rve, handle_contact_repair, handle_crossing_repair, handle_geometry_guard,
    handle_gmsh_interfaces, handle_guard_relax, handle_joint_remesh, handle_pre_joint_quality,
    handle_pre_tet_validation, handle_smooth_interfaces, handle_surface_gate,
    handle_surface_repair, handle_unpinch,
)
from upxo.meshing.gbconformant.pipeline.stage_volume import (
    handle_dihedral_gate, handle_insert_nodes, handle_tet_mesh, handle_tet_smooth,
    handle_tet_smooth_after, handle_tet_swaps,
)
from upxo.meshing.gbconformant.pipeline.state import PipelineState

Handler = Callable[[PipelineState], dict]


def default_handlers() -> dict[str, Handler]:
    """The handler of every stage."""
    return {
        "load_lgi": handle_load_lgi,
        "subset_extract": handle_subset_extract,
        "grain_id_map": handle_grain_id_map,
        "small_grains": handle_small_grains,
        "spikes": handle_spikes,
        "pinches": handle_pinches,
        "voxel_topology": handle_voxel_topology,
        "freeze_thin": handle_freeze_thin,
        "upscale": handle_upscale,
        "smooth_interfaces": handle_smooth_interfaces,
        "geometry_guard": handle_geometry_guard,
        "guard_relax": handle_guard_relax,
        "unpinch": handle_unpinch,
        "gmsh_interfaces": handle_gmsh_interfaces,
        "close_rve": handle_close_rve,
        "pre_joint_quality": handle_pre_joint_quality,
        "joint_remesh": handle_joint_remesh,
        "pre_tet_validation": handle_pre_tet_validation,
        "contact_repair": handle_contact_repair,
        "crossing_repair": handle_crossing_repair,
        "surface_repair": handle_surface_repair,
        "surface_gate": handle_surface_gate,
        "tet_mesh": handle_tet_mesh,
        "tet_smooth": handle_tet_smooth,
        "tet_swaps": handle_tet_swaps,
        "tet_smooth_after": handle_tet_smooth_after,
        "dihedral_gate": handle_dihedral_gate,
        "insert_nodes": handle_insert_nodes,
        "region_quality": handle_region_quality,
        "compare_before_after": handle_compare_before_after,
        "export_surfaces": handle_export_surfaces,
        "export_volume": handle_export_volume,
        "export_abaqus": handle_export_abaqus,
    }
