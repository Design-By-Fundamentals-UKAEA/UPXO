"""Which stages' per-iteration history is collected, grouped by the config flag that gates it.

Each iterative stage belongs to one of the four pipeline groups (Cleaning, Surface, Validation,
Volume). Its ``COMPUTE_ITERATION_METRICS_<GROUP>`` setting decides whether that stage's algorithm
builds a per-pass ``history`` list; when off, the stage still runs but the list stays empty, so
nothing downstream can read a per-pass history that was never built.
"""
from __future__ import annotations

GROUP_FLAGS: dict[str, str] = {
    "CLEANING": "COMPUTE_ITERATION_METRICS_CLEANING",
    "SURFACE": "COMPUTE_ITERATION_METRICS_SURFACE",
    "VALIDATION": "COMPUTE_ITERATION_METRICS_VALIDATION",
    "VOLUME": "COMPUTE_ITERATION_METRICS_VOLUME",
}

STAGE_GROUPS: dict[str, str] = {
    # Cleaning
    "small_grains": "CLEANING",
    "spikes": "CLEANING",
    "pinches": "CLEANING",
    "voxel_topology": "CLEANING",
    # Surface
    "smooth_interfaces": "SURFACE",
    "geometry_guard": "SURFACE",
    "guard_relax": "SURFACE",
    "unpinch": "SURFACE",
    "pre_joint_quality": "SURFACE",
    # Validation
    "crossing_repair": "VALIDATION",
    "surface_repair": "VALIDATION",
    # Volume
    "tet_smooth": "VOLUME",
    "tet_swaps": "VOLUME",
    "tet_smooth_after": "VOLUME",
    "dihedral_gate": "VOLUME",
    "insert_nodes": "VOLUME",
}


def metrics_enabled(config: dict, stage_id: str) -> bool:
    """Whether ``stage_id`` should build its per-iteration history, per its group's flag."""
    group = STAGE_GROUPS.get(stage_id)
    if group is None:
        return False
    return bool(config.get(GROUP_FLAGS[group], True))


def has_iteration_history(stage_id: str) -> bool:
    """Whether ``stage_id`` is one that can carry a per-iteration history at all."""
    return stage_id in STAGE_GROUPS
