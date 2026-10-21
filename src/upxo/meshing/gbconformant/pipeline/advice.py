"""What to try when a stage fails: settings that bear on the failure it reported.

Each rule names a stage (or any), a pattern in the error text, and the settings to change. The advice
only proposes; nothing changes unless the person applies it. A proposal is dropped when the setting
already has the proposed value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, coerce_value
from upxo.meshing.gbconformant.pipeline.stages import stage_def


@dataclass(frozen=True)
class Suggestion:
    """One proposed change; ``setting`` is ``None`` for a hint that is not a setting."""

    setting: Optional[str]
    value: Any
    why: str

    @property
    def stage(self) -> Optional[str]:
        """The first stage whose results depend on the setting (a change restarts from there)."""
        setting = SCHEMA.get(self.setting) if self.setting else None
        return setting.stages[0] if setting and setting.stages else None

    def text(self) -> str:
        if self.setting is None:
            return self.why
        shown = "automatic" if self.value is None else self.value
        return f"{self.setting} = {shown}: {self.why}"


Rule = tuple  # (stage or None, regex, [(setting, value or callable(config) -> value, why), ...])


def _more(factor: float, limit: Optional[float] = None, integer: bool = False):
    def compute(config: dict, key: str):
        value = config.get(key)
        if value is None:
            return None
        new = value * factor
        if limit is not None:
            new = min(new, limit)
        return int(round(new)) if integer else float(f"{new:.4g}")
    return compute


_NEXT_ALGORITHM = {6: 5, 5: 1, 1: 2, 2: 6}


def _next_algorithm(config: dict, key: str):
    return _NEXT_ALGORITHM.get(int(config.get(key) or 6), 5)


RULES: list[Rule] = [
    ("voxel_topology", r"Unresolved voxel boundary defects", [
        ("REQUIRE_CLEAN_TOPOLOGY", False, "carry on with the defects that remain instead of stopping"),
        ("TOPOLOGY_MAX_PASSES", _more(2, integer=True), "allow more cleaning passes"),
        ("TOPOLOGY_MAX_VOLUME_FRACTION", _more(1.5, 1.0), "allow the cleaning to change more volume"),
    ]),
    ("geometry_guard", r"exhausted passes", [
        ("GEOMETRY_MAX_PASSES", _more(2, integer=True), "allow more rollback passes"),
        ("LAPLACE_ITERATIONS", _more(0.5, integer=True), "smooth less, so there is less to roll back"),
    ]),
    ("geometry_guard", r"exceeds max_correction", [
        ("GEOMETRY_MAX_CORRECTION", None, "let the guard decide how far it may move nodes"),
        ("LAPLACE_ITERATIONS", _more(0.5, integer=True), "smooth less"),
    ]),
    ("geometry_guard", r"intersection check failed", [
        ("LAPLACE_ITERATIONS", _more(0.5, integer=True), "smooth less"),
        ("LAPLACE_RELAXATION", _more(0.6), "smooth in smaller steps"),
    ]),
    ("guard_relax", r"added facet openings|intersection check failed", [
        ("GUARD_RELAX_ITERATIONS", 1, "relax for fewer iterations"),
        ("RELAX_GUARD_ROLLBACK", False, "skip the relaxation"),
    ]),
    (None, r"conflicts with frozen grains|Frozen|frozen", [
        ("THIN_THICKNESS_VOXELS", 1, "freeze only the thinnest grains"),
        ("FREEZE_THIN_GRAINS", False, "do not freeze thin grains"),
    ]),
    ("pre_tet_validation", r"Surface validation blocked", [
        ("STOP_ON_VALIDATION_ERRORS", False, "continue to the repair stages, which may fix what the check found"),
    ]),
    ("surface_gate", r"minimum triangle angle", [
        ("SURFACE_ANGLE_MAX_PASSES", _more(2, integer=True), "allow more angle-repair passes"),
        ("APPROVE_TET_MESHING_WITH_POOR_SURFACE", True, "mesh the volume with the surface as it is"),
        ("SURFACE_MIN_ANGLE_DEGREES", lambda c, k: max(0.0, float(c.get(k, 30.0)) - 5.0),
         "lower the angle the gate asks for"),
    ]),
    ("surface_gate", r"not resolved all surface defects", [
        ("SURFACE_QUALITY_MAX_PASSES", _more(2, integer=True), "allow more quality-repair passes"),
        ("SURFACE_ANGLE_MAX_PASSES", _more(2, integer=True), "allow more angle-repair passes"),
    ]),
    ("export_abaqus", r"below quality thresholds", [
        ("REQUIRE_READY_TET_MESH", False, "export the mesh as it is"),
        ("TET_SMOOTHING_PASSES", _more(2, integer=True), "smooth the tetrahedra for more passes"),
    ]),
    ("export_abaqus", r"Deck check failed", [
        ("REQUIRE_DECK_CHECK", False, "write the deck anyway and read deck_check.json for what failed"),
    ]),
    ("export_abaqus", r"output_elset", [
        ("FM_OUTPUT_ELSET", "", "write element output for the whole model"),
    ]),
    ("export_abaqus", r"Select the FM-steel deck", [
        ("EXPORT_FM_DECK", True, "write the FM-steel deck"),
    ]),
    ("export_abaqus", r"needs an FM-steel raw folder", [
        ("EXPORT_MESH_ONLY", True, "write the mesh-only deck, which needs no raw folder"),
    ]),
    ("gmsh_interfaces", r".", [
        ("GMSH_ALGORITHM", _next_algorithm, "try another Gmsh surface algorithm"),
        ("GMSH_SIZE_FACTOR", _more(1.25), "try a coarser interface mesh"),
    ]),
    ("joint_remesh", r".", [
        ("FINAL_GMSH_ALGORITHM", _next_algorithm, "try another Gmsh surface algorithm"),
        ("GMSH_SIZE_FACTOR", _more(1.25), "try a coarser surface mesh"),
    ]),
    ("tet_mesh", r".", [
        ("TET_MESH_SIZE_FACTOR", _more(1.25), "try coarser tetrahedra"),
        ("TET_OPTIMIZE_NETGEN", False, "skip the Netgen optimisation"),
    ]),
]

_RUN_FIRST = re.compile(r"(\w+) missing[^.]*run ([\w ]+) first", re.IGNORECASE)


def suggest(stage_id: str, error: str, config: dict) -> list[Suggestion]:
    """The changes worth trying after ``stage_id`` failed with ``error`` (an empty list when nothing is known)."""
    stage_def(stage_id)                                           # a typo is an error, not silence
    text = str(error or "")
    out: list[Suggestion] = []
    seen = set()
    for stage, pattern, changes in RULES:
        if stage not in (None, stage_id) or not re.search(pattern, text):
            continue
        for setting, value, why in changes:
            if setting not in SCHEMA or setting in seen:
                continue
            new = value(config, setting) if callable(value) else value
            current = config.get(setting)
            checked, problem = coerce_value(SCHEMA[setting], new)
            if problem or checked == current:
                continue
            seen.add(setting)
            out.append(Suggestion(setting, checked, why))
    match = _RUN_FIRST.search(text)
    if match:
        out.append(Suggestion(None, None, f"{match.group(1)} is missing: run {match.group(2).strip()} first"))
    return out


def apply_suggestion(config: dict, suggestion: Suggestion) -> None:
    """Make the change (the caller invalidates the stages that depend on it)."""
    if suggestion.setting is None:
        raise ValueError("This suggestion is not a setting.")
    config[suggestion.setting] = suggestion.value
