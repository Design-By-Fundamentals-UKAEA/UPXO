"""Physical sizes beside the settings that are measured in voxels."""
from __future__ import annotations

from typing import Any, Optional

from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, Setting
from upxo.meshing.gbconformant.pipeline.stage_input import active_hierarchy

_TO_METRES = {"microns": 1e-6, "mm": 1e-3, "m": 1.0}
_SHOWN = (("um", 1e-6), ("mm", 1e-3), ("m", 1.0))


def voxel_edge(state) -> Optional[tuple[float, str]]:
    """``(edge length, units)`` of one raw voxel, or ``None`` before an FM-steel input is loaded."""
    hierarchy = active_hierarchy(state)
    if hierarchy is None or hierarchy.units not in _TO_METRES or not hierarchy.voxel_size:
        return None
    return float(hierarchy.voxel_size), str(hierarchy.units)


def _shown_unit(edge_m: float) -> tuple[str, float]:
    """The unit (name, metres) in which a length of about ``edge_m`` reads best."""
    for name, size in _SHOWN:
        if edge_m < 1000 * size:
            return name, size
    return _SHOWN[-1]


def format_length(metres: float, shown: Optional[tuple[str, float]] = None) -> str:
    name, size = shown or _shown_unit(abs(metres) if metres else 1e-6)
    return f"{metres / size:.4g} {name}"


def physical_text(state, setting: Setting | str, value: Any) -> str:
    """The physical meaning of ``value`` for ``setting`` (``""`` when it has none or the voxel size is unknown)."""
    setting = SCHEMA[setting] if isinstance(setting, str) else setting
    if not setting.unit or value is None or isinstance(value, bool):
        return ""
    edge = voxel_edge(state)
    if edge is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    size, units = edge
    edge_m = size * _TO_METRES[units]
    if setting.unit in ("length", "factor"):
        return f"{number:g} x voxel edge = {format_length(number * edge_m)}"
    if setting.unit == "thickness":
        return f"{number:g} voxels across = {format_length(number * edge_m)}"
    if setting.unit == "voxels":
        volume = number * edge_m ** 3
        name, unit_m = _shown_unit(volume ** (1.0 / 3.0) if volume > 0 else 1e-6)
        return f"{number:g} voxels = {volume / unit_m ** 3:.4g} {name}^3"
    return ""
