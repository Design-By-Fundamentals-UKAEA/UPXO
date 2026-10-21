"""Named groups of settings for a kind of material and size of crop.

A preset is a set of values for the settings it controls. Applying one sets those settings to its
values and every other setting a preset controls back to its default, so presets can be switched
between freely; settings no preset controls are left as they are. The values come from the development
notebooks or from runs on the development crop of the FM-steel data, as each preset's notes say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, coerce_value, default_config

SIZE_CLASSES = (("small", 0, 12_000), ("medium", 12_000, 60_000), ("large", 60_000, None))
MATERIALS = ("fm_steel", "any")


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    material: str                       # one of MATERIALS
    size_class: str                     # "small", "medium", "large" or "any"
    summary: str
    settings: dict
    notes: tuple = ()
    source: str = ""
    measured: Optional[dict] = field(default=None, compare=False)   # what the development crop gave, as ratios to the default

    def describe(self) -> str:
        lines = [self.title, self.summary]
        if self.settings:
            lines.append("sets: " + ", ".join(f"{k} = {v}" for k, v in self.settings.items()))
        else:
            lines.append("sets: nothing (the defaults)")
        lines += list(self.notes)
        if self.measured:
            lines.append(measured_text(self.measured))
        if self.source:
            lines.append(f"source: {self.source}")
        return "\n".join(lines)


def measured_text(measured: dict) -> str:
    """One line on what the development crop (8,800 voxels, 28 features) gave under the preset."""
    parts = []
    if measured.get("tets_vs_default") is not None:
        parts.append(f"{measured['tets_vs_default']:.2f}x the tetrahedra of the default")
    if measured.get("time_vs_default") is not None:
        parts.append(f"{measured['time_vs_default']:.2f}x its time")
    if measured.get("tets") is not None:
        parts.append(f"{measured['tets']:,} tetrahedra")
    return "on the development crop: " + ", ".join(parts) if parts else ""


# The measured ratios are from runs of the 8,800-voxel development crop (seed 42), through node insertion.
PRESETS: dict = {p.key: p for p in (
    Preset(
        "default", "Defaults", "any", "any",
        "The settings the pipeline starts with.", {},
        notes=("Developed on the FM-steel development data.",),
        source="settings table"),
    Preset(
        "quick_look", "Quick look: coarser surface mesh", "any", "any",
        "A coarser grain-boundary surface mesh for a first look at a crop, with fewer tetrahedra.",
        {"GMSH_SIZE_FACTOR": 1.5},
        notes=("The interface and volume meshes get larger elements; thin features are resolved less well.",
               "Time did not fall measurably on the development crop: runs of one setting varied from 170 to 345 s.",
               "TET_MESH_SIZE_FACTOR is left alone: it changes the tetrahedron count only when it is smaller "
               "than the surface size, and 1.0 and 2.5 both gave the count of the default."),
        source="runs on the development crop",
        measured={"tets": 82_995, "tets_vs_default": 0.74}),
    Preset(
        "finer", "Finer mesh", "any", "small",
        "A finer surface mesh, and with it a finer volume mesh, for a small crop where the detail matters.",
        {"GMSH_SIZE_FACTOR": 0.5},
        notes=("The mesh grows quickly as the factor falls: use it on small crops.",
               "Peak memory of the main process rose from 440 MB to 592 MB on the development crop.",
               "TET_MESH_SIZE_FACTOR is left alone: a value of 1.0 on top of this gave the same mesh."),
        source="runs on the development crop",
        measured={"tets": 271_545, "tets_vs_default": 2.43}),
    Preset(
        "large_crop", "Large crop: bounded cost", "fm_steel", "large",
        "A coarser mesh and no checkpoint history for crops of 60,000 voxels or more.",
        {"GMSH_SIZE_FACTOR": 1.5, "CHECKPOINT_KEEP_HISTORY": False},
        notes=("Without history a stage that fails part-way can be resumed only from the stage before it.",
               "Check the forecast (the estimate command) before starting: time and memory grow faster than the voxel count."),
        source="runs on the development crop and a crop of 27,000 voxels",
        measured={"tets": 82_995, "tets_vs_default": 0.74}),
    Preset(
        "strict", "Strict: stop on defects it cannot clean", "fm_steel", "small",
        "The values the development notebook sets, for results that match its runs.",
        {"REMOVAL_MAX_PASSES": 10, "REQUIRE_CLEAN_TOPOLOGY": True,
         "FREEZE_THIN_GRAINS": False, "APPROVE_TET_MESHING_WITH_POOR_SURFACE": True},
        notes=("Stops at voxel boundary defects it cannot clean, and meshes through a surface below the angle target.",
               "The notebooks also set GRAIN_ID_SEED = 42; the seed is left to you so that a preset never changes it."),
        source="development notebook"),
    Preset(
        "upscaled", "Upscaled: voxel grid refined by two", "fm_steel", "small",
        "The values the upscaling notebook sets: the voxel grid is refined by two before the surfaces are made.",
        {"REMOVAL_MAX_PASSES": 10, "REQUIRE_CLEAN_TOPOLOGY": True, "UPSCALE_FACTOR": 2,
         "MIN_FACET_OPENING_DEGREES": 1.0},
        notes=("Upscaling multiplies the voxel count by 8 inside the pipeline: use it on small crops only.",
               "The notebooks also set GRAIN_ID_SEED = 42; the seed is left to you so that a preset never changes it."),
        source="upscaling notebook"),
)}


def preset_keys() -> list:
    return list(PRESETS)


def get_preset(key: str) -> Preset:
    try:
        return PRESETS[key]
    except KeyError:
        raise KeyError(f"{key!r} is not a preset; the presets are: {', '.join(PRESETS)}.") from None


def controlled_keys() -> list:
    """Every setting some preset sets."""
    seen: dict = {}
    for preset in PRESETS.values():
        for key in preset.settings:
            seen.setdefault(key, None)
    return list(seen)


def values_of(key: str) -> dict:
    """The value of every controlled setting under preset ``key`` (default where the preset says nothing)."""
    preset = get_preset(key)
    defaults = default_config()
    return {name: preset.settings.get(name, defaults[name]) for name in controlled_keys()}


def changes(config: dict, key: str) -> list:
    """``[(setting, current, new)]`` for what applying ``key`` would change in ``config``."""
    return [(name, config.get(name), value) for name, value in values_of(key).items() if config.get(name) != value]


def apply_preset(config: dict, key: str) -> list:
    """Set the controlled settings of ``config`` for preset ``key``; returns the changes made."""
    made = changes(config, key)
    for name, _old, new in made:
        config[name] = new
    return made


def which_preset(config: dict) -> Optional[str]:
    """The preset whose values ``config`` holds for every controlled setting, or ``None``."""
    for key in PRESETS:
        if not changes(config, key):
            return key
    return None


def size_class(voxels: int) -> str:
    for name, low, high in SIZE_CLASSES:
        if voxels >= low and (high is None or voxels < high):
            return name
    return "large"


def recommend(voxels: int, material: str = "fm_steel") -> list:
    """Presets that suit a crop of ``voxels`` voxels, best first."""
    wanted = size_class(voxels)
    fits = [p for p in PRESETS.values() if p.material in (material, "any") and p.size_class in (wanted, "any")]
    return sorted(fits, key=lambda p: (p.size_class != wanted, p.material != material, list(PRESETS).index(p.key)))


def validate_presets() -> None:
    """Every preset names real settings with values that pass the settings checks (raises ``ValueError``)."""
    for preset in PRESETS.values():
        for name, value in preset.settings.items():
            if name not in SCHEMA:
                raise ValueError(f"preset {preset.key}: {name} is not a setting")
            _checked, problem = coerce_value(SCHEMA[name], value)
            if problem:
                raise ValueError(f"preset {preset.key}: {problem}")
