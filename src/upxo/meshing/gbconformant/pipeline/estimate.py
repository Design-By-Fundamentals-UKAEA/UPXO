"""A forecast of a run's size, time and memory before it starts.

The forecast is a power law in the number of voxels, scaled by the surface and volume size
factors, with constants fitted to runs of the FM-steel development data (a crop of 8,800 voxels
under eight settings, and a crop of 27,000 voxels). When the data directory holds earlier runs of
this computer, the forecast is scaled by how far those runs differ from it, which also absorbs the
speed of the computer. The result is a range: the mesh depends on the grain shapes, which no
formula sees, and the time of a run changed by up to 1.9 times between runs of one computer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from upxo.meshing.gbconformant.pipeline.settings import default_config

# Development crop: 8,800 voxels, 28 features, grain id seed 42, default settings through node insertion.
REFERENCE_VOXELS = 8800
REFERENCE_TETS = 111_628
REFERENCE_SECONDS = 245.0          # mean of ten runs on the crop, 171 to 345 s
NODES_PER_TET = 0.1994
TETS_VOXEL_EXPONENT = 0.91         # 111,628 tets at 8,800 voxels, 308,481 at 27,000
SECONDS_VOXEL_EXPONENT = 0.80      # about 245 s at 8,800 voxels, about 590 s at 27,000
MEMORY_BASE_MB = 330.0             # main process: the libraries and the label array
MEMORY_MB_PER_TET = 0.00096        # 410 MB at 83,000 tets, 591 MB at 272,000
GMSH_FACTOR_CURVE = ((0.5, 2.4326), (0.75, 1.0), (1.0, 0.7911), (1.5, 0.7435))   # tets relative to the default
GMSH_FACTOR_RANGE = (0.5, 1.5)     # the measured range; outside it the curve is extended with its end slopes
TET_FACTOR_EXPONENT = 1.9          # tets grow as (surface factor / volume factor) ** 1.9 once the volume size is smaller
MIN_SPREAD = 0.25                  # the range is never narrower than a factor of exp(0.25) either way
MIN_TIME_SPREAD = 0.35


@dataclass
class Observation:
    """What an earlier run turned out to be."""

    name: str
    voxels: int
    tets: Optional[int] = None
    nodes: Optional[int] = None
    seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    tet_size_factor: Optional[float] = None
    gmsh_size_factor: Optional[float] = None
    complete: bool = False


@dataclass
class Forecast:
    """The expected size of a run; every figure comes with the range it is likely to fall in."""

    voxels: int
    tets: int
    tets_range: tuple
    nodes: int
    seconds: float
    seconds_range: tuple
    memory_mb: float
    memory_range: tuple
    basis: str
    runs_used: int = 0
    notes: list = field(default_factory=list)
    output: Optional[dict] = None             # size figures of the analysis output, see fm_steel_abaqus.outputs
    output_range: Optional[tuple] = None      # megabytes

    def as_dict(self) -> dict:
        return {
            "voxels": self.voxels, "tets": self.tets, "tets_range": list(self.tets_range), "nodes": self.nodes,
            "seconds": self.seconds, "seconds_range": list(self.seconds_range), "memory_mb": self.memory_mb,
            "memory_range": list(self.memory_range), "basis": self.basis, "runs_used": self.runs_used,
            "notes": list(self.notes), "output": self.output,
            "output_range": None if self.output_range is None else list(self.output_range),
        }

    def text(self) -> str:
        """The forecast as one short paragraph for a readout or the command line."""
        lines = [f"about {_count(self.tets)} tetrahedra ({_count(self.tets_range[0])} to {_count(self.tets_range[1])}), "
                 f"{_count(self.nodes)} nodes",
                 f"about {duration_text(self.seconds)} ({duration_text(self.seconds_range[0])} to "
                 f"{duration_text(self.seconds_range[1])})",
                 f"about {memory_text(self.memory_mb)} of memory in the main process at the peak "
                 f"({memory_text(self.memory_range[1])} at most); worker processes add their own"]
        if self.output is not None:
            from upxo.meshing.gbconformant.fm_steel_abaqus.outputs import text as output_text
            lines.append(output_text(self.output) + f" ({memory_text(self.output_range[0])} to "
                                                    f"{memory_text(self.output_range[1])})")
        lines.append(self.basis)
        return "\n".join(lines + self.notes)


def _count(value) -> str:
    value = float(value)
    if value >= 1e6:
        return f"{value / 1e6:.2f} million"
    if value >= 1e4:
        return f"{round(value / 1e3):,} thousand"
    return f"{round(value):,}"


def duration_text(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{round(seconds)} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def memory_text(megabytes: float) -> str:
    return f"{megabytes / 1024:.1f} GB" if megabytes >= 1024 else f"{megabytes:.0f} MB"


# --- the model ---------------------------------------------------------------------------------------------------
def _curve(points, x: float) -> float:
    """Log-log interpolation through ``points``, extended past the ends with the slope of the last segment."""
    xs = [math.log(p[0]) for p in points]
    ys = [math.log(p[1]) for p in points]
    lx = math.log(max(x, 1e-9))
    if lx <= xs[0]:
        i = 0
    elif lx >= xs[-1]:
        i = len(xs) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= lx)
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return math.exp(ys[i] + slope * (lx - xs[i]))


def _sizes(tet_size_factor, gmsh_size_factor) -> tuple:
    defaults = default_config()
    tet = tet_size_factor if tet_size_factor else defaults["TET_MESH_SIZE_FACTOR"]
    gmsh = gmsh_size_factor if gmsh_size_factor else defaults["GMSH_SIZE_FACTOR"]
    return float(tet), float(gmsh)


def model_tets(voxels: float, tet_size_factor: Optional[float] = None, gmsh_size_factor: Optional[float] = None) -> float:
    """Tetrahedra the fitted law gives for ``voxels`` at the given size factors (default settings when omitted).

    The surface size factor sets the count; the volume size factor adds tetrahedra only when its size is
    smaller than the surface size (the interior is then meshed finer than the boundary).
    """
    tet, gmsh = _sizes(tet_size_factor, gmsh_size_factor)
    finer_inside = max(1.0, gmsh / tet) ** TET_FACTOR_EXPONENT
    return (REFERENCE_TETS * (max(float(voxels), 1.0) / REFERENCE_VOXELS) ** TETS_VOXEL_EXPONENT
            * _curve(GMSH_FACTOR_CURVE, gmsh) * finer_inside)


def model_seconds(voxels: float) -> float:
    """Time of a run: it followed the number of voxels, not the number of tetrahedra, on the crops measured."""
    return REFERENCE_SECONDS * (max(float(voxels), 1.0) / REFERENCE_VOXELS) ** SECONDS_VOXEL_EXPONENT


def model_memory_mb(tets: float) -> float:
    """Peak memory of the main process; worker processes add their own."""
    return MEMORY_BASE_MB + MEMORY_MB_PER_TET * max(float(tets), 0.0)


def _log_mean_and_spread(ratios: list) -> tuple:
    logs = [math.log(r) for r in ratios if r and r > 0]
    if not logs:
        return 1.0, MIN_SPREAD
    mean = sum(logs) / len(logs)
    spread = (math.sqrt(sum((v - mean) ** 2 for v in logs) / (len(logs) - 1)) if len(logs) > 1 else 0.0)
    return math.exp(mean), max(MIN_SPREAD, spread)


def forecast(voxels: int, config: Optional[dict] = None, history: Iterable[Observation] = ()) -> Forecast:
    """The size, time and memory to expect for a run of ``voxels`` with ``config`` (default settings when omitted).

    ``history`` holds earlier runs (see :func:`load_observations`); those that finished are used to
    scale the fitted law to this computer and to this kind of data.
    """
    config = default_config() if config is None else config
    voxels = max(int(voxels), 1)
    tet_factor, gmsh_factor = config.get("TET_MESH_SIZE_FACTOR"), config.get("GMSH_SIZE_FACTOR")
    notes = []
    if config.get("TET_MESH_SIZE") or config.get("GMSH_SIZE") or config.get("FINAL_GMSH_SIZE"):
        notes.append("An explicit mesh size is set; the forecast assumes the size the factors would give.")
    gmsh_used = _sizes(tet_factor, gmsh_factor)[1]
    if not GMSH_FACTOR_RANGE[0] <= gmsh_used <= GMSH_FACTOR_RANGE[1]:
        notes.append(f"GMSH_SIZE_FACTOR {gmsh_used:g} is outside the measured range "
                     f"{GMSH_FACTOR_RANGE[0]:g} to {GMSH_FACTOR_RANGE[1]:g}; the tetrahedron count is extrapolated.")
    base_tets = model_tets(voxels, tet_factor, gmsh_factor)

    used = [o for o in history if o.complete and o.tets and o.voxels]
    tet_scale, tet_spread = _log_mean_and_spread(
        [o.tets / model_tets(o.voxels, o.tet_size_factor, o.gmsh_size_factor) for o in used])
    tets = base_tets * tet_scale
    timed = [o for o in used if o.seconds]
    time_scale, time_spread = _log_mean_and_spread([o.seconds / model_seconds(o.voxels) for o in timed])
    time_spread = max(time_spread, MIN_TIME_SPREAD)
    seconds = model_seconds(voxels) * time_scale
    sized = [o for o in used if o.peak_memory_mb]
    memory_scale, memory_spread = _log_mean_and_spread([o.peak_memory_mb / model_memory_mb(o.tets) for o in sized])
    memory_spread = min(memory_spread, 0.5)
    memory = model_memory_mb(tets) * memory_scale

    if used:
        basis = f"scaled from {len(used)} earlier run{'s' if len(used) != 1 else ''} in the data folder"
    else:
        basis = ("from the built-in fit to the FM-steel development data; no earlier run here to scale it by, "
                 "and the time of one run varies a lot")
    if len(used) == 1:
        tet_spread = max(tet_spread, 0.35)
    low_t, high_t = math.exp(-tet_spread), math.exp(tet_spread)
    low_s, high_s = math.exp(-math.hypot(tet_spread, time_spread)), math.exp(math.hypot(tet_spread, time_spread))
    low_m, high_m = math.exp(-math.hypot(tet_spread / 2, memory_spread)), math.exp(math.hypot(tet_spread / 2, memory_spread))
    output, output_range = _output_figures(config, tets, notes, low_t, high_t)
    return Forecast(
        voxels=voxels, tets=int(round(tets)), tets_range=(int(tets * low_t), int(tets * high_t)),
        nodes=int(round(tets * NODES_PER_TET)), seconds=seconds, seconds_range=(seconds * low_s, seconds * high_s),
        memory_mb=memory, memory_range=(memory * low_m, memory * high_m), basis=basis, runs_used=len(used),
        notes=notes, output=output, output_range=output_range)


def _output_figures(config: dict, tets: float, notes: list, low: float, high: float):
    """Size of the analysis output for the load case and outputs in ``config`` (``None`` when they cannot be read)."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import outputs
    from upxo.meshing.gbconformant.fm_steel_abaqus.loading import plan_load
    from upxo.meshing.gbconformant.pipeline.stage_output import deck_settings

    try:
        settings = deck_settings(config)
        settings.validate()
        figures = outputs.estimate_for_plan(settings, plan_load(settings, 1.0), int(round(tets * NODES_PER_TET)),
                                            int(round(tets)))
    except (ValueError, KeyError) as error:
        notes.append(f"The output size is not estimated: {error}")
        return None, None
    if settings.output_elset:
        notes.append(f"The output size is for the whole model; limiting element output to {settings.output_elset} makes it smaller.")
    size = figures["megabytes"]
    return figures, (size * low, size * high)


# --- earlier runs -----------------------------------------------------------------------------------------------
def observation_from_folder(folder, manifest: Optional[dict] = None) -> Optional[Observation]:
    """What the run in ``folder`` was, or ``None`` when it holds no manifest or no voxel count.

    ``manifest`` is the folder's already-read ``run_manifest.json``, if the caller has it.
    """
    folder = Path(folder)
    if manifest is None:
        try:
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            return None
    results = manifest.get("results") or {}
    config = manifest.get("config") or {}
    box = ((manifest.get("subset") or {}).get("box") or {})
    voxels = results.get("voxels") or box.get("voxels")
    if not voxels:
        return None
    tets, nodes = results.get("tets"), results.get("nodes")
    if not tets:                                                  # manifests made before results were recorded
        tets = None
        for name in ("verified_tets.json", "CM04_dev_verified_tets.json"):   # the second name: runs made earlier
            try:
                tets = json.loads((folder / name).read_text(encoding="utf8")).get("tetrahedra")
                break
            except (OSError, ValueError, AttributeError):
                continue
    seconds = results.get("seconds")
    if seconds is None:
        durations = [t.get("duration_s") for t in manifest.get("stage_timings") or []]
        seconds = round(sum(d for d in durations if d), 2) or None
    saved = ((manifest.get("saved_results") or {}).get("stages") or {})
    complete = any(stage in saved for stage in ("tet_smooth_after", "dihedral_gate", "insert_nodes", "export_abaqus"))
    return Observation(
        name=folder.name, voxels=int(voxels), tets=int(tets) if tets else None, nodes=nodes, seconds=seconds,
        peak_memory_mb=results.get("peak_memory_mb"),
        tet_size_factor=config.get("TET_MESH_SIZE_FACTOR"), gmsh_size_factor=config.get("GMSH_SIZE_FACTOR"),
        complete=complete)


def load_observations(root: Optional[Path] = None) -> list:
    """The earlier runs under ``root`` (default: the data directory) that tell something about size and time."""
    from upxo.meshing.gbconformant.pipeline.runs import runs_directory
    base = runs_directory(root)
    if not base.is_dir():
        return []
    out = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            observation = observation_from_folder(folder)
            if observation is not None:
                out.append(observation)
    return out
