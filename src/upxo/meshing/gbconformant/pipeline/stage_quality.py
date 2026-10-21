"""The region-quality stage: the quality of the elements at the grain boundary surface, the junction lines and the
junction points of the finished tetrahedral mesh (report only; nothing downstream depends on it)."""
from __future__ import annotations

from upxo.meshing.gbconformant.pipeline.region_quality import (
    region_quality, region_settings, region_tables, summary_numbers, write_region_files)
from upxo.meshing.gbconformant.pipeline.runfiles import ensure_run_dir
from upxo.meshing.gbconformant.pipeline.state import PipelineState

REGION_KEY = "region_quality.result"
FOLDER = "region_quality"


def region_text(result: dict) -> str:
    """The stage's tables as text, with what was found above them."""
    from upxo.meshing.gbconformant.pipeline.results import Block

    c, s = result["counts"], result["settings"]
    lines = [f"Found {c['junction_lines']} junction lines ({c['junction_edges']} mesh edges) and "
             f"{c['junction_points']} junction points; {c['grain_boundary_triangles']} grain boundary triangles.",
             f"Layers from each feature: {s['order']} ({s['mode']}); an element touches a feature when it shares "
             f"{'a face (surface) or an edge (lines)' if s['touch'] == 'face or edge' else 'a node'}.",
             ""]
    if c.get("excluded_tetrahedra") or c.get("excluded_triangles"):
        lines.insert(2, f"Left out of the grain boundary surface: {c['excluded_tetrahedra']} tetrahedra and "
                        f"{c['excluded_triangles']} triangles.")
    return "\n".join(lines) + "\n\n".join(Block(*table).text() for table in region_tables(result))


def handle_region_quality(state: PipelineState) -> dict:
    cfg = state.config
    tets = state.get_artefact("verified_tets")
    if tets is None:
        raise RuntimeError("verified_tets missing — run tetrahedral meshing first.")
    result, arrays = region_quality(tets.points, tets.tetrahedra, tets.grain_ids, getattr(tets, "quality", None),
                                    region_settings(cfg))
    state.store_artefact(REGION_KEY, result)
    files = []
    if cfg.get("REGION_WRITE_FILES", True):
        files = write_region_files(ensure_run_dir(cfg) / FOLDER, tets.points, tets.tetrahedra, tets.grain_ids,
                                   result, arrays)
    return {**summary_numbers(result), "files": files, "text": region_text(result)}
