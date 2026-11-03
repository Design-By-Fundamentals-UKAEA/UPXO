"""The UPXO Sigma3 twin/parent pipeline, run on a label field + per-grain Euler angles.

Shared between the synthetic-structure tests (known ground truth) and the local real-data runs
(no ground truth, aggregate output only) so the same code is checked against a known answer
before it is trusted on real EBSD maps.
"""
import numpy as np

from upxo.xtalphy.crystal_orientation import (
    CUBIC_CSL,
    compute_mdf_from_quats,
    csl_volume_fractions,
    identify_parent_grains,
    segregate_csl_pairs,
)

from .synthetic import grain_areas, neighbour_graph, per_pixel_stored_quats

TWIN_LABEL = "S3  (twin)"


def run(lfi: np.ndarray, euler_by_id: np.ndarray, csl_tol: float = 2.0) -> dict:
    """Run the pipeline and return mdf, csl_grains, parent_info, fractions and prop.

    ``selected_peaks`` is given directly as {'angles': [60.0]} rather than taken from
    ``detect_mdf_peaks``, so the test is of the CSL/parent/fraction logic itself and not also of
    peak detection on a histogram.
    """
    quat = per_pixel_stored_quats(lfi, euler_by_id)
    neigh = neighbour_graph(lfi)
    mdf = compute_mdf_from_quats(lfi, quat, neigh)
    csl_grains = segregate_csl_pairs(mdf, {"angles": [60.0], "indices": [0]},
                                      csl=CUBIC_CSL, csl_tol=csl_tol)
    prop = {i: {"area": a} for i, a in grain_areas(lfi).items()}
    parent_info = identify_parent_grains(csl_grains, prop) if csl_grains else {}
    fractions = csl_volume_fractions(lfi, csl_grains)
    return {"mdf": mdf, "csl_grains": csl_grains, "parent_info": parent_info,
            "fractions": fractions, "prop": prop}


def truth_pairs(truth: dict) -> set[tuple[int, int]]:
    """{(a, b)} (unordered, a < b) of every parent-twin pair in a ground-truth dict
    (see ``synthetic.insert_sigma3_twins``)."""
    return {tuple(sorted((info["parent"], tid))) for tid, info in truth.items()}


def detected_pairs(csl_grains: dict) -> set[tuple[int, int]]:
    """{(a, b)} (unordered, a < b) of every grain pair in the Sigma3 CSL bucket, if present."""
    info = csl_grains.get(TWIN_LABEL)
    if info is None or len(info["pairs"]) == 0:
        return set()
    return {tuple(sorted((int(a), int(b)))) for a, b in info["pairs"]}
