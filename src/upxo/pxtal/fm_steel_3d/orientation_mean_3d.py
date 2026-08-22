"""
orientation_mean_3d.py

Crystallographically-correct averaging of several Bunge ZXZ Euler-angle
orientations (degrees), and the specific "packet mean orientation" derived
quantity for this pipeline's hierarchy.

Why not just average the Euler angles directly
------------------------------------------------
Euler angles are cyclic/non-linear coordinates on SO(3) -- arithmetic mean
of e.g. two phi1 values either side of the 0/360 wrap, or of two
orientations related by the quaternion double-cover (q and -q represent the
same rotation), gives a physically meaningless result. The standard fix is
to average in quaternion space instead: convert each orientation to a unit
quaternion, put every quaternion on the same hemisphere (w >= 0 -- this
resolves the q/-q sign ambiguity, which is unrelated to crystal symmetry),
average the quaternion components, and renormalise. This mirrors
xtalphy.crystal_orientation.grain_avg_quats' own three-step approach
(hemisphere-fix, mean, renormalise), reused here for consistency.

Deliberately NOT performed: crystal-symmetry-equivalence search (picking
whichever of e.g. 24 cubic-symmetry-equivalent representations of each
input orientation is closest to a reference before averaging). That search
is the right move when averaging several *noisy measurements of one true
orientation* (e.g. EBSD pixels within a grain). It is the WRONG move here:
a packet's up to 6 constituent block orientations are 6 deliberately
distinct KS (Kurdjumov-Sachs) variants assigned to that packet -- forcing
them into one symmetry-equivalence bucket before averaging would silently
erase the very variant diversity being summarised.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from upxo.viz.xphy.pole_figure import (
    _euler_deg_to_matrix, matrix_to_quat, matrix_to_euler_bunge,
)
from upxo.texOps.fcc import quat_to_matrix


def mean_orientation_euler_deg(eulers_deg: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """Mean of several Bunge ZXZ Euler-angle orientations (degrees), computed
    correctly in quaternion space (see module docstring). Returns a single
    (phi1, Phi, phi2) tuple in degrees.

    Raises ValueError if `eulers_deg` is empty.
    """
    eulers_deg = np.asarray(eulers_deg, dtype=np.float64)
    if eulers_deg.size == 0:
        raise ValueError("mean_orientation_euler_deg: need at least one orientation.")
    eulers_deg = np.atleast_2d(eulers_deg)
    if eulers_deg.shape[0] == 1:
        return tuple(float(v) for v in eulers_deg[0])

    R = _euler_deg_to_matrix(eulers_deg)
    q = matrix_to_quat(R)  # already hemisphere-corrected (w >= 0) by matrix_to_quat
    q_mean = q.mean(axis=0)
    norm = np.linalg.norm(q_mean)
    q_mean = q_mean / norm if norm > 0 else q_mean
    R_mean = quat_to_matrix(q_mean[None, :])
    euler_mean = matrix_to_euler_bunge(R_mean)
    euler_mean = np.atleast_2d(euler_mean)[0]
    return (float(euler_mean[0]), float(euler_mean[1]), float(euler_mean[2]))


def compute_packet_mean_orientations(
    grain_to_blocks_map: Dict[int, List[str]],
    block_orientations: Dict[str, Tuple[float, float, float]],
) -> Dict[int, Tuple[float, float, float]]:
    """{packet_id: mean orientation (phi1, Phi, phi2) in degrees}, for every
    packet (packet_id == grain_id in this pipeline) that has at least one
    block with an assigned orientation. A packet's own KS-variant-group
    identity isn't tracked as a single orientation anywhere else in the
    model -- this is the representative summary orientation used for
    plotting/overlaying "packet orientation" on a pole figure (per the
    up-to-6-block-orientations-per-packet model)."""
    out: Dict[int, Tuple[float, float, float]] = {}
    for packet_id, block_ids in grain_to_blocks_map.items():
        eulers = [block_orientations[bid] for bid in block_ids if bid in block_orientations]
        if not eulers:
            continue
        out[int(packet_id)] = mean_orientation_euler_deg(eulers)
    return out
