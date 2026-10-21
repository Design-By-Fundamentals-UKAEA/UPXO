"""Linear-intercept grain sizes along the directions blocks are sliced in.

Block and sub-block thicknesses have to fit inside the grains they slice. These
functions measure the intercept grain size along the direction each grain's
blocks will be sliced in, or along each axis, so a thickness range can be
checked against it. Nothing is generated or modified.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from upxo.pxtal.gridops import _group_ids_by_direction, directional_intercept_grain_size

__all__ = ['prospective_slicing_normals', 'intercept_sizes_by_direction',
           'thickness_range_from_intercepts']


def prospective_slicing_normals(fm_pag) -> Dict[int, np.ndarray]:
    """Unit normal of the plane each grain's blocks would be sliced along.

    Reproduces how block generation picks candidate {111} planes for the
    packets of a PAG and cycles them over its grains, without generating any
    block. Grains without voxels are left out.
    """
    from .block_generator_3d import BlockGenerator3D

    block_gen = BlockGenerator3D()
    normals = {}
    for pag_id, grain_ids in fm_pag.clusters_dict.items():
        candidate_planes = block_gen.get_slicing_planes_for_packet(
            pag_id=pag_id, pag_orientations=fm_pag.pag_orientations)
        n_planes = len(candidate_planes)
        for packet_idx, grain_id in enumerate(grain_ids):
            if grain_id not in fm_pag.grain_locs:
                continue
            normals[grain_id] = candidate_planes[packet_idx % n_planes].unit_normal
    return normals


def intercept_sizes_by_direction(lgi: np.ndarray, voxel_size: float, normals: dict,
                                 line_budget: int = 300,
                                 phase_array: Optional[np.ndarray] = None,
                                 valid_phase_ids: Optional[Sequence[int]] = None) -> dict:
    """Intercept grain sizes along each distinct direction of ``normals``.

    Parameters
    ----------
    normals : dict
        ``{id: unit normal}``; ids with the same direction share one measurement.
    line_budget : int
        Intercept lines in total, split over the distinct directions (at
        least 20 per direction), so the cost does not grow with the number of
        directions.
    phase_array, valid_phase_ids :
        Passed on to ``directional_intercept_grain_size``.

    Returns
    -------
    dict
        ``combined`` (all intercept lengths, pooled), ``direction_means`` (mean
        length of each direction that had any), ``total_lines`` and
        ``n_directions``.
    """
    groups = _group_ids_by_direction(normals)
    n_directions = len(groups)
    per_direction_target = max(20, line_budget // max(1, n_directions))

    segments, direction_means, total_lines = [], [], 0
    extra = {}
    if phase_array is not None:
        extra = {'phase_array': phase_array, 'valid_phase_ids': valid_phase_ids}
    for direction_key in groups:
        s = directional_intercept_grain_size(
            lgi, voxel_size, np.array(direction_key), n_lines_target=per_direction_target, **extra)
        if s['n_segments'] > 0:
            segments.append(s['raw_segments'])
            direction_means.append(s['mean'])
        total_lines += s['n_lines']

    return {'combined': np.concatenate(segments) if segments else np.empty(0),
            'direction_means': direction_means, 'total_lines': total_lines,
            'n_directions': n_directions}


def thickness_range_from_intercepts(stats_by_axis: dict) -> Optional[Tuple[float, float]]:
    """Interquartile range (25th and 75th percentile) of the intercept lengths
    pooled over the x, y and z axes; None when there are none.

    ``stats_by_axis`` maps 'x', 'y' and 'z' to dicts with ``raw_segments``.
    """
    pooled = np.concatenate([stats_by_axis[axis]['raw_segments'] for axis in ('x', 'y', 'z')])
    if pooled.size == 0:
        return None
    q1, q3 = np.percentile(pooled, [25.0, 75.0])
    return float(q1), float(q3)
